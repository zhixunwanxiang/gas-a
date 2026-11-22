"""Developer CPU controls on authorized train/validation; NEVER an SDK candidate.

Compare full training with a parent-preserving response-shape feature prototype.
No DAT test files, provider calls, deployment or model selection on future labels.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.data import load_model, load_pair, score_predictions
from gas_a.io import CLASSES, digest, write_json
from gas_a.optimize import classification_delta, fit_weighted
from gas_a.candidates import assess_candidate
from gas_a.pipeline import load_config
from gas_a import robustness


def response_shape(raw):
    values = raw.copy().reshape(-1, 16, 8)
    values[:, :, 1] -= 1  # UCI #224: maximum/baseline resistance ratio.
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.divide(values, maximum, out=np.zeros_like(values), where=maximum > 0)
    norm = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return (4 * np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0)).reshape(-1, 128)


def kinetic_shape(raw):
    result = response_shape(raw).reshape(-1, 16, 8)
    source = raw.reshape(-1, 16, 8)
    denominator = np.abs(source[:, :, :1])
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        ratios = np.divide(source[:, :, 2:], denominator, out=np.zeros_like(source[:, :, 2:]), where=denominator > 0)
        ratios /= np.asarray([.001, .01, .1, .001, .01, .1])
    result[:, :, 2:] = np.clip(ratios, -10, 10)
    return result.reshape(-1, 128)


class Prototype:
    def __init__(self, parent, expanded):
        self.meta, self.mean, self.scale = copy.deepcopy(parent.meta), parent.mean.copy(), parent.scale.copy()
        self.expanded = expanded
        self.adapter_mean, self.adapter_scale = None, None
        for name in ("params", "first", "second"):
            values = {key: value.copy() for key, value in getattr(parent, name).items()}
            if expanded is True:
                values["w1"] = np.concatenate([values["w1"], np.zeros_like(values["w1"])], axis=0)
            elif expanded == 'blend':
                values['gate']=np.zeros(8)
            elif isinstance(expanded, str):
                values.update(wa=np.zeros((128, 6)), ba=np.zeros(6))
            setattr(self, name, values)

    def normalized(self, raw):
        base = np.clip((raw-self.mean)/self.scale, -10, 10)
        if self.expanded:
            extra = kinetic_shape(raw) if self.expanded == "kinetic" else response_shape(raw)
            if self.adapter_mean is not None:
                extra = np.clip((extra-self.adapter_mean)/self.adapter_scale, -10, 10)
            return np.concatenate([base, extra], axis=1)
        return base

    def probability(self, x):
        if 'gate' in self.params:
            base=x[:,:128]+np.tile(self.params['gate'],16)*(x[:,128:]-x[:,:128])
        else:base = x[:, :128] if "wa" in self.params else x
        hidden = np.tanh(base @ self.params["w1"] + self.params["b1"])
        logits = hidden @ self.params["w2"] + self.params["b2"]
        if "wa" in self.params:
            logits += x[:, 128:] @ self.params["wa"] + self.params["ba"]
        value = np.exp(logits-logits.max(axis=1, keepdims=True))
        return hidden, value / value.sum(axis=1, keepdims=True)

    def predict(self, raw):
        return self.probability(self.normalized(raw))[1]

    def epoch(self, x, y, spec):
        weights = (len(y)/(6*np.maximum(np.bincount(y, minlength=6), 1)))**spec["balance"]
        weights[3] *= spec.get("target_boost", 1)
        weights /= weights[y].mean()
        rng = np.random.default_rng(np.random.SeedSequence([self.meta["seed"], self.meta["epochs"]]))
        order = rng.permutation(len(x))
        for start in range(0, len(x), spec["batch_size"]):
            ids = order[start:start+spec["batch_size"]]
            batch, target = x[ids], y[ids]
            if spec.get('gain',0):
                aug=np.random.default_rng(np.random.SeedSequence([self.meta['seed'],self.meta['epochs'],start,2718]))
                raw=self.raw_training[ids].copy().reshape(-1,16,8);raw[:,:,1]-=1
                sigma=spec['gain'];raw*=np.exp(aug.normal(-sigma*sigma/2,sigma,(len(ids),1,1)));raw[:,:,1]+=1
                perturbed=self.normalized(raw.reshape(-1,128))
                batch=np.where(aug.random((len(ids),1))<.25,batch,perturbed)
            hidden, probability = self.probability(batch)
            _, delta = classification_delta(probability, target, weights[target])
            gradient = {"w2": hidden.T @ delta + spec.get("l2", 0)*self.params["w2"], "b2": delta.sum(axis=0)}
            dh = (delta @ self.params["w2"].T)*(1-hidden**2)
            if 'gate' in self.params:
                base=batch[:,:128]+np.tile(self.params['gate'],16)*(batch[:,128:]-batch[:,:128])
            else:base = batch[:, :128] if "wa" in self.params else batch
            gradient.update(w1=base.T @ dh + spec.get("l2", 0)*self.params["w1"], b1=dh.sum(axis=0))
            if 'gate' in self.params:
                di=dh@self.params['w1'].T
                gradient['gate']=(di*(batch[:,128:]-batch[:,:128])).reshape(-1,16,8).sum(axis=(0,1))
            if "wa" in self.params:
                aux_logits=batch[:,128:]@self.params['wa']+self.params['ba']
                aux_prob=np.exp(aux_logits-aux_logits.max(axis=1,keepdims=True));aux_prob/=aux_prob.sum(axis=1,keepdims=True)
                _,aux_delta=classification_delta(aux_prob,target,weights[target])
                head_delta=delta+spec.get('aux_weight',0)*aux_delta
                gradient.update(wa=batch[:, 128:].T @ head_delta+spec.get('adapter_l2',0)*self.params['wa'], ba=head_delta.sum(axis=0))
            norm = math.sqrt(sum(float(np.sum(g*g)) for g in gradient.values()))
            self.meta["steps"] += 1
            step = self.meta["steps"]
            for name, value in gradient.items():
                value *= min(1., 5/max(norm, 1e-12))
                self.first[name] = .9*self.first[name]+.1*value
                self.second[name] = .999*self.second[name]+.001*value**2
                first = self.first[name]/(1-.9**step)
                second = self.second[name]/(1-.999**step)
                self.params[name] -= spec["learning_rate"]*first/(np.sqrt(second)+1e-8)
                if name=='gate':self.params[name]=np.clip(self.params[name],0,1)
        self.meta["epochs"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve();out.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/a.json")
    train, val = load_pair(config["train"], config["validation"])
    parent = load_model(config["baseline"])
    before = score_predictions(parent.predict(val["x"]), val["y"])
    # Check the control trainer against production Adam before trusting the probe.
    control = Prototype(parent, False)
    spec = {"learning_rate": .003, "batch_size": 32, "balance": 0., "l2": 0.}
    control.epoch(control.normalized(train["x"]), train["y"], spec)
    reference = copy.deepcopy(parent)
    fit_weighted(reference, train, {**spec, "weak_boost": 1., "epochs": 1}, "Acetaldehyde", time.monotonic()+30)
    error = max(float(np.max(np.abs(getattr(control, g)[k]-v))) for g in ("params", "first", "second") for k, v in getattr(reference, g).items())
    assert error < 1e-12
    expanded = Prototype(parent, True)
    initial_error = float(np.max(np.abs(expanded.predict(val["x"])-parent.predict(val["x"]))))
    assert initial_error < 1e-12
    plans = [
        ("identity_long", False, .003, 0., 1.),
        ("identity_balanced", False, .001, .5, 1.),
        ("identity_balanced_strong", False, .003, 1., 2.),
        ("shape_long", True, .003, 0., 1.),
        ("shape_balanced", True, .003, .5, 1.),
        ("shape_low_lr", True, .001, .5, 1.),
        ("shape_strong", True, .003, 1., 2.),
        ("residual_shape", "shape", .003, .5, 1.),
        ("residual_shape_low", "shape", .001, .5, 1.),
        ("residual_kinetic", "kinetic", .003, .5, 1.),
        ("residual_kinetic_low", "kinetic", .001, .5, 1.),
        ("residual_kinetic_strong", "kinetic", .003, 1., 2.),
        ("aux_shape", "shape", .003, .5, 1.),
        ("aux_shape_strong", "shape", .003, 1., 2.),
        ("aux_shape_regularized", "shape", .003, .5, 1.),
        ("blend_shape", "blend", .003, .5, 1.),
        ("blend_shape_gain", "blend", .003, .5, 1.),
        ("blend_shape_gain_strong", "blend", .003, 1., 2.),
    ]
    suite = tuple(robustness.validation_suite(parent, val))
    stress_before = robustness.evaluate_suite(parent, suite)
    records = []
    for name, use_shape, rate, balance, boost in plans:
        model = Prototype(parent, use_shape)
        model.raw_training=train['x']
        if name.startswith(('aux_','blend_')):
            features=response_shape(train['x']);model.adapter_mean=features.mean(axis=0);model.adapter_scale=features.std(axis=0)
            model.adapter_scale[model.adapter_scale<1e-12]=1
        x = model.normalized(train["x"])
        spec = {"learning_rate": rate, "balance": balance, "batch_size": 32, "l2": 0., "target_boost": boost}
        if name.startswith('aux_'):spec.update(aux_weight=1.,adapter_l2=.2 if name.endswith('regularized') else .01)
        if 'blend_shape_gain' in name:spec['gain']=1.2
        began = time.monotonic();history = [];best_key=None;best=None
        for epoch in range(1, 33):
            if time.monotonic()-began > 30:break
            model.epoch(x, train["y"], spec)
            metric = score_predictions(model.predict(val["x"]), val["y"])
            clean = assess_candidate(metric, before, before, config["gate"])
            stress = robustness.evaluate_suite(model, suite) if clean["accepted"] else None
            robust = robustness.gate(stress, stress_before, robustness.settings(config["optimization"])) if stress else None
            accepted = clean["accepted"] and robust["accepted"]
            row = {"epoch": epoch, "steps": model.meta["steps"], "validation": metric,
                   "clean_gate": clean, "robust_gate": robust, "eligible": bool(accepted)}
            history.append(row)
            key = (bool(accepted), clean["accepted"], metric["per_class"]["Acetaldehyde"]["recall"], metric["macro_f1"])
            if best_key is None or key > best_key:
                best_key, best = key, copy.deepcopy(model)
        chosen = max(history, key=lambda e: (e["eligible"], e["clean_gate"]["accepted"], e["validation"]["per_class"]["Acetaldehyde"]["recall"],e["validation"]["macro_f1"]))
        result = {"name": name, "prototype_only": True, "sdk_candidate": False, "shape_features": use_shape,
                  "configuration": spec, "best": chosen, "history": history, "seconds": time.monotonic()-began}
        write_json(out / (name+".json"), result);records.append(result)
        print(json.dumps({"name":name,"f1":chosen["validation"]["macro_f1"],"recall":chosen["validation"]["per_class"]["Acetaldehyde"]["recall"],"eligible":chosen["eligible"],"epoch":chosen["epoch"]}),flush=True)
    normalized = (val["x"]-parent.mean)/parent.scale
    summary = {"control_gradient_max_error": error, "parent_preserving_feature_expansion_max_error": initial_error,
        "official_train_sha256": train["sha256"], "official_validation_sha256": val["sha256"],
        "parent_sha256": digest(config["baseline"]), "baseline_validation": before,
        "per_class_clipped_fraction": {name:float(np.mean(np.abs(normalized[val['y']==i])>10)) for i,name in enumerate(CLASSES)},
        "rows": [{"name":r["name"],"f1":r["best"]["validation"]["macro_f1"],"recall":r["best"]["validation"]["per_class"]["Acetaldehyde"]["recall"],"eligible":r["best"]["eligible"]} for r in records],
        "provider_calls":0,"future_dataset_used":False,"deployment_modified":False,
        "feature_source":"https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset"}
    write_json(out / "summary.json",summary)


if __name__ == '__main__':main()
