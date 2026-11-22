"""Public-only research: additive response-shape head, not an SDK candidate."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from gas_a.data import load_pair, load_model, score_predictions
from gas_a.io import write_json
from gas_a.pipeline import load_config
from gas_a.candidates import assess_candidate
from gas_a.optimize import fit_weighted
from gas_a import robustness
from probe_generalization import response_shape


class Adapter:
    def __init__(self, parent, mean, scale, coefficients, strength, clip=0., family=-1, bounded=False):
        self.parent, self.mean, self.scale = parent, mean, scale
        self.coefficients, self.strength = coefficients, strength
        self.clip = clip
        self.family = family
        self.bounded = bounded
        self.guard = None

    def predict(self, raw):
        _, logits = self.parent.forward(self.parent.normalized(raw))
        features=response_shape(raw)
        if self.family>=0: features=features[:,self.family::8]
        features = (features - self.mean) / self.scale
        if self.clip: features = np.clip(features, -self.clip, self.clip)
        x = np.column_stack([features, np.ones(len(raw))])
        correction = x @ self.coefficients
        if self.bounded: correction=np.clip(correction,0,1)
        if self.guard is not None:
            matrix, threshold = self.guard
            distance=np.sum((features@matrix)*features,axis=1)
            correction=correction*(distance<=threshold)[:,None]
        logits = logits + self.strength * correction
        result = np.exp(logits - logits.max(axis=1, keepdims=True))
        return result / result.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); p.add_argument('--clip',type=float,default=0.); p.add_argument('--family',type=int,default=-1); p.add_argument('--bounded',action='store_true'); p.add_argument('--guard',type=float,default=0.); p.add_argument('--guard-scale',type=float,default=1.); args = p.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    cfg = load_config(ROOT/'configs/a.json'); train, val = load_pair(cfg['train'], cfg['validation'])
    parent = load_model(cfg['baseline']); before = score_predictions(parent.predict(val['x']), val['y'])
    base = copy.deepcopy(parent)
    fit_weighted(base, train, dict(learning_rate=.0001, epochs=1, balance=.5, weak_boost=1., l2=0.), 'Acetaldehyde', time.monotonic()+30)
    raw = response_shape(train['x'])
    if args.family>=0:raw=raw[:,args.family::8]
    mean = raw.mean(axis=0); scale = raw.std(axis=0); scale[scale<1e-12]=1
    def norm(raw):
        value=response_shape(raw)
        if args.family>=0:value=value[:,args.family::8]
        value=(value-mean)/scale
        return np.clip(value,-args.clip,args.clip) if args.clip else value
    x = np.column_stack([norm(train['x']), np.ones(len(raw))]); y = train['y']
    guard=None
    if args.guard:
        inverse=np.linalg.inv(x[:,:-1].T@x[:,:-1]/len(x)+.01*np.eye(x.shape[1]-1))
        distance=np.einsum('ni,ij,nj->n',x[:,:-1],inverse,x[:,:-1])
        guard=(inverse,float(np.quantile(distance,args.guard))*args.guard_scale)
    weights = len(y)/(6*np.bincount(y, minlength=6)[y]); targets = np.eye(6)[y]
    suite = tuple(robustness.validation_suite(parent, val)); stress_before = robustness.evaluate_suite(parent, suite)
    rows=[]
    for outage_weight in ((0.,) if args.guard else (0., .25, 1., 4.)):
      gram=x.T@(x*weights[:,None]); rhs=x.T@(targets*weights[:,None])
      if outage_weight:
        for sensor in range(16):
          changed=train['x'].copy();changed[:,sensor*8:(sensor+1)*8]=parent.mean[sensor*8:(sensor+1)*8]
          augmented=np.column_stack([norm(changed),np.ones(len(changed))])
          ww=weights*outage_weight/16
          gram+=augmented.T@(augmented*ww[:,None]);rhs+=augmented.T@(targets*ww[:,None])
      for ridge in (1., 10., 100.):
        coeff = np.linalg.solve(gram+ridge*np.eye(gram.shape[0]), rhs)
        for strength in (4., 8., 16.):
            model = Adapter(base, mean, scale, coeff, strength, args.clip,args.family,args.bounded)
            model.guard=guard
            metric = score_predictions(model.predict(val['x']), val['y'])
            clean = assess_candidate(metric, before, before, cfg['gate'])
            stress = robustness.evaluate_suite(model, suite)
            robust = robustness.gate(stress, stress_before, robustness.settings(cfg['optimization']))
            row = dict(outage_weight=outage_weight,ridge=ridge, strength=strength, validation=metric, clean_gate=clean, robust_gate=robust,
                       worst_f1=stress['worst_macro_f1'], eligible=clean['accepted'] and robust['accepted'])
            rows.append(row)
            print(json.dumps(dict(outage_weight=outage_weight,ridge=ridge, strength=strength, f1=metric['macro_f1'], recall=metric['per_class']['Acetaldehyde']['recall'],
                                  clean=clean['accepted'],robust=robust['accepted'], worst=row['worst_f1'])), flush=True)
    write_json(out/'summary.json',dict(family=args.family,clip=args.clip,research_only=True, provider_calls=0, future_data_used=False,
        baseline_validation=before, baseline_robustness=stress_before, rows=rows))


if __name__ == '__main__': main()
