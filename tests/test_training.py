"""训练与续训：数据隔离、参考模型兼容、优化器状态和训练产物。"""
import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from gas_a.contest import predict
from gas_a.data import evaluate, load_data, load_model, load_pair, load_unlabelled
from gas_a.io import CLASSES, digest, history, lock, read_json
from gas_a.monitor import diagnose, monitor
from gas_a.optimize import (
    bounded_experiment,
    classification_delta,
    confusion_pair_delta,
    fit_weighted,
    gate,
    train_candidate,
)
from gas_a.pipeline import load_config, ordered_specs, run
from gas_a.reference.model import Model
from gas_a.registry import Registry
from gas_a.service import ActivePredictor, make_server

ROOT = Path(__file__).resolve().parents[1]


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.json")
        cls.cfg["optimization"]["select_best_epoch"] = False  # 以下验证旧版完整训练轮数的续训契约。
        cls.cfg["gate"].pop("minimum_class_recall", None)  # 验证原始 gate 契约；弱类修复下限由单独的测试覆盖。
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.baseline = Path(cls.cfg["baseline"])
        cls.metric = evaluate(load_model(cls.baseline), cls.val)
        cls.drift = monitor(cls.train["x"], cls.val["x"], cls.cfg["monitor"])
        cls.diagnosis = diagnose(cls.train, cls.val, cls.metric, cls.drift)

    def test_official_baseline_and_diagnosis(self):
        self.assertEqual((len(self.train["y"]), len(self.val["y"])), (445, 1244))
        self.assertAlmostEqual(self.metric["macro_f1"], .7140297220788709, places=12)
        self.assertAlmostEqual(self.metric["accuracy"], .8786173633440515, places=12)
        self.assertEqual(self.diagnosis["weakest_class"], "Acetaldehyde")
        self.assertEqual(self.diagnosis["top_confusions"][0]["count"], 108)

    def test_drift_identical_shifted_and_small_sample(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(100, 128))
        self.assertFalse(monitor(x, x.copy(), self.cfg["monitor"])["drift_detected"])
        shifted = monitor(np.zeros((100, 128)), np.ones((100, 128)), self.cfg["monitor"])
        self.assertEqual(shifted["flagged_features"], 128)
        small = monitor(x[:5], x[:5] + 100, self.cfg["monitor"])
        self.assertFalse(small["sufficient_samples"])
        self.assertFalse(small["drift_detected"])

    def test_input_leakage_overlap_and_nonfinite_rejected(self):
        row = json.loads(Path(self.cfg["train"]).read_text(encoding="utf-8").splitlines()[0])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.jsonl"
            for inputs in ({**row["inputs"], "id": 1}, {"features": [True] * 128}, {"features": [float("nan")] * 128}):
                p.write_text(json.dumps({**row, "inputs": inputs}) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_data(p)
            p.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_pair(p, p)

    def test_unlabelled_monitoring_requires_no_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "window.jsonl"
            row = {"inputs": {"features": self.val["x"][0].tolist()}, "meta": {"ignored": 999}}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            values = load_unlabelled(path)
            np.testing.assert_array_equal(values, self.val["x"][:1])
            self.assertFalse(monitor(self.train["x"], values, self.cfg["monitor"])["sufficient_samples"])

    def test_continuation_and_validation_never_changes_gradients(self):
        spec = {**self.cfg["experiments"][1], "epochs": 1}
        original = digest(self.baseline)
        altered = {**self.val, "x": self.val["x"] * -3, "y": (self.val["y"] + 1) % 6}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = train_candidate(self.baseline, self.train, self.val, root / "one", spec, self.diagnosis, self.cfg["optimization"])
            train_candidate(self.baseline, self.train, altered, root / "two", spec, self.diagnosis, self.cfg["optimization"])
            first, second, parent = [load_model(p) for p in (root / "one/checkpoint.npz", root / "two/checkpoint.npz", self.baseline)]
            for name in first.params:
                np.testing.assert_array_equal(first.params[name], second.params[name])
                np.testing.assert_array_equal(first.first[name], second.first[name])
                np.testing.assert_array_equal(first.second[name], second.second[name])
            np.testing.assert_array_equal(first.mean, parent.mean)
            np.testing.assert_array_equal(first.scale, parent.scale)
            self.assertEqual(first.meta["seed"], parent.meta["seed"])
            self.assertEqual(report["final_steps"], report["initial_steps"] + 4)
            self.assertEqual(original, digest(self.baseline))
            # 从父模型非零 Adam 状态出发，精确复现一阶矩。
            probe = load_model(self.baseline)
            fit_weighted(probe, self.train, spec, self.diagnosis["weakest_class"], time.monotonic() + 30)
            for name in first.first:
                np.testing.assert_array_equal(first.first[name], probe.first[name])

    def test_a_checkpoint_rejects_incompatible_hidden_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            for hidden in (16, 64):
                with self.subTest(hidden=hidden):
                    model = Model("classification", self.train["paths"], CLASSES,
                                  self.train["x"], self.train["y"], hidden=hidden)
                    model.meta["profile"] = "A"
                    checkpoint = Path(tmp) / f"hidden-{hidden}.npz"
                    model.save(checkpoint)
                    with self.assertRaisesRegex(ValueError, "hidden width must be 32"):
                        load_model(checkpoint, self.train["paths"])

    def test_candidate_evidence_records_each_epoch_and_follows_registered_version(self):
        spec = {**self.cfg["experiments"][1], "epochs": 2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "round-001" / spec["name"]
            report = train_candidate(self.baseline, self.train, self.val, directory,
                                     spec, self.diagnosis, self.cfg["optimization"])
            training = read_json(directory / "training.json")
            epochs = history(directory / "history.jsonl")
            parent = load_model(self.baseline)
            child = load_model(directory / "checkpoint.npz")
            self.assertEqual(len(epochs), spec["epochs"])
            self.assertEqual(epochs, report["epoch_history"])
            self.assertEqual(training["stage_id"], "round-001/balanced")
            self.assertEqual(training["format"], "contest.cpu-baseline-report.v1")
            self.assertEqual(training["initial_epochs"], parent.meta["epochs"])
            self.assertEqual(training["final_epochs"], child.meta["epochs"])
            self.assertEqual(training["initial_steps"], parent.meta["steps"])
            self.assertEqual(training["final_steps"], child.meta["steps"])
            self.assertEqual(training["parent_checkpoint_sha256"], digest(self.baseline))
            self.assertEqual(training["checkpoint_sha256"], digest(directory / "checkpoint.npz"))
            self.assertEqual(training["train_sha256"], self.train["sha256"])
            self.assertEqual(training["validation_sha256"], self.val["sha256"])
            self.assertEqual(training["source_sha256"]["optimize.py"], digest(ROOT / "src/gas_a/optimize.py"))
            self.assertEqual(training["configuration"]["hidden"], 32)
            self.assertEqual(training["validation_before"], report["validation_before"])
            self.assertEqual(training["validation_after"], epochs[-1]["validation"])

            # 每条记录都与对应 epoch 的模型指标核对，
            # 避免统一使用最终 checkpoint 的指标比较全部记录。
            probe = load_model(self.baseline)
            for index, record in enumerate(epochs, 1):
                fit_weighted(probe, self.train, {**spec, "epochs": 1},
                             self.diagnosis["weakest_class"], time.monotonic() + 30)
                measured = evaluate(probe, self.val)
                self.assertEqual(record["epoch"], parent.meta["epochs"] + index)
                self.assertEqual(record["step"], probe.meta["steps"])
                self.assertEqual(record["steps"], record["step"])
                self.assertEqual(record["stage_id"], training["stage_id"])
                self.assertEqual(record["seed"], parent.meta["seed"])
                for key in ("macro_f1", "accuracy", "confusion_matrix"):
                    self.assertEqual(record["validation"][key], measured[key])
            for group in ("params", "first", "second"):
                for name in getattr(child, group):
                    np.testing.assert_array_equal(getattr(child, group)[name], getattr(probe, group)[name])

            files = ("training.json", "history.jsonl", "checkpoint.npz", "experiment.json")
            originals = {name: (directory / name).read_bytes() for name in files}
            with self.assertRaises(FileExistsError):
                train_candidate(self.baseline, self.train, self.val, directory,
                                spec, self.diagnosis, self.cfg["optimization"])
            self.assertEqual(originals, {name: (directory / name).read_bytes() for name in files})

            registry = Registry(root / "registered")
            registry.initialize(self.baseline, self.metric)
            registry.register("candidate", directory / "checkpoint.npz", "baseline", report["validation_after"], report)
            registered = registry.checkpoint("candidate").parent
            for name in ("training.json", "history.jsonl"):
                self.assertEqual((registered / name).read_bytes(), originals[name])

    def test_quality_gates(self):
        candidate = copy.deepcopy(self.metric)
        candidate["macro_f1"] += .01
        self.assertTrue(gate(candidate, self.metric, self.metric, self.cfg["gate"])["accepted"])
        candidate["accuracy"] -= .1
        self.assertIn("accuracy_regression", gate(candidate, self.metric, self.metric, self.cfg["gate"])["reasons"])
        candidate["accuracy"] = self.metric["accuracy"]
        candidate["per_class"]["Ethanol"]["recall"] = 0
        self.assertIn("class_recall_regression:Ethanol", gate(candidate, self.metric, self.metric, self.cfg["gate"])["reasons"])
        self.assertFalse(gate(self.metric, self.metric, self.metric, self.cfg["gate"])["accepted"])

    def test_registry_live_switch_corruption_and_performance_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(tmp)
            registry.initialize(self.baseline, self.metric)
            inflated = {**self.metric, "macro_f1": .99}
            registry.register("candidate", self.baseline, "baseline", inflated, {})
            registry.activate("candidate", "test")
            live = ActivePredictor(tmp)
            inputs = {"features": self.val["x"][0].tolist()}
            self.assertEqual(live.predict(inputs)["version"], "candidate")
            # 当前 validation 指标相对上线记录下降时触发恢复。
            active, _, _, event = registry.ensure_healthy(self.val, self.cfg["gate"])
            self.assertEqual(active, "baseline")
            self.assertEqual(event["cause"], "validation_quality_regression")
            self.assertEqual(live.predict(inputs)["version"], "baseline")
            registry.activate("candidate", "corruption test")
            (Path(tmp) / "versions/candidate/checkpoint.npz").write_bytes(b"corrupted")
            active, _, _, event = registry.ensure_healthy(self.val, self.cfg["gate"])
            self.assertEqual(active, "baseline")
            self.assertIn("digest mismatch", event["cause"])
            registry.activate("baseline", "manual test")
            self.assertEqual(registry.active_id(), "baseline")

    def test_lock_rejects_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with lock(tmp):
                with self.assertRaises(RuntimeError):
                    with lock(tmp):
                        pass

    def test_http_health_prediction_switch_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(tmp)
            registry.initialize(self.baseline, self.metric)
            registry.register("copy", self.baseline, "baseline", self.metric, {})
            server = make_server(tmp, port=0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/health", timeout=5) as response:
                    self.assertEqual(json.load(response)["status"], "ok")
                body = json.dumps({"inputs": {"features": self.val["x"][0].tolist()}}).encode()
                def request():
                    return urlopen(Request(base + "/predict", data=body, headers={"Content-Type": "application/json"}), timeout=5)
                with request() as response:
                    self.assertEqual(json.load(response)["version"], "baseline")
                registry.activate("copy", "HTTP live switch test")
                with request() as response:
                    self.assertEqual(json.load(response)["version"], "copy")
                (Path(tmp) / "versions/copy/checkpoint.npz").write_bytes(b"bad")
                with request() as response:
                    self.assertEqual(json.load(response)["version"], "baseline")
                (Path(tmp) / "versions/baseline/checkpoint.npz").write_bytes(b"bad")
                with self.assertRaises(HTTPError) as failed:
                    request()
                self.assertEqual(failed.exception.code, 503)
                failed.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                worker.join(5)

    def test_history_avoids_repeated_experiment(self):
        spec = self.cfg["experiments"][0]
        past = [{"experiments": [{"report": {"name": spec["name"], "configuration": spec,
                "parent_checkpoint_sha256": "same", 'train_sha256':self.train['sha256'],'validation_sha256':self.val['sha256'],
                "validation_before": {"macro_f1": .7}, "validation_after": {"macro_f1": .69}}}]}]
        selected = ordered_specs(self.cfg, past, "same", train_sha=self.train['sha256'], validation_sha=self.val['sha256'])
        self.assertTrue(selected)
        self.assertFalse(any(s == spec for s in selected))

    def test_full_lifecycle_evidence_and_saved_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg)
            cfg["runtime"] = tmp
            summary = run(cfg, rounds=2)
            registry = Registry(tmp)
            measured = evaluate(load_model(registry.checkpoint(summary["active_version"])), self.val)
            self.assertEqual(measured["macro_f1"], summary["metrics"]["macro_f1"])
            self.assertGreaterEqual(measured["macro_f1"], self.metric["macro_f1"])
            for i in (1, 2):
                directory = Path(tmp) / "rounds" / f"round-{i:03d}"
                for name in ("monitor.json", "diagnosis.json", "round.json"):
                    self.assertTrue((directory / name).is_file())
                record = read_json(directory / "round.json")
                self.assertEqual(record["history_rounds_read"], list(range(1, i)))
                self.assertEqual(len(record["experiments"]), 3)
            self.assertTrue((Path(tmp) / "active.json").is_file())
            self.assertTrue((Path(tmp) / "deployments.jsonl").is_file())


class OptimizationTests(unittest.TestCase):
    def test_focal_logits_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(3, 6))
        target, weights = np.array([0, 2, 5]), np.array([.5, 1., 1.5])

        def loss(values, gamma):
            p = np.exp(values - values.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            return classification_delta(p, target, weights, gamma)

        for gamma in (0, 1, 1.5):
            _, analytical = loss(logits, gamma)
            numerical = np.zeros_like(logits)
            for index in np.ndindex(logits.shape):
                plus, minus = logits.copy(), logits.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                numerical[index] = (loss(plus, gamma)[0] - loss(minus, gamma)[0]) / (2e-6 * len(target))
            np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-9)

    def test_best_epoch_restores_adam_and_counts_discarded_training(self):
        cfg = load_config(ROOT / "configs/a.json")
        train, val = load_pair(cfg["train"], cfg["validation"])
        parent = load_model(cfg["baseline"])
        initial = parent.meta["epochs"]
        spec = {**cfg["experiments"][0], "epochs": 8, "focal_gamma": 1., "anchor_l2": .01,
                "schedule": "cosine", "final_lr_ratio": .2}

        def controlled_metric(model, data):
            result = evaluate(model, data)
            if data is val:
                result["macro_f1"] = {0: .7, 1: .72, 2: .75}.get(model.meta["epochs"] - initial, .71)
            return result

        with tempfile.TemporaryDirectory() as tmp, patch("gas_a.optimize.evaluate", side_effect=controlled_metric):
            report = train_candidate(cfg["baseline"], train, val, Path(tmp) / "trial", spec,
                                     {"weakest_class": "Acetaldehyde"},
                                     {**cfg["optimization"], "select_best_epoch": True, "early_stopping_patience": 1,
                                      "robustness": {"enabled": False}})  # 本项只验证干净验证分数的最佳 epoch；稳健选版另有专项。
            selected = load_model(Path(tmp) / "trial/checkpoint.npz")
            self.assertEqual(selected.meta["epochs"], initial + 2)
            self.assertEqual(report["executed_epochs"], initial + 3)
            self.assertEqual(len(history(Path(tmp) / "trial/history.jsonl")), 3)
            self.assertTrue(report["checkpoint_selection"]["stopped_early"])
            self.assertGreater(report["executed_steps"], report["final_steps"])
            probe = load_model(cfg["baseline"])
            fit_weighted(probe, train, spec, "Acetaldehyde", time.monotonic() + 30, epochs_to_run=2)
            for group in ("params", "first", "second"):
                for key in getattr(probe, group):
                    np.testing.assert_array_equal(getattr(selected, group)[key], getattr(probe, group)[key])
            np.testing.assert_array_equal(parent.mean, selected.mean)
            np.testing.assert_array_equal(parent.scale, selected.scale)

    def test_cosine_and_anchor_replay_uses_original_parent_reference(self):
        cfg = load_config(ROOT / "configs/a.json")
        train, _ = load_pair(cfg["train"], cfg["validation"])
        continuous, replay = load_model(cfg["baseline"]), load_model(cfg["baseline"])
        anchor = {k: v.copy() for k, v in replay.params.items()}
        spec = {**cfg["experiments"][0], "epochs": 3, "anchor_l2": .1, "feature_lr_scale": .25,
                "schedule": "cosine", "final_lr_ratio": .2, "batch_size": 64}
        fit_weighted(continuous, train, spec, "Acetaldehyde", time.monotonic() + 30)
        for offset in range(3):
            fit_weighted(replay, train, spec, "Acetaldehyde", time.monotonic() + 30,
                         epoch_offset=offset, epochs_to_run=1, anchor_params=anchor)
        for group in ("params", "first", "second"):
            for key in getattr(replay, group):
                np.testing.assert_array_equal(getattr(replay, group)[key], getattr(continuous, group)[key])

    def test_stalled_family_gives_menu_space_to_other_directions(self):
        cfg = load_config(ROOT / "configs/a.json")
        base = cfg["experiments"][0]
        past = [{"round": r, "action": "retain", "parent_checkpoint_sha256": "parent", "experiments": [
            {"name": base["name"], "report": {"name": base["name"], "configuration": {**base, "learning_rate": rate},
             "parent_checkpoint_sha256": "parent", "validation_before": {"macro_f1": .72},
             "validation_after": {"macro_f1": .719}}}]} for r, rate in ((1, .0005), (2, .00025))]
        self.assertNotIn(base["name"], [s["name"] for s in ordered_specs(cfg, past, "parent")])
        failed = {"round": 3, "parent_checkpoint_sha256": "parent", "experiments": [
            {"name": "focal", "configuration": cfg["experiments"][2], "error": "fixture failure"}]}
        menu = ordered_specs(cfg, [failed], "parent")
        self.assertFalse(any(s == cfg["experiments"][2] for s in menu))

    def test_menu_keeps_both_conservative_and_class_loss_alternatives(self):
        cfg = load_config(ROOT / "configs/a.json")
        menu = ordered_specs(cfg, [], "parent", {"fit_comparison": {"macro_f1_gap": .25}})
        targeted = [s for s in menu if s["balance"] or s["weak_boost"] > 1 or s.get("focal_gamma", 0)]
        self.assertTrue(targeted)
        self.assertLess(len(targeted), len(menu))


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.live.json")
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.baseline = evaluate(load_model(cls.cfg["baseline"]), cls.val)

    def test_original_checkpoint_predictions_remain_bit_identical(self):
        old, new = Model.load(self.cfg["baseline"]), load_model(self.cfg["baseline"])
        np.testing.assert_array_equal(old.predict(self.val["x"]), new.predict(self.val["x"]))

    def test_macro_gain_cannot_hide_zero_recall_or_other_class_collapse(self):
        metrics = copy.deepcopy(self.baseline)
        metrics["macro_f1"] = .85
        decision = gate(metrics, self.baseline, self.baseline, self.cfg["gate"])
        self.assertIn("class_recall_below_minimum:Acetaldehyde", decision["reasons"])
        metrics["per_class"]["Acetaldehyde"]["recall"] = 1.
        metrics["per_class"]["Ethanol"]["recall"] = .5
        self.assertIn("class_recall_regression:Ethanol", gate(metrics, self.baseline, self.baseline, self.cfg["gate"])["reasons"])

    def test_old_gain_does_not_inject_a_fixed_sensor_repair(self):
        spec = self.cfg["experiments"][0]
        past = [{"round": 1, "action": "promote", "parent_checkpoint_sha256": "older", "experiments": [
            {"name": spec["name"], "report": {"name": spec["name"], "configuration": spec,
             "parent_checkpoint_sha256": "older", "validation_before": {"macro_f1": .7},
             "validation_after": {"macro_f1": .72}}}]}]
        diagnosis = {"fit_comparison": {"acetaldehyde_validation_recall": 0, "macro_f1_gap": .25}}
        menu = ordered_specs(self.cfg, past, "current", diagnosis)
        self.assertTrue(menu)
        self.assertTrue(all("kept_sensors" not in item for item in menu))

    def test_reference_checkpoint_and_original_scaler_with_preserved_adam(self):
        spec = self.cfg["experiments"][0]
        parent = load_model(self.cfg["baseline"])
        original_sha = digest(self.cfg["baseline"])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "candidate"
            report = train_candidate(self.cfg["baseline"], self.train, self.val, directory, spec,
                                     {"weakest_class": "Acetaldehyde"},
                                     {**self.cfg["optimization"], "checkpoint_gate": self.cfg["gate"]})
            candidate = load_model(directory / "checkpoint.npz")
            measured = evaluate(candidate, self.val)
            self.assertEqual((candidate.meta["hidden"], len(candidate.meta["paths"])), (32, 128))
            self.assertIs(type(candidate), Model)
            self.assertNotIn("active_features", candidate.meta)
            self.assertNotIn("input_transform", candidate.meta)
            self.assertEqual(report["freezing"]["projected_parameter_count"], 0)
            for sensor in range(16):
                rows = slice(sensor * 8, (sensor + 1) * 8)
                self.assertTrue(np.any(candidate.params["w1"][rows] != parent.params["w1"][rows]))
            np.testing.assert_array_equal(candidate.normalized(self.val["x"]), parent.normalized(self.val["x"]))
            np.testing.assert_array_equal(Model.load(directory / "checkpoint.npz").predict(self.val["x"]), candidate.predict(self.val["x"]))
            np.testing.assert_array_equal(parent.mean, candidate.mean)
            np.testing.assert_array_equal(parent.scale, candidate.scale)
            fit_weighted(parent, self.train, spec, "Acetaldehyde", time.monotonic() + 30,
                         epochs_to_run=candidate.meta["epochs"] - parent.meta["epochs"])
            for group in ("params", "first", "second"):
                for name, values in getattr(candidate, group).items():
                    np.testing.assert_array_equal(values, getattr(parent, group)[name])
            self.assertEqual(original_sha, digest(self.cfg["baseline"]))
            training = read_json(directory / "training.json")
            self.assertTrue(all(digest(directory / "source" / name) == value
                                for name, value in training["source_sha256"].items()))
            index = 0
            sample = SimpleNamespace(task="classification", dataset_key="A", inputs={"features": self.val["x"][index].tolist()})
            reply = predict(sample, directory)
            scores = [item["score"] for item in reply]
            self.assertEqual(int(np.argmax(scores)), int(candidate.predict(self.val["x"][[index]]).argmax(axis=1)[0]))
            self.assertAlmostEqual(sum(scores), 1.)

    def test_experimental_preprocessing_is_rejected_by_default(self):
        parent = Model.load(self.cfg["baseline"])
        parent.meta["active_features"] = list(range(40))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "historical.npz"
            parent.save(checkpoint)
            with self.assertRaisesRegex(ValueError, "experimental preprocessing"):
                load_model(checkpoint)
        base = self.cfg["experiments"][0]
        with self.assertRaisesRegex(ValueError, "Custom prediction preprocessing"):
            bounded_experiment({**base, "active_sensors": [3,13,14,15,16]}, self.cfg["optimization"])

    def test_confusion_pair_gradient_matches_finite_differences(self):
        logits = np.random.default_rng(11).normal(size=(3, 6))
        labels, weights = np.array([0, 3, 2]), np.array([1., 1.5, .8])
        _, analytical = confusion_pair_delta(logits, labels, weights, 3, 0, 3., 2.)
        numerical = np.zeros_like(logits)
        for index in np.ndindex(logits.shape):
            plus, minus = logits.copy(), logits.copy()
            plus[index] += 1e-6
            minus[index] -= 1e-6
            numerical[index] = (confusion_pair_delta(plus, labels, weights, 3, 0, 3., 2.)[0]
                                - confusion_pair_delta(minus, labels, weights, 3, 0, 3., 2.)[0]) / 6e-6
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
