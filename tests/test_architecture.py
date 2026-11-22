"""架构扩展必须从真实父模型续训，保留旧状态、任务边界和可恢复证据。"""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gas_a.architecture import MigratedModel, migrate_parent, shape, target_shape
from gas_a.cli import prediction
from gas_a.contest import predict
from gas_a.data import load_model, load_pair, evaluate
from gas_a.io import CLASSES, digest, read_json
from gas_a.optimize import fit_weighted, train_candidate, classification_delta
from gas_a.pipeline import load_config, ordered_specs
from gas_a.registry import Registry
from gas_a.service import ActivePredictor

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.architecture.json")
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.baseline = Path(cls.cfg["baseline"])
        cls.sha = digest(cls.baseline)

    def parent(self):
        return load_model(self.baseline)

    def migrated(self):
        return migrate_parent(self.parent(), {"hidden": 64, "hidden_layers": 2}, self.sha)[0]

    def test_migration_preserves_parent_predictions_and_every_old_optimizer_element(self):
        for request in ({"hidden": 64}, {"add_layers": 1}, {"width_multiplier": 2, "add_layers": 2}):
            with self.subTest(request=request):
                parent = self.parent()
                old = copy.deepcopy(parent)
                child, evidence = migrate_parent(parent, request, self.sha)
                self.assertTrue(evidence["performed"])
                self.assertIsInstance(child, MigratedModel)
                self.assertEqual(evidence["parent_checkpoint_sha256"], self.sha)
                self.assertEqual(len(evidence["copied_state"]), 12)
                for group in ("params", "first", "second"):
                    for key, value in getattr(parent, group).items():
                        slices = tuple(slice(0, s) for s in value.shape)
                        np.testing.assert_array_equal(getattr(child, group)[key][slices], value)
                        np.testing.assert_array_equal(getattr(parent, group)[key], getattr(old, group)[key])
                    if group != "params":
                        for key, value in getattr(child, group).items():
                            added = value.copy()
                            if key in getattr(parent, group):
                                added[tuple(slice(0, s) for s in getattr(parent, group)[key].shape)] = 0
                            self.assertFalse(np.any(added))
                for key in ("seed", "steps", "epochs", "target_mean", "target_scale", "batch_size"):
                    self.assertEqual(child.meta[key], parent.meta[key])
                np.testing.assert_array_equal(child.mean, parent.mean)
                np.testing.assert_array_equal(child.scale, parent.scale)
                for dataset in (self.train, self.val):
                    a, b = parent.predict(dataset["x"]), child.predict(dataset["x"])
                    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
                    np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
        self.assertEqual(digest(self.baseline), self.sha)

    def test_migrated_checkpoint_roundtrip_and_continuation_matches_split_run(self):
        child = self.migrated()
        spec = {"epochs": 2, "learning_rate": .0005, "balance": .25, "weak_boost": 1.5, "l2": .001}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.npz"
            child.save(path)
            replay = load_model(path, self.train["paths"])
            continuous = copy.deepcopy(replay)
            anchor = {k: v.copy() for k, v in replay.params.items()}
            fit_weighted(continuous, self.train, spec, "Acetaldehyde", time.monotonic() + 30)
            for offset in range(2):
                fit_weighted(replay, self.train, spec, "Acetaldehyde", time.monotonic() + 30,
                             epoch_offset=offset, epochs_to_run=1, anchor_params=anchor)
                out = Path(tmp) / f"part-{offset}.npz"
                replay.save(out)
                replay = load_model(out)
            self.assertEqual(replay.meta["epochs"], child.meta["epochs"] + 2)
            self.assertGreater(replay.meta["steps"], child.meta["steps"])
            for group in ("params", "first", "second"):
                for key in getattr(replay, group):
                    np.testing.assert_array_equal(getattr(replay, group)[key], getattr(continuous, group)[key])
            self.assertGreater(np.linalg.norm(replay.params["w3"]), 0)
            np.testing.assert_array_equal(replay.predict(self.val["x"]), continuous.predict(self.val["x"]))

    def test_backpropagation_includes_residual_layers_and_matches_finite_differences(self):
        model = self.migrated()
        rng = np.random.default_rng(19)
        model.params["w3"][:] = rng.normal(0, .01, model.params["w3"].shape)
        x = model.normalized(self.train["x"][:4])
        y = self.train["y"][:4]

        def loss():
            _, logits = model.forward(x)
            probability = np.exp(logits - logits.max(axis=1, keepdims=True))
            probability /= probability.sum(axis=1, keepdims=True)
            return classification_delta(probability, y, np.ones(4))[0] / 4

        _, logits = model.forward(x)
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        _, delta = classification_delta(probability, y, np.ones(4))
        gradient, input_gradient = model.backward(x, delta)
        for key, index in (("w1", (2, 3)), ("b1", (3,)), ("w3", (1, 2)), ("b3", (2,)), ("w2", (5, 1)), ("b2", (1,))):
            old = model.params[key][index]
            model.params[key][index] = old + 1e-6
            plus = loss()
            model.params[key][index] = old - 1e-6
            minus = loss()
            model.params[key][index] = old
            self.assertAlmostEqual(gradient[key][index], (plus - minus) / 2e-6, places=7)
        old = x[0, 0]
        x[0, 0] = old + 1e-6
        plus = loss()
        x[0, 0] = old - 1e-6
        minus = loss()
        x[0, 0] = old
        self.assertAlmostEqual(input_gradient[0, 0], (plus - minus) / 2e-6, places=7)

    def test_export_records_migration_and_resumes_actual_selected_migrated_parent(self):
        spec = {**self.cfg["experiments"][2], "epochs": 2}
        limits = {**self.cfg["optimization"], "select_best_epoch": False}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = root / "one"
            report = train_candidate(self.baseline, self.train, self.val, one, spec,
                                     {"weakest_class": "Acetaldehyde"}, limits)
            migrated = load_model(one / "migrated_parent.npz")
            np.testing.assert_allclose(migrated.predict(self.val["x"]), self.parent().predict(self.val["x"]), rtol=0, atol=1e-12)
            self.assertEqual(migrated.meta["steps"], self.parent().meta["steps"])
            migration = read_json(one / "migration.json")
            self.assertEqual(migration["migrated_parent_sha256"], digest(one / "migrated_parent.npz"))
            self.assertTrue(all(p["labels_identical"] for p in migration["prediction_checks"]))
            self.assertEqual(report["architecture"], {"hidden": 64, "hidden_layers": 2})
            self.assertEqual(read_json(one / "training.json")["source_sha256"]["architecture.py"], digest(ROOT / "src/gas_a/architecture.py"))
            child = load_model(one / "checkpoint.npz")
            self.assertGreater(child.meta["steps"], migrated.meta["steps"])
            registry = Registry(root / "deployment")
            registry.initialize(self.baseline, evaluate(self.parent(), self.val))
            registry.register("migrated", one / "checkpoint.npz", "baseline", report["validation_after"], report)
            registry.activate("migrated", "offline_test")
            for name in ("migration.json", "migrated_parent.npz"):
                self.assertEqual((registry.root / "versions/migrated" / name).read_bytes(), (one / name).read_bytes())
            selected = registry.checkpoint(registry.active_id())
            spec = {"name": "continue", "epochs": 1, "learning_rate": .0005, "balance": 0., "weak_boost": 1., "l2": .0001}
            two = train_candidate(selected, self.train, self.val, root / "two", spec, {"weakest_class": "Acetaldehyde"}, limits)
            continued = load_model(root / "two/checkpoint.npz")
            self.assertEqual(two["parent_checkpoint_sha256"], digest(selected))
            self.assertEqual(two["initial_steps"], child.meta["steps"])
            self.assertEqual(continued.meta["epochs"], child.meta["epochs"] + 1)
            expected = copy.deepcopy(child)
            fit_weighted(expected, self.train, spec, "Acetaldehyde", time.monotonic() + 30)
            for group in ("params", "first", "second"):
                for key in getattr(expected, group):
                    np.testing.assert_array_equal(getattr(continued, group)[key], getattr(expected, group)[key])

    def test_migrated_model_prediction_contract_and_mixed_version_recovery(self):
        child = self.migrated()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child.save(root / "checkpoint.npz")
            inputs = {"features": self.val["x"][0].tolist()}
            answer = predict(SimpleNamespace(task="classification", dataset_key="A", inputs=inputs), root)
            self.assertEqual([r["label"] for r in answer], CLASSES)
            self.assertEqual(len(answer), 6)
            self.assertAlmostEqual(sum(r["score"] for r in answer), 1.)
            registry = Registry(root / "deployment")
            registry.initialize(self.baseline, evaluate(self.parent(), self.val))
            registry.register("expanded", root / "checkpoint.npz", "baseline", evaluate(child, self.val), {})
            service = ActivePredictor(registry.root)
            for version in ("expanded", "baseline", "expanded"):
                registry.activate(version, "offline_recovery_test")
                result = service.predict(inputs)
                self.assertEqual(result["version"], version)
                self.assertEqual([r["label"] for r in result["outputs"]], CLASSES)
                self.assertEqual(len(result["outputs"]), 6)
            for values in ([0.] * 127, [0.] * 129, [True] * 128, [float("nan")] * 128):
                with self.assertRaises(ValueError):prediction(child, {"features": values})
            fake = SimpleNamespace(meta={"classes": CLASSES}, predict=lambda x: np.ones((1, 7)) / 7)
            with self.assertRaisesRegex(ValueError, "six finite probabilities"):prediction(fake, inputs)

    def test_malformed_migration_io_changes_and_destructive_architecture_are_rejected(self):
        child = self.migrated()
        for request in ({"hidden": 32}, {"hidden_layers": 1}, {"hidden": True}, {"input_size": 129},
                        {"classes": ["x"]}, {"hidden": 256}, {"hidden_layers": 5},
                        {"hidden": 64, "width_multiplier": 2}, {"add_layers": 4}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                target_shape(child, request)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-proof.npz"
            child.save(path)
            with np.load(path, allow_pickle=False) as arrays:
                content = {k: arrays[k].copy() for k in arrays.files}
            meta = json.loads(str(content["metadata"]))
            meta.pop("architecture_migration")
            content["metadata"] = np.array(json.dumps(meta))
            np.savez(path, **content)
            with self.assertRaisesRegex(ValueError, "migration"):load_model(path)
        for key, value in (("paths", self.train["paths"][:-1]), ("classes", list(reversed(CLASSES)))):
            broken = copy.deepcopy(child)
            broken.meta[key] = value
            with self.assertRaises(ValueError):broken.validate()
        maxed, _ = migrate_parent(child, {"hidden": 128, "hidden_layers": 4}, self.sha)
        self.assertEqual([s["name"] for s in ordered_specs(self.cfg, [], self.sha, model=maxed)], ["gentle"])

    def test_second_migration_preserves_already_trained_residual_state(self):
        parent = self.migrated()
        parent.fit(self.train["x"], self.train["y"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actual-parent.npz"
            parent.save(path)
            child, proof = migrate_parent(load_model(path), {"hidden": 96, "add_layers": 1}, digest(path))
            self.assertEqual(proof["parent_checkpoint_sha256"], digest(path))
            self.assertEqual(shape(child), {"hidden": 96, "hidden_layers": 3})
            for group in ("params", "first", "second"):
                for key, value in getattr(parent, group).items():
                    np.testing.assert_array_equal(getattr(child, group)[key][tuple(slice(0, d) for d in value.shape)], value)
            np.testing.assert_allclose(child.predict(self.val["x"]), parent.predict(self.val["x"]), rtol=0, atol=1e-12)

    def test_migrated_continuation_gradients_do_not_use_validation_samples(self):
        spec = {**self.cfg["experiments"][2], "epochs": 1}
        limits = {**self.cfg["optimization"], "select_best_epoch": False}
        changed = {**self.val, "x": -3 * self.val["x"], "y": (self.val["y"] + 1) % 6}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, validation in (("original", self.val), ("altered", changed)):
                train_candidate(self.baseline, self.train, validation, root / name, spec,
                                {"weakest_class": "Acetaldehyde"}, limits)
            one, two = [load_model(root / name / "checkpoint.npz") for name in ("original", "altered")]
            for group in ("params", "first", "second"):
                for key in getattr(one, group):
                    np.testing.assert_array_equal(getattr(one, group)[key], getattr(two, group)[key])

    def test_robust_migrated_epoch_restores_all_adam_tensors(self):
        initial = self.parent().meta["epochs"]
        spec = {**self.cfg["experiments"][2], "epochs": 8}

        def measured(model, dataset):
            value = evaluate(model, dataset)
            if dataset is self.val:
                value["macro_f1"] = {0: .7, 1: .8, 2: .75}.get(model.meta["epochs"] - initial, .71)
            return value

        def stress(model, suite):
            value = {0: .5, 1: .6, 2: .65}.get(model.meta["epochs"] - initial, .61)
            return {"worst_macro_f1": value, "mean_macro_f1": value,
                    "cases": {name: {"macro_f1": value} for name, _ in suite}}

        with tempfile.TemporaryDirectory() as tmp, \
                patch("gas_a.optimize.evaluate", side_effect=measured), \
                patch("gas_a.robustness.evaluate_suite", side_effect=stress):
            root = Path(tmp)
            report = train_candidate(self.baseline, self.train, self.val, root / "trial", spec,
                                     {"weakest_class": "Acetaldehyde"},
                                     {**self.cfg["optimization"], "early_stopping_patience": 1,
                                      "robustness": {"enabled": True, "max_scenario_f1_drop": .02}})
            # 干净指标在第一轮最高，稳健指标在第二轮最高，检查实际导出状态。
            self.assertEqual(report["checkpoint_selection"]["selected_epoch"], initial + 2)
            self.assertEqual(report["executed_epochs"], initial + 3)
            self.assertEqual(report["robust_validation_after"]["worst_macro_f1"], .65)
            selected = load_model(root / "trial/checkpoint.npz")
            replay = load_model(root / "trial/migrated_parent.npz")
            fit_weighted(replay, self.train, spec, "Acetaldehyde", time.monotonic() + 30, epochs_to_run=2)
            for group in ("params", "first", "second"):
                for key in getattr(selected, group):
                    np.testing.assert_array_equal(getattr(selected, group)[key], getattr(replay, group)[key])

    def test_best_migrated_epoch_restores_all_adam_tensors(self):
        initial = self.parent().meta["epochs"]
        spec = {**self.cfg["experiments"][2], "epochs": 8}

        def measured(model, dataset):
            value = evaluate(model, dataset)
            if dataset is self.val:
                value["macro_f1"] = {0: .7, 1: .72, 2: .75}.get(model.meta["epochs"] - initial, .71)
            return value

        with tempfile.TemporaryDirectory() as tmp, patch("gas_a.optimize.evaluate", side_effect=measured):
            root = Path(tmp)
            report = train_candidate(self.baseline, self.train, self.val, root / "trial", spec,
                                     {"weakest_class": "Acetaldehyde"},
                                     {**self.cfg["optimization"], "early_stopping_patience": 1,
                                      "robustness": {"enabled": False}})  # 本项验证干净指标选 epoch；稳健排序另测。
            self.assertEqual(report["checkpoint_selection"]["selected_epoch"], initial + 2)
            self.assertEqual(report["executed_epochs"], initial + 3)
            selected = load_model(root / "trial/checkpoint.npz")
            replay = load_model(root / "trial/migrated_parent.npz")
            fit_weighted(replay, self.train, spec, "Acetaldehyde", time.monotonic() + 30, epochs_to_run=2)
            for group in ("params", "first", "second"):
                for key in getattr(selected, group):
                    np.testing.assert_array_equal(getattr(selected, group)[key], getattr(replay, group)[key])


if __name__ == "__main__":
    unittest.main()
