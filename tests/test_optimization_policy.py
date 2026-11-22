"""全传感器训练、真实目标进展与合法降权，不自动注入固定子集配方。"""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from gas_a.adaptive import fixed_gate, window_gate, policy
from gas_a.controller_policy import measured_evidence
from gas_a.data import load_model, load_pair, evaluate
from gas_a.io import digest
from gas_a.monitor import monitor, diagnose
from gas_a.pipeline import load_config
from gas_a.planning import build_prompt, configurations, resolve_plan
from gas_a.optimize import train_candidate, fit_weighted

ROOT = Path(__file__).resolve().parents[1]


class OptimizationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.adaptive.json")
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.model = load_model(cls.cfg["baseline"])
        cls.sha = digest(cls.cfg["baseline"])
        cls.metric = evaluate(cls.model, cls.val)
        drift = monitor(cls.train["x"], cls.val["x"], cls.cfg["monitor"])
        cls.diagnosis = diagnose(cls.train, cls.val, cls.metric, drift)
        cls.evidence = measured_evidence(cls.metric, cls.metric, evaluate(cls.model, cls.train),
            cls.diagnosis, drift, [], cls.sha, [], validation_sha=cls.val["sha256"])

    def raw(self, **updates):
        return {"action": "experiment", "hypothesis": "检验类别贡献调整能否改善当前错误。", "reason": "由数据提出，效果待验证。",
                "experiments": [{"name": "data_driven", "learning_rate": .0005, "batch_size": 128, "epochs": 2, **updates}]}

    def test_downweight_and_equivalent_provider_format_preserve_original_proposal(self):
        for boosts in ({"Acetaldehyde": .5}, {"classes": ["Acetaldehyde"], "multiplier": .5}):
            raw = self.raw(class_boosts=boosts); original = copy.deepcopy(raw)
            plan, specs = resolve_plan(raw, self.cfg, self.model, [], self.evidence, 1, [], self.sha)
            self.assertEqual(raw, original)
            self.assertEqual(specs[0]["class_boosts"], {"Acetaldehyde": .5})
            self.assertEqual(plan["experiments"], raw["experiments"])
            if "classes" in boosts:
                self.assertEqual(plan["configuration_normalizations"][0]["origin"], "lossless_format_expansion_not_parameter_change")
        for value in (0, -.5, 5.1, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configurations(self.raw(class_boosts={"Acetaldehyde": value}), self.cfg, self.model, [], self.sha, 1)

    def test_total_f1_gain_cannot_hide_unresolved_zero_target_or_target_regression(self):
        cfg = policy(self.cfg)
        active = copy.deepcopy(self.metric)
        candidate = copy.deepcopy(self.metric)
        candidate["macro_f1"] = .8
        self.assertIn("unresolved_zero_target_recall:Acetaldehyde", fixed_gate(candidate, active, active, cfg)["reasons"])
        self.assertFalse(window_gate(candidate, active, active, cfg)["accepted"])
        active["per_class"]["Acetaldehyde"]["recall"] = .9
        candidate["per_class"]["Acetaldehyde"]["recall"] = .5
        self.assertIn("target_recall_regression:Acetaldehyde", fixed_gate(candidate, active, self.metric, cfg)["reasons"])
        candidate["per_class"]["Acetaldehyde"]["recall"] = .89
        self.assertTrue(fixed_gate(candidate, active, self.metric, cfg)["accepted"])

    def test_fixed_validation_floor_also_limits_regression_from_actual_parent(self):
        active = copy.deepcopy(self.metric); active["macro_f1"] = .8
        active["per_class"]["Acetaldehyde"]["recall"] = .8
        candidate = copy.deepcopy(active); candidate["macro_f1"] = .75
        result = fixed_gate(candidate, active, self.metric, policy(self.cfg))
        self.assertIn("fixed_f1_regression_from_parent", result["reasons"])

    def test_default_training_updates_connections_for_every_sensor(self):
        plan, specs = resolve_plan(self.raw(), self.cfg, self.model, [], self.evidence, 2, [], self.sha)
        self.assertEqual(len(specs), 1)
        self.assertEqual(plan["experiment_names"], ["data_driven"])
        self.assertEqual(specs[0]["proposal_origin"], "ai_composed_configuration")
        self.assertNotIn("kept_sensors", specs[0])
        with tempfile.TemporaryDirectory() as tmp:
            limits = {**self.cfg["optimization"], "checkpoint_gate": policy(self.cfg), "checkpoint_reference": self.metric}
            report = train_candidate(self.cfg["baseline"], self.train, self.val, Path(tmp) / "control", specs[0],
                                     {"weakest_class": "Acetaldehyde"}, limits)
            child = load_model(Path(tmp) / "control/checkpoint.npz")
            self.assertEqual(report["parent_checkpoint_sha256"], self.sha)
            self.assertEqual(report["initial_steps"], self.model.meta["steps"])
            self.assertGreater(child.meta["steps"], self.model.meta["steps"])
            self.assertEqual(child.meta["seed"], self.model.meta["seed"])
            np.testing.assert_array_equal(child.mean, self.model.mean)
            np.testing.assert_array_equal(child.scale, self.model.scale)
            self.assertEqual(report["freezing"]["projected_parameter_count"], 0)
            for sensor in range(16):
                rows = slice(sensor * 8, (sensor + 1) * 8)
                self.assertTrue(np.any(child.params["w1"][rows] != self.model.params["w1"][rows]))
        self.assertEqual(digest(self.cfg["baseline"]), self.sha)

    def test_feedback_remains_visible_without_fixed_program_controls(self):
        report = {"configuration": {"name": "previous", "learning_rate": .003, "epochs": 10},
                  "initial_steps": 80, "final_steps": 88, "executed_steps": 112,
                  "train_before": self.metric, "train_after": copy.deepcopy(self.metric),
                  "validation_before": self.metric, "validation_after": self.metric,
                  "parent_checkpoint_sha256": self.sha, "weak_class": "Acetaldehyde"}
        report["train_after"]["per_class"]["Acetaldehyde"]["recall"] = 1.
        past = [{"parent_checkpoint_sha256": self.sha, "shadow_before": self.metric,
                 "experiments": [{"name": "previous", "report": report, "shadow_metrics": self.metric}]}]
        prompt = build_prompt(self.cfg, self.model, [], self.evidence, 1, 30, mode="sdk", index=1,
                              past=past, parent_sha=self.sha)
        self.assertNotIn("allowed_experiments", prompt)
        self.assertNotIn("program_controls", prompt)
        self.assertEqual(prompt["training_work"]["updates_per_epoch_examples"], {"32": 14, "128": 4})
        feedback = prompt["recent_configurations"][0]
        self.assertEqual(feedback["selected_updates"], 8)
        self.assertEqual(feedback["executed_updates"], 32)
        self.assertEqual(feedback["target_recall_feedback"]["Acetaldehyde"]["train_after"], 1.)
        self.assertEqual(feedback["target_recall_feedback"]["Acetaldehyde"]["fixed_after"], 0.)

    def test_unmasked_later_parent_resumes_learning_with_exact_adam_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = load_model(self.cfg["baseline"])
            first = {"name": "later", "learning_rate": .001, "epochs": 1, "batch_size": 128,
                     "balance": 0., "weak_boost": 1., "l2": 0.}
            fit_weighted(parent, self.train, first, "Acetaldehyde", time.monotonic() + 30)
            # 模拟旧候选屏蔽一个传感器后的实际父状态，保持原有 Adam 矩。
            parent.params["w1"][8:16] = 0
            path = root / "actual-selected-parent.npz"
            parent.save(path)
            metric = evaluate(parent, self.val)
            self.assertEqual(metric["per_class"]["Acetaldehyde"]["recall"], 0)
            _, specs = resolve_plan(self.raw(), self.cfg, parent, [], self.evidence, 2, [], digest(path))
            limits = {**self.cfg["optimization"], "checkpoint_gate": policy(self.cfg), "checkpoint_reference": self.metric}
            report = train_candidate(path, self.train, self.val, root / "resumed", specs[0],
                                     {"weakest_class": "Acetaldehyde"}, limits)
            child = load_model(root / "resumed/checkpoint.npz")
            self.assertEqual(report["parent_checkpoint_sha256"], digest(path))
            self.assertEqual(report["initial_steps"], parent.meta["steps"])
            self.assertGreater(report["initial_steps"], self.model.meta["steps"])
            self.assertTrue(np.any(child.params["w1"][8:16] != 0))
            self.assertEqual(report["freezing"]["projected_parameter_count"], 0)
            expected = load_model(path)
            fit_weighted(expected, self.train, specs[0], "Acetaldehyde", time.monotonic() + 30,
                         epochs_to_run=report["final_epochs"] - report["initial_epochs"])
            for group in ("params", "first", "second"):
                for name in child.params:
                    self.assertEqual(getattr(child, group)[name].tobytes(), getattr(expected, group)[name].tobytes())

    def test_shipped_configs_do_not_supply_fixed_sensor_candidates(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(config=path.name):
                cfg = load_config(path)
                self.assertNotIn("reference_configurations", cfg["optimization"])
                self.assertTrue(all("kept_sensors" not in spec for spec in cfg["experiments"]))

    def test_old_automatic_reference_config_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-config.json"
            path.write_text(json.dumps({"extends": str(ROOT / "configs/a.json"), "optimization": {
                "reference_configurations": [{"name": "old_reference"}]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Automatic reference configurations are disabled"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
