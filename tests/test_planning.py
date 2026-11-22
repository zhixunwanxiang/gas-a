"""默认AI组合配置、执行兼容性与拒绝后再次请求AI修正的端到端验证。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gas_a.controller_policy import measured_evidence
from gas_a.data import load_model, load_pair, evaluate
from gas_a.io import digest, history, read_json
from gas_a.live import run_live
from gas_a.model_records import collect_model_calls, purpose_for
from gas_a.monitor import monitor, diagnose
from gas_a.optimize import train_candidate
from gas_a.pipeline import load_config
from gas_a.planning import build_prompt, configurations, resolve_plan, correction_prompt
from gas_a.temporal import run_temporal

ROOT = Path(__file__).resolve().parents[1]


def proposal(**updates):
    spec = {"name": "ai_joint_update", "learning_rate": .0005, "batch_size": 64, "epochs": 2,
            "loss": "focal", "focal_gamma": 1., "balance": .3, "class_boosts": {"Acetaldehyde": 2},
            "freeze": "first_hidden", "architecture": {"hidden": 48, "hidden_layers": 2},
            "anchor_l2": .01, "schedule": "cosine", "final_lr_ratio": .2, "input_noise": .01, **updates}
    return {"action": "experiment", "experiments": [spec], "hypothesis": "保留父特征并调整容量、损失和类别贡献，验证弱类识别。",
            "reason": "基于训练与当前分类差异设计组合实验，收益待检验。"}


class Controller:
    model = "explicit-offline-composition-fixture"

    def __init__(self, reject=False, always_invalid=False):
        self.requests = []
        self.reject, self.always_invalid = reject, always_invalid

    def complete(self, messages, **options):
        request = json.loads(messages[-1]["content"])
        self.requests.append(request)
        if "search_space" in request:
            response = proposal()
            if self.always_invalid or self.reject and "validation_error" not in request:
                response["experiments"][0]["batch_size"] = 0
        else:
            response = {"selected_artifact_id": next(iter(request["eligible_artifacts"])), "reason": "保留实测合格版本。"}
        text = json.dumps(response, ensure_ascii=False)
        return {"call_id": f"offline-composition-{len(self.requests)}", "text": text,
                "response": {"choices": [{"message": {"content": text}}]}, "seconds": .001,
                "usage": {"prompt_tokens": 9, "completion_tokens": 7}, "receipt_origin": "explicit_offline_test_fixture"}


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / "configs/a.adaptive.json")
        cls.train, cls.val = load_pair(cls.cfg["train"], cls.cfg["validation"])
        cls.model = load_model(cls.cfg["baseline"])
        cls.sha = digest(cls.cfg["baseline"])
        metric = evaluate(cls.model, cls.val)
        drift = monitor(cls.train["x"], cls.val["x"], cls.cfg["monitor"])
        cls.diagnosis = diagnose(cls.train, cls.val, metric, drift)
        cls.evidence = measured_evidence(metric, metric, evaluate(cls.model, cls.train), cls.diagnosis,
                                         drift, [], cls.sha, [], validation_sha=cls.val["sha256"])

    def compile(self, raw, past=(), model=None):
        return configurations(raw, self.cfg, model or self.model, past, self.sha, 2, self.diagnosis["weakest_class"],
                              train_sha=self.train['sha256'], validation_sha=self.val['sha256'])

    def test_default_prompt_exposes_composable_parameters_not_a_menu(self):
        prompt = build_prompt(self.cfg, self.model, [], self.evidence, 2, 30, mode="sdk", index=1, parent_sha=self.sha)
        self.assertEqual(self.cfg["controller"]["planning_mode"], "compose")
        self.assertTrue(prompt["decision_context"]["training_allowed"])
        self.assertNotIn("allowed_experiments", prompt)
        self.assertNotIn("experiment_guide", prompt)
        self.assertNotIn("parameter_changes", prompt["plan_contract"])
        self.assertEqual(prompt["search_space"]["batch_size"], [16, 256])
        self.assertEqual(prompt["search_space"]["architecture"]["hidden"], [32, 128])
        self.assertEqual(purpose_for([{"content": json.dumps(prompt)}]), "experiment_planning")
        self.assertLess(len(json.dumps(prompt, ensure_ascii=False)), 8500)

    def test_many_combined_fields_execute_real_parent_continuation_and_freezing(self):
        raw = proposal()
        original = copy.deepcopy(raw)
        plan, specs = resolve_plan(raw, self.cfg, self.model, [], self.evidence, 2, [], self.sha,
                                   weak_class=self.diagnosis["weakest_class"])
        self.assertEqual(raw, original)
        self.assertEqual(plan["experiments"], raw["experiments"])
        self.assertGreater(len(specs[0]), 10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "combo"
            report = train_candidate(self.cfg["baseline"], self.train, self.val, root, specs[0], self.diagnosis,
                                     {**self.cfg["optimization"], "select_best_epoch": False})
            child, migrated = load_model(root / "checkpoint.npz"), load_model(root / "migrated_parent.npz")
            self.assertEqual(report["architecture"], {"hidden": 48, "hidden_layers": 2})
            self.assertEqual(report["final_steps"], self.model.meta["steps"] + 14)
            self.assertEqual(report["parent_checkpoint_sha256"], self.sha)
            self.assertEqual(child.meta["seed"], self.model.meta["seed"])
            for group in ("params", "first", "second"):
                for name in ("w1", "b1"):
                    self.assertEqual(getattr(child, group)[name].tobytes(), getattr(migrated, group)[name].tobytes())
            self.assertFalse(np.array_equal(child.params["w2"], migrated.params["w2"]))
            np.testing.assert_array_equal(child.mean, self.model.mean)
            np.testing.assert_array_equal(child.scale, self.model.scale)
            self.assertEqual(child.predict(self.val["x"]).shape, (1244, 6))
        self.assertEqual(digest(self.cfg["baseline"]), self.sha)

    def test_unsafe_unknown_conflicting_and_over_budget_proposals_are_rejected(self):
        for update in ({"seed": 1}, {"optimizer": "SGD"}, {"input_features": 127}, {"classes": ["x"]},
                       {"epochs": 33}, {"batch_size": 0}, {"batch_size": True}, {"learning_rate": float("nan")},
                       {"loss": "new_algorithm"}, {"loss": "cross_entropy", "focal_gamma": 1},
                       {"freeze": "features", "kept_sensors": [3, 4]}, {"architecture": {"hidden": 16}},
                       {"frozen_parameters": ["w1", "b1", "w2", "b2", "w3", "b3"]},
                       {"architecture": {"hidden_layers": 1}, "frozen_parameters": ["w3"]},
                       {"epochs": 2, "minimum_epochs": 3}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.compile(proposal(**update))
        with self.assertRaises(ValueError):
            self.compile({**proposal(), "experiments": proposal()["experiments"] * 3})
        with self.assertRaises(ValueError):
            self.compile({**proposal(), "quality_gate": {"min_macro_f1_gain": -1}})

    def test_configuration_identity_does_not_depend_on_ai_name(self):
        first = self.compile(proposal())[0]
        past = [{"parent_checkpoint_sha256": self.sha, 'train_sha256':self.train['sha256'],
                 'validation_sha256':self.val['sha256'], "experiments": [{"configuration": first}]}]
        with self.assertRaisesRegex(ValueError, "already attempted"):
            self.compile(proposal(name="renamed_same_configuration"), past)
        changed = self.compile(proposal(name="changed", learning_rate=.0007), past)
        self.assertEqual(changed[0]["learning_rate"], .0007)
        self.assertTrue(self.compile(proposal(), [{**past[0], "parent_checkpoint_sha256": "different"}]))

    def test_long_but_bounded_reason_preserves_raw_text_without_a_paid_revision(self):
        raw = proposal()
        raw["reason"] = "当前弱类识别不足，按诊断调整训练贡献，实际收益仍需要评估。" * 4
        self.assertGreater(len(raw["reason"]), 80)
        plan, specs = resolve_plan(raw, self.cfg, self.model, [], self.evidence, 2, [], self.sha)
        self.assertEqual(plan["reason"], raw["reason"])
        self.assertEqual(len(specs), 1)

    def test_compiled_configuration_cannot_be_program_repaired_as_an_ai_reply(self):
        with self.assertRaisesRegex(ValueError, "revised by AI"):
            resolve_plan(proposal(batch_size=0), self.cfg, self.model, [], self.evidence, 2, [], self.sha, repair=True)
        prompt = build_prompt(self.cfg, self.model, [], self.evidence, 2, 30, mode="sdk", index=1)
        retry = correction_prompt(prompt, proposal(batch_size=0), ValueError("batch_size must be 16..256"))
        self.assertEqual(retry["previous_response"]["experiments"][0]["batch_size"], 0)
        self.assertEqual(retry["search_space"]["batch_size"], [16, 256])
        self.assertIn("16..256", retry["validation_error"])
        self.assertNotIn("recent_configurations", retry)

    def temporal(self, root, client):
        cfg = copy.deepcopy(self.cfg)
        cfg["runtime"], cfg["experiments"] = str(root / "runtime"), []
        for batch, offset in ((3, 0), (4, 40)):
            rows = [str(int(y)+1) + " " + " ".join(f"{i+1}:{v:.17g}" for i, v in enumerate(x))
                    for x, y in zip(self.val["x"][offset:offset+40], self.val["y"][offset:offset+40])]
            (root / f"batch{batch}.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
        summary = run_temporal(cfg, root, (3, 4), cloud=False, client=client)
        return summary, read_json(Path(cfg["runtime"]) / "windows/batch03/result.json"), cfg

    def test_temporal_rejected_configuration_gets_a_real_second_ai_proposal_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = Controller(reject=True)
            summary, row, cfg = self.temporal(Path(tmp), client)
            self.assertTrue(summary["workflow_completed"])
            self.assertEqual(summary["controller_failed_windows"], 0)
            self.assertEqual(len(client.requests), 3)
            self.assertIn("validation_error", client.requests[1])
            self.assertEqual(row["planning_rejections"][0]["proposal"]["experiments"][0]["batch_size"], 0)
            self.assertEqual(row["plan"]["experiments"][0]["batch_size"], 64)
            self.assertNotIn("error", row["experiments"][0])
            self.assertEqual(row["experiments"][0]["report"]["configuration"]["batch_size"], 64)
            self.assertEqual(len(collect_model_calls(Path(cfg["runtime"]) / "session")), 3)

    def test_live_rejected_configuration_is_revised_and_all_calls_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = copy.deepcopy(self.cfg)
            cfg["runtime"], cfg["experiments"] = str(Path(tmp) / "live"), []
            client = Controller(reject=True)
            run_live(cfg, cloud=False, client=client)
            row = history(Path(cfg["runtime"]) / "history.jsonl")[0]
            self.assertEqual(len(client.requests), 3)
            self.assertEqual(row["planning_attempts"][0]["accepted"], False)
            self.assertTrue(row["planning_attempts"][1]["accepted"])
            self.assertEqual(row["planning_attempts"][0]["proposal"]["experiments"][0]["batch_size"], 0)
            self.assertEqual(row["plan"]["experiments"][0]["batch_size"], 64)
            self.assertEqual(row["experiments"][0]["configuration"]["batch_size"], 64)
            self.assertEqual(row["review"]["costs"]["calls"], 3)

    def test_repeated_rejections_are_bounded_and_never_execute_an_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, patch("gas_a.candidates.train_candidate", side_effect=AssertionError("invalid config trained")):
            client = Controller(always_invalid=True)
            summary, row, _ = self.temporal(Path(tmp), client)
            self.assertTrue(summary["workflow_completed"])
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(row["plan"]["action"], "observe")
            self.assertEqual(row["experiments"], [])
            self.assertEqual(row["plan"]["decision_origin"], "program_safe_observe")


if __name__ == "__main__":
    unittest.main()
