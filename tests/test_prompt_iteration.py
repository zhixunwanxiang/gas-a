"""Prompt feedback must preserve real scope, failed mechanisms and model provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gas_a.controller_policy import measured_evidence, build_selection_prompt
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import digest, history
from gas_a.live import run_live
from gas_a.monitor import diagnose, monitor
from gas_a.pipeline import load_config
from gas_a.planning import build_prompt, configurations, correction_prompt, iteration_brief, mechanism_tags

ROOT = Path(__file__).resolve().parents[1]


class PromptIterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs/a.json")
        cls.train, cls.val = load_pair(cls.config["train"], cls.config["validation"])
        cls.model = load_model(cls.config["baseline"])
        cls.sha = digest(cls.config["baseline"])
        metric = evaluate(cls.model, cls.val)
        drift = monitor(cls.train["x"], cls.val["x"], cls.config["monitor"])
        diagnosis = diagnose(cls.train, cls.val, metric, drift)
        cls.metric = metric
        cls.evidence = measured_evidence(metric, metric, evaluate(cls.model, cls.train), diagnosis,
            drift, [], cls.sha, [], validation_sha=cls.val["sha256"], training_sha=cls.train["sha256"])

    def previous(self, index=1, **configuration):
        before = copy.deepcopy(self.metric)
        after = copy.deepcopy(before)
        after["macro_f1"] += .003
        after["per_class"]["Acetaldehyde"]["recall"] = .01
        spec = {"name": f"renamed_{index}", "learning_rate": .001/index, "epochs": 10+index,
                "class_boosts": {"Acetaldehyde": 1+index}, **configuration}
        return {"round": index, "parent_checkpoint_sha256": self.sha,
                "train_sha256": self.train["sha256"], "validation_sha256": self.val["sha256"],
                "selection_feedback": {"next_test": "检验结构化传感器扰动是否减少跨批漏检。"},
                "experiments": [{"name": spec["name"], "configuration": spec, "eligible": False,
                    "local_gate": {"accepted": False, "reasons": ["class_recall_below_minimum:Acetaldehyde"]},
                    "report": {"configuration": spec, "validation_before": before, "validation_after": after}}]}

    def test_numeric_nudges_and_ai_names_are_grouped_without_inventing_results(self):
        brief = iteration_brief(self.config, self.model, self.evidence,
                                [self.previous(1), self.previous(2)], self.sha)
        groups = brief["same_parent_same_public_data"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["attempts"], 2)
        self.assertEqual(groups[0]["eligible"], 0)
        self.assertEqual(groups[0]["best_f1_gain"], .003)
        self.assertEqual(groups[0]["best_target_recall"], {"Acetaldehyde": .01})
        self.assertEqual(groups[0]["failures"]["class_recall_below_minimum:Acetaldehyde"], 2)
        self.assertEqual(brief["last_decision_advice"][-1]["origin"], "model_hypothesis_not_measured_fact")

    def test_developer_controls_require_matching_data_parent_and_algorithm_source(self):
        from unittest.mock import patch
        from gas_a.io import read_json
        controls=read_json(ROOT/'configs/evidence/response_controls.json')
        evidence=copy.deepcopy(self.evidence)
        prompt=build_prompt(self.config,self.model,[],evidence,2,30,mode='sdk',index=1,parent_sha=self.sha)
        self.assertEqual(prompt['developer_controls']['scope'],controls['scope'])
        self.assertEqual(evidence['facts']['P']['real_model_calls'],0)
        self.assertEqual(prompt['iteration_brief']['available_history_rounds'],[])
        for field in ('train_sha256','validation_sha256','parent_checkpoint_sha256'):
            wrong=copy.deepcopy(controls);wrong[field]='sha256:'+'0'*64
            with patch('gas_a.planning.read_json',return_value=wrong):
                p=build_prompt(self.config,self.model,[],evidence,2,30,mode='sdk',index=1,parent_sha=self.sha)
            self.assertNotIn('developer_controls',p);self.assertNotIn('P',evidence['facts'])
        wrong=copy.deepcopy(controls);wrong['source_sha256']['response.py']='sha256:'+'0'*64
        with patch('gas_a.planning.read_json',return_value=wrong):
            p=build_prompt(self.config,self.model,[],evidence,2,30,mode='sdk',index=1,parent_sha=self.sha)
        self.assertNotIn('developer_controls',p)

    def test_different_parent_or_data_cannot_be_presented_as_repeated_failure(self):
        rows = [self.previous(i) for i in range(1, 5)]
        rows[1]["train_sha256"] = "other-train"
        rows[2]["validation_sha256"] = "other-validation"
        rows[3]["parent_checkpoint_sha256"] = "other-parent"
        brief = iteration_brief(self.config, self.model, self.evidence, rows, self.sha)
        self.assertEqual(brief["excluded_other_or_unknown_scope_trials"], 3)
        self.assertEqual(brief["same_parent_same_public_data"][0]["attempts"], 1)
        unknown = copy.deepcopy(self.evidence)
        unknown["facts"]["T"]["sha256"] = None
        self.assertEqual(iteration_brief(self.config, self.model, unknown, rows, self.sha)["same_parent_same_public_data"], [])

    def test_structured_augmentation_is_distinguished_from_unstructured_noise(self):
        self.assertEqual(mechanism_tags({"input_noise": .1}, self.model), ["input_noise"])
        self.assertEqual(mechanism_tags({"sensor_gain": .1, "sensor_shift": .1}, self.model), ["sensor_affine"])
        self.assertEqual(mechanism_tags({"learning_rate": .001, "epochs": 30}, self.model), ["optimization_only"])

    def test_component_used_in_a_combination_is_not_reported_as_untried(self):
        brief = iteration_brief(self.config, self.model, self.evidence,
                                [self.previous(1, sensor_gain=.1, pair_weight=1.)], self.sha)
        coverage = {r["mechanism"]: r for r in brief["component_coverage"]}
        self.assertEqual(coverage["confusion_pair"]["attempts_including_combinations"], 1)
        self.assertEqual(coverage["confusion_pair"]["standalone_attempts"], 0)
        self.assertEqual(coverage["sensor_affine"]["rounds"], [1])

    def test_selection_sees_actual_intervention_and_clean_class_results(self):
        trial = self.previous()["experiments"][0]
        prompt = build_selection_prompt({"parent": {"kind": "retain", "local": self.metric}},
                                        [trial], "parent", mode="sdk", index=2)
        view = prompt["experiments"][0]
        self.assertEqual(view["configuration"], trial["configuration"])
        target_index = self.evidence["class_order"].index("Acetaldehyde")
        self.assertEqual(view["clean_validation"]["classes"][target_index][1], .01)
        self.assertEqual(prompt["comparison"]["measured_preference"], "parent")

    def test_correction_preserves_pause_and_mechanism_feedback_without_changing_proposal(self):
        evidence = copy.deepcopy(self.evidence)
        evidence["facts"]["S"] = {"pause": True}
        evidence["facts"]["R"] = {"scope": "authorized_validation_only", "cases": {"sensor_01_missing": {"macro_f1": .5}}}
        prompt = build_prompt(self.config, self.model, [], evidence, 0, 30,
                              mode="sdk", index=2, past=[self.previous()], parent_sha=self.sha)
        raw = {"action": "experiment", "experiments": [{"name": "invalid", "batch_size": 0}]}
        original = copy.deepcopy(raw)
        retry = correction_prompt(prompt, raw, ValueError("budget exhausted"))
        self.assertEqual(raw, original)
        self.assertEqual(retry["previous_response"], original)
        self.assertFalse(retry["decision_context"]["training_allowed"])
        self.assertEqual(retry["iteration_brief"], prompt["iteration_brief"])
        self.assertEqual(retry["robust_validation"], prompt["robust_validation"])
        self.assertNotIn("recent_configurations", retry)

    def test_iteration_references_real_history_and_exact_experiment_names(self):
        raw = {"action": "experiment", "hypothesis": "检验传感器扰动的稳健性。", "reason": "已有公开误判证据。",
               "experiments": [{"name": "sensor_test", "learning_rate": .001, "batch_size": 64, "epochs": 2, "sensor_gain": .1}],
               "iteration": {"history_rounds": [1], "lesson": "目标类尚未达标。", "tests": {"sensor_test": "验证扰动是否改善目标类召回。"}}}
        past = [self.previous()]
        self.assertEqual(len(configurations(raw, self.config, self.model, past, self.sha, 2)), 1)
        bad = copy.deepcopy(raw)
        bad["iteration"]["history_rounds"] = [99]
        with self.assertRaisesRegex(ValueError, "actual available"):
            configurations(bad, self.config, self.model, past, self.sha, 2)
        bad = copy.deepcopy(raw)
        bad["iteration"]["tests"] = {"made_up": "虚构实验"}
        with self.assertRaisesRegex(ValueError, "exact name"):
            configurations(bad, self.config, self.model, past, self.sha, 2)

    def test_sdk_selection_advice_is_saved_with_real_receipt_provenance(self):
        class Client:
            model = "explicit_offline_prompt_iteration_fixture"
            def __init__(self): self.calls = 0
            def complete(self, messages, **options):
                self.calls += 1
                prompt = json.loads(messages[-1]["content"])
                reply = ({"action": "experiment", "hypothesis": "检验小幅扰动。", "reason": "公开验证证据。",
                          "experiments": [{"name": "noise_probe", "learning_rate": .001, "batch_size": 64,
                                           "epochs": 2, "input_noise": .01}]} if "search_space" in prompt else
                         {"selected_artifact_id": prompt["comparison"]["measured_preference"], "reason": "按实测与门槛选版。",
                          "next_test": "检验原响应增益与标准化噪声的差异。"})
                return {"call_id": f"offline-{self.calls}", "text": json.dumps(reply, ensure_ascii=False),
                        "response": {}, "usage": {}, "seconds": 0, "receipt_origin": "explicit_offline_fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(ROOT / "configs/a.live.json")
            cfg["runtime"] = str(Path(tmp) / "live")
            run_live(cfg, rounds=1, cloud=False, client=Client())
            note = history(Path(cfg["runtime"]) / "history.jsonl")[0]["selection_feedback"]
            self.assertEqual(note["call_id"], "offline-2")
            self.assertEqual(note["origin"], "model_hypothesis_not_measured_fact")


if __name__ == "__main__":
    unittest.main()
