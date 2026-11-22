"""真实 SDK 的离线宿主：验证缺项成绩、父文件绑定和实际数值续训。"""
import base64
from collections import deque
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from contest_harness.round import Round
from gas_a.contest import train, validation_comparison
from gas_a.controller_policy import _accepted
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import digest, read_json
from gas_a.pipeline import load_config
from fixtures import ChunkedUploads

ROOT = Path(__file__).resolve().parents[1]


class SdkHost:
    """回执来自离线替身；训练、checkpoint 重载和验证计算实际执行。"""
    def __init__(self, root, config, metric_mode, parent_name, multiple, raw_digest):
        self.root, self.config, self.metric_mode = root, config, metric_mode
        self.pending, self.calls, self.selected_prompt, self.record = deque(), [], None, None
        self.uploads, self.frames = ChunkedUploads(), []
        parent = root / "previous" / parent_name
        parent.parent.mkdir(parents=True)
        shutil.copyfile(config["baseline"], parent)
        paths = [parent_name]
        if multiple:
            other = load_model(parent)
            other.meta["stage_id"] = "different-declared-checkpoint"
            other.save(root / "previous" / "unused.npz")
            paths.insert(0, "unused.npz")
        self.parent = parent
        descriptor = {"checkpoint_paths": paths, "code_paths": [],
                      "predict_argv": ["python3", "/app/harness.py", "predict"]}
        receipt = {"artifact_id": "parent-a", "artifact": descriptor, "files": [
            {"path": name, "sha256": digest(root / "previous" / name),
             "size_bytes": (root / "previous" / name).stat().st_size} for name in paths]}
        _, self.validation = load_pair(config["train"], config["validation"])
        self.before = evaluate(load_model(parent), self.validation)
        self.frame = {"protocol": "contest.harness.workspace.v2", "prompt": "Offline compatibility fixture",
            "prompt_version": "test", "dataset_key": "A", "round_index": 1,
            "task_contract": {"task_type": "classification", "public_projection": {
                "train_path": config["train"], "validation_path": config["validation"]}},
            "model_name": "offline-fixture", "max_output_tokens": 2048, "wall_seconds": 60,
            "parent_artifact": receipt, "parent_checkpoint_sha256": digest(parent), "artifacts": [receipt],
            "resource_limits": dict(ChunkedUploads.limits)}
        if raw_digest:
            self.frame["parent_checkpoint_sha256"] = digest(parent).removeprefix("sha256:")
        if metric_mode != "missing_parent":
            self.frame["current_validation"] = {"metrics": self.before}
        self.pending.append(self.frame)

    def readline(self):
        return json.dumps(self.pending.popleft()) + "\n"

    def flush(self):
        pass

    def write(self, text):
        frame = json.loads(text)
        self.frames.append(frame)
        reply = {"id": frame.get("id")}
        if frame["type"].startswith("upload_"):
            reply.update(self.uploads.handle(frame))
        elif frame["type"] == "history_request":
            reply.update(type="history_response", records=[{"id": "prior-actual-read"}])
        elif frame["type"] == "model_request":
            prompt = json.loads(frame["messages"][0]["content"])
            if "search_space" in prompt:
                answer = {"action": "experiment", "hypothesis": "Measure a bounded continuation.",
                    "reason": "Offline test uses only the authorized public projection.",
                    "experiments": [{"name": "compatibility_trial", "learning_rate": .003,
                        "epochs": 24, "batch_size": 128, "balance": .5, "weak_boost": 1.5,
                        "class_boosts": {"Acetaldehyde": 2}, "l2": .001}]}
            else:
                self.selected_prompt = prompt
                answer = {"selected_artifact_id": prompt["comparison"]["measured_preference"],
                          "reason": "Use the explicitly identified common validation scope."}
            call_id = f"offline-call-{len(self.calls) + 1}"
            self.calls.append(call_id)
            reply.update(type="model_response", call_id=call_id,
                         response={"choices": [{"message": {"content": json.dumps(answer)}}]})
        elif frame["type"] == "candidate_request":
            if frame["parent_checkpoint_sha256"] != self.frame["parent_checkpoint_sha256"]:
                raise AssertionError("Candidate lost the host parent binding")
            self.candidate = self.root / "measured-candidate"
            self.candidate.mkdir()
            for item in self.uploads.files(frame, "candidate"):
                path = self.candidate / item["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(item["data_base64"], validate=True))
            measured = evaluate(load_model(self.candidate / "checkpoint.npz"), self.validation)
            if self.metric_mode == "regressing_candidate":
                measured = {**measured, "macro_f1": .1}
            elif self.metric_mode == "missing_candidate":
                measured = {}
            reply.update(type="candidate_response", artifact_id="candidate-a", summary={"metrics": measured})
        elif frame["type"] == "state_request":
            self.state = {f["path"]: base64.b64decode(f["data_base64"]) for f in self.uploads.files(frame, "state")}
            reply.update(type="state_response", state_sha256="sha256:" + "1" * 64)
        elif frame["type"] == "round_complete":
            self.record = frame["record"]
            reply = {"type": "round_ack", "accepted": True}
        else:
            raise AssertionError(frame["type"])
        self.pending.append(reply)
        return len(text)


class SdkCompatibilityTests(unittest.TestCase):
    def run_round(self, mode="complete", *, parent_name="checkpoint.npz", multiple=False, raw_digest=False):
        temporary = tempfile.TemporaryDirectory(prefix="gas-sdk-compatibility-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = load_config(ROOT / "configs/a.json")
        # 单独验证宿主比较与路径合同；默认类别/稳健保护由专项测试覆盖。
        config["optimization"]["robustness"]["enabled"] = False
        config["gate"].update(minimum_class_recall={}, reject_zero_target_recall=False,
            max_accuracy_drop=1., max_class_recall_drop=1., max_parent_fixed_f1_drop=1.)
        host = SdkHost(root, config, mode, parent_name, multiple, raw_digest)
        rnd = Round.start(host, host)
        rnd.config = config
        rnd.parent.root = root / "previous"
        with patch.dict(os.environ, {"CONTEST_WORK": str(root / "work")}):
            train(rnd)
        self.assertTrue(rnd._finished)
        self.assertEqual(host.record["model_call_ids"], host.calls)
        self.assertEqual(host.record["history_ids"], ["prior-actual-read"])
        detail = read_json(next((root / "work").glob("gas-a-*/round.json")))
        self.assertTrue(detail["experiments"][0]["local_gate"]["accepted"])
        self.assertGreater(load_model(host.candidate / "checkpoint.npz").meta["steps"],
                           load_model(host.parent).meta["steps"])
        return host, detail["experiments"][0]

    def test_missing_parent_score_promotes_by_explicit_common_local_validation(self):
        host, trial = self.run_round("missing_parent")
        self.assertEqual(host.record["selected_artifact_id"], "candidate-a")
        self.assertTrue(trial["eligible"])
        self.assertTrue(_accepted(trial))
        self.assertIsNone(trial["host_gate_passed"])
        self.assertEqual(trial["validation_comparison"]["basis"], "local")
        self.assertEqual(host.selected_prompt["comparison"]["basis"], "local")
        self.assertNotIn("host", host.selected_prompt["eligible_artifacts"]["parent-a"])
        self.assertIn("host", host.selected_prompt["eligible_artifacts"]["candidate-a"])
        self.assertEqual(host.selected_prompt["experiments"][0]["validation_comparison"],
                         trial["validation_comparison"])

    def test_complete_host_scores_keep_host_gate_and_regression_is_rejected(self):
        for mode, selected in (("complete", "candidate-a"), ("regressing_candidate", "parent-a")):
            with self.subTest(mode=mode):
                host, trial = self.run_round(mode)
                self.assertEqual(host.record["selected_artifact_id"], selected)
                self.assertEqual(trial["validation_comparison"]["basis"], "host")
                self.assertEqual(trial["host_gate_passed"], mode == "complete")
                self.assertEqual(trial["eligible"], mode == "complete")

    def test_missing_candidate_host_measurement_cannot_be_replaced_by_local_score(self):
        host, trial = self.run_round("missing_candidate")
        self.assertEqual(host.record["selected_artifact_id"], "parent-a")
        self.assertFalse(trial["eligible"])
        self.assertEqual(trial["validation_comparison"]["reason"], "candidate_host_measurement_missing")

    def test_parent_is_selected_by_digest_among_declared_nested_checkpoints(self):
        host, _ = self.run_round(parent_name="nested/actual-weights.npz", multiple=True, raw_digest=True)
        self.assertEqual(host.record["selected_artifact_id"], "candidate-a")
        saved = json.loads(host.state["lifecycle.json"])
        self.assertEqual(saved["baseline_sha256"], digest(host.parent))
        self.assertEqual(digest(host.parent), digest(ROOT / "data/official_a/checkpoint.npz"))

    def test_local_fallback_cannot_mix_high_host_score_with_local_parent(self):
        result = validation_comparison({"macro_f1": .8}, {"macro_f1": .7}, None,
                                       {"macro_f1": .99}, .002)
        self.assertEqual(result["basis"], "local")
        self.assertFalse(result["accepted"])
        self.assertAlmostEqual(result["gain_macro_f1"], -.1)

    def test_parent_digest_mismatch_is_rejected_before_any_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(ROOT / "configs/a.json")
            host = SdkHost(Path(tmp), config, "complete", "weights.npz", False, False)
            host.frame["parent_checkpoint_sha256"] = "sha256:" + "0" * 64
            rnd = Round.start(host, host)
            rnd.parent.root = Path(tmp) / "previous"
            with self.assertRaisesRegex(ValueError, "matches the host binding"):
                train(rnd)
            self.assertEqual(host.calls, [])
