"""比赛 SDK 协议：入口、候选、回执、状态、确认及逐行预测。"""
import base64
import contest_harness
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from gas_a.contest import measured_metrics
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import digest, read_json
from gas_a.pipeline import load_config

from fixtures import fixture_plan, ChunkedUploads

ROOT = Path(__file__).resolve().parents[1]


def environment(app_root=ROOT):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(app_root / "src"), str(app_root), str(Path(contest_harness.__file__).parent.parent)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    return env


class ProtocolTests(unittest.TestCase):
    def test_sdk_prediction_replies_before_reading_next_sample(self):
        from contest_harness.predict import serve_predictions, Sample
        from gas_a.contest import predict
        row = json.loads((ROOT / "data/official_a/public/validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
        request = json.dumps({"task": "classification", "inputs": row["inputs"]}) + "\n"
        output = io.StringIO()

        def source():
            yield request
            self.assertEqual(len(output.getvalue().splitlines()), 1)
            yield request

        with patch("contest_harness.predict.model_dir", return_value=ROOT / "data/official_a"), patch.dict(os.environ, {}, clear=True):
            serve_predictions(predict, source(), output)
            with self.assertRaises(ValueError):
                predict(Sample("regression", row["inputs"]), ROOT / "data/official_a")
        lines = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], lines[1])
        scores = [piece["score"] for piece in lines[0]["outputs"]]
        self.assertEqual(len(scores), 6)
        self.assertEqual(scores.count(max(scores)), 1)
        self.assertAlmostEqual(sum(scores), 1)

    def test_describe(self):
        result = subprocess.run([sys.executable, str(ROOT / "harness.py"), "describe"], cwd=ROOT,
                                env=environment(), capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["tasks"], ["classification"])
        self.assertEqual(value["model_access"], "host_gateway_rpc")
        self.assertEqual(value["sdk_version"], "1.0.4")

    def test_missing_host_measurements_do_not_become_local_scores(self):
        self.assertIsNone(measured_metrics({"artifact_id": "x", "summary": {"metrics": {}}}))
        self.assertIsNone(measured_metrics({"macro_f1": True}))
        self.assertIsNone(measured_metrics({"macro_f1": float("nan")}))

    def test_sdk_round_real_candidate_receipts_history_and_state(self):
        self.exercise_round()

    def test_sdk_single_eligible_decision_and_long_chinese_reason(self):
        self.exercise_round(only_parent=True, selection_reason="审核依据：保留当前父检查点。" * 400)

    def test_sdk_corrected_selection_cites_all_actual_calls(self):
        self.exercise_round(retry_selection=True)

    def test_sdk_migrated_candidates_keep_parent_and_task_a_contract(self):
        self.exercise_round(architecture=True)

    def test_sdk_executes_freezing_and_class_focus_in_the_existing_two_calls(self):
        self.exercise_round(training_focus=True)

    def test_sdk_default_27b_can_choose_each_architecture_without_architecture_config(self):
        for choice in ("widen_hidden", "deepen_hidden", "widen_deepen"):
            with self.subTest(choice=choice):
                self.exercise_round(architecture_choice=choice)

    def test_sdk_default_composed_configuration_and_ai_revision_execute(self):
        self.exercise_round(composition=True, rejected_configuration=True)

    def test_sdk_response_head_is_uploaded_measured_selected_and_predictable(self):
        self.exercise_round(composition=True, response_adapter=True)

    def test_sdk_revises_response_freeze_conflict_before_training(self):
        self.exercise_round(composition=True,response_adapter=True,rejected_configuration=True,freeze_response_conflict=True)

    def test_sdk_parent_fallback_is_authorized_by_a_real_round_model_receipt(self):
        self.exercise_round(authorized_fallback=True)

    def test_contest_image_sources_work_without_direct_clients(self):
        with tempfile.TemporaryDirectory(prefix="gas-image-sources-") as temporary:
            app = Path(temporary)
            # 根据实际 Docker COPY 指令准备输入，避免维护第二份文件清单。
            dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
            for line in dockerfile.splitlines():
                parts = shlex.split(line)
                if not parts or parts[0] != "COPY" or not parts[-1].startswith("/app/"):
                    continue
                destination = app / parts[-1].removeprefix("/app/")
                for name in parts[1:-1]:
                    source = ROOT / name
                    self.assertTrue(source.is_file(), "Unsupported Docker COPY source: " + name)
                    target = destination / source.name if parts[-1].endswith("/") else destination
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            absent = ("live_access", "live", "live_commands", "cloud_records")
            code = ("import importlib.util, gas_a.contest, sys; "
                    f"assert all(importlib.util.find_spec('gas_a.'+name) is None for name in {absent!r}); "
                    "assert not any(name.startswith('gas_a.live') for name in sys.modules)")
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=app, env=environment(app),
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.exercise_round(app_root=app)

    def exercise_round(self, *, app_root=ROOT, only_parent=False, retry_selection=False, architecture=False, training_focus=False,
                       architecture_choice=None, composition=False, rejected_configuration=False,
                       authorized_fallback=False, response_adapter=False, freeze_response_conflict=False,
                       selection_reason="Offline fixture selects measured eligible maximum"):
        cfg = load_config(ROOT / "configs" / ("a.architecture.json" if architecture else "a.json"))
        if not composition:
            cfg["controller"]["planning_mode"] = "menu"  # 显式覆盖原菜单兼容协议。
        else:
            self.assertEqual(cfg["controller"]["planning_mode"], "compose")
        if training_focus or architecture_choice:
            cfg["experiments"] = [{**cfg["experiments"][0], "epochs": 2}]
            self.assertNotIn("architecture", cfg["experiments"][0])
        train, validation = load_pair(cfg["train"], cfg["validation"])
        metric = evaluate(load_model(cfg["baseline"]), validation)
        parent = {"artifact_id": "baseline-a", "artifact": {"checkpoint_paths": ["checkpoint.npz"],
                  "code_paths": [], "predict_argv": ["python3", "/app/harness.py", "predict"]},
                  "files": [{"path": "checkpoint.npz", "sha256": digest(cfg["baseline"]),
                             "size_bytes": Path(cfg["baseline"]).stat().st_size}]}
        start = {"type": "round_start", "protocol": "contest.harness.workspace.v2", "prompt": "Offline test prompt",
                 "prompt_version": "test-1", "dataset_key": "A", "round_index": 1,
                 "task_contract": {"task_type": "classification", "public_projection": {"train_path": cfg["train"], "validation_path": cfg["validation"]}},
                 "model_name": "offline-fixture-only", "max_output_tokens": 768, "wall_seconds": 60,
                 "parent_artifact": parent, "parent_checkpoint_sha256": digest(cfg["baseline"]),
                 "artifacts": [parent], "current_validation": {"metrics": metric},
                 "resource_limits": dict(ChunkedUploads.limits)}
        with tempfile.TemporaryDirectory() as tmp:
            env = environment(app_root)
            env.update(CONTEST_WORK=tmp, GAS_A_PARENT_ROOT=str(Path(cfg["baseline"]).parent), GAS_A_CONFIG=str(app_root / "configs/a.json"))
            config_path = Path(tmp) / "test-config.json"
            config_path.write_text(json.dumps(cfg), encoding="utf-8")
            env["GAS_A_CONFIG"] = str(config_path)
            frames, calls, candidates, failures = [], [], {}, []
            uploads = ChunkedUploads()
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errors:
                process = subprocess.Popen([sys.executable, str(app_root / "harness.py"), "train"], cwd=app_root, env=env,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                           text=True, encoding="utf-8", bufsize=1)

                def send(frame):
                    process.stdin.write(json.dumps(frame) + "\n")
                    process.stdin.flush()

                def host():
                    try:
                        send(start)
                        sequence = 0
                        while True:
                            line = process.stdout.readline()
                            if not line:
                                raise AssertionError("Participant exited before round_complete")
                            frame = json.loads(line)
                            frames.append(frame)
                            kind = frame["type"]
                            if kind == "round_complete":
                                record = frame["record"]
                                self.assertEqual(record["history_ids"], ["history-test-1"])
                                self.assertEqual(record["model_call_ids"], calls)
                                self.assertEqual(len(calls), 3 if retry_selection or rejected_configuration or authorized_fallback else 2)
                                self.assertIn(record["selected_artifact_id"], ["baseline-a", *candidates])
                                if authorized_fallback:
                                    self.assertEqual(record["selected_artifact_id"], "baseline-a")
                                    self.assertIn("显式兜底", record["reason"])
                                else:
                                    self.assertTrue(selection_reason.startswith(record["reason"]))
                                self.assertTrue(record["reason"].strip())
                                self.assertLessEqual(len(record["reason"].encode("utf-8")), 4000)
                                if len(selection_reason.encode("utf-8")) > 4000:
                                    self.assertGreater(len(record["reason"].encode("utf-8")), 3996)
                                elif not authorized_fallback:
                                    self.assertEqual(record["reason"], selection_reason)
                                send({"type": "round_ack", "accepted": True})
                                process.stdin.close()
                                return
                            if kind == "round_failed":
                                raise AssertionError(frame)
                            self.assertEqual(frame["id"], sequence)
                            sequence += 1
                            if kind.startswith("upload_"):
                                reply = uploads.handle(frame)
                            elif kind == "history_request":
                                reply = {"type": "history_response", "records": [{"id": "history-test-1", "reason": "Prior test record"}]}
                            elif kind == "model_request":
                                # 官方提示词由 SDK 网关宿主绑定，参赛模块不重复注入。
                                self.assertFalse(any("Offline test prompt" in m.get("content", "") for m in frame["messages"]))
                                prompt = json.loads(frame["messages"][0]["content"])
                                if "search_space" in prompt:
                                    self.assertTrue(composition)
                                    self.assertNotIn("allowed_experiments", prompt)
                                    self.assertNotIn("experiment_guide", prompt)
                                    self.assertNotIn("program_controls", prompt)
                                    self.assertEqual(prompt["budget"]["max_experiments"], 2)
                                    cfgs = [{"name": f"ai_combo_{i}", "learning_rate": rate, "epochs": 2, "batch_size": 64,
                                             "loss": "focal", "focal_gamma": 1., "balance": .3, "class_boosts": {"Acetaldehyde": 2},
                                             "freeze": "first_hidden", "architecture": {"hidden": 48, "hidden_layers": 2},
                                             "anchor_l2": .01, "schedule": "cosine", "final_lr_ratio": .2, "input_noise": .01}
                                            for i, rate in enumerate((.0005, .0007))]
                                    cfgs = cfgs[:prompt["budget"]["max_experiments"]]
                                    if response_adapter:
                                        cfgs = [{"name":"ai_response", "learning_rate":.0001, "epochs":2, "batch_size":32,
                                                 "response_adapter":{"ridge":10.,"strength":16.,"guard_multiplier":1.5}}]
                                    if rejected_configuration and not calls:
                                        if freeze_response_conflict:
                                            cfgs[0]['freeze']='features'
                                        else:
                                            cfgs[0]["batch_size"] = 0
                                    elif rejected_configuration:
                                        if freeze_response_conflict:
                                            self.assertIn('frozen response',prompt['validation_error'])
                                            self.assertEqual(prompt['previous_response']['experiments'][0]['freeze'],'features')
                                        else:
                                            self.assertIn("batch", prompt["validation_error"])
                                            self.assertEqual(prompt["previous_response"]["experiments"][0]["batch_size"], 0)
                                    answer = {"action": "experiment", "experiments": cfgs,
                                              "hypothesis": "比较两组组合参数，检验弱类识别。", "reason": "按实测错误设计配置，效果待检验。"}
                                elif "allowed_experiments" in prompt:
                                    answer = fixture_plan(prompt)
                                    if training_focus:
                                        self.assertIn("features", prompt["permitted_parameter_changes"]["freeze"])
                                        answer["parameter_changes"] = {answer["experiment_names"][0]:
                                            {"freeze": "features", "class_boosts": {"Acetaldehyde": 2}}}
                                    if architecture_choice:
                                        self.assertEqual(prompt["permitted_parameter_changes"]["architecture_choice"], cfg["architecture_choices"])
                                        self.assertEqual(prompt["permitted_parameter_changes"]["parent_architecture"], {"hidden": 32, "hidden_layers": 1})
                                        answer["parameter_changes"] = {answer["experiment_names"][0]:
                                            {"architecture_choice": architecture_choice, "freeze": "first_hidden"}}
                                else:
                                    eligible = prompt["eligible_artifacts"]
                                    if only_parent:
                                        self.assertEqual(list(eligible), ["baseline-a"])
                                    selected = max(eligible, key=lambda k: eligible[k].get("host", eligible[k]["local"])["macro_f1"])
                                    answer = {"selected_artifact_id": selected, "reason": selection_reason}
                                    if authorized_fallback:
                                        answer["selected_artifact_id"] = "not-eligible"
                                    if retry_selection:
                                        if len(calls) == 1:
                                            answer["selected_artifact_id"] = "not-eligible"
                                        else:
                                            self.assertIn("correction", prompt)
                                call_id = f"offline-call-{len(calls) + 1}"
                                calls.append(call_id)
                                reply = {"type": "model_response", "call_id": call_id,
                                         "response": {"choices": [{"message": {"content": json.dumps(answer)}}]}}
                            elif kind == "candidate_request":
                                submitted_plan = read_json(next(Path(tmp).glob("gas-a-*/model-plan.json")))["bound_plan"]
                                self.assertTrue(frame["hypothesis"].startswith(submitted_plan["hypothesis"]))
                                actual_test = submitted_plan.get("iteration", {}).get("tests", {}).get(frame["configuration"]["name"])
                                if actual_test:
                                    self.assertIn(actual_test, frame["hypothesis"])
                                files = uploads.files(frame, "candidate")
                                self.assertEqual(frame["parent_checkpoint_sha256"], start["parent_checkpoint_sha256"])
                                self.assertEqual(frame["artifact"]["checkpoint_paths"], ["checkpoint.npz"])
                                self.assertLessEqual(len(files), 64)
                                self.assertLessEqual(sum(len(base64.b64decode(item["data_base64"])) for item in files), start["resource_limits"]["artifact_bytes"])
                                self.assertFalse(any(Path(item["path"]).name == "live_access.py" for item in files))
                                self.assertTrue({"checkpoint.npz", "training.json", "history.jsonl"}
                                                <= {item["path"] for item in files})
                                identity = f"offline-candidate-{len(candidates) + 1}"
                                directory = Path(tmp) / identity
                                directory.mkdir()
                                for item in files:
                                    path = directory / item["path"]
                                    self.assertTrue(path.resolve().is_relative_to(directory.resolve()))
                                    path.parent.mkdir(parents=True, exist_ok=True)
                                    path.write_bytes(base64.b64decode(item["data_base64"], validate=True))
                                measured = evaluate(load_model(directory / "checkpoint.npz", train["paths"]), validation)
                                self.assertGreater(load_model(directory / "checkpoint.npz").meta["steps"], 80)
                                training = read_json(directory / "training.json")
                                if response_adapter:
                                    self.assertIn('response.py',frame['artifact']['code_paths'])
                                    self.assertTrue(training['architecture']['response_adapter'])
                                    self.assertTrue(training['robust_gate']['accepted'])
                                    self.assertGreater(training['validation_after']['macro_f1'],.85)
                                    self.assertEqual(training['response_adapter_training']['train_sha256'],train['sha256'])
                                if composition and not response_adapter:
                                    self.assertEqual(training["configuration"]["proposal_origin"], "ai_composed_configuration")
                                    self.assertNotIn("kept_sensors", training["configuration"])
                                    self.assertTrue(training["architecture_migration"]["performed"])
                                    self.assertEqual(training["architecture"], {"hidden": 48, "hidden_layers": 2})
                                    self.assertEqual(training["configuration"]["batch_size"], 64)
                                    self.assertEqual(training["configuration"]["class_boosts"], {"Acetaldehyde": 2})
                                    self.assertTrue(training["freezing"]["verified_unchanged"])
                                    self.assertEqual(training["initial_steps"], load_model(cfg["baseline"]).meta["steps"])
                                    self.assertTrue((directory / "source/planning.py").exists())
                                if training_focus:
                                    self.assertTrue(training["freezing"]["verified_unchanged"])
                                    self.assertEqual(training["configuration"]["class_boosts"], {"Acetaldehyde": 2})
                                    exported, original = load_model(directory / "checkpoint.npz"), load_model(cfg["baseline"])
                                    for group in ("params", "first", "second"):
                                        for name in ("w1", "b1"):
                                            self.assertEqual(getattr(exported, group)[name].tobytes(), getattr(original, group)[name].tobytes())
                                if architecture or architecture_choice:
                                    self.assertTrue(training["architecture_migration"]["performed"])
                                    self.assertEqual(training["architecture_migration"]["parent_checkpoint_sha256"], start["parent_checkpoint_sha256"])
                                    self.assertTrue((directory / "architecture.py").is_file())
                                    initial_model = load_model(directory / "migrated_parent.npz", train["paths"])
                                    if architecture_choice:
                                        self.assertEqual(training["configuration"]["architecture"], cfg["architecture_choices"][architecture_choice])
                                        self.assertTrue(training["freezing"]["verified_unchanged"])
                                        exported = load_model(directory / "checkpoint.npz")
                                        for group in ("params", "first", "second"):
                                            for name in ("w1", "b1"):
                                                self.assertEqual(getattr(exported, group)[name].tobytes(), getattr(initial_model, group)[name].tobytes())
                                    np.testing.assert_allclose(initial_model.predict(validation["x"]),
                                        load_model(cfg["baseline"]).predict(validation["x"]), rtol=0, atol=1e-12)
                                epoch_history = [json.loads(line) for line in
                                                 (directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
                                self.assertEqual(training["parent_checkpoint_sha256"], start["parent_checkpoint_sha256"])
                                self.assertEqual(training["checkpoint_sha256"], digest(directory / "checkpoint.npz"))
                                self.assertEqual(len(epoch_history), training["checkpoint_selection"]["executed_epoch_count"])
                                self.assertLessEqual(len(epoch_history), training["configuration"]["requested_epochs"])
                                self.assertEqual(epoch_history[-1]["step"], training["executed_steps"])
                                selected_epoch = next(row for row in epoch_history if row["step"] == training["final_steps"])
                                self.assertEqual(selected_epoch["validation"]["macro_f1"], measured["macro_f1"])
                                candidates[identity] = measured
                                reply = {"type": "candidate_response", "artifact_id": identity,
                                         "summary": {"metrics": {} if only_parent else measured}}
                            elif kind == "state_request":
                                saved = {f["path"]: base64.b64decode(f["data_base64"], validate=True) for f in uploads.files(frame, "state")}
                                memory = json.loads(saved["lifecycle.json"])
                                self.assertEqual(memory["baseline_artifact_id"], "baseline-a")
                                self.assertEqual(memory["rounds"][0]["model_call_ids"], calls)
                                self.assertEqual(memory["rounds"][0]["review"]["costs"]["calls"], len(calls))
                                self.assertEqual(len(saved["baseline_checkpoint.npz"]), Path(cfg["baseline"]).stat().st_size)
                                reply = {"type": "state_response", "state_sha256": "sha256:" + "11" * 32}
                            else:
                                raise AssertionError("Unexpected frame " + kind)
                            send({"id": frame["id"], **reply})
                    except BaseException as exc:
                        failures.append(exc)
                        process.kill()

                worker = threading.Thread(target=host, daemon=True)
                worker.start()
                worker.join(30)
                if worker.is_alive():
                    process.kill()
                    self.fail("SDK host exchange timed out")
                process.wait(timeout=10)
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
                errors.seek(0)
                stderr = errors.read()
                self.assertFalse(failures, f"{failures!r}\n{stderr}")
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(len(candidates), 1 if training_focus or architecture_choice or response_adapter else 2)
                self.assertEqual(frames[-1]["type"], "round_complete")


if __name__ == "__main__":
    unittest.main()
