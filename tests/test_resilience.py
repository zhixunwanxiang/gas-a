"""跨监测、诊断、规划、训练、评估、切版、回滚的故障注入。"""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import os
import unittest
from unittest.mock import patch

import numpy as np

from fixtures import ReplayModelFixture, CloudFixture, ModelFixture
from gas_a.cloud_records import Audit, import_records
from gas_a.live import run_live
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import digest, history, read_json, write_json
from gas_a.io import lock, writer_active
from gas_a.live_access import ProviderCallError
from fixtures import legacy_config as load_config
from gas_a.registry import Registry
from gas_a.resilience import measured_selection, monitor_safely, bounded_model_time
from gas_a.temporal import run_temporal, integrity_guard

ROOT = Path(__file__).resolve().parents[1]


class ResilienceTests(unittest.TestCase):
    def setup_run(self, root):
        cfg = load_config(ROOT / "configs/a.adaptive.json")
        cfg["runtime"] = str(root / "result")
        cfg["experiments"] = [{**cfg["experiments"][0], "epochs": 1}]
        cfg["optimization"]["max_selected_experiments"] = 1
        train, val = load_pair(cfg["train"], cfg["validation"])
        for batch, offset in ((3, 0), (4, 40)):
            rows = [str(int(y) + 1) + " " + " ".join(f"{i+1}:{v:.17g}" for i, v in enumerate(x))
                    for x, y in zip(val["x"][offset:offset+40], val["y"][offset:offset+40])]
            (root / f"batch{batch}.dat").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return cfg, train, val

    def result(self, cfg):
        return read_json(Path(cfg["runtime"]) / "windows/batch03/result.json")

    def test_monitoring_calculation_failure_is_unknown_and_does_not_erase_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            with patch("gas_a.temporal.monitor", side_effect=ArithmeticError("injected statistics failure")):
                summary = run_temporal(cfg, root, (3,4), cloud=False, client=ReplayModelFixture())
            row = self.result(cfg)
            self.assertTrue(summary["workflow_completed"])
            self.assertFalse(summary["full_monitoring_and_diagnosis_evidence"])
            self.assertIsNone(row["drift"]["drift_detected"])
            self.assertIsNone(row["drift"]["flagged_features"])
            self.assertEqual(digest(Path(cfg["runtime"]) / "windows/batch03/predictions.npz"), row["prediction_commit"]["predictions_sha256"])
            self.assertTrue(any(r["stage"] == "monitoring" for r in row["recoveries"]))

    def test_diagnostic_failure_uses_measured_weak_class_without_inventing_cause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            with patch("gas_a.temporal.diagnose", side_effect=ValueError("injected diagnosis failure")):
                summary = run_temporal(cfg, root, (3,4), cloud=False, client=ReplayModelFixture())
            row = self.result(cfg)
            self.assertTrue(summary["workflow_completed"])
            self.assertEqual(row["diagnosis"]["status"], "limited_measured_summary")
            self.assertEqual(row["diagnosis"]["top_confusions"], [])
            self.assertTrue(any(r["stage"] == "diagnosis" for r in row["recoveries"]))

    def test_unverified_historical_best_comment_is_rebound_and_raw_reply_is_preserved(self):
        class HistoryClaim(ReplayModelFixture):
            def complete(self, messages, **options):
                result = super().complete(messages, **options)
                prompt = json.loads(messages[-1]["content"])
                if "allowed_experiments" in prompt:
                    plan = json.loads(result["text"])
                    plan["reason"] = "该方案历史最佳。"
                    result["text"] = json.dumps(plan, ensure_ascii=False)
                    result["response"]["choices"][0]["message"]["content"] = result["text"]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            summary = run_temporal(cfg, root, (3,4), cloud=False, client=HistoryClaim())
            row = self.result(cfg)
            self.assertTrue(summary["success"])
            self.assertEqual(summary["controller_failed_windows"], 0)
            self.assertEqual(summary["controller_recovered_windows"], 1)
            self.assertEqual(len(row["experiments"]), 1)
            self.assertEqual(row["plan"]["decision_origin"], "program_repaired_model_menu_intent")
            raw = "\n".join(p.read_text(encoding="utf-8") for p in (Path(cfg["runtime"])/"session/model_calls").glob('call-*.json'))
            self.assertIn("历史最佳", raw)
            self.assertNotIn("历史最佳", row["plan"]["reason"])

    def test_transient_provider_error_retries_and_keeps_unknown_cost_explicit(self):
        class TimeoutOnce(ReplayModelFixture):
            def __init__(self):super().__init__();self.attempts=0
            def complete(self, messages, **options):
                self.attempts += 1
                if self.attempts == 1:raise TimeoutError("injected network timeout")
                return super().complete(messages, **options)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            summary = run_temporal(cfg, root, (3,4), cloud=False, client=TimeoutOnce())
            self.assertTrue(summary["success"])
            self.assertEqual(summary["controller_recovered_windows"], 1)
            self.assertEqual(summary["costs"]["calls"], 3)
            self.assertEqual(summary["costs"]["unknown_usage_calls"], 1)
            self.assertIsNone(summary["costs"]["tokens_total"])

    def test_training_timeout_rejects_candidate_and_finishes_later_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            with patch("gas_a.temporal.execute_candidate", side_effect=TimeoutError("injected CPU timeout")):
                summary = run_temporal(cfg, root, (3,4), cloud=False, client=ReplayModelFixture())
            self.assertTrue(summary["success"])
            self.assertEqual(summary["finished_windows"], 2)
            self.assertEqual(summary["failed_candidate_trials"], 1)
            self.assertEqual(Registry(Path(cfg["runtime"])/"deployment").active_id(), "baseline")
            self.assertTrue(any(r["stage"] == "candidate_evaluation" for r in self.result(cfg)["recoveries"]))

    def test_missing_or_malformed_window_is_rejected_before_model_and_audit_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);cfg, _, _ = self.setup_run(root)
            (root / "batch4.dat").write_text("1 1:nan\n", encoding="utf-8")
            client = ReplayModelFixture()
            with patch("gas_a.temporal.Audit.start") as start:
                with self.assertRaises(ValueError):run_temporal(cfg, root, (3,4), cloud=False, client=client)
            self.assertFalse(start.called)
            self.assertEqual(client.requests, [])
            self.assertFalse(Path(cfg["runtime"]).exists())

    def test_unknown_selection_cannot_bypass_eligible_measurements(self):
        prompt = {"eligible_artifacts": {"old": {}, "new": {}},
                  "comparison": {"measured_preference": "old", "rows": [
                      {"artifact_id": "old", "score": .7}, {"artifact_id": "new", "score": .8}]}}
        with self.assertRaises(ValueError):measured_selection(prompt)
        prompt["comparison"]["measured_preference"] = "new"
        self.assertEqual(measured_selection(prompt)["selected_artifact_id"], "new")
        prompt["comparison"]["rows"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):measured_selection(prompt)

    def registry(self, root):
        cfg = load_config(ROOT / "configs/a.json")
        _, val = load_pair(cfg["train"], cfg["validation"])
        registry = Registry(root)
        metric = evaluate(load_model(cfg["baseline"]), val)
        registry.initialize(cfg["baseline"], metric)
        registry.register("next", cfg["baseline"], "baseline", metric, {})
        return registry, val

    def test_transient_activation_write_failure_is_retried_without_duplicate_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, _ = self.registry(Path(tmp));attempts=[]
            from gas_a.registry import write_json as real_write
            def fail_once(path, value):
                if Path(path).name == "active.json" and not attempts:
                    attempts.append(True);raise OSError("injected temporary write failure")
                return real_write(path,value)
            with patch("gas_a.registry.write_json", side_effect=fail_once):registry.activate("next", "test")
            self.assertEqual(registry.active_id(), "next")
            self.assertEqual(len(history(Path(tmp)/"deployments.jsonl")), 1)
            self.assertFalse((Path(tmp)/"activation_pending.json").exists())

    def test_activation_interruption_resumes_pointer_and_log_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry,_=self.registry(root)
            with patch("gas_a.registry.write_jsonl", side_effect=OSError("injected disk unavailable")):
                with self.assertRaises(OSError):registry.activate("next", "interrupted")
            self.assertTrue((root/"activation_pending.json").exists())
            self.assertEqual(registry.active_id(), "next")
            registry.recover_activation();registry.recover_activation()
            self.assertEqual(len(history(root/"deployments.jsonl")), 1)
            self.assertFalse((root/"activation_pending.json").exists())

    def test_interrupted_activation_with_corrupt_target_restores_prior_intact_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry,_=self.registry(root)
            with patch("gas_a.registry.write_jsonl", side_effect=OSError("injected interrupted publish")):
                with self.assertRaises(OSError):registry.activate("next", "test")
            (root/"versions/next/checkpoint.npz").write_bytes(b"damaged after interruption")
            registry.recover_activation()
            self.assertEqual(registry.active_id(), "baseline")
            self.assertEqual(history(root/"deployments.jsonl")[0]["reason"], "activation_aborted_restore_intact_version")

    def test_corrupt_active_pointer_recovers_before_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry,val=self.registry(root)
            (root/"active.json").write_text("{incomplete", encoding="utf-8")
            event=integrity_guard(registry,val)
            self.assertEqual(event["to"], "baseline")
            self.assertEqual(registry.active_id(), "baseline")

    def test_authentication_failure_is_not_hidden_by_safe_observation(self):
        class Denied(ReplayModelFixture):
            def complete(self, messages, **options):
                raise ProviderCallError("injected authentication failure",http_status=401)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg,_,_=self.setup_run(root)
            with self.assertRaises(ProviderCallError):run_temporal(cfg,root,(3,4),cloud=False,client=Denied())
            self.assertEqual(Registry(Path(cfg["runtime"])/"deployment").active_id(), "baseline")
            self.assertFalse(read_json(Path(cfg["runtime"])/"summary.json")["success"])

    def test_round_budget_caps_and_restores_http_timeout(self):
        from types import SimpleNamespace
        import time
        client = SimpleNamespace(timeout=90)
        with bounded_model_time(client, time.monotonic() + 2):
            self.assertLessEqual(client.timeout, 2)
        self.assertEqual(client.timeout, 90)
        with self.assertRaises(TimeoutError):
            with bounded_model_time(client, time.monotonic()):
                self.fail("No external request is allowed after the deadline")

    def test_process_crash_releases_os_lock_without_manual_stale_file_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = ("import os,sys; from gas_a.io import lock; "
                    "guard=lock(sys.argv[1]); guard.__enter__(); "
                    "print('locked',flush=True); sys.stdin.readline(); os._exit(7)")
            child = subprocess.Popen([sys.executable,"-B","-c",code,tmp],cwd=ROOT,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "locked")
                self.assertTrue(writer_active(tmp))
                with self.assertRaises(RuntimeError):
                    with lock(tmp):pass
                child.communicate("crash\n",timeout=10)
                self.assertEqual(child.returncode,7)
                self.assertTrue((Path(tmp)/".run.lock").exists())
                self.assertFalse(writer_active(tmp))
                with lock(tmp):pass
            finally:
                if child.poll() is None:child.kill();child.communicate(timeout=5)

    def test_legacy_live_writer_is_not_bypassed_by_new_lock_protocol(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/".run.lock"
            path.write_text(str(os.getpid()), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "legacy writer"):
                with lock(tmp):pass

    def test_recovery_after_log_commit_records_separate_restore_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry,_=self.registry(root)
            real_unlink=Path.unlink
            def blocked(path,*args,**kwargs):
                if path.name=="activation_pending.json":raise OSError("injected cleanup interruption")
                return real_unlink(path,*args,**kwargs)
            with patch.object(Path,"unlink",blocked):
                with self.assertRaises(OSError):registry.activate("next","test")
            self.assertEqual(len(history(root/"deployments.jsonl")),1)
            (root/"versions/next/checkpoint.npz").write_bytes(b"corrupted after commit")
            registry.recover_activation()
            events=history(root/"deployments.jsonl")
            self.assertEqual(len(events),2)
            self.assertNotEqual(events[0]["activation_id"],events[1]["activation_id"])
            self.assertEqual(events[1]["to"],"baseline")

    def test_cloud_upload_transient_failure_retries_without_duplicate_event(self):
        class Flaky(CloudFixture):
            def init(self, **options):
                run=super().init(**options);send=run.log;self.attempts=0
                def log(*args,**kwargs):
                    self.attempts+=1
                    if self.attempts==1:raise RuntimeError("injected transient cloud failure")
                    return send(*args,**kwargs)
                run.log=log;return run
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"SWANLAB_API_KEY":"fixture-cloud-secret"}):
            module=Flaky();audit=Audit(Path(tmp),cloud=True,module=module)
            audit.start({"test":"cloud retry"});receipt=audit.finish(True)
            self.assertTrue(receipt["upload_complete"])
            self.assertEqual(receipt["upload_retries"],1)
            self.assertEqual(len(module.records),1)

    def test_cloud_outage_preserves_training_and_full_call_records_for_resync(self):
        class Unavailable(CloudFixture):
            def init(self, **options):
                run=super().init(**options)
                def log(*args,**kwargs):raise RuntimeError("injected cloud outage")
                run.log=log;return run
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"SWANLAB_API_KEY":"fixture-cloud-secret"}):
            root=Path(tmp);cfg=load_config(ROOT/"configs/a.live.json")
            cfg["runtime"]=str(root/"live");cfg["gate"].pop("minimum_class_recall",None)
            summary=run_live(cfg,cloud=True,client=ModelFixture(),
                audit_factory=lambda path,cloud:Audit(path,cloud,module=Unavailable()))
            self.assertEqual(summary["completed_rounds"],1)
            self.assertFalse(summary["cloud"]["upload_complete"])
            self.assertEqual(summary["cloud"]["status"],"upload_pending")
            source=Path(summary["audit_directory"]);queue=history(source/"cloud-outbox.jsonl")
            calls=[e for e in queue if e["kind"]=="model_call"]
            self.assertEqual(len(calls),2)
            self.assertTrue(all("response" in e["event"]["payload"]["record"] for e in calls))
            self.assertNotIn("fixture-cloud-secret",(source/"cloud-outbox.jsonl").read_text(encoding="utf-8"))
            original=digest(source/"cloud-outbox.jsonl")
            resend=Audit(root/"resync",cloud=True,module=CloudFixture());resend.start({"mode":"test resync"})
            self.assertEqual(import_records(source,resend),len(queue))
            self.assertTrue(resend.finish(True)["upload_complete"])
            self.assertEqual(digest(source/"cloud-outbox.jsonl"),original)

    def test_cloud_finish_failure_is_explicit_and_does_not_overwrite_local_result(self):
        class FailedFinish(CloudFixture):
            def finish(self,**kwargs):raise RuntimeError("injected finish acknowledgement outage")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"SWANLAB_API_KEY":"fixture-cloud-secret"}):
            audit=Audit(Path(tmp),cloud=True,module=FailedFinish());audit.start({"test":"finish"})
            audit.record("local_result",{"passed":True})
            receipt=audit.finish(True)
            self.assertTrue(receipt["session_succeeded"])
            self.assertFalse(receipt["upload_complete"])
            self.assertEqual(receipt["status"],"finish_pending")
            self.assertEqual(history(Path(tmp)/"events.jsonl")[-1]["payload"],{"passed":True})
            with patch("gas_a.io._process_alive", return_value=False):
                with lock(tmp):pass


if __name__ == "__main__":unittest.main()
