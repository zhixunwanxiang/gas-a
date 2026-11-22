"""联网运行流程；模型和云端均使用离线替身或 localhost 服务。"""
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

from gas_a.cloud_records import Audit, import_records
from gas_a.io import history, read_json, write_json
from gas_a.live import recover_commit, run_live
from gas_a.live_access import PyTrioChat, load_credentials, redact
from gas_a.model_records import expand_receipts, read_call, resolve_call
from fixtures import legacy_config as load_config
from gas_a.registry import Registry

from fixtures import CloudFixture, ModelFixture, fixture_plan

ROOT = Path(__file__).resolve().parents[1]


class LiveTests(unittest.TestCase):
    def config(self, root):
        cfg = load_config(ROOT / "configs/a.live.json")
        cfg["gate"].pop("minimum_class_recall", None)  # 现有测试替身用于验证旧版候选选择行为。
        cfg["runtime"] = str(Path(root) / "live")
        return cfg

    def test_actual_http_payload_and_provider_receipt_parsing(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append({"path": self.path, "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                                 "authorization": self.headers["Authorization"]})
                body = json.dumps({"id": "fixture-provider-id", "choices": [{"message": {"content": "OK"}}],
                                   "usage": {"prompt_tokens": 5, "completion_tokens": 1}}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        try:
            result = PyTrioChat("fixture-model", "fixture-secret", base_url=f"http://127.0.0.1:{server.server_port}/v1").complete(
                [{"role": "user", "content": "test"}], max_tokens=32, enable_thinking=False)
            self.assertEqual(result["call_id"], "fixture-provider-id")
            self.assertEqual(result["usage"]["completion_tokens"], 1)
            self.assertEqual(received[0]["body"]["max_tokens"], 32)
            self.assertIs(received[0]["body"]["enable_thinking"], False)
            self.assertEqual(received[0]["authorization"], "Bearer fixture-secret")
            self.assertEqual(received[0]["path"], "/v1/chat/completions")
        finally:
            server.shutdown(); server.server_close(); worker.join(5)

    def test_billing_failure_is_not_retried(self):
        class Reply(Exception):
            pass
        from urllib.error import HTTPError
        import io
        error = HTTPError("https://fixture.invalid", 409, "test", {}, io.BytesIO(b'{"code":"billing_insufficient_balance"}'))
        with patch("gas_a.live_access.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, "balance"):
                PyTrioChat("fixture-model", "fixture-secret").complete([{"role": "user", "content": "test"}])
            self.assertEqual(request.call_count, 1)

    def test_empty_text_keeps_receipts_and_never_fabricates_an_sdk_decision(self):
        class Empty(ModelFixture):
            def complete(self, messages, **kwargs):
                receipt = super().complete(messages, **kwargs)
                receipt["text"] = ""
                receipt["response"]["choices"][0].update(finish_reason="length")
                receipt["response"]["choices"][0]["message"]["content"] = None
                return receipt
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            with self.assertRaisesRegex(RuntimeError, "cannot fabricate an SDK decision"):
                run_live(cfg, cloud=False, client=Empty())
            session = next((Path(cfg["runtime"])/"sessions").iterdir())
            events = history(session/"events.jsonl")
            captured = next(e for e in events if e["event"] == "model_call")
            self.assertNotIn("response", captured["payload"])
            canonical = resolve_call(captured["payload"], cfg["runtime"])
            self.assertEqual(canonical["provider_call_id"], "fixture-call-1")
            self.assertEqual(canonical["response"]["choices"][0]["finish_reason"], "length")
            self.assertEqual(Registry(cfg["runtime"]).active_id(), "baseline")
            self.assertEqual(history(Path(cfg["runtime"]) / "history.jsonl"), [])
            self.assertEqual(len(list((session/"model_calls").glob("call-*.json"))), 3)

    def test_configured_token_budget_reaches_actual_requests(self):
        class Budget(ModelFixture):
            def complete(self, messages, **kwargs):
                if kwargs["max_tokens"] != 1536:
                    raise AssertionError("Configured budget was not forwarded")
                return super().complete(messages, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp); cfg["live"] = {"max_output_tokens":1536}
            run_live(cfg, cloud=False, client=Budget())

    def test_complete_round_cloud_records_and_commit_recovery(self):
        module, client = CloudFixture(), ModelFixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SWANLAB_API_KEY": "fixture-cloud-secret"}):
            cfg = self.config(tmp)
            summary = run_live(cfg, client=client, audit_factory=lambda path, cloud: Audit(path, cloud, module))
            self.assertEqual(client.calls, 2)
            self.assertGreaterEqual(summary["metrics"]["macro_f1"], .7140297220788709)
            self.assertFalse(summary["official_competition_result"])
            self.assertEqual(module.finished, 1)
            self.assertEqual(summary["cloud"]["status"], "sdk_finish_returned")
            root = Path(cfg["runtime"])
            rows = history(root / "history.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model_call_ids"], ["fixture-call-1", "fixture-call-2"])
            events = history(Path(summary["audit_directory"]) / "events.jsonl")
            self.assertTrue(any(e["event"] == "candidate_epoch" for e in events))
            self.assertNotIn("fixture-cloud-secret", json.dumps(events))
            self.assertTrue(module.records)
            self.assertTrue(any("review/summary" in record for record, _ in module.records))
            self.assertTrue(any("performance/macro_f1" in record for record, _ in module.records))
            self.assertTrue(any("audit/details" in record for record, _ in module.records))
            self.assertIn(("*", {"hidden": True}), module.definitions)
            self.assertEqual({options.get("section_name") for _, options in module.definitions if options.get("hidden") is False},
                             {"模型效果", "资源成本", "训练曲线", "每轮摘要"})
            uploaded_events = [json.loads("".join(part["text"] for part in data["audit/details"]))
                               for data, _ in module.records]
            for event in uploaded_events:
                if event["event"] == "model_call":
                    self.assertIn("request", event["payload"]["record"])
                    self.assertIn("response", event["payload"]["record"])
                    event["payload"] = event["payload"]["reference"]
            self.assertEqual(uploaded_events, events[1:])
            compact = next(data["review/summary"]["text"] for data, _ in module.records if "review/summary" in data)
            self.assertLessEqual(len(compact.splitlines()), 6)
            self.assertIn("已知 Token", compact)
            report_text = (root / "review/report.txt").read_text(encoding="utf-8")
            self.assertIn("完整 2 个候选，缺项 0 个", report_text)
            calls = [read_call(p) for p in (Path(summary["audit_directory"]) / "model_calls").glob("call-*.json")]
            self.assertEqual(len(calls), 2)
            self.assertEqual({c["purpose"] for c in calls}, {"experiment_planning", "artifact_selection"})
            self.assertTrue(all(c["request"]["body"]["messages"] and c["reply_text"] for c in calls))
            pending = read_json(root / "pending_commit.json"); pending["status"] = "prepared"
            write_json(root / "pending_commit.json", pending)
            recover_commit(Registry(root)); recover_commit(Registry(root))
            self.assertEqual(len(history(root / "history.jsonl")), 1)

    def test_hidden_cloud_audit_keeps_long_response_and_all_local_fields(self):
        module = CloudFixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SWANLAB_API_KEY": "fixture-cloud-secret"}):
            audit = Audit(tmp, module=module)
            audit.start({"model": "fixture-model"})
            payload = {"round": 1, "response": {"content": "完整回复" * 8000 + "末尾校验", "usage": None},
                       "future_field": {"attempts": 2}, "status": "error"}
            event = audit.record("model_response", payload)
            data, _ = module.records[-1]
            self.assertEqual(set(data), {"audit/details"})
            self.assertGreater(len(data["audit/details"]), 1)
            cloud_event = json.loads("".join(part["text"] for part in data["audit/details"]))
            self.assertEqual(cloud_event, event)
            self.assertEqual(history(Path(tmp) / "events.jsonl")[-1], event)
            self.assertEqual(cloud_event["payload"], payload)
            audit.finish(False)

    def test_online_failure_prevents_paid_model_call(self):
        module, client = CloudFixture(fail_init=True), ModelFixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SWANLAB_API_KEY": "fixture-cloud-secret"}):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                run_live(self.config(tmp), client=client, audit_factory=lambda path, cloud: Audit(path, cloud, module))
            self.assertEqual(client.calls, 0)

    def test_invalid_model_selection_recovers_only_from_measured_eligible_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            client = ModelFixture(invalid_selection=True)
            run_live(cfg, cloud=False, client=client)
            self.assertEqual(client.calls, 3)  # 一次规划调用，加上两次无效的最终决策调用。
            rows = history(Path(cfg["runtime"]) / "history.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertNotEqual(Registry(cfg["runtime"]).active_id(), "not-eligible")
            self.assertTrue((Path(cfg["runtime"]) / "state_pointer.json").exists())
            choice = read_json(next(Path(cfg["runtime"]).glob("sessions/*/work/*/model-choice.json")))
            self.assertEqual(choice["selection_mode"], "model_authorized_parent_fallback")
            self.assertEqual(choice["authorization_call_id"], "fixture-call-1")
            self.assertEqual(len(choice["attempts"]), 2)
            self.assertTrue(all(attempt["error"] for attempt in choice["attempts"]))

    def test_invalid_selection_can_be_corrected_by_a_real_second_reply(self):
        class Corrected(ModelFixture):
            def complete(self, messages, **kwargs):
                self.invalid_selection = self.calls == 1
                return super().complete(messages, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            cfg, client = self.config(tmp), Corrected()
            run_live(cfg, cloud=False, client=client)
            row = history(Path(cfg["runtime"]) / "history.jsonl")[0]
            self.assertEqual(client.calls, 3)
            self.assertEqual(row["selection_mode"], "model_retry")
            self.assertIsNone(row["selection_parse_error"])
            self.assertEqual(row["review"]["costs"]["calls"], 3)
            self.assertEqual(row["review"]["costs"]["tokens_total"], 60)
            self.assertEqual(row["model_call_ids"], ["fixture-call-1", "fixture-call-2", "fixture-call-3"])
            self.assertTrue(row["selection_attempts"][0]["error"])
            self.assertIsNone(row["selection_attempts"][1]["error"])
            corrected_prompt = json.loads(expand_receipts(row, cfg["runtime"])[-1]["messages"][0]["content"])
            self.assertIn("correction", corrected_prompt)

    def test_two_rounds_continue_actual_selected_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, client = self.config(tmp), ModelFixture()
            summary = run_live(cfg, rounds=2, cloud=False, client=client)
            rows = history(Path(cfg["runtime"]) / "history.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["parent_artifact_id"], rows[0]["selected_version"])
            self.assertEqual(client.calls, 4)
            self.assertEqual(rows[1]["selection_mode"], "model")
            self.assertEqual(rows[1]["review"]["costs"]["calls"], 2)
            self.assertEqual(rows[1]["review"]["costs"]["tokens_total"], 40)
            self.assertEqual(summary["cloud"]["status"], "local_only")

    def test_invalid_plan_observes_without_gradient_updates_and_finishes(self):
        class InvalidPlan(ModelFixture):
            def complete(self, messages, **kwargs):
                receipt = super().complete(messages, **kwargs)
                if "allowed_experiments" in json.loads(messages[0]["content"]):
                    receipt["text"] = '{"experiment_names":["unknown"],"reason":"无效测试方案"}'
                    receipt["response"]["choices"][0]["message"]["content"] = receipt["text"]
                return receipt
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            run_live(cfg, cloud=False, client=InvalidPlan())
            row = history(Path(cfg["runtime"]) / "history.jsonl")[0]
            self.assertTrue(row["model_plan_parse_error"])
            self.assertEqual(len(row["experiments"]), 0)
            self.assertTrue(row["review"]["fallback"])
            self.assertTrue(row["review"]["baseline_floor_passed"])

    def test_valid_observe_has_no_training_and_keeps_real_sdk_decisions(self):
        class Observe(ModelFixture):
            def complete(self, messages, **kwargs):
                result = super().complete(messages, **kwargs)
                prompt = json.loads(messages[0]["content"])
                if "allowed_experiments" in prompt:
                    result["text"] = json.dumps(fixture_plan(prompt, action="observe"))
                    result["response"]["choices"][0]["message"]["content"] = result["text"]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            client = Observe()
            with patch("gas_a.candidates.train_candidate", side_effect=AssertionError("Observe cannot train")):
                run_live(cfg, cloud=False, client=client)
            row = history(Path(cfg["runtime"]) / "history.jsonl")[0]
            self.assertFalse(row["experiments"])
            self.assertIsNone(row["model_plan_parse_error"])
            self.assertEqual(row["plan"]["action"], "observe")
            self.assertEqual(client.calls, 2)

    def test_sdk_no_gain_pause_has_no_candidates_and_keeps_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.config(tmp)
            cfg["gate"]["min_macro_f1_gain"] = 1.0  # 构造连续三个候选被拒的窗口，使用实际指标，不伪造分数。
            client = ModelFixture()
            run_live(cfg, rounds=4, cloud=False, client=client)
            rows = history(Path(cfg["runtime"]) / "history.jsonl")
            self.assertEqual(rows[-1]["decision_source"], "budget_pause")
            self.assertFalse(rows[-1]["experiments"])
            self.assertEqual(rows[-1]["plan"]["action"], "observe")
            self.assertEqual(client.calls, 8)

    def test_literal_env_and_redaction(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            p = Path(tmp) / ".env.local"
            p.write_text('PYTRIO_API_KEY="fixture-secret"\nPYTRIO_MODEL=$(never_execute)\nSWANLAB_PROJECT=fixture-project\n', encoding="utf-8")
            load_credentials(p)
            self.assertEqual(os.environ["PYTRIO_MODEL"], "$(never_execute)")
            self.assertEqual(os.environ["GAS_A_SWANLAB_PROJECT"], "fixture-project")
            self.assertNotIn("SWANLAB_PROJECT", os.environ)
            self.assertEqual(redact({"api_key": "fixture-secret", "note": "fixture-secret"}),
                             {"api_key": "[REDACTED]", "note": "[REDACTED]"})

    def test_import_preserves_original_rule_controller_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "history.jsonl"
            source.write_text(json.dumps({"round": 1, "mode": "local_cpu_rule_controller", "macro_f1": .7})+'\n', encoding="utf-8")
            audit = Audit(root / "import", cloud=False); audit.start({})
            self.assertEqual(import_records(source, audit), 1)
            record = history(audit.root / "events.jsonl")[-1]
            self.assertEqual(record["payload"]["original_record"]["mode"], "local_cpu_rule_controller")
            audit.finish(True)


if __name__ == "__main__":
    unittest.main()
