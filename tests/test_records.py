"""运行记录与审计：模型原文、成本、摘要及公共配置。"""
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from gas_a import data
from gas_a.candidates import execute_candidate
from gas_a.cloud_records import Audit
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import append_jsonl, digest, read_config, read_json, write_json
from gas_a.live_access import InvalidProviderResponse, ProviderCallError, PyTrioChat
from gas_a.model_records import (
    call_reference,
    capture_call,
    collect_model_calls,
    export_model_calls,
    read_call,
    recorded_chat,
    recorded_round_model,
    resolve_call,
    transcript,
)
from gas_a.monitor import diagnose, monitor
from gas_a.optimize import train_candidate
from fixtures import legacy_config as load_config
from gas_a.review import export_review, planning_context, usage_totals

from fixtures import ModelFixture

ROOT = Path(__file__).resolve().parents[1]


class ModelRecordTests(unittest.TestCase):
    def test_actual_wire_payload_and_reply_are_paired_without_credentials(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"id": "provider-fixture-1", "choices": [{"message": {"content": "收到中文数据"}}],
                                   "usage": {"prompt_tokens": 23, "completion_tokens": 7}}, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                messages = [{"role": "user", "content": json.dumps({"allowed_experiments": ["balanced"], "诊断": "乙醛弱类"}, ensure_ascii=False)}]
                client = PyTrioChat("fixture-model", "fixture-key", base_url=f"http://127.0.0.1:{server.server_port}/v1")
                receipt = recorded_chat(client, tmp, messages, max_tokens=1536, temperature=.2, enable_thinking=False)
                record = read_call(next(Path(tmp).glob("call-*.json")))
                self.assertEqual(record["request"]["body"], received[0])
                self.assertEqual(record["request"]["body"]["messages"], messages)
                self.assertEqual(record["response"], receipt["response"])
                self.assertEqual(record["reply_text"], "收到中文数据")
                self.assertEqual(record["provider_call_id"], "provider-fixture-1")
                self.assertEqual(record["usage"]["completion_tokens"], 7)
                self.assertEqual(record["purpose"], "experiment_planning")
                index = json.loads((Path(tmp) / "index.jsonl").read_text(encoding="utf-8"))
                self.assertNotIn("request", index)
                self.assertTrue((Path(tmp) / index["file"]).is_file())
                self.assertEqual(index["usage"], record["usage"])
                self.assertNotIn("fixture-key", "".join(p.read_text(encoding="utf-8") for p in Path(tmp).iterdir()))
                self.assertFalse(list(Path(tmp).glob("call-*.txt")))
                self.assertIn("乙醛弱类", transcript(record, full=True))
        finally:
            server.shutdown(); server.server_close(); worker.join(5)

    def test_request_is_durable_before_call_and_full_http_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            detail = "上游拒绝请求" * 3000
            response = {"error": {"message": detail, "api_key": "fixture-key"}}
            error = HTTPError("https://fixture.invalid", 409, "fixture", {"x-request-id": "failure-1"},
                              io.BytesIO(json.dumps(response, ensure_ascii=False).encode()))
            def fail(request, **kwargs):
                record = read_call(next(Path(tmp).glob("call-*.json")))
                self.assertEqual(record["status"], "pending")
                self.assertEqual(record["request"]["body"]["messages"][0]["content"], "test")
                raise error
            with patch("gas_a.live_access.urlopen", side_effect=fail) as transport:
                with self.assertRaisesRegex(ProviderCallError, "balance"):
                    recorded_chat(PyTrioChat("fixture-model", "fixture-key"), tmp, [{"role": "user", "content": "test"}])
                self.assertEqual(transport.call_count, 1)
            record = read_call(next(Path(tmp).glob("call-*.json")))
            self.assertEqual(record["status"], "error")
            self.assertEqual(record["response"]["error"]["message"], detail)
            self.assertEqual(record["response"]["error"]["api_key"], "[REDACTED]")
            self.assertEqual(record["provider_call_id"], "failure-1")
            self.assertEqual(record["http_status"], 409)

    def test_empty_and_invalid_responses_are_kept_without_fabricated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = {"call_id": "empty-1", "text": "", "response": {"id": "empty-1", "choices": [{"message": {"content": None}, "finish_reason": "length"}]},
                       "usage": {"completion_tokens": 768}}
            capture_call(tmp, {"messages": []}, lambda: receipt, record_id="empty")
            def invalid():
                raise InvalidProviderResponse("invalid JSON", "<html>provider error</html>", 200, "invalid-1")
            with self.assertRaises(InvalidProviderResponse):
                capture_call(tmp, {"messages": []}, invalid, record_id="invalid")
            records = [read_call(p) for p in Path(tmp).glob("call-*.json")]
            self.assertEqual({r["status"] for r in records}, {"empty_response", "error"})
            self.assertEqual(next(r for r in records if r["status"] == "empty_response")["usage"]["completion_tokens"], 768)
            self.assertEqual(next(r for r in records if r["status"] == "error")["response"], "<html>provider error</html>")
            def missing_id():
                raise InvalidProviderResponse("missing provider ID", {"choices": [{"message": {"content": "模型实际回复"}}]})
            with self.assertRaises(InvalidProviderResponse):
                capture_call(tmp, {"messages": []}, missing_id, record_id="missing-id")
            saved = next(read_call(p) for p in Path(tmp).glob("call-*.json") if read_json(p)["record_id"] == "missing-id")
            self.assertEqual(saved["reply_text"], "模型实际回复")

    def test_export_pairs_existing_events_and_marks_missing_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root / "sessions" / "fixture-session"
            events = session / "events.jsonl"
            request = {"round": 1, "model": "fixture", "messages": [{"role": "user", "content": '{"allowed_experiments": []}'}], "max_tokens": 768}
            append_jsonl(events, {"event": "model_request", "event_id": "old-request", "time": "2026-11-18T01:00:00Z", "payload": request})
            append_jsonl(events, {"event": "model_response", "time": "2026-11-18T01:00:00Z", "payload": {"round": 1, "call_id": "old-call", "text": "旧回复", "response": {"id": "old-call"}, "usage": {"completion_tokens": 4}}})
            append_jsonl(events, {"event": "model_request", "event_id": "lost-request", "time": "2026-11-18T01:00:01Z", "payload": {**request, "round": 2}})
            append_jsonl(events, {"event": "session_failed", "time": "2026-11-18T01:00:02Z", "payload": {"error": "connection failed"}})
            result = export_model_calls(root, root / "export")
            self.assertEqual(result["calls"], 2)
            references = [json.loads(s) for s in Path(result["index"]).read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("request" not in ref and "response" not in ref for ref in references))
            self.assertFalse((root / "export/calls").exists())
            rows = collect_model_calls(root, exclude=root / "export")
            self.assertEqual(rows[0]["request"]["captured_event_payload"], request)
            self.assertEqual(rows[0]["reply_text"], "旧回复")
            self.assertIsNone(rows[1]["response"])
            self.assertIsNone(rows[1]["reply_text"])
            self.assertEqual(rows[1]["status"], "error")
            self.assertIn("无法补回", rows[1]["error"]["note"])
            self.assertEqual(export_model_calls(root, root / "export")["calls"], 2)

    def test_modern_record_and_audit_events_export_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root / "sessions" / "fixture-session"
            receipt = {"call_id": "fixture-provider", "text": "OK", "response": {"id": "fixture-provider"}}
            capture_call(session / "model_calls", {"body": {"messages": []}}, lambda: receipt, record_id="same-request")
            append_jsonl(session / "events.jsonl", {"event": "model_request", "event_id": "audit-id", "payload": {"record_id": "same-request", "round": 1, "messages": []}})
            append_jsonl(session / "events.jsonl", {"event": "model_response", "payload": {"record_id": "same-request", "round": 1, **receipt}})
            result = export_model_calls(root, root / "export")
            self.assertEqual(result["calls"], 1)
            self.assertEqual(result["status_counts"]["success"], 1)

    def test_contest_rpc_records_messages_and_returns_original_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = SimpleNamespace(call_id="host-call", text="host reply", response={"id": "host-call", "usage": {"completion_tokens": 5}})
            rnd = SimpleNamespace(round_index=3, model_name="host-model", model=lambda *a, **kw: receipt)
            messages = [{"role": "user", "content": "请选择合格版本"}]
            result = recorded_round_model(rnd, tmp, messages, max_tokens=768, temperature=0, purpose="artifact_selection")
            self.assertIs(result, receipt)
            record = read_call(next((Path(tmp) / "model_calls").glob("call-*.json")))
            self.assertEqual(record["request"]["messages"], messages)
            self.assertEqual(record["reply_text"], "host reply")
            self.assertEqual(record["usage"]["completion_tokens"], 5)


class ReviewTests(unittest.TestCase):
    def test_usage_counts_failed_and_partial_calls_without_inventing_zero(self):
        calls = [{"status": "success", "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
                 {"status": "empty_response", "usage": {"prompt_tokens": 8, "completion_tokens": 3}},
                 {"status": "error", "usage": {"prompt_tokens": 2}}, {"status": "error"}]
        costs = usage_totals(calls)
        self.assertEqual(costs["calls"], 4)
        self.assertEqual(costs["tokens_known"], 27)
        self.assertEqual(costs["unknown_usage_calls"], 2)
        self.assertEqual(costs["unsuccessful_calls"], 3)
        self.assertIsNone(costs["tokens_total"])
        self.assertEqual(usage_totals([])["tokens_total"], 0)

    def test_compact_prompt_preserves_metrics_and_has_no_duplicate_matrix(self):
        cfg = load_config(ROOT / "configs/a.json")
        train, val = load_pair(cfg["train"], cfg["validation"])
        metric = evaluate(load_model(cfg["baseline"]), val)
        diagnosis = diagnose(train, val, metric, monitor(train["x"], val["x"], cfg["monitor"]))
        compact = planning_context(metric, diagnosis, [])
        self.assertEqual(compact["parent_metrics"]["macro_f1"], metric["macro_f1"])
        self.assertEqual(compact["diagnosis"]["weakest_class_metrics"], diagnosis["weakest_class_metrics"])
        self.assertNotIn("confusion_matrix", compact["parent_metrics"])
        self.assertLess(len(json.dumps(compact)), len(json.dumps({"parent_metrics": metric, "diagnosis": diagnosis})))

    def test_failed_training_retains_actual_resource_record(self):
        cfg = load_config(ROOT / "configs/a.json")
        train, val = load_pair(cfg["train"], cfg["validation"])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            with self.assertRaises(TimeoutError):
                train_candidate(cfg["baseline"], train, val, output, cfg["experiments"][0],
                                {"weakest_class": "Acetaldehyde"}, {**cfg["optimization"], "max_seconds_per_candidate": 0})
            failure = read_json(output / "failure.json")
            self.assertGreaterEqual(failure["resources"]["wall_seconds"], 0)
            self.assertGreaterEqual(failure["resources"]["cpu_seconds"], 0)
            self.assertIsNone(failure["resources"]["platform_training_credits"])
            self.assertFalse((output / "checkpoint.npz").exists())

    def test_review_deduplicates_calls_and_counts_unfinished_session_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "sessions" / "fixture"
            payload = {"request": "中文问题", "history": ["历史内容" * 400 + "请求尾部校验"]}
            response_body = {"id": "provider-1", "choices": [{"message": {"content": '{"reason":"中文理由"}'}, "finish_reason": "stop"}],
                             "extra": "返回内容" * 400 + "响应尾部校验"}
            response = {"call_id": "provider-1", "text": '{"reason":"中文理由"}', "response": response_body,
                        "usage": {"prompt_tokens": 20, "completion_tokens": 5}}
            capture_call(session / "model_calls", {"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}], "max_tokens": 512},
                         lambda: response, context={"round": 1, "session": "fixture"}, record_id="first")
            append_jsonl(session / "events.jsonl", {"event": "model_request", "payload": {"record_id": "first", "round": 1, "messages": []}})
            append_jsonl(session / "events.jsonl", {"event": "model_response", "payload": {"record_id": "first", "round": 1, **response}})
            def fail():
                raise RuntimeError("未知服务方用量的失败调用")
            with self.assertRaises(RuntimeError):
                capture_call(session / "model_calls", {"messages": []}, fail,
                             context={"round": 2, "session": "fixture"}, record_id="failed")
            append_jsonl(root / "history.jsonl", {"id": "local-history-001-fixture", "round": 1, "before": {"macro_f1": .7},
                         "after": {"macro_f1": .72}, "static_baseline": {"macro_f1": .7}, "selected_version": "candidate",
                         "action": "promote", "reason": "实际指标改善", "experiments": []})
            original = {p: digest(p) for p in root.rglob("*") if p.is_file()}
            result = export_review(root, root / "review")
            text = Path(result["report"]).read_text(encoding="utf-8")
            self.assertEqual(result["calls"], 2)
            self.assertIn("已知输入 Token：20；已知输出 Token：5；用量缺失：1 次", text)
            self.assertIn("单位增益 Token 成本：未计算", text)
            self.assertIn("平台 GPU 时长和训练额度：未提供", text)
            self.assertIn("正式评分：尚未由平台确认", text)
            self.assertEqual([p.name for p in (root / "review").iterdir()], ["report.txt"])
            self.assertEqual(export_review(root, root / "review")["calls"], 2)
            self.assertEqual(original, {p: digest(p) for p in original})
            self.assertIn("中文问题", text)
            self.assertIn("中文理由", text)
            self.assertIn("【发送给模型的完整请求数据】", text)
            self.assertIn("【模型返回的完整数据】", text)
            self.assertIn("未取得服务方返回数据", text)
            request_text = text.split("【发送给模型的完整请求数据】\n", 1)[1].split("\n【模型返回的完整数据】", 1)[0]
            response_text = text.split("【模型返回的完整数据】\n", 1)[1].split("\n【单独捕获的用量】", 1)[0]
            self.assertEqual(json.loads(request_text)["messages"][0]["content"], payload)
            self.assertEqual(json.loads(response_text)["extra"], response_body["extra"])
            self.assertEqual(json.loads(response_text)["choices"][0]["message"]["content"], {"reason": "中文理由"})
            self.assertNotIn("摘要截断", text)


class RefactorTests(unittest.TestCase):
    def test_config_inheritance_propagates_common_fields_and_rejects_cycles(self):
        common, live = load_config(ROOT / "configs/a.json"), load_config(ROOT / "configs/a.live.json")
        self.assertEqual({k: v for k, v in live.items() if k not in ("runtime", "live")},
                         {k: v for k, v in common.items() if k != "runtime"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "base.json", {"optimization": {"epochs": 5, "batch": 32}, "experiments": [1, 2]})
            write_json(root / "child.json", {"extends": "base.json", "optimization": {"epochs": 8}, "experiments": [3]})
            value = read_config(root / "child.json")
            self.assertEqual(value, {"optimization": {"epochs": 8, "batch": 32}, "experiments": [3]})
            write_json(root / "base.json", {"extends": "child.json"})
            with self.assertRaisesRegex(ValueError, "cycle"):
                read_config(root / "child.json")

    def test_canonical_raw_exists_once_and_events_reports_expand_references(self):
        request_text, reply = "REQUEST_CANARY_" * 150, "REPLY_CANARY_" * 150
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "sessions" / "one" / "model_calls"
            response = {"id": "fixture-provider", "choices": [{"message": {"content": reply}}]}
            capture_call(directory, {"messages": [{"role": "user", "content": request_text}]},
                lambda: {"call_id": "fixture-provider", "text": reply, "response": response,
                         "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
                record_id="one", context={"round": 1, "session": "one"})
            path = next(directory.glob("call-*.json"))
            stored = read_json(path)
            self.assertNotIn("reply_text", stored)
            self.assertEqual(path.read_text(encoding="utf-8").count(reply), 1)
            reference = call_reference(path, root)
            audit = Audit(root / "sessions/one", cloud=False)
            audit.start({})
            audit.record_model_call(reference, root)
            self.assertNotIn(request_text, (audit.root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertNotIn(reply, (audit.root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(collect_model_calls(audit.root / "events.jsonl")[0]["response"], response)
            result = export_model_calls(root, root / "export")
            self.assertIn(request_text, Path(result["transcript"]).read_text(encoding="utf-8"))
            self.assertIn(reply, Path(result["transcript"]).read_text(encoding="utf-8"))
            self.assertNotIn(request_text, Path(result["index"]).read_text(encoding="utf-8"))
            self.assertEqual(len(list(root.rglob("call-*.json"))), 1)
            self.assertEqual(export_review(root, root / "review")["calls"], 1)

    def test_reference_corruption_and_escape_are_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_call(root / "model_calls", {"messages": []},
                         lambda: {"call_id": "fixture", "response": {"id": "fixture"}, "text": "OK"}, record_id="one")
            path = next((root / "model_calls").glob("call-*.json"))
            reference = call_reference(path, root)
            escaped = {**reference, "file": "../outside.json"}
            with self.assertRaises(ValueError):
                resolve_call(escaped, root)
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_call(reference, root)

    def test_shared_candidate_preserves_weights_adam_and_selected_epoch(self):
        cfg = load_config(ROOT / "configs/a.json")
        train, validation = data.load_pair(cfg["train"], cfg["validation"])
        model = data.load_model(cfg["baseline"])
        before = data.evaluate(model, validation)
        diagnosis = diagnose(train, validation, before, monitor(train["x"], validation["x"], cfg["monitor"]))
        spec = {**cfg["experiments"][0], "epochs": 2}
        limits = {**cfg["optimization"], "checkpoint_gate": cfg["gate"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct = train_candidate(cfg["baseline"], train, validation, root / "direct", spec, diagnosis, limits)
            trial = execute_candidate(cfg["baseline"], train, validation, root / "shared", spec, diagnosis, limits,
                active_metrics=before, baseline_metrics=before, policy=cfg["gate"], shadow=(validation, before, before))
            self.assertNotIn("error", trial)
            first, second = data.load_model(root / "direct/checkpoint.npz"), data.load_model(root / "shared/checkpoint.npz")
            self.assertEqual(first.meta, second.meta)
            for group in ("params", "first", "second"):
                for name in getattr(first, group):
                    np.testing.assert_array_equal(getattr(first, group)[name], getattr(second, group)[name])
            self.assertEqual(trial["report"]["validation_after"]["macro_f1"], direct["validation_after"]["macro_f1"])
            self.assertGreaterEqual(trial["resources"]["cpu_seconds"], trial["report"]["resources"]["cpu_seconds"])
            self.assertIn("shadow_gate", trial)
            failed = execute_candidate(cfg["baseline"], train, validation, root / "bad", {**spec, "epochs": 0}, diagnosis, limits,
                active_metrics=before, baseline_metrics=before, policy=cfg["gate"])
            self.assertIn("error", failed)
            self.assertGreaterEqual(failed["resources"]["cpu_seconds"], 0)

    def test_current_verifier_reads_refs_and_does_not_mislabel_fixture_calls(self):
        from fixtures import fixture_plan
        from fixtures import ModelFixture
        from gas_a.live import run_live

        class Observe(ModelFixture):
            def complete(self, messages, **options):
                receipt = super().complete(messages, **options)
                prompt = json.loads(messages[0]["content"])
                if "allowed_experiments" in prompt:
                    receipt["text"] = json.dumps(fixture_plan(prompt, action="observe"))
                    receipt["response"]["choices"][0]["message"]["content"] = receipt["text"]
                return receipt

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(ROOT / "configs/a.live.json")
            config["runtime"] = str(root / "live")
            write_json(root / "config.json", config)
            run_live(config, cloud=False, client=Observe())
            spec = importlib.util.spec_from_file_location("current_live_verifier", ROOT / "scripts/verify_live_run.py")
            verifier = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verifier)
            result = verifier.verify(root / "config.json", 1)
            rejected = [item["check"] for item in result["checks"] if not item["passed"]]
            self.assertEqual(rejected, ["round_01/real_provider_receipts"])
            self.assertEqual(result["all_attempts_costs"]["calls"], 2)


if __name__ == "__main__":
    unittest.main()
