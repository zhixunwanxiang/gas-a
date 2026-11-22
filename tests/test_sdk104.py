"""SDK 1.0.4 的分块上传、资源合同和中断会话回归；不调用外部服务。"""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

from contest_harness.errors import HarnessError
from contest_harness.round import Round
from gas_a.contest import train
from gas_a.pipeline import load_config
from test_sdk_compatibility import SdkHost

ROOT = Path(__file__).resolve().parents[1]


class InterruptedReader:
    def __init__(self, host, stage):
        self.host, self.stage, self.triggered = host, stage, False
        self.reads = 0

    def readline(self):
        self.reads += 1
        frame = self.host.pending[0]
        kind = frame.get("type")
        match = {"plan": kind == "model_response" and len(self.host.calls) == 1,
                 "selection": kind == "model_response" and len(self.host.calls) == 2,
                 "chunk": kind == "upload_chunk_response",
                 "candidate_validation": kind == "candidate_response",
                 "state": kind == "state_response", "finish": kind == "round_ack"}
        if not self.triggered and match[self.stage]:
            self.triggered = True
            raise TimeoutError("Offline interruption after the host handled the request")
        return self.host.readline()


class Sdk104Tests(unittest.TestCase):
    def prepare(self, root, *, reader_stage=None, limits=None):
        config = load_config(ROOT / "configs/a.json")
        host = SdkHost(root, config, "complete", "checkpoint.npz", False, False)
        if limits:
            host.frame["resource_limits"].update(limits)
        reader = InterruptedReader(host, reader_stage) if reader_stage else host
        rnd = Round.start(reader, host)
        rnd.config, rnd.parent.root = config, root / "previous"
        return host, reader, rnd

    def test_candidate_and_state_use_hash_checked_chunked_upload_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host, _, rnd = self.prepare(root)
            with patch.dict(os.environ, {"CONTEST_WORK": str(root / "work")}):
                train(rnd)
            self.assertTrue(rnd._finished)
            self.assertGreater(host.uploads.chunk_count, len(host.uploads.uploads))
            commits = [f for f in host.frames if f["type"] in ("candidate_request", "state_request")]
            self.assertEqual([f["type"] for f in commits], ["candidate_request", "state_request"])
            for frame in commits:
                self.assertNotIn("files", frame)
                self.assertTrue(host.uploads.uploads[frame["upload_id"]]["finished"])
            self.assertGreater(host.uploads.reserved["candidate"], 0)
            self.assertGreater(host.uploads.reserved["state"], 0)

    def test_old_host_is_rejected_before_model_calls_or_training(self):
        for limits in ({"artifact_upload_protocol": "inline"}, {}, None):
            with self.subTest(limits=limits), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                host, _, rnd = self.prepare(root)
                rnd.limits = limits
                with patch("gas_a.contest.execute_candidate") as execute:
                    with self.assertRaises(HarnessError) as caught:
                        train(rnd)
                self.assertEqual(caught.exception.code, "CHUNKED_UPLOAD_UNSUPPORTED")
                execute.assert_not_called()
                self.assertEqual(host.frames, [])
                self.assertEqual(host.calls, [])

    def test_host_candidate_and_state_size_limits_are_honored(self):
        for resource, forbidden in (("artifact_bytes", "candidate_request"), ("state_bytes", "state_request")):
            with self.subTest(resource=resource), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                host, _, rnd = self.prepare(root, limits={resource: 1})
                with patch.dict(os.environ, {"CONTEST_WORK": str(root / "work")}):
                    with self.assertRaises(HarnessError) as caught:
                        train(rnd)
                self.assertEqual(caught.exception.code, "ARTIFACT_LIMIT")
                self.assertNotIn(forbidden, [f["type"] for f in host.frames])
                self.assertNotIn("round_complete", [f["type"] for f in host.frames])

    def test_interrupted_round_does_not_retry_or_consume_late_responses(self):
        expected_calls = {"plan": 1, "selection": 2, "chunk": 1,
                          "candidate_validation": 1, "state": 2, "finish": 2}
        for stage, count in expected_calls.items():
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                host, reader, rnd = self.prepare(root, reader_stage=stage)
                with patch.dict(os.environ, {"CONTEST_WORK": str(root / "work")}), \
                        patch.object(rnd, "model", wraps=rnd.model) as model:
                    with self.assertRaisesRegex(TimeoutError, "Offline interruption"):
                        train(rnd)
                self.assertTrue(reader.triggered)
                self.assertTrue(rnd._outcome_unknown)
                self.assertEqual(model.call_count, count)
                sent, reads = len(host.frames), reader.reads
                self.assertTrue(host.pending)  # 保留迟到回执，不把它读作后续请求的回复。
                with self.assertRaises(HarnessError) as caught:
                    rnd.history()
                self.assertEqual(caught.exception.code, "RPC_OUTCOME_UNKNOWN")
                self.assertEqual(len(host.frames), sent)
                self.assertEqual(reader.reads, reads)
