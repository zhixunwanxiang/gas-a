"""Exercise current sources in an existing Linux runtime, without rebuilding an image.

Host-only PyTRiO/SwanLab, SDK 1.0.4 chunk RPC, ten public-data checkpoints,
then a committed, feature-only batch3--10 prediction stream. Not an official host.
"""
import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import numpy as np
from fixtures import ChunkedUploads
from gas_a.blind import feature_rows, score_committed
from gas_a.cloud_records import Audit
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import CLASSES, digest, read_json, write_json
from gas_a.live_access import PyTrioChat, load_credentials
from gas_a.model_records import recorded_chat, purpose_for
from gas_a.redaction import redact


def stamp():
    return datetime.now(timezone.utc).isoformat()


def manifest(root):
    return [{"path": p.relative_to(root).as_posix(), "sha256": digest(p), "size_bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file()]


def tree_digest(files):
    return "sha256:" + hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def descriptor(identity, root, artifact=None):
    artifact = artifact or {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                            "predict_argv": ["python3", "/app/harness.py", "predict"]}
    files = manifest(root)
    encoded = json.dumps({"files": files, "artifact": artifact}, sort_keys=True, separators=(",", ":")).encode()
    return {"artifact_id": identity, "artifact": artifact, "files": files,
            "artifact_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest()}


def unpack(uploads, frame, kind, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for item in uploads.files(frame, kind):
        relative = Path(item["path"])
        path = (destination / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(destination.resolve()):
            raise ValueError("Unsafe upload path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(item["data_base64"], validate=True))
    return manifest(destination)


class Process:
    def __init__(self, runner, args, name, error_file):
        self.runner, self.name = runner, name
        self.error = error_file.open("w", encoding="utf-8")
        self.wire = (error_file.parent / "protocol-duplex.jsonl").open("w", encoding="utf-8") if args[-1] == "train" else None
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.error, text=True, encoding="utf-8", bufsize=1)
        self.lines = queue.Queue()
        def reader():
            try:
                for line in self.process.stdout:
                    self.lines.put(line)
            finally:
                self.lines.put(None)
        threading.Thread(target=reader, daemon=True).start()

    def send(self, value):
        self.record_wire("host_to_participant", value)
        self.process.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def record_wire(self, direction, frame):
        if self.wire:
            self.wire.write(json.dumps({"time": stamp(), "direction": direction, "frame": frame},
                                       ensure_ascii=False, allow_nan=False) + "\n")
            self.wire.flush()

    def receive(self, timeout=120):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No protocol/prediction response before deadline") from None
        if line is None:
            raise RuntimeError("Container exited; see " + self.error.name)
        frame = json.loads(line)
        self.record_wire("participant_to_host", frame)
        return frame

    def finish(self):
        self.process.stdin.close()
        if self.lines.get(timeout=30) is not None or self.process.wait(timeout=30) != 0:
            raise RuntimeError("Unexpected trailing output or nonzero container exit")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.process.poll() is None:
            subprocess.run([self.runner.docker, "stop", "--time", "1", self.name],
                           capture_output=True, timeout=20)
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=20)
        self.error.close()
        self.process.stdout.close()
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if self.wire:
            self.wire.close()


class Runner:
    def __init__(self, args):
        self.args, self.docker = args, args.docker
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.app, self.sdk = self.out / "app", self.out / "sdk"
        self.session = uuid.uuid4().hex[:10]
        self.history, self.artifacts, self.directories, self.calls = [], {}, {}, set()
        self.state, self.state_sha = None, None
        self.audit = Audit(self.out / "audit", cloud=not args.local_records)
        self.completed = False
        self.scenario = "normal_run"

    def event(self, kind, **fields):
        fields = {"scenario": self.scenario, **fields}
        self.audit.record(kind, fields)
        print(json.dumps(redact({"time": stamp(), "event": kind, **fields}), ensure_ascii=False), flush=True)

    def command(self, args, **kwargs):
        reply = subprocess.run([self.docker, *args], capture_output=True, text=True, encoding="utf-8", timeout=60, **kwargs)
        if reply.returncode:
            raise RuntimeError("Docker command failed: " + reply.stderr[:1200])
        return reply.stdout

    @staticmethod
    def mount(args, source, target, writable=False):
        args += ["--mount", f"type=bind,source={Path(source).resolve()},target={target}" + ("" if writable else ",readonly")]

    def container(self, suffix):
        name = f"gas-sdk104-{self.session}-{suffix}"
        args = [self.docker, "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
                "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--memory", "4g", "--memory-swap", "4g", "--cpus", "2",
                "--env", "PYTHONPATH=/app/src:/app:/sdk",
                "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OPENBLAS_NUM_THREADS=1",
                "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1"]
        self.mount(args, self.app, "/app")
        self.mount(args, self.sdk, "/sdk")
        temporary = self.out / "temporary" / name
        temporary.mkdir(parents=True, exist_ok=True)
        self.mount(args, temporary, "/tmp", writable=True)
        return args, name

    def prepare(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
        for line in dockerfile.splitlines():
            parts = shlex.split(line)
            if not parts or parts[0] != "COPY" or not parts[-1].startswith("/app/"):
                continue
            for source in parts[1:-1]:
                target = self.app / parts[-1].removeprefix("/app/")
                if parts[-1].endswith("/"):
                    target /= Path(source).name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / source, target)
        archive = ROOT / "vendor/contest-harness-sdk-1.0.4.tar.gz"
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.isfile() and "/contest_harness/" in member.name:
                    relative = member.name.split("/contest_harness/", 1)[1]
                    target = (self.sdk / "contest_harness" / relative).resolve()
                    if not target.is_relative_to(self.sdk.resolve()):
                        raise ValueError("Unsafe SDK member")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        assert (self.sdk / "contest_harness/cli.py").is_file()
        self.snapshot = {"app": manifest(self.app), "sdk": manifest(self.sdk), "sdk_archive_sha256": digest(archive)}
        write_json(self.out / "source-snapshot.json", self.snapshot)
        public = self.out / "public"
        public.mkdir()
        for name in ("train.jsonl", "validation.jsonl"):
            shutil.copyfile(ROOT / "data/official_a/public" / name, public / name)
        self.train, self.validation = load_pair(public / "train.jsonl", public / "validation.jsonl")
        # Verify Dataset batch1/2 really match the authorized public projections.
        for batch, data in ((1, self.train), (2, self.validation)):
            x = np.asarray(list(feature_rows(Path(self.args.dataset) / f"batch{batch}.dat")))
            raw = (Path(self.args.dataset) / f"batch{batch}.dat").read_text(encoding="utf-8-sig")
            y = np.asarray([int(line.split()[0])-1 for line in raw.splitlines() if line.strip()])
            by_path = np.asarray([int(json.loads(path)[1][1]) for path in data["paths"]])
            np.testing.assert_array_equal(x[:, by_path], data["x"])
            np.testing.assert_array_equal(y, data["y"])
        baseline = self.out / "baseline"
        baseline.mkdir()
        shutil.copyfile(ROOT / "data/official_a/checkpoint.npz", baseline / "checkpoint.npz")
        self.baseline_id, self.active = "local-official-baseline-a", "local-official-baseline-a"
        self.directories[self.active] = baseline
        self.artifacts[self.active] = descriptor(self.active, baseline)
        self.baseline_sha = digest(baseline / "checkpoint.npz")
        self.image_info = json.loads(self.command(["image", "inspect", self.args.image]))[0]
        assert self.image_info["Os"] == "linux" and self.image_info["Architecture"] == "amd64"
        args, _ = self.container("describe")
        reply = subprocess.run([*args, self.args.image, "describe"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert reply.returncode == 0, reply.stderr
        self.description = json.loads(reply.stdout)
        assert self.description["sdk_version"] == "1.0.4" and self.description["tasks"] == ["classification"]
        self.probe()
        self.audit.start({"mode": "linux_runtime_current_source_overlay_sdk104", "rounds": self.args.rounds,
                          "controller": "real_PyTRiO_Qwen3.8-27B", "dataset": self.args.dataset,
                          "train_batch": 1, "validation_batch": 2, "prediction_batches": list(range(3, 11)),
                          "image_rebuilt": False, "official_competition_result": False,
                          "configuration": read_json(self.app / "configs/a.json")})
        self.event("prepared", sdk=self.description, train_rows=len(self.train["y"]),
                   validation_rows=len(self.validation["y"]), image_id=self.image_info["Id"])
        controls=self.app/'configs/evidence/response_controls.json'
        if controls.is_file():
            self.event('public_developer_CPU_controls_not_platform_history', evidence=read_json(controls))

    def probe(self):
        directory = self.out / "probe"
        directory.mkdir()
        for name in ("workspace", "state"):
            (directory / name).mkdir()
        args, _ = self.container("probe")
        self.mount(args, self.out / "public", "/data")
        self.mount(args, self.directories[self.active], "/previous")
        self.mount(args, directory / "state", "/state")
        self.mount(args, directory / "workspace", "/workspace", writable=True)
        code = """import os,json,hashlib,importlib.util; from pathlib import Path; import numpy,contest_harness,gas_a.contest
result={'uid':os.geteuid(),'numpy':numpy.__version__,'sdk':contest_harness.SDK_VERSION,'write_probes':{}}
for folder in ('/app','/data','/previous','/state','/workspace','/tmp'):
 try:
  p=Path(folder)/'_write_probe';p.write_text('probe');p.unlink();result['write_probes'][folder]='writable'
 except OSError:result['write_probes'][folder]='blocked'
result['direct_clients_absent']=all(importlib.util.find_spec('gas_a.'+name) is None for name in ('live','live_access','cloud_records'))
result['memory_limit_bytes']=int(Path('/sys/fs/cgroup/memory.max').read_text());result['cpu_max']=Path('/sys/fs/cgroup/cpu.max').read_text().strip()
result['source_hashes']={str(p):'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest() for root in ('/app','/sdk') for p in sorted(Path(root).rglob('*')) if p.is_file()}
print(json.dumps(result))"""
        reply = subprocess.run([*args, "--entrypoint", "python3", self.args.image, "-B", "-c", code],
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert reply.returncode == 0, reply.stderr
        result = json.loads(reply.stdout)
        assert result["uid"] == 65534 and result["direct_clients_absent"] and result["sdk"] == "1.0.4"
        assert result["memory_limit_bytes"] == 4*1024**3
        quota, period = map(int, result["cpu_max"].split())
        assert quota / period == 2
        for folder, status in result["write_probes"].items():
            assert status == ("writable" if folder in ("/workspace", "/tmp") else "blocked"), (folder, status)
        for tree in ("app", "sdk"):
            for item in self.snapshot[tree]:
                assert result["source_hashes"][f"/{tree}/{item['path']}"] == item["sha256"]
        write_json(self.out / "runtime-mount-proof.json", result)

    def resource_snapshot(self, name):
        code = "import json;from pathlib import Path;print(json.dumps({p:Path('/sys/fs/cgroup/'+p).read_text() for p in ('memory.current','memory.peak','memory.events','cpu.stat')}))"
        return json.loads(self.command(["exec", name, "python3", "-B", "-c", code]))

    def predictor(self, identity, suffix, directory):
        args, name = self.container(suffix)
        self.mount(args, self.directories[identity], "/model")
        context = {"protocol": "contest.predict-context.v1", "dataset_key": "A", "task": "classification",
                   "artifact": self.artifacts[identity]["artifact"], "artifact_id": identity}
        args += ["--env", "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz", "--env",
                 "CONTEST_PREDICTION_CONTEXT=" + json.dumps(context, separators=(",", ":")), self.args.image, "predict"]
        write_json(directory / (suffix + "-mounts.json"), {"args": args, "labels_mounted": False,
                   "public_data_mounted": False, "credentials_mounted": False})
        return Process(self, args, name, directory / (suffix + "-stderr.txt"))

    @staticmethod
    def predict_one(worker, features):
        worker.send({"task": "classification", "inputs": {"features": features}})
        reply = worker.receive(timeout=15)
        assert set(reply) == {"outputs"}
        pieces = reply["outputs"]
        assert [piece["label"] for piece in pieces] == CLASSES
        values = np.asarray([piece["score"] for piece in pieces], dtype=float)
        assert len(values) == 6 and np.isfinite(values).all() and (values >= 0).all()
        assert abs(values.sum()-1) < 1e-10 and np.count_nonzero(values == values.max()) == 1
        return values

    def measure_candidate(self, identity, directory, suffix):
        rows = []
        began = time.monotonic()
        with self.predictor(identity, suffix, directory) as worker:
            # Original JSONL order, never supply IDs/answers to the predictor.
            for line in (self.out / "public/validation.jsonl").read_text(encoding="utf-8").splitlines():
                rows.append(self.predict_one(worker, json.loads(line)["inputs"]["features"]))
            resources = self.resource_snapshot(worker.name)
            worker.finish()
        from gas_a.data import score_predictions
        scores = score_predictions(np.asarray(rows), self.validation["y"], time.monotonic()-began)
        np.savez_compressed(directory / (suffix + "-predictions.npz"), probabilities=np.asarray(rows))
        return scores, resources

    def train_round(self, index):
        began, started = time.monotonic(), stamp()
        root = self.out / f"round-{index:02d}"
        work = root / "workspace"
        work.mkdir(parents=True)
        parent = self.active
        before = evaluate(load_model(self.directories[parent] / "checkpoint.npz"), self.validation)
        args, name = self.container(f"train{index}")
        self.mount(args, self.out / "public", "/data")
        self.mount(args, self.directories[parent], "/previous")
        self.mount(args, work, "/workspace", writable=True)
        if self.state:
            self.mount(args, self.state, "/state")
        args += [self.args.image, "train"]
        prompt = ("本次为本地 SDK 1.0.4 真容器测试，非官方隐藏评测。任务 A：128 维特征六类分类，主指标 Macro-F1。"
                  "必须续训绑定父检查点。只以授权公开训练集更新梯度；公开验证用于评估与选择，未来数据及标签不可见。"
                  "按用户消息给出的实验与选版契约返回一个 JSON 对象。")
        start = {"type": "round_start", "protocol": "contest.harness.workspace.v2", "run_id": self.session,
                 "prompt": prompt, "prompt_version": "local-sdk104-docker-v1", "dataset_key": "A",
                 "round_index": index, "stage_id": "public-train1-validation2",
                 "task_contract": {"task_type": "classification", "metric": "macro_f1", "metric_direction": "maximize",
                                   "series": "gas_sensor_drift", "public_projection": {
                                       "train_path": "/data/train.jsonl", "validation_path": "/data/validation.jsonl"}},
                 "model_name": self.client.model, "max_output_tokens": 768, "wall_seconds": 300,
                 "parent_artifact": self.artifacts[parent], "parent_checkpoint_sha256": digest(self.directories[parent] / "checkpoint.npz"),
                 "artifacts": list(self.artifacts.values()), "current_validation": {"metrics": before},
                 "state_path": "/state" if self.state else None, "state_sha256": self.state_sha,
                 "resource_limits": dict(ChunkedUploads.limits)}
        write_json(root / "start.json", start)
        write_json(root / "mounts.json", {"args": args, "future_data_mounted": False, "credentials_mounted": False})
        uploads, calls, candidates, events = ChunkedUploads(), [], [], []
        next_state, next_sha, sequence = None, None, 0
        with Process(self, args, name, root / "container-stderr.txt") as process, (root / "protocol.jsonl").open("w", encoding="utf-8") as trace:
            process.send(start)
            while True:
                if time.monotonic()-began > 300:
                    raise TimeoutError("Local host round wall-time budget exceeded")
                frame = process.receive(timeout=min(120, 310-(time.monotonic()-began)))
                kind = frame["type"]
                logged = dict(frame)
                if "data_base64" in logged:
                    logged["decoded_bytes"] = len(base64.b64decode(logged.pop("data_base64")))
                trace.write(json.dumps({"time": stamp(), **redact(logged)}, ensure_ascii=False) + "\n")
                trace.flush()
                if kind == "round_failed":
                    raise RuntimeError("Container round_failed: " + str(frame))
                if kind == "round_complete":
                    complete = frame["record"]
                    assert complete["model_call_ids"] == calls
                    assert complete["history_ids"] == [record["id"] for record in self.history]
                    assert complete["selected_artifact_id"] in self.artifacts and next_state
                    resources = self.resource_snapshot(name)
                    process.send({"type": "round_ack", "accepted": True})
                    process.finish()
                    break
                assert frame["id"] == sequence
                sequence += 1
                if kind.startswith("upload_"):
                    reply = uploads.handle(frame)
                elif kind == "history_request":
                    reply = {"type": "history_response", "records": self.history}
                    events.append({"stage": "history_read", "time": stamp(), "records": len(self.history)})
                elif kind == "model_request":
                    purpose = purpose_for(frame["messages"])
                    events.append({"stage": purpose, "time": stamp()})
                    options = {key: value for key, value in frame.items() if key not in ("type", "id", "messages")}
                    assert not set(options)-{"max_tokens", "temperature", "enable_thinking"}
                    options.setdefault("enable_thinking", False)
                    binding = {key: start[key] for key in ("run_id", "dataset_key", "round_index", "task_contract", "prompt_version")}
                    messages = [{"role": "system", "content": prompt + "\n" + json.dumps(binding, ensure_ascii=False)}, *frame["messages"]]
                    self.event("model_call_started", round=index, purpose=purpose)
                    receipt = recorded_chat(self.client, root / "model-calls", messages, **options,
                                            purpose=purpose, context={"round": index, "scenario": self.scenario,
                                                                     "origin": "local_sdk_host_not_official"})
                    assert receipt["call_id"] not in self.calls
                    self.calls.add(receipt["call_id"])
                    calls.append(receipt["call_id"])
                    self.audit.record("host_model_receipt", {"round": index, "purpose": purpose, "receipt": receipt})
                    self.event("model_call_completed", round=index, purpose=purpose, call_id=receipt["call_id"], usage=receipt.get("usage", {}))
                    reply = {"type": "model_response", "call_id": receipt["call_id"], "response": receipt["response"]}
                elif kind == "candidate_request":
                    assert frame["parent_artifact_id"] == parent and frame["parent_checkpoint_sha256"] == start["parent_checkpoint_sha256"]
                    identity = f"local-r{index:02d}-candidate-{len(candidates)+1:02d}"
                    directory = root / "artifacts" / identity
                    files = unpack(uploads, frame, "candidate", directory)
                    declaration = frame["artifact"]
                    assert declaration["checkpoint_paths"] == ["checkpoint.npz"]
                    assert declaration["predict_argv"] == ["python3", "/app/harness.py", "predict"]
                    assert set(declaration["checkpoint_paths"] + declaration["code_paths"]) <= {item["path"] for item in files}
                    report = read_json(directory / "training.json")
                    pm = load_model(self.directories[parent] / "checkpoint.npz")
                    child = load_model(directory / "checkpoint.npz")
                    assert report["initial_steps"] == pm.meta["steps"] and child.meta["steps"] > pm.meta["steps"]
                    assert report["parent_checkpoint_sha256"] == start["parent_checkpoint_sha256"]
                    np.testing.assert_array_equal(child.mean, pm.mean)
                    np.testing.assert_array_equal(child.scale, pm.scale)
                    self.directories[identity], self.artifacts[identity] = directory, descriptor(identity, directory, declaration)
                    measured, prediction_resources = self.measure_candidate(identity, root, f"validation{len(candidates)+1}")
                    assert abs(measured["macro_f1"]-report["validation_after"]["macro_f1"]) < 1e-12
                    candidate = {"artifact_id": identity, "metrics": measured, "initial_steps": pm.meta["steps"],
                                 "final_steps": child.meta["steps"], "continuation_verified": True,
                                 "validation_predict_resources": prediction_resources, "configuration": frame["configuration"]}
                    candidates.append(candidate)
                    events.append({"stage": "candidate_evaluated", "time": stamp(), "artifact_id": identity})
                    self.event("candidate_measured", round=index, artifact_id=identity, macro_f1=measured["macro_f1"],
                               initial_steps=pm.meta["steps"], final_steps=child.meta["steps"])
                    reply = {"type": "candidate_response", "artifact_id": identity, "summary": {"metrics": measured}}
                elif kind == "state_request":
                    assert frame["parent_state_sha256"] == (self.state_sha or "")
                    next_state = root / "saved-state"
                    files = unpack(uploads, frame, "state", next_state)
                    memory = read_json(next_state / "lifecycle.json")
                    assert memory["rounds"][-1]["model_call_ids"] == calls
                    assert digest(next_state / "baseline_checkpoint.npz") == self.baseline_sha
                    next_sha = tree_digest(files)
                    reply = {"type": "state_response", "state_sha256": next_sha}
                elif kind == "artifact_read_request":
                    directory = self.directories[frame["artifact_id"]]
                    file = (directory / frame["path"]).resolve()
                    assert file.is_relative_to(directory.resolve())
                    declared = next(item for item in self.artifacts[frame["artifact_id"]]["files"] if item["path"] == frame["path"])
                    assert digest(file) == declared["sha256"]
                    with file.open("rb") as stream:
                        stream.seek(frame["offset"])
                        payload = stream.read(frame["length"])
                    reply = {"type": "artifact_read_response", "data_base64": base64.b64encode(payload).decode(),
                             "eof": frame["offset"]+len(payload) == declared["size_bytes"]}
                else:
                    raise ValueError("Unhandled SDK frame: " + kind)
                process.send({"id": frame["id"], **reply})
        internal = read_json(next(work.glob("gas-a-*/round.json")))
        self.active = complete["selected_artifact_id"]
        self.state, self.state_sha = next_state, next_sha
        self.history.append({"id": f"local-history-{index:02d}", "round_index": index,
                             "selected_artifact_id": self.active, "reason": complete["reason"], "metrics": internal["after"]})
        result = {"round": index, "started_at": started, "completed_at": stamp(), "seconds": time.monotonic()-began,
                  "scenario": self.scenario, "history_ids": complete["history_ids"], "finish_record": complete,
                  "action": internal["action"], "parent": parent, "selected": self.active,
                  "before": before, "after": internal["after"], "reason": complete["reason"],
                  "candidates": candidates, "call_ids": calls, "events": events, "resources": resources,
                  "chunk_count": uploads.chunk_count, "reserved_bytes": uploads.reserved,
                  "sdk_finish_acknowledged": True, "drift": read_json(next(work.glob("gas-a-*/monitor.json"))),
                  "diagnosis": internal["diagnosis"], "plan": internal["plan"], "experiments": internal["experiments"]}
        write_json(root / "result.json", result)
        self.audit.record("round_summary", internal["review"])
        self.event("round_completed", round=index, action=result["action"], selected=self.active,
                   candidates=len(candidates), macro_f1=result["after"]["macro_f1"], seconds=result["seconds"])
        return result

    def rollback_drill(self):
        """An isolated, explicitly injected degraded model. Never part of the score run."""
        saved = {name: getattr(self, name) for name in
                 ("out", "active", "state", "state_sha", "history", "artifacts", "directories", "scenario")}
        main = self.out
        root = main / "drills/rollback"
        root.mkdir(parents=True)
        try:
            self.out, self.scenario = root, "injected_rollback_drill_not_normal_score"
            shutil.copytree(main / "public", root / "public")
            self.history = saved["history"][:1]
            self.state = main / "round-01/saved-state"
            self.state_sha = tree_digest(manifest(self.state))
            self.active = "local-injected-degraded-parent"
            parent = root / "injected-parent"
            parent.mkdir()
            model = load_model(saved["directories"][self.baseline_id] / "checkpoint.npz")
            model.params["w2"].fill(0)
            model.params["b2"].fill(0)
            model.params["b2"][0] = 5
            model.save(parent / "checkpoint.npz")
            self.directories = {self.baseline_id: saved["directories"][self.baseline_id], self.active: parent}
            self.artifacts = {self.baseline_id: saved["artifacts"][self.baseline_id],
                              self.active: descriptor(self.active, parent)}
            write_json(root / "fault-injection.json", {
                "injected": True, "normal_score_run": False, "hidden_data_used": False,
                "mechanism": "Zero output weights and bias the first class in a copied model; preserve a valid checkpoint format.",
                "history_source": str(main / "round-01/result.json"),
                "state_source": str(self.state), "degraded_checkpoint_sha256": digest(parent / "checkpoint.npz"),
                "expected_recovery_id": self.baseline_id})
            self.event("rollback_drill_started", injected=True, excluded_from_normal_score=True)
            result = self.train_round(2)
            recovered, resources = self.measure_candidate(self.active, root, "recovery-validation")
            baseline = evaluate(load_model(self.directories[self.baseline_id] / "checkpoint.npz"), self.validation)
            passed = (result["action"] == "rollback" and result["selected"] == self.baseline_id
                      and abs(recovered["macro_f1"]-baseline["macro_f1"]) < 1e-12)
            outcome = {"passed": passed, "injected": True, "normal_score_run": False,
                       "action": result["action"], "selected": result["selected"], "before": result["before"],
                       "recovered": recovered, "call_ids": result["call_ids"],
                       "resources": resources, "completed_at": stamp()}
            write_json(root / "summary.json", outcome)
            self.event("rollback_drill_completed", passed=passed, action=result["action"], selected=result["selected"])
            return outcome
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def replay(self):
        root = self.out / "blind"
        root.mkdir()
        files = [Path(self.args.dataset) / f"batch{batch}.dat" for batch in range(3, 11)]
        commits = []
        bindings = {identity: digest(self.directories[identity] / "checkpoint.npz")
                    for identity in (self.active, self.baseline_id)}
        with ExitStack() as stack:
            actual = stack.enter_context(self.predictor(self.active, "stream-active", root))
            static = stack.enter_context(self.predictor(self.baseline_id, "stream-baseline", root))
            for batch, file in zip(range(3, 11), files):
                began, started, source_sha = time.monotonic(), stamp(), digest(file)
                active, baseline = [], []
                for features in feature_rows(file):
                    active.append(self.predict_one(actual, features))
                    baseline.append(self.predict_one(static, features))
                assert digest(file) == source_sha
                directory = root / "windows" / f"batch{batch:02d}"
                directory.mkdir(parents=True)
                predictions = directory / "predictions.npz"
                np.savez_compressed(predictions, active=np.asarray(active), baseline=np.asarray(baseline))
                commit = {"batch": batch, "rows": len(active), "source_sha256": source_sha,
                          "predictions": predictions.relative_to(root).as_posix(), "predictions_sha256": digest(predictions),
                          "started_at": started, "committed_at": stamp(), "seconds": time.monotonic()-began,
                          "labels_read_for_scoring": False}
                write_json(directory / "prediction_commit.json", commit)
                commits.append(commit)
                self.event("blind_window_committed", batch=batch, rows=len(active), seconds=commit["seconds"])
            resources = {"active": self.resource_snapshot(actual.name), "baseline": self.resource_snapshot(static.name)}
            actual.finish()
            static.finish()
        for identity, binding in bindings.items():
            assert digest(self.directories[identity] / "checkpoint.npz") == binding
        write_json(root / "prediction_complete.json", {"workers_exited": True, "commits": commits, "completed_at": stamp()})
        scoring_started = stamp()
        results = score_committed(root, files)
        summary = {"success": True, "mode": "actual_sdk_cli_feature_only_continuous_stream",
                   "official_competition_result": False, "labels_sent_to_predictor": False,
                   "feedback_calls": 0, "model_calls_in_prediction": 0, "online_updates": 0,
                   "scoring_started_at": scoring_started, "completed_at": stamp(), "resources": resources, **results}
        write_json(root / "summary.json", summary)
        return summary

    def run(self):
        try:
            self.prepare()
            load_credentials(ROOT / ".env.local")
            self.client = PyTrioChat(model="Qwen/Qwen3.8-27B", timeout=90)
            rounds = [self.train_round(i) for i in range(1, self.args.rounds+1)]
            normal_calls = len(self.calls)
            drill = self.rollback_drill() if self.args.rollback_drill else None
            selected = self.out / "selected-model"
            selected.mkdir()
            shutil.copyfile(self.directories[self.active] / "checkpoint.npz", selected / "checkpoint.npz")
            write_json(selected / "selection.json", {"artifact_id": self.active, "sha256": digest(selected / "checkpoint.npz")})
            blind = self.replay()
            assert manifest(self.app) == self.snapshot["app"] and manifest(self.sdk) == self.snapshot["sdk"]
            used = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
            summary = {"success": True, "started_at": rounds[0]["started_at"], "completed_at": stamp(),
                       "mode": "actual_linux_containers_current_sources_overlay_no_image_rebuild", "sdk": self.description,
                       "official_competition_result": False, "image_rebuilt": False, "runtime_image_id": self.image_info["Id"],
                       "train_rows": len(self.train["y"]), "validation_rows": len(self.validation["y"]),
                       "training_batches": [1], "validation_batches": [2], "prediction_batches": list(range(3, 11)),
                       "rounds": len(rounds), "candidate_count": sum(len(r["candidates"]) for r in rounds),
                       "promotions": sum(r["action"] == "promote" for r in rounds),
                       "rollbacks": sum(r["action"] == "rollback" for r in rounds), "actual_model_calls": normal_calls,
                       "rollback_drill": drill, "total_actual_model_calls_including_drill": len(self.calls),
                       "selected": self.active, "output_bytes": used, "blind": blind,
                       "limits": {"cpus": 2, "memory_bytes": 4*1024**3, "tmp_storage": "host_disk_bind",
                                  "tmp_disk_cap_hard_enforced": False,
                                  "workspace_disk_cap_hard_enforced": False,
                                  "note": "Windows bind workspace has no hard 4GiB disk quota; observed usage reported."}}
            write_json(self.out / "summary.json", summary)
            self.event("simulation_complete", selected=self.active, aggregate=blind["aggregate"], rounds=len(rounds))
            self.completed = True
        except Exception as exc:
            write_json(self.out / "failure.json", {"time": stamp(), "type": type(exc).__name__, "error": redact(str(exc))})
            raise
        finally:
            cloud = self.audit.finish(self.completed)
            if self.completed:
                summary["cloud"] = cloud
                write_json(self.out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", required=True)
    parser.add_argument("--image", default="gas-a-harness:current-20261121-sdk-live")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--local-records", action="store_true")
    parser.add_argument("--rollback-drill", action="store_true", help="Run a separate injected-degradation recovery check")
    args = parser.parse_args()
    load_credentials(ROOT / ".env.local")
    Runner(args).run()


if __name__ == "__main__":
    main()
