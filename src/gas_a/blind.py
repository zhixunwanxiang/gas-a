"""Freeze a complete feature-only prediction stream before independent scoring.

The worker is a separate process, given only checkpoint path and input features.
DAT paths, targets, window IDs, credentials and evaluation feedback are not sent.
No training or model selection occurs during this replay.
"""
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np

from .data import load_model, score_predictions
from .io import CLASSES, digest, read_json, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def feature_rows(path):
    """The first token is deliberately neither converted nor interpreted."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            values, seen = [0.] * 128, set()
            for token in line.split()[1:]:
                key, value = token.split(":", 1)
                index = int(key) - 1
                if index in seen or not 0 <= index < 128:
                    raise ValueError(f"Invalid feature index at line {number}")
                seen.add(index)
                values[index] = float(value)
            if len(seen) != 128 or not np.isfinite(values).all():
                raise ValueError(f"Expected 128 finite features at line {number}")
            yield values


class Predictor:
    def __init__(self, checkpoint, timeout=10):
        self.timeout, self.output, self.errors = timeout, queue.Queue(), deque(maxlen=8)
        environment = {key: value for key, value in os.environ.items() if key.upper() in {
            "SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}}
        environment.update(PYTHONPATH=str(Path(__file__).resolve().parent.parent),
                           PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", BASELINE_CPU_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.process = subprocess.Popen([sys.executable, "-B", "-m", "gas_a.blind_worker",
                                         "--checkpoint", str(Path(checkpoint).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=environment, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.error_reader = threading.Thread(target=self._errors, daemon=True)
        self.reader.start(); self.error_reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.output.put(line)
        self.output.put(None)

    def _errors(self):
        for line in self.process.stderr:
            self.errors.append(line[-1000:])

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        try:
            if kind is not None:
                self.process.kill()
            else:
                self.process.stdin.close()
            self.process.wait(timeout=self.timeout)
            self.reader.join(timeout=1); self.error_reader.join(timeout=1)
            if kind is None and self.process.returncode:
                raise RuntimeError("Prediction worker failed: " + "".join(self.errors))
        except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(timeout=5)
            if kind is None:
                raise TimeoutError("Prediction worker did not finish") from None
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()

    def predict(self, features):
        self.process.stdin.write(json.dumps({"task": "classification", "inputs": {"features": features}}, allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.output.get(timeout=self.timeout)
        except queue.Empty:
            raise TimeoutError("No prediction before the next input; feedback/EOF waits are forbidden") from None
        if line is None:
            raise RuntimeError("Prediction worker stopped: " + "".join(self.errors))
        reply = json.loads(line)
        if not isinstance(reply, dict) or set(reply) != {"outputs"} or not isinstance(reply["outputs"], list):
            raise ValueError("Invalid prediction worker response")
        outputs = reply["outputs"]
        if len(outputs) != 6 or [item["label"] for item in outputs] != CLASSES:
            raise ValueError("Prediction must contain the six official classes")
        values = np.asarray([item["score"] for item in outputs], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or abs(values.sum()-1) > 1e-10 or np.count_nonzero(values == values.max()) != 1:
            raise ValueError("Invalid classification probabilities")
        return values


def score_committed(output, files):
    """Called only after every worker has exited and all batches are committed."""
    root = Path(output)
    complete = read_json(root / "prediction_complete.json")
    if complete.get("workers_exited") is not True or len(complete["commits"]) != len(files):
        raise ValueError("Complete prediction stream must be committed before opening targets")
    rows, labels, current, reference = [], [], [], []
    for commit, source in zip(complete["commits"], files):
        frozen = root / commit["predictions"]
        if digest(source) != commit["source_sha256"] or digest(frozen) != commit["predictions_sha256"]:
            raise ValueError("Source or frozen prediction changed before scoring")
        # This is the first target conversion anywhere in the replay.
        y = np.asarray([int(line.split()[0])-1 for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()], dtype=int)
        if len(y) != commit["rows"] or np.any((y < 0) | (y >= 6)):
            raise ValueError("Invalid scoring targets")
        with np.load(frozen, allow_pickle=False) as values:
            active, base = values["active"].copy(), values["baseline"].copy()
        a, b = score_predictions(active, y), score_predictions(base, y)
        rows.append({"batch": commit["batch"], "active": a, "baseline": b,
                     "gain_macro_f1": a["macro_f1"]-b["macro_f1"]})
        labels.append(y); current.append(active); reference.append(base)
    actual = score_predictions(np.concatenate(current), np.concatenate(labels))
    baseline = score_predictions(np.concatenate(reference), np.concatenate(labels))
    return {"windows": rows, "aggregate": {"active": actual, "baseline": baseline,
            "gain_macro_f1": actual["macro_f1"]-baseline["macro_f1"]}}


def run_blind(checkpoint, baseline, dataset_directory, output, batches=tuple(range(3, 11)), timeout=10):
    batches = tuple(batches)
    if (not batches or tuple(sorted(set(batches))) != batches
            or any(type(batch) is not int or not 3 <= batch <= 10 for batch in batches)
            or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 120):
        raise ValueError("Invalid blind replay windows or prediction timeout")
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("Blind replay requires a fresh output directory")
    checkpoint, baseline = Path(checkpoint).resolve(), Path(baseline).resolve()
    load_model(checkpoint); load_model(baseline)
    bindings = {"checkpoint": digest(checkpoint), "baseline": digest(baseline)}
    files = [Path(dataset_directory).resolve()/f"batch{batch}.dat" for batch in batches]
    sources = []
    for file in files:
        if not file.is_file() or file.is_symlink():
            raise ValueError("Missing regular DAT window")
        count = sum(1 for _ in feature_rows(file))
        if not count:
            raise ValueError("Empty DAT window")
        sources.append({"sha256": digest(file), "rows": count})
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"configuration.json", {"checkpoint": str(checkpoint), "baseline": str(baseline),
        "bindings": bindings, "batches": batches, "dataset_directory": str(Path(dataset_directory).resolve()),
        "labels_in_prediction": False, "model_calls_in_prediction": 0,
        "online_training": False, "online_scaler_fitting": False,
        "worker_code_sha256": digest(Path(__file__).with_name("blind_worker.py")),
        "scoring_code_sha256": digest(Path(__file__)), "started_at": timestamp()})
    commits = []
    try:
        with ExitStack() as stack:
            active_worker = stack.enter_context(Predictor(checkpoint, timeout))
            base_worker = stack.enter_context(Predictor(baseline, timeout))
            for batch, file, source in zip(batches, files, sources):
                active, base = [], []
                for features in feature_rows(file):
                    active.append(active_worker.predict(features))
                    base.append(base_worker.predict(features))
                if len(active) != source["rows"] or digest(file) != source["sha256"]:
                    raise ValueError("DAT changed during prediction")
                directory = root/"windows"/f"batch{batch:02d}"
                directory.mkdir(parents=True, exist_ok=False)
                target = directory/"predictions.npz"
                np.savez_compressed(target, active=np.asarray(active), baseline=np.asarray(base))
                commit = {"batch": batch, "rows": len(active), "source_sha256": source["sha256"],
                    "predictions": target.relative_to(root).as_posix(), "predictions_sha256": digest(target),
                    "committed_at": timestamp(), "labels_read_for_scoring": False}
                write_json(directory/"prediction_commit.json", commit)
                commits.append(commit)
        if digest(checkpoint) != bindings["checkpoint"] or digest(baseline) != bindings["baseline"]:
            raise ValueError("Frozen prediction model changed")
        write_json(root/"prediction_complete.json", {"workers_exited": True, "commits": commits,
                                                     "completed_at": timestamp()})
        scoring_started = timestamp()
        results = score_committed(root, files)
        summary = {"success": True, "mode": "independent_scoring_without_label_feedback",
                   "official_competition_result": False, "scoring_started_at": scoring_started,
                   "completed_at": timestamp(), "prediction_models_unchanged": True, **results}
        write_json(root/"summary.json", summary)
        return summary
    except Exception as exc:
        write_json(root/"failure.json", {"error": f"{type(exc).__name__}: {exc}", "time": timestamp(),
                                        "committed_windows": len(commits), "success": False})
        raise
