"""本地模型注册表：保存版本身份、校验 checkpoint、切换活动版本和执行恢复。"""
import re
import shutil
import uuid
from pathlib import Path
from .data import load_model, evaluate
from .io import read_json, write_json, write_jsonl, digest, stamp, history
from .controller_policy import class_protection


class Registry:
    """管理不可覆盖的模型版本与 active.json 指针；写入操作由上层运行锁协调。"""
    def __init__(self, root):
        self.root = Path(root).resolve()

    def initialize(self, baseline, metrics):
        """为运行目录绑定原始 baseline；已有注册表时检查 baseline 身份一致性。"""
        path = self.root / "versions" / "baseline" / "checkpoint.npz"
        if (self.root / "registry.json").exists():
            if self.entries()["baseline"]["sha256"] != digest(baseline):
                raise ValueError("Runtime already belongs to a different baseline")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline, path)
        entry = {"path": "versions/baseline/checkpoint.npz", "sha256": digest(path),
                 "parent": None, "metrics": metrics, "created_at": stamp()}
        write_json(self.root / "registry.json", {"format": "gas-a.registry.v1", "versions": {"baseline": entry}})
        write_json(self.root / "active.json", {"version": "baseline", "updated_at": stamp(), "reason": "initialize"})

    def entries(self):
        return read_json(self.root / "registry.json")["versions"]

    def active_id(self):
        return read_json(self.root / "active.json")["version"]

    def checkpoint(self, version):
        """检查版本 ID、路径范围和 SHA-256 摘要，并确认 checkpoint 可以加载。"""
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise ValueError("Invalid version ID")
        entry = self.entries()[version]
        declared = self.root / entry["path"]
        path = declared.resolve()
        # 注册表中的相对路径必须解析到 runtime 内，文件内容也必须匹配登记的 SHA-256。
        if not path.is_relative_to(self.root) or declared.is_symlink() or digest(path) != entry["sha256"]:
            raise ValueError("Version checkpoint digest mismatch: " + version)
        load_model(path)
        return path

    def register(self, version, checkpoint, parent, metrics, report):
        """复制候选 checkpoint 和训练证据，登记父版本与指标；注册本身不激活模型。"""
        if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", version):
            raise ValueError("Invalid version ID")
        entries = self.entries()
        if version in entries:
            raise ValueError("Version already exists")
        source = Path(checkpoint)
        load_model(source)
        dest = self.root / "versions" / version / "checkpoint.npz"
        dest.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(source, dest)
        # 新注册版本同时保留逐 epoch 训练证据；旧 checkpoint
        # 仍可注册，但不补造缺失的历史记录。
        for name in ("training.json", "history.jsonl", "migration.json", "migrated_parent.npz"):
            evidence = source.parent / name
            if evidence.is_file():
                shutil.copyfile(evidence, dest.parent / name)
        if (source.parent / "source").is_dir():
            shutil.copytree(source.parent / "source", dest.parent / "source")
        write_json(dest.parent / "experiment.json", report)
        entries[version] = {"path": dest.relative_to(self.root).as_posix(), "sha256": digest(dest),
                            "parent": parent, "metrics": metrics, "created_at": stamp()}
        write_json(self.root / "registry.json", {"format": "gas-a.registry.v1", "versions": entries})

    def activate(self, version, reason):
        """先写可恢复意图，再提交活动指针和幂等日志；短暂写入错误重试一次。"""
        self.recover_activation()
        self.checkpoint(version)  # 修改活动版本指针前先校验 checkpoint。
        try:
            old = read_json(self.root / "active.json")
            previous = old["version"]
        except (ValueError, KeyError, OSError):
            previous, old = None, None
        identity = uuid.uuid4().hex
        pending = {"format": "gas-a.activation.v1", "activation_id": identity, "previous_pointer": old,
                   "pointer": {"version": version, "previous_version": previous, "reason": reason, "updated_at": stamp()},
                   "event": {"activation_id": identity, "from": previous, "to": version, "reason": reason, "time": stamp()}}
        write_json(self.root / "activation_pending.json", pending)
        for attempt in range(2):
            try:
                self._complete_activation(pending)
                return
            except OSError:
                if attempt:
                    raise  # 意图仍在，调度器重启后可补齐；不伪造持久化成功。

    def _complete_activation(self, pending):
        self.checkpoint(pending["pointer"]["version"])
        write_json(self.root / "active.json", pending["pointer"])
        path = self.root / "deployments.jsonl"
        events = history(path)
        if not any(e.get("activation_id") == pending["event"]["activation_id"] for e in events):
            write_jsonl(path, [*events, pending["event"]])
        (self.root / "activation_pending.json").unlink(missing_ok=True)

    def recover_activation(self):
        """由持运行锁的调度器恢复中断切版，HTTP读取不主动写恢复状态。"""
        path = self.root / "activation_pending.json"
        if not path.exists():
            return None
        pending = read_json(path)
        if (pending.get("format") != "gas-a.activation.v1"
                or not re.fullmatch(r"[a-f0-9]{32}", str(pending.get("activation_id")))
                or pending.get("event", {}).get("activation_id") != pending["activation_id"]
                or pending.get("event", {}).get("to") != pending.get("pointer", {}).get("version")):
            raise ValueError("Invalid interrupted activation record")
        try:
            self.checkpoint(pending["pointer"]["version"])
        except (ValueError, KeyError, OSError) as exc:
            old = pending.get("previous_pointer") or {"version": "baseline"}
            try:
                self.checkpoint(old["version"])
            except (ValueError, KeyError, OSError):
                self.checkpoint("baseline")
                old = {"version": "baseline"}
            pending["pointer"] = {**old, "reason": "activation_aborted_restore_intact_version", "updated_at": stamp()}
            try:
                actual_from = self.active_id()
            except (ValueError, KeyError, OSError):
                actual_from = None
            pending["event"] = {"activation_id": pending["activation_id"] + "_restore", "from": actual_from,
                "to": old["version"], "attempted_to": pending["event"]["to"],
                "reason": "activation_aborted_restore_intact_version", "time": stamp(), "error": str(exc)}
        self._complete_activation(pending)
        return {"activation_id": pending["activation_id"], "selected": pending["pointer"]["version"],
                "reason": pending["pointer"]["reason"]}

    def ensure_healthy(self, validation, gate):
        """在当前 validation 上复评 baseline 和活动版本，必要时重新比较可用版本并恢复。"""
        self.recover_activation()
        # baseline 在当前 validation 上重新评估；恢复判断不直接沿用其他数据窗口的分数。
        baseline_metrics = evaluate(load_model(self.checkpoint("baseline"), validation["paths"]), validation)
        error, active_id, active_metrics = None, None, None
        try:
            active_id = self.active_id()
            active_metrics = evaluate(load_model(self.checkpoint(active_id), validation["paths"]), validation)
        except (ValueError, KeyError, OSError) as exc:
            error = str(exc)
        previous_score = self.entries().get(active_id, {}).get("metrics", {}).get("macro_f1", 0)
        class_failure = error is None and active_id != "baseline" and not class_protection(active_metrics, baseline_metrics, baseline_metrics, gate)["accepted"]
        trigger = (error is not None or class_failure or active_metrics["macro_f1"] < baseline_metrics["macro_f1"]
                   or active_metrics["macro_f1"] < previous_score - gate["rollback_macro_f1_drop"])
        event = None
        if trigger:
            # 恢复时重新加载并评估各版本，损坏文件跳过；baseline 可作为显式紧急兜底。
            valid = []
            for version in self.entries():
                try:
                    metric = evaluate(load_model(self.checkpoint(version), validation["paths"]), validation)
                    protected = version == "baseline" or class_protection(metric, active_metrics or baseline_metrics, baseline_metrics, gate)["accepted"]
                    if protected and metric["macro_f1"] + 1e-12 >= baseline_metrics["macro_f1"]:
                        valid.append((metric["macro_f1"], version, metric))
                except (ValueError, KeyError, OSError):
                    continue
            if not valid:
                raise RuntimeError("No valid rollback checkpoint is available")
            _, selected, metric = max(valid, key=lambda item: item[0])
            if error is not None or selected != active_id and (class_failure or metric["macro_f1"] >= active_metrics["macro_f1"]):
                event = {"from": active_id, "to": selected, "cause": error or "validation_quality_regression",
                         "class_protection_triggered": class_failure, "baseline_emergency_fallback": selected == "baseline"}
                self.activate(selected, "automatic_rollback:" + event["cause"])
                active_id, active_metrics = selected, metric
        return active_id, active_metrics, baseline_metrics, event

