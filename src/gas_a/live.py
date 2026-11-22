"""开发联调宿主：真实 PyTRiO 对话、CPU 候选实测和云端审计。

本地 artifact/history ID 仅表示本机编号，不代表赛事宿主回执。"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from .io import digest, history, read_json, write_json, append_jsonl, lock, stamp
from .data import load_pair, load_model, evaluate
from .registry import Registry
from .live_access import PyTrioChat, redact
from .cloud_records import Audit
from .model_records import purpose_for, recorded_chat, call_path, call_reference
from .review import export_review


def recover_commit(registry):
    """重放 prepared 状态的本地提交，补齐活动指针、状态指针和历史，按记录 ID 防止重复追加。"""
    path = registry.root / "pending_commit.json"
    if not path.exists():
        return
    pending = read_json(path)
    if pending.get("status") != "prepared":
        return
    # prepared 表示部署提交已具备所需证据；中断后按已有记录补齐剩余步骤。
    selected = pending["record"]["selected_version"]
    registry.checkpoint(selected)
    if registry.active_id() != selected:
        registry.activate(selected, "local_pytrio_measured_selection")
    write_json(registry.root / "state_pointer.json", {"path": pending["state_path"]})
    recorded = history(registry.root / "history.jsonl")
    # 按记录 ID 去重，使重复恢复不会再次追加同一轮 history。
    if not any(r.get("id") == pending["record"]["id"] for r in recorded):
        append_jsonl(registry.root / "history.jsonl", pending["record"])
    write_json(path, {**pending, "status": "committed"})


class LocalRound:
    """模拟赛事 Round 接口，将真实 PyTRiO 调用、本地 CPU 实测和审计接入同一训练流程。"""
    records_model_calls = True

    def __init__(self, config, registry, client, audit, train_data, validation, session):
        self.config, self.registry, self.client, self.audit = config, registry, client, audit
        self.validation = validation
        self._history = history(registry.root / "history.jsonl")
        self.round_index = len(self._history) + 1
        tokens = config.get("live", {}).get("max_output_tokens", 768)
        if type(tokens) is not int or not 128 <= tokens <= 4096:
            raise ValueError("Invalid live output token budget")
        wall_seconds = config.get("live", {}).get("round_wall_seconds", 180)
        if type(wall_seconds) is not int or not 30 <= wall_seconds <= 600:
            raise ValueError("Invalid live round wall-time budget")
        self.dataset_key, self.max_output_tokens, self.wall_seconds = "A", tokens, wall_seconds
        self.deadline = time.monotonic() + wall_seconds - 5
        self.task = {"task_type": "classification"}
        self.data = SimpleNamespace(train_path=config["train"], validation_path=config["validation"])
        active = registry.active_id()
        checkpoint = registry.checkpoint(active)

        def parent_path(name, root=None):
            if name != "checkpoint.npz" or digest(checkpoint) != registry.entries()[active]["sha256"]:
                raise ValueError("Local parent checkpoint binding changed")
            return checkpoint

        self.parent = SimpleNamespace(artifact_id=active, checkpoint_sha256=digest(checkpoint),
                                      descriptor={"checkpoint_paths": ["checkpoint.npz"], "code_paths": []}, path=parent_path)
        self.frame = {"current_validation": {"metrics": evaluate(load_model(checkpoint), validation)},
                      "origin": "local_development_not_official_evaluation"}
        self.artifacts = [{"artifact_id": name} for name in registry.entries()]
        self.call_ids, self._calls = [], []
        self.work_root = audit.root / "work"
        self.session, self._finished, self._state = session, False, None

    def history(self):
        # 新提示词仅引用历史摘要，避免递归嵌入旧模型提示词。
        """仅返回已完成轮次的摘要，避免将历史完整对话递归加入新提示词。"""
        return [{"id": r["id"], "round": r["round"], "selected_version": r["selected_version"],
                 "macro_f1": r["after"]["macro_f1"], "accuracy": r["after"]["accuracy"],
                 "reason": r["reason"], "origin": "local_development_history"} for r in self._history]

    def model(self, messages, max_tokens, temperature):
        """先记录实际请求，再调用服务方；保留原始回执、usage 和失败证据。"""
        thinking = self.config.get("live", {}).get("enable_thinking")
        record_id = uuid.uuid4().hex
        purpose = purpose_for(messages)
        self.audit.record("model_call_started", {"round": self.round_index, "model": self.client.model,
                          "record_id": record_id, "purpose": purpose})
        try:
            from .resilience import bounded_model_time
            with bounded_model_time(self.client, self.deadline):
                receipt = recorded_chat(self.client, self.audit.root / "model_calls", messages,
                                    max_tokens=max_tokens, temperature=temperature, enable_thinking=thinking,
                                    context={"session": self.session, "round": self.round_index,
                                             "parent_version": self.parent.artifact_id, "sequence": len(self._calls) + 1},
                                    purpose=purpose, record_id=record_id)
        finally:
            path = call_path(self.audit.root / "model_calls", record_id)
            if path.is_file():
                reference = call_reference(path, self.registry.root)
                self.audit.record_model_call(reference, self.registry.root)
        if receipt["call_id"] in self.call_ids:
            raise ValueError("Provider reused a completion ID")
        self.call_ids.append(receipt["call_id"])
        self._calls.append(reference)
        if not receipt["text"].strip():
            reason = receipt["response"].get("choices", [{}])[0].get("finish_reason")
            raise RuntimeError(f"PyTRiO returned empty content; finish_reason={reason}. Actual response and usage were retained")
        return SimpleNamespace(call_id=receipt["call_id"], response=receipt["response"], text=receipt["text"],
                               usage=receipt.get("usage", {}), record_ref=reference)

    def submit_candidate(self, directory, **declaration):
        """独立加载导出的 checkpoint 进行本地评估并登记产物，供上层继续判断是否可部署。"""
        directory = Path(directory)
        report = read_json(directory / "experiment.json")
        measured = evaluate(load_model(directory / "checkpoint.npz"), self.validation)
        identity = f"local_r{self.round_index:03d}_{report['name']}_{self.session}"
        self.registry.register(identity, directory / "checkpoint.npz", self.parent.artifact_id, measured, report)
        # 本地宿主独立重载实际导出的模型并重新评估。
        result = {"artifact_id": identity, "summary": {"metrics": measured},
                  "origin": "local_cpu_measurement_not_contest_host"}
        self.artifacts.append({"artifact_id": identity})
        self.audit.record("candidate", {"round": self.round_index, "report": report,
                          "declaration": declaration, "measurement": result})
        for row in report["epoch_history"]:
            self.audit.record("candidate_epoch", {"round": self.round_index, "name": report["name"],
                              "recorded_after_training": True, **row})
        return result

    def state_workspace(self, directory):
        """复制上轮已提交状态；首轮使用静态 baseline 初始化持续学习的恢复锚点。"""
        destination = Path(directory)
        pointer = self.registry.root / "state_pointer.json"
        if pointer.exists():
            source = (self.registry.root / read_json(pointer)["path"]).resolve()
            if not source.is_relative_to(self.registry.root):
                raise ValueError("State path outside runtime")
            shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True, exist_ok=False)
            baseline = self.registry.checkpoint("baseline")
            shutil.copyfile(baseline, destination / "baseline_checkpoint.npz")
            write_json(destination / "lifecycle.json", {"baseline_artifact_id": "baseline",
                       "baseline_sha256": digest(baseline), "rounds": []})
        return destination

    def save_state(self, directory):
        self._state = Path(directory)

    def finish(self, **record):
        """审计最终决策后准备可恢复提交；云端记录成功后才推进本地部署与历史。"""
        if self._state is None or not self.call_ids:
            raise ValueError("Missing actual calls or saved state")
        memory = read_json(self._state / "lifecycle.json")
        detail = memory["rounds"][-1]
        # train() 仅提交校验通过的最终模型决策；纠正尝试
        # 仍保留各自的原始回复和实际调用 ID。
        if record["model_call_ids"] != self.call_ids:
            raise ValueError("Decision receipt references do not match real provider calls")
        selected = record["selected_artifact_id"]
        self.registry.checkpoint(selected)
        state = self.registry.root / "state" / f"round-{self.round_index:03d}-{self.session}"
        shutil.copytree(self._state, state)
        row = {**detail, "id": f"local-history-{self.round_index:03d}-{self.session}",
               "time": stamp(), "selected_version": selected, "mode": "local_pytrio_development",
               "provider_receipts": self._calls, "finish_declaration": record,
               "official_competition_result": False}
        self.audit.record("round_summary", detail["review"])
        self.audit.record("round_decision", row)  # 云端记录失败时，不提交部署切换。
        write_json(self.registry.root / "pending_commit.json", {"status": "prepared", "record": row,
                   "state_path": state.relative_to(self.registry.root).as_posix()})
        recover_commit(self.registry)
        self._finished = True


def run_live(config, rounds=1, cloud=True, client=None, audit_factory=Audit):
    """用独立会话运行真实模型闭环，并在成功或失败后整理本地报告及云端状态。"""
    if type(rounds) is not int or not 1 <= rounds <= 10:
        raise ValueError("Live rounds must be in 1..10; start with one round")
    timeout = config.get("live", {}).get("model_timeout_seconds", 45)
    if type(timeout) is not int or not 1 <= timeout <= 90:
        raise ValueError("Invalid model timeout budget")
    client = client or PyTrioChat(model=config.get("live", {}).get("controller_model"), timeout=timeout)
    root = Path(config["runtime"])
    if root.name == "a":
        raise ValueError("Use configs/a.live.json to keep live evidence separate from the earlier rule-controller run")
    session = uuid.uuid4().hex[:10]
    audit = audit_factory(root / "sessions" / session, cloud=cloud)
    succeeded = False
    try:
        with lock(root):
            audit.start({"mode": "local_pytrio_development", "model": client.model, "rounds": rounds,
                         "configuration": config, "official_competition_result": False})
            train_data, validation = load_pair(config["train"], config["validation"])
            registry = Registry(root)
            registry.initialize(config["baseline"], evaluate(load_model(config["baseline"]), validation))
            recover_commit(registry)
            from .contest import train
            for _ in range(rounds):
                active, metrics, baseline, rollback = registry.ensure_healthy(validation, config["gate"])
                audit.record("round_start", {"active": active, "metrics": metrics, "baseline": baseline,
                                             "automatic_rollback": rollback})
                rnd = LocalRound(config, registry, client, audit, train_data, validation, session)
                train(rnd)
                if not rnd._finished:
                    raise RuntimeError("Live round did not finish")
            selected = registry.active_id()
            summary = {"active_version": selected,
                       "metrics": evaluate(load_model(registry.checkpoint(selected)), validation),
                       "completed_rounds": len(history(root / "history.jsonl")),
                       "model": client.model, "mode": "local_pytrio_development",
                       "official_competition_result": False, "audit_directory": str(audit.root)}
            audit.record("session_complete", summary)
            succeeded = True
    except Exception as exc:
        try:
            audit.record("session_failed", {"error": redact(str(exc))}, upload=audit.run is not None)
        except RuntimeError:
            audit.record("failure_log_upload_failed", {"original_error": redact(str(exc))}, upload=False)
        raise
    finally:
        # 即使服务方或云端操作失败，也尝试生成简明的本地报告。
        if root.exists():
            try:
                export_review(root, root / "review")
            except (OSError, ValueError, KeyError) as exc:
                audit.record("review_export_failed", {"error": redact(str(exc))}, upload=False)
        receipt = audit.finish(succeeded)
    summary["cloud"] = receipt
    write_json(root / "live_summary.json", summary)
    return summary
