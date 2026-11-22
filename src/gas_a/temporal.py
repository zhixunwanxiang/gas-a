"""使用真实且记录用量的在线控制器与审计服务，按时间窗口回放历史数据。

本地研究流程不扩充赛事训练集。评估器读取当前窗口标签之前，
必须先提交并冻结预测。候选的 epoch 选择仅使用原始公开 validation。
随后揭示的窗口标签可用于诊断和影子选择，
更新后的部署仅影响后续窗口。"""
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from .cloud_records import Audit
from .data import evaluate, load_model, load_pair, score_predictions
from .io import CLASSES, append_jsonl, digest, history, read_json, stamp, write_json
from .live_access import PyTrioChat
from .model_records import collect_model_calls, recorded_chat, call_path, call_reference
from .monitor import diagnose, monitor
from .candidates import execute_candidate, assess_candidate, candidate_resources
from .pipeline import ordered_specs
from .registry import Registry
from .review import metrics_brief, usage_totals
from .controller_policy import (measured_evidence, build_selection_prompt, class_metrics,
                                class_protection, stopping_evidence)
from . import adaptive
from .architecture import available_choices, shape as architecture_shape
from .planning import composed, build_prompt, resolve_plan, correction_prompt
from .resilience import (fatal_provider_error, recovery, safe_observation, measured_selection,
                         monitor_safely, diagnose_safely)
from .resilience import bounded_model_time


def dat_inputs(path, paths):
    """仅读取特征字段，有意跳过标签字段的解析。"""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Expected a regular public DAT file")
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        tokens = line.split()[1:]
        values = np.empty(128, dtype=np.float64)
        indices = set()
        for token in tokens:
            index, value = token.split(":", 1)
            index = int(index) - 1
            if index in indices or not 0 <= index < 128:
                raise ValueError("DAT feature index is duplicated or out of range")
            indices.add(index)
            values[index] = float(value)
        if len(indices) != 128 or not np.isfinite(values).all():
            raise ValueError("DAT requires 128 finite numeric features")
        rows.append(values)
    if not rows:
        raise ValueError("Empty DAT window")
    return {"x": np.asarray(rows), "paths": paths, "sha256": digest(path),
            "ids": [f"{path.stem}-{i + 1:05d}" for i in range(len(rows))]}


def reveal_labels(path, features):
    """确认 DAT 文件与预测提交时的摘要一致后再解析标签，防止评分数据被替换。"""
    if digest(path) != features["sha256"]:
        raise ValueError("Window changed after prediction commitment")
    labels = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            label = int(line.split()[0]) - 1
            if not 0 <= label < 6:
                raise ValueError("DAT label outside the six reference classes")
            labels.append(label)
    if len(labels) != len(features["x"]):
        raise ValueError("Window label count changed")
    return {**features, "y": np.asarray(labels, dtype=np.int64)}


def integrity_guard(registry, validation):
    """预测前检查模型产物，不使用未来窗口的标签。"""
    active = None
    try:
        registry.recover_activation()
        active = registry.active_id()
        registry.checkpoint(active)
        return None
    except (OSError, ValueError, KeyError) as exc:
        # 使用原始不可变 baseline 作为确定的恢复锚点。
        registry.checkpoint("baseline")
        metric = evaluate(load_model(registry.checkpoint("baseline")), validation)
        registry.activate("baseline", "automatic_rollback:checkpoint_integrity")
        return {"from": active, "to": "baseline", "kind": "checkpoint_integrity",
                "error": str(exc), "fixed_validation": metric, "time": stamp()}


def window_gate(candidate, parent, baseline, config):
    """评估当前窗口上的候选质量，并明确记录无法估计 recall 的缺失类别。"""
    return assess_candidate(candidate, parent, baseline, config, allow_absent_targets=True)


def performance_rollback(registry, current, active_metric, baseline_metric, validation, config):
    """标签揭示后，在同一当前窗口上比较版本，不直接比较历史窗口分数。"""
    active = registry.active_id()
    pointer = read_json(registry.root / "active.json")
    previous = pointer.get("previous_version")
    candidates = {"baseline": baseline_metric}
    fixed_baseline = evaluate(load_model(registry.checkpoint("baseline")), validation)
    fixed_active = evaluate(load_model(registry.checkpoint(active)), validation)
    rejected = {}
    if previous and previous != active:
        try:
            metric = evaluate(load_model(registry.checkpoint(previous)), current)
            fixed = evaluate(load_model(registry.checkpoint(previous)), validation)
            protection = class_protection(metric, active_metric, baseline_metric, config)
            fixed_protection = class_protection(fixed, fixed_active, fixed_baseline, config)
            if (metric["macro_f1"] + 1e-12 >= baseline_metric["macro_f1"]
                    and fixed["macro_f1"] + 1e-12 >= fixed_baseline["macro_f1"]
                    and protection["accepted"] and fixed_protection["accepted"]):
                candidates[previous] = metric
            else:
                rejected[previous] = {"current_classes": protection, "fixed_classes": fixed_protection,
                                      "current_macro_f1": metric["macro_f1"], "fixed_macro_f1": fixed["macro_f1"]}
        except (OSError, ValueError, KeyError):
            pass
    best_id = max(candidates, key=lambda key: candidates[key]["macro_f1"])
    best = candidates[best_id]
    class_failure = active != "baseline" and not class_protection(active_metric, baseline_metric, baseline_metric, config)["accepted"]
    triggered = (class_failure or active_metric["macro_f1"] + 1e-12 < baseline_metric["macro_f1"]
                 or best["macro_f1"] - active_metric["macro_f1"] > config["rollback_macro_f1_drop"])
    if not triggered or best_id == active or not class_failure and best["macro_f1"] <= active_metric["macro_f1"]:
        return None
    return {"from": active, "to": best_id, "kind": "same_window_quality_regression",
            "before": active_metric, "after": best, "time": stamp(),
            "class_protection_triggered": class_failure, "rejected_historical_restores": rejected,
            "baseline_is_explicit_emergency_fallback": best_id == "baseline",
            "effective": "future_predictions_only; current frozen score unchanged"}


class Controller:
    """封装时间窗口中的模型请求、调用身份检查和完整审计。"""
    def __init__(self, client, audit, config):
        self.client, self.audit, self.config = client, audit, config
        self.call_ids = set()
        self.deadline = None

    def ask(self, payload, batch, purpose, max_tokens):
        """将聚合测量证据发送给控制器；请求、回复和异常均关联到同一 record ID。"""
        if getattr(self.client, "local_rules", False):
            result = self.client.decide(payload)
            if "allowed_experiments" in payload:
                result = adaptive.normalize_plan(result)
            self.audit.record("rule_decision", {"batch": batch, "purpose": purpose, "result": result,
                                              "origin": "cpu_rule_screen_not_model_call"})
            return result
        record_id = uuid.uuid4().hex
        self.audit.record("controller_request", {"batch": batch, "purpose": purpose, "record_id": record_id})
        directory = self.audit.root / "model_calls"
        try:
            with bounded_model_time(self.client, self.deadline):
                receipt = recorded_chat(self.client, directory,
                [{"role": "system", "content": "只返回一个合法JSON对象，不输出代码块或思考过程。"},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                max_tokens=max_tokens, temperature=0,
                enable_thinking=self.config["live"].get("enable_thinking", False),
                context={"round": batch - 2, "batch": batch, "session": self.audit.root.name},
                purpose=purpose, record_id=record_id)
        finally:
            # 这里保存实际持久化的请求和回复，包括失败记录；
            # 不另行合成回执，提示词仅使用聚合数据。
            path = call_path(directory, record_id)
            if path.is_file():
                self.audit.record_model_call(call_reference(path, self.config["runtime"]), self.config["runtime"])
        if receipt["call_id"] in self.call_ids:
            raise ValueError("Provider reused a completion ID")
        self.call_ids.add(receipt["call_id"])
        from .contest import parse_object
        return parse_object(receipt["text"])


def optimize_window(config, registry, train, validation, current, row, past, audit, controller, directory, recent=()):
    """在当前批预测冻结后诊断并训练候选；梯度与 epoch 选择仍仅使用原始 train/validation。"""
    started = time.monotonic()
    if controller is not None:
        controller.deadline = started + config["live"].get("round_wall_seconds", 300) - 5
    parent = registry.checkpoint(registry.active_id())
    parent_id, parent_sha = registry.active_id(), digest(parent)
    model = load_model(parent, train["paths"])
    fixed_before = evaluate(model, validation)
    current_before = evaluate(model, current)
    diagnosis, diagnostic_recovery = diagnose_safely(diagnose, train, current, current_before, row["drift"],
                                                    train_metrics=evaluate(model, train))
    if diagnostic_recovery:
        row.setdefault("recoveries", []).append(diagnostic_recovery)
        audit.record("stage_recovery", {"batch": row["batch"], **diagnostic_recovery})
    row.update(parent_artifact_id=parent_id, parent_checkpoint_sha256=parent_sha,
               train_sha256=train["sha256"], validation_sha256=validation["sha256"],
               diagnosis=diagnosis, fixed_validation_before=fixed_before, shadow_before=current_before,
               experiments=[], action="retain", selected_version=parent_id)
    specs = ordered_specs(config, past, parent_sha, diagnosis, model=model,
                          train_sha=train['sha256'], validation_sha=validation['sha256'])
    architecture_choices = available_choices(model, config.get("architecture_choices", {}))
    use_adaptive = config.get("adaptive", {}).get("enabled", False)
    feedback = adaptive.feedback_stats(past, row.get("forward_feedback", [])) if use_adaptive else {}
    budget = min(config["optimization"].get("max_selected_experiments", 2),
                 config["optimization"]["max_candidates"] if composed(config) else len(specs))
    evidence = measured_evidence(current_before, fixed_before, evaluate(model, train), diagnosis, row["drift"],
                                 past, parent_sha, [s["name"] for s in config["experiments"]], validation_sha=validation["sha256"],
                                 training_sha=train["sha256"])
    stop = stopping_evidence(past, parent_sha, train["sha256"], validation["sha256"], config)
    evidence["facts"]["S"] = stop
    if use_adaptive:
        evidence["facts"]["U"] = feedback
    write_json(directory / "controller-evidence.json", evidence)
    row["controller_evidence_sha256"] = digest(directory / "controller-evidence.json")
    audit.record("controller_measured_evidence", {"batch": row["batch"], "evidence": evidence,
                                                 "sha256": row["controller_evidence_sha256"]})
    if stop["pause"] or not budget:
        # 无新增实验时仍保留本窗口监测与评分；暂停不会修改已经冻结的预测结果。
        row.update(status="completed", decision_source="budget_pause", reason="同一训练数据和父版本连续无合格实验，观察一窗后重新判断。",
                   plan={"action": "observe", "experiment_names": [], "reason": "预算策略暂停新增训练，继续监测、评估和回滚。",
                         "source": "system_policy", "evidence_ids": ["C", "F", "S"]})
        audit.record("temporal_budget_pause", {"batch": row["batch"], "evidence": stop, "reason": row["reason"]})
        return
    prompt = build_prompt(config, model, specs, evidence, budget,
        config["optimization"]["max_seconds_per_candidate"], mode="temporal", index=row["batch"], feedback=feedback,
        parameter_bounds=(adaptive.parameter_bounds if use_adaptive else adaptive.focus_bounds)(
            specs, architecture_choices, architecture_shape(model)), past=past, parent_sha=parent_sha)
    # 规划与选择共用一次纠正预算，每个优化窗口最多进行三次真实调用。
    correction_left = 1
    plan = None
    while True:
        plan = None
        try:
            plan = controller.ask(prompt, row["batch"], "experiment_planning", config["live"]["max_output_tokens"])
            plan, chosen_specs = resolve_plan(plan, config, model, specs, evidence, budget, past, parent_sha,
                focus_only=not use_adaptive, architecture_choices=architecture_choices, weak_class=diagnosis["weakest_class"])
            names = plan["experiment_names"]
            break
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            if fatal_provider_error(exc):
                raise
            if composed(config):
                row.setdefault("planning_rejections", []).append({"error": str(exc), "proposal": plan,
                    "next_action": "ask_ai_to_revise" if correction_left else "correction_budget_exhausted"})
            try:
                repaired, chosen_specs = resolve_plan(plan, config, model, specs, evidence, budget, past, parent_sha,
                    repair=True, focus_only=not use_adaptive, architecture_choices=architecture_choices,
                    weak_class=diagnosis["weakest_class"])
                plan, names = repaired, repaired["experiment_names"]
                note = recovery("planning", exc, "verified_contract_rebinding")
                row.setdefault("recoveries", []).append(note)
                audit.record("controller_recovery", {"batch": row["batch"], **note, "bound_plan": plan})
                break
            except (ValueError, KeyError, TypeError):
                pass
            if not correction_left:
                plan = safe_observation(evidence, exc)
                names, chosen_specs = [], specs
                note = recovery("planning", exc, "safe_observe_without_training")
                row.setdefault("recoveries", []).append(note)
                audit.record("controller_recovery", {"batch": row["batch"], **note, "bound_plan": plan})
                break
            correction_left -= 1
            note = recovery("planning", exc, "bounded_correction_retry")
            row.setdefault("recoveries", []).append(note)
            audit.record("controller_recovery", {"batch": row["batch"], **note})
            prompt = correction_prompt(prompt, plan, exc)
    row["plan"] = plan
    specs = chosen_specs
    audit.record("diagnosis_and_plan", {"batch": row["batch"], "diagnosis": diagnosis, "plan": plan})
    if plan["action"] == "observe":
        row.update(status="completed", reason=plan["reason"],
                   decision_source=plan.get("decision_origin", "model_observe"))
        audit.record("temporal_model_observe", {"batch": row["batch"], "plan": plan, "selected": parent_id})
        return
    eligible = {parent_id: {"kind": "retain", "role": "当前已部署版本，允许保留；不是本轮新候选", "current_window_shadow": class_metrics(current_before),
                            "fixed_validation": class_metrics(fixed_before)}}
    if use_adaptive:
        eligible[parent_id]["selection_score"] = adaptive.ranking(model, current, recent, config["adaptive"]["recent_weights"])["score"]
    end = started + config["live"].get("round_wall_seconds", 300) - 5
    for name in names:
        spec = next(s for s in specs if s["name"] == name)
        trial = {"name": name, "configuration": spec}
        destination = directory / "candidates" / name
        try:
            remaining = end - time.monotonic()
            if remaining < 2:
                raise TimeoutError("Window optimization budget exhausted")
            limits = {**config["optimization"], "checkpoint_gate": config["gate"],
                      "max_seconds_per_candidate": min(remaining, config["optimization"]["max_seconds_per_candidate"])}
            # 固定 validation 用于选择 epoch；current 只在训练导出后进入 shadow 评估。
            gate_policy = config["gate"]
            if use_adaptive:
                gate_policy = adaptive.policy(config)
                limits.update(checkpoint_gate=gate_policy, checkpoint_reference=row["fixed_static_baseline"],
                              checkpoint_reference_sha256=digest(registry.checkpoint("baseline")))
            trial = execute_candidate(parent, train, validation, destination, spec, diagnosis, limits,
                active_metrics=fixed_before, baseline_metrics=row["fixed_static_baseline"], policy=gate_policy,
                shadow=(current, current_before, row["static_baseline"]))
            if trial.get("error"):
                row.setdefault("recoveries", []).append(recovery("training", trial["error"], "reject_candidate_keep_valid_parent"))
                row["experiments"].append(trial)
                write_json(directory / "progress.json", row)
                audit.record("temporal_candidate", {"batch": row["batch"], **trial})
                continue
            report, shadow, measured_fixed = trial["report"], trial["shadow_metrics"], trial["metrics"]
            fixed_gate, current_gate = trial.pop("gate"), trial.pop("shadow_gate")
            identity = f"batch{row['batch']:02d}_{name}"
            registry.register(identity, destination / "checkpoint.npz", parent_id, measured_fixed, report)
            trial.update(report=report, evidence_directory=str(destination), shadow_metrics=shadow,
                         fixed_gate=fixed_gate, window_gate=current_gate, artifact_id=identity,
                         eligible=fixed_gate["accepted"] and current_gate["accepted"])
            if use_adaptive:
                rank_start, rank_cpu = time.monotonic(), time.process_time()
                ranked = adaptive.ranking(load_model(destination / "checkpoint.npz"), current, recent, config["adaptive"]["recent_weights"])
                trial["resources"]["cpu_seconds"] += time.process_time() - rank_cpu
                trial["resources"]["wall_seconds"] += time.monotonic() - rank_start
                trial["selection_ranking"] = ranked
                if ranked["score"] < eligible[parent_id]["selection_score"] + config["gate"]["min_macro_f1_gain"]:
                    trial["eligible"] = False
                    trial["ranking_rejection"] = "recent_window_utility_gain_below_minimum"
            if trial["eligible"]:
                eligible[identity] = {"kind": "promote", "role": "本轮新训练且已通过双重门槛的候选", "current_window_shadow": class_metrics(shadow),
                                      "fixed_validation": class_metrics(measured_fixed)}
                if use_adaptive:
                    eligible[identity]["selection_score"] = trial["selection_ranking"]["score"]
            for epoch in report["epoch_history"]:
                audit.record("candidate_epoch", {"batch": row["batch"], "name": name,
                                                "recorded_after_training": True, **epoch})
        except (OSError, ValueError, TimeoutError) as exc:
            trial["error"] = str(exc)
            row.setdefault("recoveries", []).append(recovery("candidate_evaluation", exc, "reject_candidate_keep_valid_parent"))
            if (destination / "failure.json").exists():
                trial["resources"] = read_json(destination / "failure.json")["resources"]
        row["experiments"].append(trial)
        write_json(directory / "progress.json", row)
        audit.record("temporal_candidate", {"batch": row["batch"], **trial})
    selection = build_selection_prompt(eligible, row["experiments"], parent_id,
        mode="temporal", index=row["batch"])
    choice = None
    while True:
        try:
            choice = controller.ask(selection, row["batch"], "artifact_selection",
                                    config["live"].get("selection_max_output_tokens", 512))
            selected, reason = choice.get("selected_artifact_id"), choice.get("reason")
            if not isinstance(selected, str) or selected not in eligible or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Invalid quality-gated artifact selection")
            if use_adaptive and eligible[selected]["selection_score"] < max(v["selection_score"] for v in eligible.values()) - 1e-12:
                raise ValueError("Selection must use the highest measured recent-window score")
            break
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            if fatal_provider_error(exc):
                raise
            if not correction_left:
                choice = measured_selection(selection)
                selected, reason = choice["selected_artifact_id"], choice["reason"]
                note = recovery("selection", exc, "highest_measured_eligible_version")
                row.setdefault("recoveries", []).append(note)
                audit.record("controller_recovery", {"batch": row["batch"], **note, "bound_choice": choice})
                break
            correction_left -= 1
            note = recovery("selection", exc, "bounded_correction_retry")
            row.setdefault("recoveries", []).append(note)
            audit.record("controller_recovery", {"batch": row["batch"], **note})
            selection.update(correction=str(exc), previous_response=choice)
    row.update(selected_version=selected, action=eligible[selected]["kind"], reason=reason,
               selection=choice, status="completed", eligible_artifacts=eligible)
    audit.record("temporal_deployment_prepared", {"batch": row["batch"], "choice": choice,
                                               "effective_from_batch": row["batch"] + 1})
    if selected != registry.active_id():
        registry.activate(selected, f"temporal_pytrio_selection_after_batch_{row['batch']}")


def rollback_drill(root, registry, validation):
    """对一次性副本注入损坏故障，不改动实际部署产物。"""
    original_pointer = digest(registry.root / "active.json")
    drill = Registry(Path(root) / "rollback_drill")
    drill.initialize(registry.checkpoint("baseline"), evaluate(load_model(registry.checkpoint("baseline")), validation))
    drill.register("fault_injection", registry.checkpoint(registry.active_id()), "baseline", {}, {})
    drill.activate("fault_injection", "explicit_fault_injection_drill")
    drill.checkpoint("fault_injection").write_bytes(b"deliberately corrupted disposable checkpoint")
    event = integrity_guard(drill, validation)
    result = {"scope": "isolated_fault_injection_not_natural_drift", "event": event,
              "passed": event is not None and drill.active_id() == "baseline",
              "actual_deployment_unchanged": original_pointer == digest(registry.root / "active.json")}
    write_json(drill.root / "result.json", result)
    return result


def export_temporal(root, rows, aggregate, audit_receipt=None, error=None):
    """汇总冻结的实际预测、候选资源和调用成本，分别呈现运行完成状态与性能增益。"""
    root = Path(root)
    calls = collect_model_calls(root / "session")
    configuration = read_json(root / "configuration.json")
    prior_attempts = []
    for previous in configuration.get("prior_attempts", []):
        previous = Path(previous).resolve()
        if previous == root.resolve():
            raise ValueError("Retry cannot count itself as a prior attempt")
        previous_summary = read_json(previous / "summary.json")
        calls.extend(collect_model_calls(previous / "session"))
        prior_attempts.append({"directory": str(previous), "costs": previous_summary["costs"],
                               "controller_failed_windows": previous_summary["controller_failed_windows"]})
    costs = usage_totals(calls)
    trials = [trial for row in rows for trial in row.get("experiments", [])]
    resources = [candidate_resources(t) for t in trials]
    costs.update(candidate_cpu_seconds_known=sum(r.get("cpu_seconds") or 0 for r in resources)
                     + sum(p["costs"]["candidate_cpu_seconds_known"] for p in prior_attempts),
                 candidate_wall_seconds_known=sum(r.get("wall_seconds") or 0 for r in resources)
                     + sum(p["costs"]["candidate_wall_seconds_known"] for p in prior_attempts),
                 candidates_with_unknown_resources=sum(not r for r in resources)
                     + sum(p["costs"]["candidates_with_unknown_resources"] for p in prior_attempts),
                 actual_label_service_requests=0, revealed_public_labels=sum(r["operational"]["rows"] for r in rows),
                 local_gpu_training_seconds=0, platform_training_credits=None, provider_gpu_seconds=None)
    gain = aggregate.get("gain_vs_static_baseline")
    costs["tokens_per_operational_f1_gain"] = costs["tokens_total"] / gain if gain and gain > 0 and costs["tokens_total"] is not None else None
    mode = configuration.get("controller_mode", "pytrio")
    summary = {"format": "gas-a.temporal-replay.v1", "mode": "historical_cpu_rule_screen" if mode == "rule" else "historical_replay_with_online_services",
               "official_competition_result": False, "finished_windows": len(rows),
               "optimization_windows": sum(r.get("status") != "terminal_evaluation" for r in rows),
               "controller_failed_windows": sum(r.get("status") == "controller_failed" for r in rows),
               "controller_recovered_windows": sum(any(e["stage"] in ("planning", "selection") for e in r.get("recoveries", [])) for r in rows),
               "recovered_windows": sum(bool(r.get("recoveries")) for r in rows),
               "failed_candidate_trials": sum(bool(t.get("error")) for t in trials),
               "promotions": sum(r.get("action") == "promote" for r in rows),
               "natural_rollbacks": sum(bool(r.get("performance_rollback") or r.get("integrity_rollback")) for r in rows),
               "aggregate_frozen_operational": aggregate, "costs": costs,
               "prior_attempts": prior_attempts, "cost_scope": "current_run_plus_all_listed_prior_attempts",
               "cloud": audit_receipt, "error": error, "report": str(root / "report.txt")}
    summary["forward_feedback_observations"] = sum(len(r.get("forward_feedback", [])) for r in rows)
    costs["probe_cpu_seconds_known"] = sum(r.get("probe_resources", {}).get("cpu_seconds", 0) for r in rows) + sum(
        p["costs"].get("probe_cpu_seconds_known", 0) for p in prior_attempts)
    costs["history_selection_cpu_seconds_known"] = sum(r.get("history_selection_resources", {}).get("cpu_seconds", 0) for r in rows) + sum(
        p["costs"].get("history_selection_cpu_seconds_known", 0) for p in prior_attempts)
    summary["success"] = (bool((audit_receipt or {}).get("session_succeeded")) and not error
                          and len(rows) == len(configuration["batches"])
                          and summary["controller_failed_windows"] == 0)
    summary["all_model_decisions_valid_without_recovery"] = not summary["controller_recovered_windows"] and not summary["controller_failed_windows"]
    summary["workflow_completed"] = summary["success"]
    summary["full_monitoring_and_diagnosis_evidence"] = all(
        r.get("drift", {}).get("status") != "unavailable"
        and r.get("diagnosis", {}).get("status") != "limited_measured_summary" for r in rows)
    write_json(root / "summary.json", summary)
    title = "后续时间批次闭环测试（CPU规则筛选，无真实大模型调用）" if mode == "rule" else "后续时间批次闭环测试（真实在线服务，历史数据按时间回放）"
    lines = [title,
             "训练：仅原 train.jsonl；选 epoch：仅原 validation.jsonl；固定128维输入、6类输出，Adam及标准化沿用父检查点，架构变更须显式迁移。",
             "当前批先预测落盘，再开放标签做诊断、影子评估与回滚；部署只影响未来预测。末批仅评估，节省无后续检验的训练。",
             "这是扩展时间漂移研究，不是赛事隐藏集成绩或正式 Harness 回执。", "",
             "批次 | 更新前版本 | 实际 Macro-F1 | 静态 Macro-F1 | 乙醛召回 | 后续部署决策"]
    lines.insert(4, f"控制器未恢复失败窗口：{summary['controller_failed_windows']}；程序已恢复窗口：{summary['recovered_windows']}；失败候选：{summary['failed_candidate_trials']}。原始错误和恢复来源保留。")
    if prior_attempts:
        lines.insert(4, f"成本包含此前 {len(prior_attempts)} 次运行，原始失败记录保留；下列性能仅对应本次完整运行。")
    for row in rows:
        metric = row["operational"]
        action = {"promote": "采用新候选", "retain": "保留", "rollback": "回滚"}.get(row.get("action"), row.get("action", "保留"))
        if row.get("performance_rollback"):
            action = "先切换历史版本，再" + action
        lines.append(f"batch{row['batch']} | {row['prediction_version']} | {metric['macro_f1']:.6f} | "
                     f"{row['static_baseline']['macro_f1']:.6f} | "
                     f"{metric['per_class']['Acetaldehyde']['recall']:.4f} | {action} → {row.get('selected_version')}")
    actual, static = aggregate.get("operational", {}), aggregate.get("static_baseline", {})
    if actual:
        lines.extend(["", f"累计 {actual['rows']} 个样本：实际 Macro-F1 {actual['macro_f1']:.6f}；"
                      f"静态基线 {static['macro_f1']:.6f}；增益 {gain:+.6f}；实际准确率 {actual['accuracy']:.4f}。",
                      f"累计乙醛召回 {actual['per_class']['Acetaldehyde']['recall']:.4f}；"
                      f"乙醇召回 {actual['per_class']['Ethanol']['recall']:.4f}。"])
        lines.extend(["", "累计各类召回（静态基线 → 实际更新前预测）："])
        for name in CLASSES:
            lines.append(f"{name}: {static['per_class'][name]['recall']:.4f} → {actual['per_class'][name]['recall']:.4f}")
        unmet = [name for name, target in configuration["gate"].get("minimum_class_recall", {}).items()
                 if actual["per_class"][name]["recall"] < target]
        lines.append("总体增益不代表各类均改善；累计类别目标未达标：" + (", ".join(unmet) or "无") + "。各批次目标仍需单独核对。")
    lines.extend([f"全部成本：调用 {costs['calls']} 次；已知 Token {costs['tokens_known']}；用量缺失 {costs['unknown_usage_calls']} 次；"
                  f"候选 CPU {costs['candidate_cpu_seconds_known']:.3f} 秒，耗时 {costs['candidate_wall_seconds_known']:.3f} 秒。",
                  f"开放公开标签 {costs['revealed_public_labels']} 个；新增标注服务请求 0 次；平台训练额度和服务端 GPU 时长未提供。",
                  f"自然数据触发历史版本切换 {summary['natural_rollbacks']} 次；故障注入演练见 rollback_drill/result.json，单独计数。",
                  f"SwanLab：{(audit_receipt or {}).get('experiment_url') or '连接状态见 session/cloud.json'}", ""])
    for row in rows:
        lines.extend([f"batch{row['batch']}｜{row.get('status')}｜{row.get('reason', '')}",
                      f"漂移特征 {row['drift']['flagged_features']}；性能回滚：{bool(row.get('performance_rollback'))}"])
        if row.get("plan"):
            lines.append("规划：" + (", ".join(row["plan"]["experiment_names"]) or "观察，不训练") + "；" + row["plan"]["reason"])
            if row["plan"].get("evidence_ids"):
                lines.append("依据：" + ", ".join(row["plan"]["evidence_ids"]) + "；假设：" + row["plan"].get("hypothesis", "预算暂停策略") +
                             "；停止条件：未通过质量门槛则不晋升，观察后重新评估。")
        for trial in row.get("experiments", []):
            shadow = trial.get("shadow_metrics", {})
            def gate_text(value):
                if value is None:
                    return "未完成"
                if value["accepted"]:
                    return "通过" + ("；预警=" + ",".join(value["warnings"]) if value.get("warnings") else "")
                names = {"macro_f1_gain_below_minimum": "增益不足", "below_static_baseline": "低于静态基线",
                         "accuracy_regression": "准确率下降超限", "class_recall_below_minimum": "类别召回未达标",
                         "class_recall_regression": "类别召回下降超限"}
                return "、".join(names.get(reason.split(":")[0], reason) +
                                 ("（" + reason.split(":", 1)[1] + "）" if ":" in reason else "")
                                 for reason in value["reasons"])
            lines.append(f"实验 {trial['name']}：影子 F1={shadow.get('macro_f1')}，"
                         f"{'合格' if trial.get('eligible', False) else '未采用'}；"
                         f"固定验证：{gate_text(trial.get('fixed_gate'))}；当前批：{gate_text(trial.get('window_gate'))}；"
                         f"异常：{trial.get('error') or '无'}")
        if row.get("error"):
            lines.append("异常：" + row["error"])
        lines.append("")
    for filename, label in (("independent-verification.json", "训练状态与冻结预测独立复核"),
                            ("cloud-readback.json", "SwanLab云端指标读回"),
                            ("service-verification.json", "选定版本HTTP部署验证")):
        if (root / filename).is_file():
            lines.append(f"{label}：{'通过' if read_json(root / filename)['passed'] else '未通过'}（{filename}）。")
    lines.append("完整证据：windows/ 各批预测与实验，session/events.jsonl 全流程，session/model_calls/ 完整模型请求回复，source/ 本次代码快照。")
    if error:
        lines.append("会话异常：" + error)
    (root / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_temporal(config, dataset_directory, batches=tuple(range(3, 11)), *, cloud=True, client=None, audit_factory=Audit):
    """按时间顺序回放批次：先冻结预测，再揭示标签、检查恢复并为后续窗口选择版本。"""
    batches = tuple(batches)
    if not batches or any(type(b) is not int or not 3 <= b <= 10 for b in batches) or tuple(sorted(set(batches))) != batches:
        raise ValueError("Replay windows must be unique and chronological in batch3..10")
    root = Path(config["runtime"])
    if root.exists():
        raise ValueError("Temporal run requires a fresh runtime directory")
    files = [Path(dataset_directory) / f"batch{batch}.dat" for batch in batches]
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("Missing regular public DAT window")
    train, validation = load_pair(config["train"], config["validation"])
    base_model = load_model(config["baseline"], train["paths"])
    preflight = []
    for batch, path in zip(batches, files):
        features = dat_inputs(path, train["paths"])
        preflight.append({"batch": batch, "rows": len(features["x"]), "sha256": features["sha256"]})
    client = client or PyTrioChat(model=config["live"]["controller_model"], timeout=config["live"]["model_timeout_seconds"])
    mode = "rule" if getattr(client, "local_rules", False) else "pytrio"
    if mode == "rule" and cloud:
        raise ValueError("CPU rule screening must use local records and cannot be labeled a real cloud/model run")
    config = {**config, "controller_mode": mode}
    use_adaptive = config.get("adaptive", {}).get("enabled", False)
    root.mkdir(parents=True, exist_ok=False)
    for path in Path(__file__).parent.rglob("*.py"):
        destination = root / "source" / path.relative_to(Path(__file__).parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(root / "configuration.json", {**config, "batches": batches, "dataset_directory": str(Path(dataset_directory).resolve())})
    write_json(root / "preflight.json", {"checked_at": stamp(), "labels_read": False, "windows": preflight,
                                         "baseline_sha256": digest(config["baseline"])})
    audit = audit_factory(root / "session", cloud=cloud)
    registry = Registry(root / "deployment")
    rows, all_y, all_active, all_baseline, recent = [], [], [], [], []
    aggregate, succeeded, error, final_receipt = {}, False, None, None
    bindings = {"train": train["sha256"], "validation": validation["sha256"], "baseline": digest(config["baseline"])}
    try:
        audit.start({"mode": "historical_cpu_rule_screen" if mode == "rule" else "historical_temporal_replay_online_services", "controller_model": client.model,
                     "batches": batches, "gradients": "original_train_only", "epoch_selection": "original_validation_only",
                     "window_labels": "reveal_only_after_prediction_commit;diagnosis_and_shadow_evaluation_only",
                     "configuration": config, "bindings": bindings, "official_competition_result": False})
        for previous in config.get("prior_attempts", []):
            previous_summary = read_json(Path(previous) / "summary.json")
            audit.record("prior_attempt_costs", {"directory": str(previous), "summary_sha256": digest(Path(previous) / "summary.json"),
                         "costs": previous_summary["costs"], "controller_failed_windows": previous_summary["controller_failed_windows"]})
        controller = Controller(client, audit, config)
        base_model = load_model(config["baseline"], train["paths"])
        fixed_baseline = evaluate(base_model, validation)
        registry.initialize(config["baseline"], fixed_baseline)
        for index, (batch, path) in enumerate(zip(batches, files)):
            window = root / "windows" / f"batch{batch:02d}"
            window.mkdir(parents=True, exist_ok=False)
            integrity = integrity_guard(registry, validation)
            features = dat_inputs(path, train["paths"])
            if features["sha256"] != preflight[index]["sha256"]:
                raise ValueError("Public replay window changed after preflight")
            probe_started, probe_cpu = time.monotonic(), time.process_time()
            probes = adaptive.capture_probes(registry, features, rows) if use_adaptive and config["adaptive"].get("forward_feedback") else []
            probe_resources = {"cpu_seconds": time.process_time() - probe_cpu, "wall_seconds": time.monotonic() - probe_started}
            active = registry.active_id()
            checkpoint = registry.checkpoint(active)
            started = time.perf_counter()
            probabilities = load_model(checkpoint).predict(features["x"])
            prediction_seconds = time.perf_counter() - started
            baseline_probabilities = base_model.predict(features["x"])
            # 先保存活动版本与静态 baseline 的预测及其哈希，再允许评估器读取标签。
            # 此后的诊断、训练和版本切换都不能回写本批 operational 成绩。
            prediction_file = window / "predictions.npz"
            extra = {}
            if probes:
                extra = {"probe_values": np.stack([p["prediction"] for p in probes]),
                         "probe_parent_values": np.stack([p["parent_prediction"] for p in probes])}
            np.savez_compressed(prediction_file, active=probabilities, static_baseline=baseline_probabilities, **extra)
            frozen = {"batch": batch, "version": active, "checkpoint_sha256": digest(checkpoint),
                      "source_sha256": features["sha256"], "rows": len(features["x"]),
                      "predictions_sha256": digest(prediction_file), "committed_at": stamp(), "labels_revealed": False}
            frozen["probe_bindings"] = [{k: v for k, v in probe.items() if k not in ("prediction", "parent_prediction")} for probe in probes]
            write_json(window / "prediction_commit.json", frozen)
            drift, monitor_recovery = monitor_safely(monitor, train["x"], features["x"], config["monitor"])
            audit.record("prediction_committed_before_labels", frozen)
            audit.record("temporal_drift", {"batch": batch, **drift})
            # 到此处评估器才首次解析当前窗口的标签。
            current = reveal_labels(path, features)
            operational = score_predictions(probabilities, current["y"], prediction_seconds)
            baseline = score_predictions(baseline_probabilities, current["y"])
            forward_feedback = adaptive.score_probes(probes, current["y"], batch)
            row = {"round": index + 1, "batch": batch, "prediction_version": active,
                   "prediction_commit": frozen, "labels_revealed_at": stamp(),
                   "operational": operational, "static_baseline": baseline,
                   "fixed_static_baseline": fixed_baseline, "drift": drift, "integrity_rollback": integrity,
                   "status": "evaluated", "experiments": [], "action": "retain"}
            row.update(forward_feedback=forward_feedback, probe_resources=probe_resources)
            if monitor_recovery:
                row.setdefault("recoveries", []).append(monitor_recovery)
                audit.record("stage_recovery", {"batch": batch, **monitor_recovery})
            if forward_feedback:
                audit.record("next_window_feedback", {"batch": batch, "observations": forward_feedback})
            audit.record("labels_revealed_and_frozen_predictions_scored", {"batch": batch, "operational": operational,
                                                                         "static_baseline": baseline})
            selection_start, selection_cpu = time.monotonic(), time.process_time()
            if use_adaptive:
                rollback, selection_review = adaptive.select_history(registry, current, recent, validation, config, rows, forward_feedback)
                row["history_selection"] = selection_review
            else:
                rollback = performance_rollback(registry, current, operational, baseline, validation, config["gate"])
            row["history_selection_resources"] = {"cpu_seconds": time.process_time() - selection_cpu,
                                                   "wall_seconds": time.monotonic() - selection_start}
            if rollback:
                audit.record("temporal_performance_rollback", {"batch": batch, **rollback})
                registry.activate(rollback["to"], f"automatic_rollback:batch_{batch}_same_window_quality")
            row["performance_rollback"] = rollback
            row["selected_version"] = registry.active_id()
            if index == len(batches) - 1:
                # 末批没有后续窗口检验新版本，因此仅评估和检查恢复，不消耗新增训练预算。
                row.update(status="terminal_evaluation", reason="末批仅评分和检查回滚；不执行没有后续窗口检验的新增训练。")
            else:
                try:
                    optimize_window(config, registry, train, validation, current, row, rows, audit, controller, window, recent)
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    row.update(status="controller_failed", error=f"{type(exc).__name__}: {exc}", action="retain",
                               selected_version=registry.active_id(), reason="控制器或实验失败，保留安全版本；失败调用和实验计入成本。")
                    audit.record("temporal_controller_failed", {"batch": batch, "error": row["error"]})
                    # 认证或计费错误无法通过继续请求后续批次解决，需立即中止。
                    if getattr(exc, "http_status", None) in (401, 402, 403, 409):
                        raise
            if digest(prediction_file) != frozen["predictions_sha256"]:
                raise ValueError("Frozen operational prediction evidence changed")
            write_json(window / "result.json", row)
            rows.append(row)
            recent = [*recent, current][-2:]
            append_jsonl(root / "history.jsonl", row)
            all_y.append(current["y"])
            all_active.append(probabilities)
            all_baseline.append(baseline_probabilities)
            aggregate = {"operational": score_predictions(np.concatenate(all_active), np.concatenate(all_y)),
                         "static_baseline": score_predictions(np.concatenate(all_baseline), np.concatenate(all_y))}
            aggregate["gain_vs_static_baseline"] = aggregate["operational"]["macro_f1"] - aggregate["static_baseline"]["macro_f1"]
            calls = collect_model_calls(audit.root)
            window_calls = [c for c in calls if c.get("context", {}).get("batch") == batch]
            resources = [candidate_resources(t) for t in row["experiments"]]
            costs = usage_totals(window_calls)
            costs.update(candidate_cpu_seconds_known=sum(r.get("cpu_seconds") or 0 for r in resources),
                         candidate_wall_seconds_known=sum(r.get("wall_seconds") or 0 for r in resources),
                         candidates_with_unknown_resources=sum(not r for r in resources))
            audit.record("temporal_summary", {"batch": batch, "version": active,
                         "operational": metrics_brief(operational), "static_baseline": metrics_brief(baseline),
                         "gain": operational["macro_f1"] - baseline["macro_f1"], "costs": costs,
                         "action": row["action"], "selected": row["selected_version"],
                         "rollback": bool(rollback or integrity), "status": row["status"], "reason": row["reason"]})
            export_temporal(root, rows, aggregate, audit.receipt)
            print(json.dumps({"batch": batch, "status": row["status"], "frozen_macro_f1": operational["macro_f1"],
                              "static_macro_f1": baseline["macro_f1"], "action": row["action"],
                              "next_version": row["selected_version"], "calls": costs["calls"]}, ensure_ascii=False), flush=True)
        drill = rollback_drill(root, registry, validation)
        audit.record("isolated_rollback_drill", drill)
        checks = {"source_data_unchanged": all(digest(config[name]) == value for name, value in bindings.items()),
                  "predictions_still_frozen": all(digest(root / "windows" / f"batch{r['batch']:02d}" / "predictions.npz")
                                                  == r["prediction_commit"]["predictions_sha256"] for r in rows),
                  "rollback_drill_passed": drill["passed"] and drill["actual_deployment_unchanged"],
                  "all_windows_finished": len(rows) == len(batches),
                  "all_controller_windows_completed": not any(r["status"] == "controller_failed" for r in rows)}
        write_json(root / "verification.json", checks)
        audit.record("temporal_verification", checks)
        succeeded = all(checks.values())
        audit.record("temporal_session_complete", {"checks": checks, "aggregate": aggregate,
                     "active_version": registry.active_id(), "scope": "historical_public_data_not_official_hidden_evaluation"})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        try:
            audit.record("session_failed", {"error": error})
        except RuntimeError:
            pass
        raise
    finally:
        try:
            final_receipt = audit.finish(succeeded)
        finally:
            summary = export_temporal(root, rows, aggregate, final_receipt or audit.receipt, error)
    return summary
