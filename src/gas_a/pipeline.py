"""本地规则闭环：加载配置、安排受限实验、比较候选并维护活动模型版本。"""
import json
import math
from pathlib import Path
from .data import load_pair, load_model, evaluate
from .monitor import monitor, diagnose
from .optimize import bounded_experiment, gate
from .registry import Registry
from .io import CLASSES, read_json, read_config, write_json, append_jsonl, history, lock, stamp, digest
from .review import write_round_review
from .candidates import execute_candidate
from .architecture import target_shape, validate_choices
from .resilience import monitor_safely, diagnose_safely, recovery
from .controller_policy import scoped_experiments


def load_config(path, root=None):
    """解析配置继承、统一项目路径，并在启动训练前校验轮数、预算和实验参数。"""
    path = Path(path).resolve()
    config = read_config(path)
    root = Path(root).resolve() if root else path.parent.parent
    for name in ("train", "validation", "baseline", "runtime"):
        p = Path(config[name])
        config[name] = str(p if p.is_absolute() else (root / p).resolve())
    if type(config["rounds"]) is not int or not 1 <= config["rounds"] <= 100:
        raise ValueError("rounds must be in 1..100")
    if not 1 <= config["optimization"]["max_candidates"] <= 6:
        raise ValueError("Invalid candidate count")
    if not 0 < config["optimization"]["max_seconds_per_candidate"] <= 300:
        raise ValueError("Invalid per-candidate time budget")
    limits = config["optimization"]
    from .robustness import settings as robustness_settings
    robustness_settings(limits)
    if limits.get("reference_configurations"):
        raise ValueError("Automatic reference configurations are disabled; candidates must come from the current AI proposal")
    if type(config["gate"].get("reject_zero_target_recall", False)) is not bool:
        raise ValueError("Invalid zero-target-recall gate")
    for name in ("target_recall_max_drop", "max_parent_fixed_f1_drop"):
        value = config["gate"].get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid " + name)
    pause = config.get("controller", {}).get("rejected_windows_before_pause", 3)
    if config.get("controller", {}).get("planning_mode", "compose") not in ("compose", "menu"):
        raise ValueError("Invalid planning mode")
    if type(pause) is not int or not 2 <= pause <= 10:
        raise ValueError("Invalid controller no-gain pause threshold")
    adaptive = config.get("adaptive", {})
    if type(adaptive.get("enabled", False)) is not bool:
        raise ValueError("Invalid adaptive mode")
    weights = adaptive.get("recent_weights", [1.0])
    if (not isinstance(weights, list) or not 1 <= len(weights) <= 3
            or any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights)
            or not sum(weights) > 0):
        raise ValueError("Invalid recent-window weights")
    margin = adaptive.get("switch_margin", .005)
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not 0 <= margin <= .1:
        raise ValueError("Invalid historical switch margin")
    for name in ("forward_feedback", "compact_prompts"):
        if type(adaptive.get(name, False)) is not bool:
            raise ValueError("Invalid adaptive " + name)
    hard = adaptive.get("hard_minimum_class_recall", {})
    if not isinstance(hard, dict) or any(k not in CLASSES or isinstance(v, bool) or not isinstance(v, (float, int))
                                       or not math.isfinite(v) or not 0 <= v <= 1 for k, v in hard.items()):
        raise ValueError("Invalid adaptive hard class minimum")
    if limits.get("search_mode", "adaptive") != "adaptive":
        raise ValueError("Invalid search mode")
    if type(limits.get("select_best_epoch", False)) is not bool:
        raise ValueError("Invalid checkpoint selection mode")
    patience = limits.get("early_stopping_patience", 0)
    if type(patience) is not int or not 0 <= patience <= limits["max_epochs"]:
        raise ValueError("Invalid early stopping patience")
    for spec in config["experiments"]:
        bounded_experiment(spec, limits)
    validate_choices(config.get("architecture_choices", {}))
    if len({s["name"] for s in config["experiments"]}) != len(config["experiments"]):
        raise ValueError("Experiment names must be unique")
    targets = config["gate"].get("minimum_class_recall", {})
    if not isinstance(targets, dict) or any(name not in CLASSES or isinstance(value, bool)
           or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
           for name, value in targets.items()):
        raise ValueError("Invalid minimum class recall targets")
    return config


def search_evidence(past, parent_sha):
    """汇总精简的实测搜索历史，包括被拒绝和失败的尝试。"""
    families = {}
    stalled = 0
    for row in reversed(past):
        if row.get("parent_checkpoint_sha256") != parent_sha or row.get("action") == "promote":
            break
        stalled += 1
    for row in past:
        for trial in row.get("experiments", []):
            report = trial.get("report", {})
            name = trial.get("name", report.get("name"))
            if not name:
                continue
            stat = families.setdefault(name, {"attempts": 0, "same_parent_attempts": 0,
                                              "best_gain": None, "cpu_seconds_known": 0, "failures": 0})
            stat["attempts"] += 1
            stat["same_parent_attempts"] += report.get("parent_checkpoint_sha256", row.get("parent_checkpoint_sha256")) == parent_sha
            stat["failures"] += bool(trial.get("error"))
            stat["cpu_seconds_known"] += (trial.get("resources") or report.get("resources", {})).get("cpu_seconds") or 0
            if report:
                gain = report["validation_after"]["macro_f1"] - report["validation_before"]["macro_f1"]
                stat["best_gain"] = max(stat["best_gain"] if stat["best_gain"] is not None else -1, gain)
    return {"stalled_rounds_on_parent": stalled, "families": families,
            "policy": "先覆盖未试方向；已获益方向最多占一个菜单位；同一父模型连续停滞后优先换方向；精确配置不重复。"}


def ordered_specs(config, past, parent_sha=None, diagnosis=None, model=None, *, train_sha=None, validation_sha=None):
    """按父 checkpoint、已试配置、实测收益和诊断信号生成去重后的受限实验菜单。"""
    evidence = search_evidence(past, parent_sha)
    stats = evidence["families"]
    previous = []
    for row, trial in scoped_experiments(past, parent_sha, train_sha, validation_sha):
        spec = trial.get('report', {}).get('configuration', trial.get('configuration'))
        if spec:
            previous.append(spec)

    def signature(spec):
        # 名称不参与配置身份，省略的参数按默认值补齐后再判断等价性。
        defaults = {"focal_gamma": 0, "anchor_l2": 0, "feature_lr_scale": 1,
                    "schedule": "constant", "final_lr_ratio": 1, "batch_size": 128,
                    "freeze": "none", "frozen_parameters": [], "class_boosts": {}}
        return json.dumps({**defaults, **{k: v for k, v in spec.items() if k != "name"}}, sort_keys=True)

    used = {signature(s) for s in previous}
    proposals = []
    for base in config["experiments"]:
        if model is not None and "architecture" in base:
            try:
                target_shape(model, base["architecture"])
            except ValueError:
                continue  # 当前父模型已达到扩宽/加层上限，不把不可执行的方案交给控制器。
        variants = [dict(base)]
        if base.get("focal_gamma", 0):
            variants += [{**base, "focal_gamma": value} for value in (.5, 1.5)]
        elif base.get("anchor_l2", 0):
            variants += [{**base, "anchor_l2": value} for value in (.1, .001)]
        elif base.get("feature_lr_scale", 1) != 1:
            variants += [{**base, "feature_lr_scale": value} for value in (.1, .5)]
        elif base["balance"]:
            variants += [{**base, "balance": value, "weak_boost": min(base["weak_boost"], 1.5)} for value in (.25, .75)]
        variants += [{**base, "learning_rate": base["learning_rate"] * factor} for factor in (.5, 2)]
        variants += [{**base, "batch_size": 64}, {**base, "schedule": "cosine", "final_lr_ratio": .2}]
        for trial in variants:
            try:
                bounded_experiment(trial, config["optimization"])
            except ValueError:
                continue
            key = signature(trial)
            if key not in used:
                proposals.append(trial)
                used.add(key)  # 不同实验方向之间的等价配置也需要去重。
                break
    # 即使 LLM 多次偏好历史获益方案，菜单仍为尚未尝试的实验方向保留空间。
    order = {s["name"]: i for i, s in enumerate(config["experiments"])}
    generalization_gap = (diagnosis or {}).get("fit_comparison", {}).get("macro_f1_gap", 0)

    def priority(spec):
        stat = stats.get(spec["name"], {})
        measured_gain = stat.get("best_gain") or 0
        direction = -1 if generalization_gap > .15 and (spec.get("anchor_l2", 0) or spec.get("feature_lr_scale", 1) < 1
                                                       or spec.get("freeze") in ("features", "first_hidden")) else 0
        return (stat.get("same_parent_attempts", 0), direction, stat.get("failures", 0),
                -max(0, measured_gain) / max(stat.get("cpu_seconds_known", 0), .01), order[spec["name"]])

    proposals.sort(key=priority)
    if evidence["stalled_rounds_on_parent"] < 2:
        proven = [s for s in proposals if (stats.get(s["name"], {}).get("best_gain") or 0) >= config["gate"]["min_macro_f1_gain"]
                  and stats[s["name"]]["same_parent_attempts"] < 2]
        if proven:
            winner = min(proven, key=priority)
            proposals.remove(winner)
            proposals.insert(0, winner)
    # train/validation 差距较大并不能证明某种修正必然有效；菜单同时保留
    # 保护特征表示的方向，以及调整类别权重或 loss 的方向。
    selected = []
    while proposals and len(selected) < config["optimization"]["max_candidates"]:
        if selected:
            def direction(spec):
                return "class_loss" if spec["balance"] or spec["weak_boost"] > 1 or spec.get("focal_gamma", 0) else "conservative"
            fresh = [s for s in proposals if stats.get(s["name"], {}).get("same_parent_attempts", 0)
                     == stats.get(proposals[0]["name"], {}).get("same_parent_attempts", 0)]
            pick = min(fresh, key=lambda s: sum(direction(s) == direction(old) for old in selected))
        else:
            pick = proposals[0]
        selected.append(pick)
        proposals.remove(pick)
    if (diagnosis or {}).get("drift_detected") and generalization_gap > .15:
        robust = lambda s: any(s.get(k, 0) for k in ("input_noise", "sensor_gain", "sensor_shift", "sensor_dropout"))
        alternatives = [s for s in proposals if robust(s)]
        if alternatives and selected and not any(robust(s) for s in selected):
            selected[-1] = alternatives[0]  # 菜单保留一个可以实际检验的鲁棒性方向。
    return selected


def run(config, rounds=None):
    """在单写入锁内执行规则闭环，先检查活动模型，再训练候选、比较门槛并记录本轮结果。"""
    count = config["rounds"] if rounds is None else rounds
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("rounds must be in 1..100")
    root = Path(config["runtime"])
    registry = Registry(root)
    with lock(root):
        train, validation = load_pair(config["train"], config["validation"])
        model = load_model(config["baseline"], train["paths"])
        registry.initialize(config["baseline"], evaluate(model, validation))
        for _ in range(count):
            # 轮次编号包含中断后留下的目录，避免覆盖已有证据。
            existing = [int(p.name.split("-")[-1]) for p in (root / "rounds").glob("round-*")] if (root / "rounds").exists() else []
            index = max(existing, default=0) + 1
            directory = root / "rounds" / f"round-{index:03d}"
            directory.mkdir(parents=True, exist_ok=False)
            active, before, baseline, rollback = registry.ensure_healthy(validation, config["gate"])
            # 先完成完整性与恢复检查，再绑定本轮父模型，确保所有候选的起点一致。
            parent_path = registry.checkpoint(active)
            drift, monitoring_recovery = monitor_safely(monitor, train["x"], validation["x"], config["monitor"])
            diagnosis, diagnostic_recovery = diagnose_safely(diagnose, train, validation, before, drift,
                                                            train_metrics=evaluate(load_model(parent_path), train))
            write_json(directory / "monitor.json", drift)
            write_json(directory / "diagnosis.json", diagnosis)
            past = history(root / "history.jsonl")
            record = {"round": index, "time": stamp(), "mode": "local_cpu_rule_controller", "parent_version": active,
                      "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
                      "parent_checkpoint_sha256": digest(parent_path),
                      "history_rounds_read": [r["round"] for r in past], "before": before, "static_baseline": baseline,
                      "automatic_rollback": rollback, "diagnosis": diagnosis, "quality_gate": config["gate"], "experiments": []}
            record["recoveries"] = [r for r in (monitoring_recovery, diagnostic_recovery) if r]
            selected, selected_metrics, best_trial = active, before, None
            needs_work = drift["drift_detected"] or before["macro_f1"] < .98
            record["search"] = search_evidence(past, digest(parent_path))
            specs = ordered_specs(config, past, digest(parent_path), diagnosis, model=load_model(parent_path),
                                  train_sha=train['sha256'], validation_sha=validation['sha256']) if needs_work else []
            # 同轮候选分别从同一父模型续训，候选之间不串联更新，便于公平比较。
            for spec in specs:
                limits = {**config["optimization"], "checkpoint_gate": config["gate"]}
                trial = execute_candidate(parent_path, train, validation, directory / spec["name"], spec, diagnosis, limits,
                    active_metrics=before, baseline_metrics=baseline, policy=config["gate"], validation_gate=gate)
                if not trial.get("error") and trial["gate"]["accepted"] and trial["metrics"]["macro_f1"] > selected_metrics["macro_f1"]:
                    selected_metrics, best_trial = trial["metrics"], trial
                record["experiments"].append(trial)
                if trial.get("error"):
                    record["recoveries"].append(recovery("training", trial["error"], "reject_candidate_keep_valid_parent"))
            if best_trial:
                # 只有通过门槛且优于当前已选指标的候选才会进入发布分支。
                selected = f"r{index:03d}_{best_trial['name']}"
                registry.register(selected, directory / best_trial["name"] / "checkpoint.npz", active, selected_metrics, best_trial["report"])
                registry.activate(selected, "candidate_passed_quality_gates")
            record.update(selected_version=selected, after=selected_metrics,
                          action="promote" if best_trial else "retain", selection_mode="rule_controller",
                          reason="采用通过全部质量门槛且 Macro-F1 最高的候选。" if best_trial else "没有候选通过全部质量门槛，保留当前合格模型。")
            record["review"] = write_round_review(directory, record)
            write_json(directory / "round.json", record)
            append_jsonl(root / "history.jsonl", record)
            print(json.dumps({"round": index, "action": record["action"], "selected": selected,
                              "macro_f1": selected_metrics["macro_f1"], "accuracy": selected_metrics["accuracy"],
                              "weak_class": diagnosis["weakest_class"], "drift_features": drift["flagged_features"]}), flush=True)
        summary = {"format": "gas-a.run-summary.v1", "active_version": registry.active_id(),
                   "completed_rounds": len(history(root / "history.jsonl")),
                   "metrics": evaluate(load_model(registry.checkpoint(registry.active_id()), train["paths"]), validation),
                   "static_baseline": baseline, "mode": "local_cpu", "official_competition_result": False}
        write_json(root / "summary.json", summary)
        return summary

