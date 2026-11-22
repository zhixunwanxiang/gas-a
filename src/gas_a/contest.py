"""Workspace v2 适配器，由宿主 SDK 执行真实模型调用和用量计量。"""
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from .data import load_pair, load_model, evaluate
from .monitor import monitor, diagnose
from .candidates import execute_candidate
from .pipeline import ordered_specs, search_evidence
from .io import read_json, read_config, write_json, append_jsonl, digest
from .cli import prediction
from .adaptive import focus_bounds
from .architecture import available_choices, shape as architecture_shape
from .planning import composed, build_prompt, resolve_plan, correction_prompt
from .model_records import recorded_round_model, receipt_reference, collect_model_calls
from .resilience import (fatal_provider_error, recovery, safe_observation,
                         monitor_safely, diagnose_safely)
from .review import write_round_review
from .controller_policy import (measured_evidence, build_selection_prompt,
                                class_protection, stopping_evidence)

TASKS = ["classification"]
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]


def configuration():
    """读取宿主指定或镜像内的配置，并解析共享配置继承。"""
    default = Path(__file__).resolve().parents[2] / "configs" / "a.json"
    path = Path(os.environ.get("GAS_A_CONFIG", "/app/configs/a.json"))
    return read_config(path if path.is_file() else default)


def parse_object(text):
    """去除可选的 think 块和 Markdown 代码围栏，要求回复正文为单个 JSON 对象。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Model response must be one JSON object")
    return result


def sdk_text(text, max_bytes=4000):
    """按 SDK 文本字段的字节上限截取前缀，保持 UTF-8 编码有效。"""
    return text.strip().encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def measured_metrics(response):
    """从 SDK 回执的 summary.metrics 读取指标；宿主分数缺失时不以自报指标替代。"""
    if not isinstance(response, dict):
        return None
    obj = response.get("summary", response)
    obj = obj.get("metrics", obj) if isinstance(obj, dict) else None
    if not isinstance(obj, dict):
        return None
    v = obj.get("macro_f1")
    if isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 <= v <= 1:
        return None
    return {k: obj[k] for k in ("macro_f1", "accuracy") if k in obj}


def parent_checkpoint(rnd):
    """按宿主绑定摘要定位已声明的父检查点，路径与文件摘要仍由 SDK 核验。"""
    expected = rnd.parent.checkpoint_sha256
    if not isinstance(expected, str):
        raise ValueError("Invalid host parent checkpoint digest")
    expected = expected.removeprefix("sha256:").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid host parent checkpoint digest")
    for name in rnd.parent.descriptor["checkpoint_paths"]:
        path = rnd.parent.path(name, root=os.environ.get("GAS_A_PARENT_ROOT"))
        if digest(path).removeprefix("sha256:") == expected:
            return path
    raise ValueError("No declared parent checkpoint matches the host binding")


def validation_comparison(local_parent, local_candidate, host_parent, host_candidate, minimum_gain):
    """仅比较同一来源的父子指标；缺少候选宿主测量时不能采用新产物。"""
    use_host = host_parent is not None and host_candidate is not None
    before, after = (host_parent, host_candidate) if use_host else (local_parent, local_candidate)
    gain = after["macro_f1"] - before["macro_f1"]
    measured = host_candidate is not None
    accepted = measured and gain + 1e-12 >= minimum_gain
    return {"basis": "host" if use_host else "local",
            "scope": "host_current_validation" if use_host else "same_local_validation_for_parent_and_candidate",
            "parent_macro_f1": before["macro_f1"], "candidate_macro_f1": after["macro_f1"],
            "gain_macro_f1": gain, "minimum_gain": minimum_gain,
            "host_parent_available": host_parent is not None, "host_candidate_available": measured,
            "accepted": accepted,
            "reason": ("candidate_host_measurement_missing" if not measured else
                       "minimum_gain_not_met" if not accepted else
                       "host_gain_passed" if use_host else "same_local_validation_gain_passed")}


def train(rnd):
    """执行一轮宿主闭环：核验父模型、规划受限实验、提交实测候选、选择版本并保存状态。"""
    started = time.monotonic()
    if rnd.dataset_key != "A" or rnd.task.get("task_type", rnd.task.get("task", "classification")) != "classification":
        raise ValueError("This image supports task A classification only")
    limits = getattr(rnd, "limits", None)
    if hasattr(rnd, "limits") and (not isinstance(limits, dict)
                                   or limits.get("artifact_upload_protocol") != "chunked-v1"):
        from contest_harness.errors import HarnessError
        raise HarnessError("CHUNKED_UPLOAD_UNSUPPORTED", "SDK 1.0.4 requires a chunked-v1 host")
    config = getattr(rnd, "config", None) or configuration()
    model_tokens = min(config.get("live", {}).get("max_output_tokens", 768), rnd.max_output_tokens)
    train_data, validation = load_pair(rnd.data.train_path, rnd.data.validation_path)
    parent = parent_checkpoint(rnd)
    parent_model = load_model(parent, train_data["paths"])
    parent_metric = evaluate(parent_model, validation)
    work = Path(getattr(rnd, "work_root", os.environ.get("CONTEST_WORK", "/workspace"))) / f"gas-a-{rnd.round_index:03d}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=False)
    state = rnd.state_workspace(work / "state")
    state_file = state / "lifecycle.json"
    if state_file.exists():
        memory = read_json(state_file)
        if digest(state / "baseline_checkpoint.npz") != memory["baseline_sha256"]:
            raise ValueError("Saved baseline anchor digest mismatch")
    else:
        if rnd.round_index != 1:
            raise ValueError("Original baseline state is missing after round 1")
        shutil.copyfile(parent, state / "baseline_checkpoint.npz")
        memory = {"baseline_artifact_id": rnd.parent.artifact_id, "baseline_sha256": digest(parent), "rounds": []}
    baseline = evaluate(load_model(state / "baseline_checkpoint.npz", train_data["paths"]), validation)
    recoveries = []
    drift, note = monitor_safely(monitor, train_data["x"], validation["x"], config["monitor"])
    if note:
        recoveries.append(note)
    diagnosis, note = diagnose_safely(diagnose, train_data, validation, parent_metric, drift,
                                    train_metrics=evaluate(parent_model, train_data))
    if note:
        recoveries.append(note)
    historical = rnd.history()
    search = search_evidence(memory["rounds"], digest(parent))
    specs = ordered_specs(config, memory["rounds"], digest(parent), diagnosis, model=parent_model,
                          train_sha=train_data['sha256'], validation_sha=validation['sha256'])
    architecture_choices = available_choices(parent_model, config.get("architecture_choices", {}))
    names = [s["name"] for s in specs]
    selected_budget = config["optimization"].get("max_selected_experiments", len(specs))
    if type(selected_budget) is not int or selected_budget < 1:
        raise ValueError("No valid candidate budget is available")
    selected_budget = min(selected_budget, config["optimization"]["max_candidates"] if composed(config) else len(specs))
    evidence = measured_evidence(parent_metric, parent_metric, evaluate(parent_model, train_data), diagnosis, drift,
                                  memory["rounds"], digest(parent), [s["name"] for s in config["experiments"]],
                                  validation_sha=validation["sha256"], fixed_scope="host_current_authorized_validation",
                                  training_sha=train_data["sha256"])
    stop = stopping_evidence(memory["rounds"], digest(parent), train_data["sha256"], validation["sha256"], config)
    evidence["facts"]["S"] = stop
    from . import robustness
    if robustness.settings(config["optimization"])["enabled"]:
        evidence["facts"]["R"] = robustness.compact(robustness.evaluate_suite(
            parent_model, robustness.validation_suite(parent_model, validation)))
    if stop["pause"]:
        specs, selected_budget = [], 0  # SDK 仍接收实际规划和版本选择回执。
    write_json(work / "controller-evidence.json", evidence)
    planning_prompt = build_prompt(config, parent_model, specs, evidence, selected_budget,
        config["optimization"]["max_seconds_per_candidate"], mode="sdk", index=rnd.round_index,
        parameter_bounds=focus_bounds(specs, architecture_choices, architecture_shape(parent_model)),
        past=memory["rounds"], parent_sha=digest(parent))
    plan_error = None
    plan_receipt = None
    plan = {}
    correction_left, planning_attempts = 1, []
    parent_fallback_call_id = None
    while True:
        try:
            plan_receipt = recorded_round_model(rnd, work,
                    [{"role": "user", "content": json.dumps(planning_prompt, ensure_ascii=False)}],
                    max_tokens=model_tokens, temperature=0, purpose="experiment_planning")
            plan = parse_object(plan_receipt.text)
            parent_fallback_call_id = (plan_receipt.call_id if isinstance(plan, dict) and plan.get("fallback") == "retain_parent" else None)
            plan, specs = resolve_plan(plan, config, parent_model, specs, evidence, selected_budget,
                memory["rounds"], digest(parent), focus_only=True, architecture_choices=architecture_choices,
                weak_class=diagnosis["weakest_class"])
            plan_error = None
            planning_attempts.append({"call_id": plan_receipt.call_id, "accepted": True})
            break
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            if fatal_provider_error(exc, rnd):
                raise
            plan_error = str(exc)
            planning_attempts.append({"call_id": getattr(plan_receipt, "call_id", None), "accepted": False,
                                      "error": plan_error, "proposal": plan})
            if composed(config) and correction_left:
                correction_left -= 1
                planning_prompt = correction_prompt(planning_prompt, plan, exc)
                recoveries.append(recovery("planning", exc, "ask_ai_to_revise_configuration"))
                continue
            try:
                plan, specs = resolve_plan(plan, config, parent_model, specs, evidence, selected_budget,
                    memory["rounds"], digest(parent), repair=True, focus_only=True,
                    architecture_choices=architecture_choices, weak_class=diagnosis["weakest_class"])
                recoveries.append(recovery("planning", exc, "verified_contract_rebinding"))
            except (ValueError, KeyError, TypeError):
                plan, specs = safe_observation(evidence, exc), []
                recoveries.append(recovery("planning", exc, "safe_observe_without_training"))
            break
    write_json(work / "monitor.json", drift)
    write_json(work / "diagnosis.json", diagnosis)
    write_json(work / "model-plan.json", {"call": receipt_reference(plan_receipt, work) if plan_receipt else None,
               "parse_error": plan_error, "bound_plan": plan, "recoveries": recoveries, "attempts": planning_attempts})
    record = {"round": rnd.round_index, "parent_artifact_id": rnd.parent.artifact_id,
              "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
              "parent_checkpoint_sha256": digest(parent), "before": parent_metric,
              "static_baseline": baseline, "diagnosis": diagnosis, "search": search, "quality_gate": config["gate"], "experiments": [],
              "plan": {**plan, "experiments": plan.get("experiments", [s["name"] for s in specs]),
                       "reason": plan.get("reason")},
              "recoveries": recoveries, "planning_attempts": planning_attempts,
              "model_plan_parse_error": plan_error, "history_ids": [r["id"] for r in historical]}
    if stop["pause"]:
        record["decision_source"] = "budget_pause"
    eligible = {}
    baseline_id = memory["baseline_artifact_id"]
    unsafe_parent = (rnd.parent.artifact_id != baseline_id
                     and not class_protection(parent_metric, baseline, baseline, config["gate"])["accepted"])
    if not unsafe_parent and parent_metric["macro_f1"] + 1e-12 >= baseline["macro_f1"]:
        eligible[rnd.parent.artifact_id] = {"local": parent_metric, "kind": "retain"}
    # 回滚引用真实的历史 artifact ID，保持与本轮新训练候选的身份区分。
    known = {a["artifact_id"] for a in rnd.artifacts}
    baseline_id = memory["baseline_artifact_id"]
    if (unsafe_parent or parent_metric["macro_f1"] < baseline["macro_f1"]) and baseline_id in known:
        eligible[baseline_id] = {"local": baseline, "kind": "rollback"}
    host_parent = measured_metrics(rnd.frame.get("current_validation", {}))
    if host_parent is not None and rnd.parent.artifact_id in eligible:
        eligible[rnd.parent.artifact_id]["host"] = host_parent
    if "R" in evidence["facts"] and rnd.parent.artifact_id in eligible:
        eligible[rnd.parent.artifact_id]["robust_validation"] = evidence["facts"]["R"]
    end_time = started + max(1, rnd.wall_seconds - 5) if rnd.wall_seconds else float("inf")
    for spec in specs[:config["optimization"]["max_candidates"]]:
        trial = {"name": spec["name"], "configuration": spec}
        candidate = work / spec["name"]
        trial["evidence_directory"] = str(candidate)
        try:
            remaining = end_time - time.monotonic()
            if remaining < 2:
                trial["error"] = "round_wall_budget_exhausted"
                record["experiments"].append(trial)
                break
            limits = {**config["optimization"], "checkpoint_gate": config["gate"],
                      "max_seconds_per_candidate": min(config["optimization"]["max_seconds_per_candidate"], remaining)}
            trial = execute_candidate(parent, train_data, validation, candidate, spec, diagnosis, limits,
                active_metrics=parent_metric, baseline_metrics=baseline, policy=config["gate"])
            if trial.get("error"):
                record["experiments"].append(trial)
                continue
            report = trial["report"]
            shutil.copyfile(Path(__file__).parent / "reference" / "model.py", candidate / "model.py")
            shutil.copyfile(Path(__file__).parent / "architecture.py", candidate / "architecture.py")
            shutil.copyfile(Path(__file__).parent / "response.py", candidate / "response.py")
            shutil.copyfile(Path(__file__).parent / "quality.py", candidate / "quality.py")
            shutil.copyfile(Path(__file__).parent / "resilience.py", candidate / "resilience.py")
            shutil.copyfile(Path(__file__).parent / "optimize.py", candidate / "optimizer.py")
            shutil.copyfile(Path(__file__).parent / "data.py", candidate / "input_adapter.py")
            shutil.copyfile(Path(__file__).parent / "cli.py", candidate / "predict_adapter.py")
            write_json(candidate / "diagnosis.json", diagnosis)
            candidate_hypothesis = plan["hypothesis"]
            candidate_test = plan.get("iteration", {}).get("tests", {}).get(spec["name"])
            if candidate_test:
                candidate_hypothesis += "\n该候选检验：" + candidate_test
            submitted = rnd.submit_candidate(candidate,
                hypothesis=sdk_text(candidate_hypothesis),
                change_summary=json.dumps(spec, ensure_ascii=False), configuration=spec, seed=parent_model.meta["seed"],
                checkpoint_paths=["checkpoint.npz"], code_paths=["model.py", "architecture.py", "response.py", "quality.py", "resilience.py", "optimizer.py", "input_adapter.py", "predict_adapter.py"], predict_argv=PREDICT_ARGV)
            host_candidate = measured_metrics(submitted)
            decision = trial.pop("gate")
            comparison = validation_comparison(parent_metric, report["validation_after"], host_parent,
                                               host_candidate, config["gate"]["min_macro_f1_gain"])
            qualified = decision["accepted"] and comparison["accepted"]
            # 缺少父宿主分数时，宿主增益保持未知；明确记录同口径本地比较及候选回执。
            host_ok = comparison["accepted"] if comparison["basis"] == "host" else None
            trial.update(report=report, local_gate=decision, host_receipt=submitted, host_gate_passed=host_ok,
                         validation_comparison=comparison, eligible=qualified)
            if qualified:
                eligible[submitted["artifact_id"]] = {"local": report["validation_after"], "host": host_candidate, "kind": "promote"}
                if "robust_validation_after" in report:
                    eligible[submitted["artifact_id"]]["robust_validation"] = report["robust_validation_after"]
        except (ValueError, OSError, TimeoutError) as exc:
            if fatal_provider_error(exc, rnd):
                raise
            trial["error"] = f"{type(exc).__name__}: {exc}"
            if (candidate / "experiment.json").is_file():
                trial["report"] = read_json(candidate / "experiment.json")
            elif (candidate / "failure.json").is_file():
                trial["resources"] = read_json(candidate / "failure.json")["resources"]
        record["experiments"].append(trial)
    if not eligible:
        raise ValueError("No eligible artifact satisfies the original-baseline quality floor")
    selection_prompt = build_selection_prompt(eligible, record["experiments"], rnd.parent.artifact_id,
        mode="sdk", index=rnd.round_index, history_ids=record["history_ids"])
    selection_receipts, selection_attempts = [], []
    selection_error = None
    selection_mode = "model"
    selected = None
    choice_receipt = None
    next_test = None
    selection_budget = 1 + correction_left
    for attempt in range(selection_budget):
        request = dict(selection_prompt)
        if attempt:
            request["correction"] = {"error": selection_error, "previous_response": getattr(choice_receipt, "text", None),
                                     "request": "上一回复未通过校验。请只从合格版本中返回一个完整编号及非空理由。"}
        choice_receipt = None
        try:
            choice_receipt = recorded_round_model(rnd, work,
                          [{"role": "user", "content": json.dumps(request, ensure_ascii=False)}],
                          max_tokens=min(model_tokens, config.get("live", {}).get("selection_max_output_tokens", model_tokens)),
                          temperature=0, purpose="artifact_selection")
            selection_receipts.append(choice_receipt)
            choice = parse_object(choice_receipt.text)
            identity = choice["selected_artifact_id"]
            reason = choice["reason"]
            proposed_next = choice.get("next_test")
            if proposed_next is not None and (not isinstance(proposed_next, str)
                    or not proposed_next.strip() or len(proposed_next) > 256):
                raise ValueError("next_test must be a concise nonempty mechanism hypothesis")
            # 最终选择只能引用 eligible 中的完整 artifact ID，模型不能绕过程序门槛。
            if not isinstance(identity, str) or identity not in eligible or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Model selection is outside the quality-gated eligible set")
            submitted_reason = sdk_text(reason)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            if fatal_provider_error(exc, rnd):
                raise
            selection_error = str(exc)
            selection_attempts.append({"call_id": getattr(choice_receipt, "call_id", None), "error": selection_error})
            if attempt + 1 < selection_budget:
                recoveries.append(recovery("selection", exc, "bounded_correction_retry"))
            continue
        selected = identity
        next_test = proposed_next
        selection_error = None
        selection_mode = "model_retry" if attempt else "model"
        selection_attempts.append({"call_id": choice_receipt.call_id, "error": None})
        break
    if selected is None:
        if not parent_fallback_call_id or rnd.parent.artifact_id not in eligible:
            raise RuntimeError("No valid model selection or explicit model-authorized eligible-parent fallback; cannot fabricate an SDK decision")
        selected = rnd.parent.artifact_id
        reason = "依据模型规划中的显式兜底指令，在最终选版回复未通过校验时保留本轮合格父模型。"
        submitted_reason = sdk_text(reason)
        selection_mode = "model_authorized_parent_fallback"
        recoveries.append(recovery("selection", selection_error, "model_authorized_parent_retention"))
        record["selection_authorization_call_id"] = parent_fallback_call_id
    record.update(selected_artifact_id=selected, action=eligible[selected]["kind"], after=eligible[selected]["local"],
                  reason=reason, selection_mode=selection_mode, selection_parse_error=selection_error,
                  selection_attempts=selection_attempts, model_call_ids=list(rnd.call_ids))
    if next_test:
        record["selection_feedback"] = {"next_test": next_test, "call_id": choice_receipt.call_id,
                                        "origin": "model_hypothesis_not_measured_fact"}
    call_root = rnd.audit.root if getattr(rnd, "records_model_calls", False) else work
    call_records = [r for r in collect_model_calls(call_root) if r.get("context", {}).get("round") == rnd.round_index]
    record["review"] = write_round_review(work, record, call_records)
    write_json(work / "round.json", record)
    write_json(work / "model-choice.json", {"selection_mode": selection_mode,
               "authorization_call_id": parent_fallback_call_id if selection_mode == "model_authorized_parent_fallback" else None,
               "call_id": choice_receipt.call_id if choice_receipt else None,
               "call": receipt_reference(choice_receipt, work) if choice_receipt else None, "selected": selected,
               "reason": reason, "submitted_reason": submitted_reason, "attempts": selection_attempts})
    memory["rounds"].append(record)
    # 限制持久状态中的历史长度；完整实验记录仍由平台保留。
    memory["rounds"] = memory["rounds"][-10:]
    write_json(state_file, memory)
    rnd.save_state(state)
    rnd.finish(selected_artifact_id=selected,
               hypothesis=plan.get("hypothesis", "诊断当前公开数据的分布变化，并比较受约束的检查点续训实验。"),
               change_summary=f"A 任务闭环第 {rnd.round_index} 轮，动作代号：{record['action']}",
               reason=submitted_reason, configuration={"experiments": [s["name"] for s in specs], "action": record["action"],
                                                   "proposed_configurations": specs,
                                                   "plan_parse_error": plan_error, "selection_parse_error": selection_error},
               seed=parent_model.meta["seed"], history_ids=record["history_ids"], model_call_ids=list(rnd.call_ids))


_MODEL_CACHE = {}


def predict(sample, model_dir):
    """仅从选中模型的挂载目录加载任务 A checkpoint，按文件摘要复用预测缓存。"""
    if sample.task != "classification" or getattr(sample, "dataset_key", None) not in (None, "A"):
        raise ValueError("Only A classification predictions are supported")
    root = Path(model_dir).resolve()
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH", root / "checkpoint.npz")).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Prediction checkpoint must be inside the selected model mount")
    key = (str(path), digest(path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = load_model(path)
    return prediction(_MODEL_CACHE[key], sample.inputs)

