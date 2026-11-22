"""控制器异常的受控恢复：保留原始证据，不降低质量门槛或伪造模型回执。"""
import copy
import math
import time
from contextlib import contextmanager

from .controller_policy import validate_plan
from .redaction import redact
from .io import CLASSES


@contextmanager
def bounded_model_time(client, deadline):
    """把单次HTTP超时限制在本轮剩余时间内，留出状态提交和恢复时间。"""
    remaining = deadline - time.monotonic() if deadline is not None else float("inf")
    if remaining <= 1:
        raise TimeoutError("Round request budget exhausted; use safe measured recovery")
    original = getattr(client, "timeout", None)
    changed = isinstance(original, (float, int)) and not isinstance(original, bool) and original > 0
    if changed:
        client.timeout = min(original, remaining)
    try:
        yield
    finally:
        if changed:
            client.timeout = original


def monitor_safely(compute, reference, current, config):
    """坏输入不能伪装成无漂移；统计计算异常时明确标为未知，继续有效预测的评估。"""
    import numpy as np
    x, y = np.asarray(reference, float), np.asarray(current, float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != 128 or y.shape[1] != 128 or not len(x) or not len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Monitoring preflight requires finite nonempty 128-feature matrices")
    try:
        return compute(x, y, config), None
    except (ValueError, KeyError, RuntimeError, ArithmeticError) as exc:
        note = recovery("monitoring", exc, "explicit_unknown_drift_keep_prediction_evidence")
        return {"status": "unavailable", "reference_rows": len(x), "current_rows": len(y),
                "sufficient_samples": min(len(x), len(y)) >= config["minimum_samples"],
                "drift_detected": None, "flagged_features": None, "sensor_ranking": [], "top_features": [],
                "features": [], "error": note["error"]}, note


def diagnose_safely(compute, train, current, metric, drift, *, train_metrics=None):
    """详细归因异常时只使用已测逐类指标，不伪造传感器或物理原因。"""
    try:
        return compute(train, current, metric, drift, train_metrics=train_metrics), None
    except (ValueError, KeyError, RuntimeError, ArithmeticError) as exc:
        import numpy as np
        observed = [c for c in CLASSES if metric["per_class"][c]["support"] > 0]
        if not observed:
            raise ValueError("No measured class supports available for diagnosis recovery") from exc
        weak = min(observed, key=lambda c: metric["per_class"][c]["f1"])
        a, b = np.bincount(train["y"], minlength=6), np.bincount(current["y"], minlength=6)
        note = recovery("diagnosis", exc, "measured_class_metrics_without_causal_claims")
        return {"status": "limited_measured_summary", "weakest_class": weak,
                "weakest_class_metrics": metric["per_class"][weak], "fit_comparison": {}, "top_confusions": [],
                "class_prior_total_variation": .5 * float(np.abs(a / a.sum() - b / b.sum()).sum()),
                "drift_detected": drift["drift_detected"], "top_sensors": drift["sensor_ranking"][:4],
                "hypotheses": ["仅保留实测分类错误，详细统计归因暂不可用。"], "error": note["error"]}, note


def fatal_provider_error(error, rnd=None):
    """认证失败或结果未知的 SDK 会话必须中止，不能继续纠正、提交或自动重试。"""
    return (getattr(error, "http_status", None) in (401, 402, 403, 409)
            or getattr(error, "code", None) in ("RPC_OUTCOME_UNKNOWN", "RPC_MISMATCH",
                                                  "HOST_STREAM_CLOSED", "CHUNKED_UPLOAD_UNSUPPORTED")
            or bool(getattr(rnd, "_outcome_unknown", False)))


def recovery(stage, error, method):
    return {"stage": stage, "error": str(redact(str(error)))[:1200], "method": method,
            "origin": "program_recovery_not_model_reply"}


def repair_plan(plan, menu, budget, evidence, *, text_limits=None):
    """仅保留合法菜单意图；将无依据的自由文本及错误引用替换成程序可核验的契约。

    不生成实验名称，不删除超范围参数修改。调用方仍必须执行参数白名单校验。
    原始回复由 model_records 保留，本函数的结果明确属于程序修复后的提案。
    """
    if not isinstance(plan, dict):
        raise ValueError("Cannot recover a plan without a JSON object")
    names, action = plan.get("experiment_names"), plan.get("action")
    permitted = {s["name"] for s in menu}
    if (action not in ("experiment", "observe") or not isinstance(names, list)
            or any(not isinstance(n, str) or n not in permitted for n in names)
            or len(names) != len(set(names)) or action == "observe" and names
            or action == "experiment" and not 1 <= len(names) <= budget):
        raise ValueError("Recovery cannot authorize unknown experiments or exceed the budget")
    result = copy.deepcopy(plan)
    result.update(evidence_ids=["C", "F"], claims=[], success_condition="configured_quality_gates",
        stop_condition="observe_then_reassess" if action == "observe" else "reject_if_gates_fail",
        hypothesis="可能存在当前数据与既有分类边界差异，需要通过受限实验验证。" if action == "experiment" else "继续观察后续数据和已提交预测的实际表现。",
        reason="根据当前与固定验证的测量检验菜单方案，收益仍需实际评估。" if action == "experiment" else "不新增训练，保留当前模型并继续监测与恢复检查。",
        decision_origin="program_repaired_model_menu_intent")
    validate_plan(result, menu, budget, evidence, text_limits=text_limits)
    return result


def safe_observation(evidence, reason):
    result = {"action": "observe", "experiment_names": [], "evidence_ids": ["C", "F"], "claims": [],
              "success_condition": "configured_quality_gates", "stop_condition": "observe_then_reassess",
              "hypothesis": "当前控制器未给出可安全执行的方案，继续观察已提交预测的实际表现。",
              "reason": "控制器回复无法安全执行，保留可用模型，不新增梯度更新。",
              "decision_origin": "program_safe_observe", "recovery_error": str(redact(str(reason)))[:1200]}
    validate_plan(result, [], 0, evidence)
    return result


def measured_selection(prompt):
    """只使用程序已生成的同口径比较表，绝不把错误回复中的版本放进合格集合。"""
    eligible = prompt.get("eligible_artifacts", {})
    comparison = prompt.get("comparison", {})
    rows = comparison.get("rows", [])
    if not eligible or len(rows) != len(eligible) or {r.get("artifact_id") for r in rows} != set(eligible):
        raise ValueError("No complete measured eligible set is available for safe recovery")
    if any(isinstance(r.get("score"), bool) or not isinstance(r.get("score"), (int, float))
           or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in rows):
        raise ValueError("Recovery requires finite measured scores")
    preferred = comparison.get("measured_preference")
    best = max(r["score"] for r in rows)
    if preferred not in eligible or next(r["score"] for r in rows if r["artifact_id"] == preferred) < best - 1e-12:
        raise ValueError("Recovery preference is not the highest measured eligible score")
    return {"selected_artifact_id": preferred, "reason": "控制器选择无效，程序按同口径实测分数及原质量门槛恢复选择。",
            "decision_origin": "program_measured_selection"}
