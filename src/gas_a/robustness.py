"""Authorized-validation stress tests; no fitting or hidden-stream adaptation."""
import math
import numpy as np

from .data import evaluate


def settings(limits):
    value = limits.get("robustness", {"enabled": False})
    if (not isinstance(value, dict) or set(value) - {"enabled", "max_scenario_f1_drop", "max_worst_f1_drop", "ranking"}
            or type(value.get("enabled", False)) is not bool):
        raise ValueError("Invalid robustness settings")
    drop = value.get("max_scenario_f1_drop", 0.)
    if isinstance(drop, bool) or not isinstance(drop, (int, float)) or not math.isfinite(drop) or not 0 <= drop <= 1:
        raise ValueError("Invalid robustness regression allowance")
    worst_drop = value.get('max_worst_f1_drop', 0.)
    if isinstance(worst_drop, bool) or not isinstance(worst_drop, (int,float)) or not math.isfinite(worst_drop) or not 0 <= worst_drop <= 1:
        raise ValueError('Invalid worst-scenario regression allowance')
    ranking = value.get('ranking', 'worst_case')
    if ranking not in ('macro_f1', 'worst_case'):
        raise ValueError('Invalid guarded checkpoint ranking')
    return {"enabled": value.get("enabled", False), "max_scenario_f1_drop": drop,
            'max_worst_f1_drop': worst_drop, 'ranking': ranking}


def validation_suite(parent, validation):
    """Test every sensor once; its eight inputs are replaced by the parent train mean.

    No sensor is selected from evaluation-label feedback. This is a deterministic
    missing-response stress test, not a reconstruction of physical drift.
    The parent scaler is read, never refitted or written.
    """
    for sensor in range(16):
        x = validation["x"].copy()
        x[:, sensor * 8:(sensor + 1) * 8] = parent.mean[sensor * 8:(sensor + 1) * 8]
        yield f"sensor_{sensor + 1:02d}_missing", {**validation, "x": x}


def evaluate_suite(model, suite):
    cases = {}
    for name, data in suite:
        metric = evaluate(model, data)
        cases[name] = {"macro_f1": metric["macro_f1"], "accuracy": metric["accuracy"],
                       "rows": metric["rows"], "per_class_recall": {
                           label: value["recall"] for label, value in metric["per_class"].items()}}
    if not cases:
        raise ValueError("Empty robustness suite")
    scores = [value["macro_f1"] for value in cases.values()]
    return {"scope": "authorized_validation_only; not_hidden_evaluation",
            "scenario": "each_of_16_sensors_replaced_by_unchanged_parent_training_mean",
            "worst_macro_f1": min(scores), "mean_macro_f1": sum(scores) / len(scores),
            "cases": cases}


def gate(candidate, parent, config):
    reasons = []
    if set(candidate["cases"]) != set(parent["cases"]):
        raise ValueError("Robustness cases differ from parent reference")
    if candidate["worst_macro_f1"] < parent["worst_macro_f1"] - config.get('max_worst_f1_drop',0.) - 1e-12:
        reasons.append("robust_worst_case_regression")
    for name, metric in candidate["cases"].items():
        if metric["macro_f1"] < parent["cases"][name]["macro_f1"] - config["max_scenario_f1_drop"] - 1e-12:
            reasons.append("robust_scenario_regression:" + name)
    return {"accepted": not reasons, "reasons": reasons, "reference": "actual_parent_on_same_authorized_validation_cases",
            'policy': dict(config), 'worst_f1_change': candidate['worst_macro_f1']-parent['worst_macro_f1'],
            'mean_f1_change': candidate['mean_macro_f1']-parent['mean_macro_f1']}


def compact(value):
    return {key: value[key] for key in ("scope", "scenario", "worst_macro_f1", "mean_macro_f1", "cases")}
