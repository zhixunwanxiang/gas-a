"""Shared validation admission rules for epoch selection and final release."""
import math

from .adaptive import fixed_gate, window_gate
from .controller_policy import protect_decision


def gate(candidate, active, baseline, config):
    """Apply configured gains, original-baseline floors and class protection."""
    for key in ('macro_f1', 'accuracy'):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            return {'accepted': False, 'reasons': ['invalid_metrics']}
    reasons = []
    if candidate['macro_f1'] < active['macro_f1'] + config['min_macro_f1_gain'] - 1e-12:
        reasons.append('macro_f1_gain_below_minimum')
    if candidate['macro_f1'] < baseline['macro_f1'] - 1e-12:
        reasons.append('below_static_baseline')
    if candidate['accuracy'] < active['accuracy'] - config['max_accuracy_drop'] - 1e-12:
        reasons.append('accuracy_regression')
    for name, minimum in config.get('minimum_class_recall', {}).items():
        value = candidate.get('per_class', {}).get(name, {}).get('recall')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value + 1e-12 < minimum:
            reasons.append('class_recall_below_minimum:' + name)
    return protect_decision({'accepted': not reasons, 'reasons': reasons or ['quality_gates_passed']},
                            candidate, active, baseline, config)


def assess_candidate(metrics, active, baseline, policy, *, allow_absent_targets=False):
    if policy.get('adaptive'):
        return window_gate(metrics, active, baseline, policy) if allow_absent_targets else fixed_gate(metrics, active, baseline, policy)
    unobserved = [name for name in policy.get('minimum_class_recall', {})
                  if not baseline.get('per_class', {}).get(name, {}).get('support', 0)]
    if allow_absent_targets:
        policy = {**policy, 'minimum_class_recall': {name: value
            for name, value in policy.get('minimum_class_recall', {}).items()
            if baseline.get('per_class', {}).get(name, {}).get('support', 0) > 0}}
    result = gate(metrics, active, baseline, policy)
    if allow_absent_targets:
        result['unobserved_recall_targets'] = unobserved
    return result
