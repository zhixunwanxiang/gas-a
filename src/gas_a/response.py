"""Train-only fitted additive response head; inference uses frozen per-row features.

All 16 sensors are retained. This is an explicitly migrated auxiliary head,
not a replacement checkpoint or a hidden-stream adaptation rule.
"""
import hashlib
import math
import numpy as np

ARRAYS = ('response_mean', 'response_scale', 'response_precision')
METHOD = 'preserve_parent_blocks_zero_adam_additive_response_head'
FORMAT = 'gas-a.response-mlp.v1'
CLIP = 3.


def enabled(model):
    return model.meta.get('response_adapter') is True


def validate_spec(value):
    ranges = {'ridge': (.01, 1000.), 'strength': (1., 32.), 'guard_multiplier': (1., 4.)}
    if not isinstance(value, dict) or set(value) != set(ranges):
        raise ValueError('response_adapter requires ridge, strength and guard_multiplier')
    for name, (lo, hi) in ranges.items():
        v = value[name]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError('Invalid response_adapter ' + name)


def shape_features(raw):
    """Per-feature response pattern across every sensor; no fitted/test statistics.

    UCI #224: the second feature is Rmax/R0 (zero response is 1).
    Stable max scaling avoids squaring the potentially large original values.
    """
    values = raw.copy().reshape(-1, 16, 8)
    values[:, :, 1] -= 1
    maximum = np.max(np.abs(values), axis=1, keepdims=True)
    scaled = np.divide(values, maximum, out=np.zeros_like(values), where=maximum > 0)
    norm = np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))
    return (4 * np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0)).reshape(-1, 128)


def normalized(model, raw):
    return np.clip((shape_features(raw) - model.response_mean) / model.response_scale, -CLIP, CLIP)


def guard(model, features):
    if not model.meta['response_fitted']:
        return np.ones((len(features), 1))
    distance = np.sum((features @ model.response_precision) * features, axis=1)
    threshold = model.meta['response_radius'] * model.meta['response_guard_multiplier']
    return (distance <= threshold).astype(np.float64)[:, None]


def fit_head(model, train, spec):
    """Weighted ridge increment about parent head, then ordinary Adam continuation.

    Only train is accepted here. Existing normalizers/Adam moments are never reset.
    The new head's first optimizer is an explicit least-squares update, not Adam;
    subsequent minibatches update the whole network with the inherited Adam step.
    """
    validate_spec(spec)
    raw, labels = train['x'], train['y']
    if not model.meta['response_fitted']:
        features = shape_features(raw)
        model.response_mean = features.mean(axis=0)
        model.response_scale = features.std(axis=0)
        model.response_scale[model.response_scale < 1e-12] = 1
        x = normalized(model, raw)
        model.response_precision = np.linalg.inv(x.T @ x / len(x) + .01 * np.eye(128))
        # Symmetrize rounding error; this is a shrinkage precision, not a causal diagnosis.
        model.response_precision = (model.response_precision + model.response_precision.T) / 2
        model.meta.update(response_radius=float(np.max(np.sum((x @ model.response_precision) * x, axis=1))),
                          response_fitted=True, response_statistics_train_sha256=train['sha256'])
    x = np.column_stack([normalized(model, raw), np.ones(len(raw))])
    weights = len(labels) / (6 * np.maximum(np.bincount(labels, minlength=6), 1)[labels])
    old = np.vstack([model.params['wa'], model.params['ba']])
    target = spec['strength'] * np.eye(6)[labels]
    increment = np.linalg.solve(x.T @ (x * weights[:, None]) + spec['ridge'] * np.eye(129),
                                x.T @ ((target - x @ old) * weights[:, None]))
    fitted = old + increment
    model.params['wa'], model.params['ba'] = fitted[:-1].copy(), fitted[-1].copy()
    model.meta['response_guard_multiplier'] = float(spec['guard_multiplier'])
    record = {'method': 'class_balanced_ridge_increment_about_actual_parent_head_then_joint_Adam',
              'train_sha256': train['sha256'], 'rows': len(raw), 'configuration': dict(spec),
              'normalizer': 'fit_once_on_authorized_train_only; parent_scaler_unchanged',
              'coverage': 'shrinkage_Mahalanobis; covariance_ridge=0.01; radius=max_train_distance',
              'feature_clip': CLIP, 'sensors': 'all_16; no_fixed_subset',
              'parent_head_sha256': 'sha256:' + hashlib.sha256(old.tobytes()).hexdigest(),
              'increment_norm': float(np.linalg.norm(increment)), 'adam_reset': False}
    model.meta['response_training'] = record
    model.validate()
    return record
