"""任务 A 的显式架构迁移：嵌入父权重与 Adam 状态，扩宽及添加恒等残差层。"""
import copy
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from .io import CLASSES
from .reference.model import Model, FORMAT as LEGACY_FORMAT, PARAMETERS, MAX_CHECKPOINT_BYTES
from . import response

FORMAT = "gas-a.residual-mlp.v2"
METHOD = "preserve_parent_blocks_zero_pad_adam_identity_residual"


def shape(model):
    return {"hidden": model.meta["hidden"], "hidden_layers": model.meta.get("hidden_layers", 1),
            **({"response_adapter": True} if response.enabled(model) else {})}


def validate_request(request):
    """只允许改变隐藏宽度和深度，输入、输出、种子和训练状态不属于可变配置。"""
    allowed = {"hidden", "hidden_layers", "width_multiplier", "add_layers", "response_adapter"}
    if not isinstance(request, dict) or not request or set(request) - allowed:
        raise ValueError("Architecture accepts hidden/hidden_layers or width_multiplier/add_layers only")
    for key, value in request.items():
        if key == "response_adapter":
            if value is not True:
                raise ValueError("response_adapter migration can only add/preserve the head")
            continue
        low, high = {"hidden": (32, 128), "hidden_layers": (1, 4),
                     "width_multiplier": (2, 4), "add_layers": (1, 3)}[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid architecture " + key)
    if {"hidden", "width_multiplier"} <= set(request) or {"hidden_layers", "add_layers"} <= set(request):
        raise ValueError("Use absolute or relative architecture dimensions, not both")


def target_shape(model, request):
    validate_request(request)
    old = shape(model)
    hidden = request.get("hidden", old["hidden"] * request.get("width_multiplier", 1))
    layers = request.get("hidden_layers", old["hidden_layers"] + request.get("add_layers", 0))
    if hidden < old["hidden"] or layers < old["hidden_layers"]:
        raise ValueError("Shrinking/removing layers would discard parent state; continuation requires expansion only")
    if hidden > 128 or layers > 4:
        raise ValueError("Architecture continuation exceeds width 128 or depth 4")
    adapter = old.get("response_adapter", False) or request.get("response_adapter", False)
    count = 128 * hidden + hidden + hidden * 6 + 6 + (layers - 1) * (hidden * hidden + hidden) + (774 if adapter else 0)
    if count >= 100000:
        raise ValueError("Architecture must have fewer than 100,000 parameters")
    return {"hidden": hidden, "hidden_layers": layers, **({"response_adapter": True} if adapter else {})}


def validate_choices(choices):
    """控制器只接受配置中最多三种命名架构，不接受任意结构或状态修改。"""
    if not isinstance(choices, dict) or len(choices) > 3:
        raise ValueError("Architecture choices must be a map of at most three named requests")
    for name, request in choices.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
            raise ValueError("Invalid architecture choice name")
        validate_request(request)


def available_choices(model, choices):
    """在调用模型之前，按真实父网络过滤超限及无变化的架构选项。"""
    validate_choices(choices)
    available = {}
    for name, request in choices.items():
        try:
            target = target_shape(model, request)
        except ValueError:
            continue
        if target != shape(model):
            available[name] = dict(request)
    return available


def _shapes(hidden, layers, adapter=False):
    result = {"w1": (128, hidden), "b1": (hidden,), "w2": (hidden, 6), "b2": (6,)}
    for layer in range(2, layers + 1):
        result.update({f"w{layer + 1}": (hidden, hidden), f"b{layer + 1}": (hidden,)})
    if adapter:
        result.update(wa=(128, 6), ba=(6,))
    return result


class MigratedModel(Model):
    """第一层为 tanh；新增隐藏层为 h + tanh(hW+b)，零初始化时是恒等映射。"""

    def validate(self):
        m = self.meta
        adapter = response.enabled(self)
        if (m.get("format") != (response.FORMAT if adapter else FORMAT) or m.get("task") != "classification" or m.get("profile") != "A"
                or m.get("classes") != CLASSES or len(m.get("paths", [])) != 128):
            raise ValueError("Migrated checkpoints require task A, 128 features and the fixed six classes")
        if type(m.get("hidden_layers")) is not int or not 1 <= m["hidden_layers"] <= 4:
            raise ValueError("Invalid migrated hidden layer count")
        if type(m.get("hidden")) is not int or not 32 <= m["hidden"] <= 128:
            raise ValueError("Invalid migrated hidden width")
        expected = _shapes(m["hidden"], m["hidden_layers"], adapter)
        for group in (self.params, self.first, self.second):
            if set(group) != set(expected):
                raise ValueError("Migrated checkpoint tensors missing")
            for key, dimensions in expected.items():
                value = group[key]
                if value.shape != dimensions or value.dtype != np.dtype("float64") or not np.isfinite(value).all():
                    raise ValueError("Invalid migrated checkpoint tensor: " + key)
        if type(m.get("parameter_count")) is not int or m["parameter_count"] != sum(v.size for v in self.params.values()) or m["parameter_count"] >= 100000:
            raise ValueError("Invalid migrated parameter count")
        # 复用原检查器校验基础张量、scaler、seed、steps、epochs 和 Adam 二阶矩。
        base = Model.__new__(Model)
        base.meta = {**m, "format": LEGACY_FORMAT,
                     "parameter_count": sum(self.params[k].size for k in PARAMETERS)}
        for group in ("params", "first", "second"):
            setattr(base, group, {k: getattr(self, group)[k] for k in PARAMETERS})
        base.mean, base.scale = self.mean, self.scale
        Model.validate(base)
        if any(np.any(v < 0) for v in self.second.values()):
            raise ValueError("Invalid migrated Adam second moment")
        migration = m.get("architecture_migration", {})
        old = migration.get("from", {})
        if (migration.get("method") != (response.METHOD if adapter else METHOD) or migration.get("to") != shape(self)
                or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(migration.get("parent_checkpoint_sha256")))
                or set(old) not in ({"hidden", "hidden_layers"}, {"hidden", "hidden_layers", "response_adapter"})
                or ("response_adapter" in old and old["response_adapter"] is not True)
                or any(type(old.get(k)) is not int for k in ("hidden", "hidden_layers"))
                or not 32 <= old["hidden"] <= m["hidden"]
                or not 1 <= old["hidden_layers"] <= m["hidden_layers"]
                or old == shape(self)):
            raise ValueError("Missing or invalid explicit parent architecture migration")
        if adapter:
            for name, dimensions in zip(response.ARRAYS, ((128,), (128,), (128, 128))):
                value = getattr(self, name)
                if value.shape != dimensions or value.dtype != np.dtype('float64') or not np.isfinite(value).all():
                    raise ValueError('Invalid response adapter statistics')
            if (np.any(self.response_scale <= 0) or not np.allclose(self.response_precision, self.response_precision.T, rtol=0, atol=1e-10)
                    or np.min(np.linalg.eigvalsh(self.response_precision)) <= 0
                    or type(m.get('response_fitted')) is not bool
                    or isinstance(m.get('response_radius'), bool) or not isinstance(m.get('response_radius'), (int, float))
                    or not math.isfinite(m['response_radius']) or m['response_radius'] < 0
                    or isinstance(m.get('response_guard_multiplier'), bool)
                    or not isinstance(m.get('response_guard_multiplier'), (int, float))
                    or not 1 <= m['response_guard_multiplier'] <= 4):
                raise ValueError('Invalid response adapter coverage guard')
            if not m['response_fitted'] and any(np.any(g[k]) for g in (self.params, self.first, self.second) for k in ('wa', 'ba')):
                raise ValueError('Unfitted response head must preserve zero weights and Adam moments')
            if m['response_fitted'] and not re.fullmatch(r'sha256:[a-f0-9]{64}', str(m.get('response_statistics_train_sha256'))):
                raise ValueError('Missing train-only response statistics provenance')

    def normalized(self, raw):
        base = super().normalized(raw)
        return np.column_stack([base, response.normalized(self, raw)]) if response.enabled(self) else base

    def _forward_cache(self, x):
        if response.enabled(self):
            x = x[:, :128]
        first = np.tanh(x @ self.params["w1"] + self.params["b1"])
        hidden, residuals = first, []
        for layer in range(2, self.meta["hidden_layers"] + 1):
            key = layer + 1
            branch = np.tanh(hidden @ self.params[f"w{key}"] + self.params[f"b{key}"])
            residuals.append((key, hidden, branch))
            hidden = hidden + branch
        return hidden, (x, first, residuals)

    def forward(self, x):
        hidden, _ = self._forward_cache(x)
        logits = hidden @ self.params["w2"] + self.params["b2"]
        if response.enabled(self):
            logits += response.guard(self, x[:, 128:]) * (x[:, 128:] @ self.params['wa'] + self.params['ba'])
        return hidden, logits

    def backward(self, x, delta):
        """delta 是 loss 对 logits 的导数；返回全部参数梯度和输入梯度。"""
        extra = x[:, 128:] if response.enabled(self) else None
        hidden, (x, first, residuals) = self._forward_cache(x)
        gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
        dh = delta @ self.params["w2"].T
        for key, previous, branch in reversed(residuals):
            dz = dh * (1 - branch ** 2)
            gradients.update({f"w{key}": previous.T @ dz, f"b{key}": dz.sum(axis=0)})
            dh = dh + dz @ self.params[f"w{key}"].T
        dz = dh * (1 - first ** 2)
        gradients.update(w1=x.T @ dz, b1=dz.sum(axis=0))
        input_gradient = dz @ self.params["w1"].T
        if response.enabled(self):
            head_delta = delta * response.guard(self, extra)
            gradients.update(wa=extra.T @ head_delta, ba=head_delta.sum(axis=0))
            input_gradient = np.column_stack([input_gradient, head_delta @ self.params['wa'].T])
        return gradients, input_gradient

    def save(self, path):
        if not response.enabled(self):
            return super().save(path)
        self.validate()
        arrays = {**self.params, **{'adam_first_' + k: v for k, v in self.first.items()},
                  **{'adam_second_' + k: v for k, v in self.second.items()}, 'mean': self.mean, 'scale': self.scale,
                  **{k: getattr(self, k) for k in response.ARRAYS},
                  'metadata': np.array(json.dumps(self.meta, sort_keys=True, separators=(',', ':'), allow_nan=False))}
        with Path(path).open('xb') as stream, zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO(); np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0)); info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue())

    def fit(self, x, y, epochs):
        """经共用训练器续训，避免沿用仅更新两层的旧 Model.fit。"""
        if type(epochs) is not int or not 1 <= epochs <= 10000:
            raise ValueError("epochs must be between 1 and 10000")
        self.normalized(x)
        if (y.shape != (len(x),) or not len(x) or not np.issubdtype(y.dtype, np.integer)
                or np.any((y < 0) | (y >= 6))):
            raise ValueError("Invalid task A training targets")
        from .optimize import fit_weighted
        records, _ = fit_weighted(self, {"x": x, "y": y},
            {"learning_rate": self.meta["learning_rate"], "epochs": epochs, "balance": 0., "weak_boost": 1., "l2": 0.},
            CLASSES[0], float("inf"))
        return [r["loss"] for r in records]

    @classmethod
    def load(cls, path):
        model = load_checkpoint(path)
        if not isinstance(model, cls):
            raise ValueError("Expected an explicitly migrated checkpoint")
        return model


def load_checkpoint(path):
    """保留原格式加载器，新增格式单独严格校验，不让任意宽度的随机旧模型混入续训。"""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint must be a regular file no larger than 16 MiB")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [v.filename for v in entries]
            if (len(names) != len(set(names)) or "metadata.npy" not in names or len(names) > 42
                    or sum(v.file_size for v in entries) > MAX_CHECKPOINT_BYTES):
                raise ValueError("Invalid checkpoint archive")
        with np.load(path, allow_pickle=False) as arrays:
            metadata = arrays["metadata"]
            if metadata.ndim != 0 or metadata.dtype.kind != "U":
                raise ValueError("Invalid checkpoint metadata")
            meta = json.loads(str(metadata))
            if not isinstance(meta, dict):
                raise ValueError("Invalid checkpoint metadata")
            if meta.get("format") == LEGACY_FORMAT:
                return Model.load(path)
            hidden, layers = meta.get("hidden"), meta.get("hidden_layers")
            adapter = meta.get('response_adapter') is True
            if (meta.get("format") != (response.FORMAT if adapter else FORMAT) or type(hidden) is not int or not 32 <= hidden <= 128
                    or type(layers) is not int or not 1 <= layers <= 4):
                raise ValueError("Unsupported migrated architecture")
            keys = _shapes(hidden, layers, adapter)
            expected = {*keys, *("adam_first_" + k for k in keys), *("adam_second_" + k for k in keys),
                        "mean", "scale", "metadata"}
            if adapter:
                expected.update(response.ARRAYS)
            if set(names) != {n + ".npy" for n in expected}:
                raise ValueError("Invalid migrated checkpoint archive")
            model = MigratedModel.__new__(MigratedModel)
            model.meta = meta
            for group, prefix in (("params", ""), ("first", "adam_first_"), ("second", "adam_second_")):
                setattr(model, group, {k: arrays[prefix + k].copy() for k in keys})
            model.mean, model.scale = arrays["mean"].copy(), arrays["scale"].copy()
            if adapter:
                for key in response.ARRAYS:
                    setattr(model, key, arrays[key].copy())
        model.validate()
        return model
    except (zipfile.BadZipFile, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid checkpoint archive or metadata") from exc


def migrate_parent(parent, request, parent_sha):
    """只扩展父模型；旧张量及 Adam 矩逐元素保留，新张量的 Adam 矩为零。"""
    parent.validate()
    if (parent.meta.get("profile") != "A" or parent.meta["task"] != "classification"
            or parent.meta["classes"] != CLASSES or len(parent.meta["paths"]) != 128
            or (not isinstance(parent, MigratedModel) and parent.meta["hidden"] != 32)):
        raise ValueError("Architecture migration requires a compatible task A parent")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(parent_sha)):
        raise ValueError("Migration requires the actual parent checkpoint SHA-256")
    old, new = shape(parent), target_shape(parent, request)
    adapter = new.get('response_adapter', False)
    evidence = {"performed": old != new, "method": response.METHOD if adapter else METHOD, "parent_checkpoint_sha256": parent_sha,
                "from": old, "to": new, "copied_state": [],
                "preserved": ["seed", "steps", "epochs", "mean", "scale", "target_mean", "target_scale"],
                "new_adam_state": "zero; bias correction retains the parent cumulative step"}
    if old == new:
        return copy.deepcopy(parent), evidence
    model = MigratedModel.__new__(MigratedModel)
    model.meta = copy.deepcopy(parent.meta)
    model.meta.update(format=response.FORMAT if adapter else FORMAT, **new)
    model.meta["architecture_migration"] = {k: evidence[k] for k in ("method", "parent_checkpoint_sha256", "from", "to")}
    model.mean, model.scale = parent.mean.copy(), parent.scale.copy()
    dimensions = _shapes(new["hidden"], new["hidden_layers"], adapter)
    for group in ("params", "first", "second"):
        values = {k: np.zeros(size, dtype=np.float64) for k, size in dimensions.items()}
        for key, value in getattr(parent, group).items():
            slices = tuple(slice(0, s) for s in value.shape)
            values[key][slices] = value
            evidence["copied_state"].append({"group": group, "tensor": key, "parent_shape": list(value.shape),
                                           "child_shape": list(values[key].shape), "copied_elements": value.size})
        setattr(model, group, values)
    if adapter:
        for key, value in zip(response.ARRAYS, (np.zeros(128), np.ones(128), np.eye(128))):
            setattr(model, key, getattr(parent, key).copy() if response.enabled(parent) else value)
        if not response.enabled(parent):
            model.meta.update(response_fitted=False, response_radius=0., response_guard_multiplier=1.)
    # 新单元的输入连接确定性初始化；输出及跨接连接为零，因此迁移前后函数一致。
    added = new["hidden"] - old["hidden"]
    if added:
        rng = np.random.default_rng(np.random.SeedSequence([parent.meta["seed"], parent.meta["steps"],
                                                           new["hidden"], new["hidden_layers"], 731]))
        model.params["w1"][:, old["hidden"]:] = rng.normal(0, math.sqrt(1 / 128), (128, added))
    model.meta["parameter_count"] = sum(v.size for v in model.params.values())
    model.validate()
    return model, evidence
