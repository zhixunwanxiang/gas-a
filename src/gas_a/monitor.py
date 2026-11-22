"""提供描述性的分布诊断；特征排序不构成因果归因。"""
import numpy as np
from .io import CLASSES

KINDS = ["delta_R", "normalized_response", "EMAi_0.001", "EMAi_0.01", "EMAi_0.1",
         "EMAd_0.001", "EMAd_0.01", "EMAd_0.1"]
SCHEMA_SOURCE = "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset"


def monitor(reference, current, config):
    """以固定训练分布为参照计算 128 维 PSI、KS 和均值变化，并按每个传感器 8 维汇总。"""
    x, y = np.asarray(reference, float), np.asarray(current, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 128 or y.shape[1] != 128 or not len(x) or not len(y):
        raise ValueError("Monitor needs nonempty 128-column reference/current matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite monitor input")
    enough = min(len(x), len(y)) >= config["minimum_samples"]
    features = []
    for j in range(128):
        a, b = x[:, j], y[:, j]
        # PSI 分箱仅由 reference 分位数确定；两端开放以接收越界样本。
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            value = float(a[0])
            delta = max(abs(value) * 1e-6, 1e-8)
            edges = np.array([-np.inf, value - delta, value + delta, np.inf])
        else:
            edges = np.r_[-np.inf, edges[1:-1], np.inf]
        pa, pb = np.histogram(a, edges)[0].astype(float), np.histogram(b, edges)[0].astype(float)
        # 加入平滑计数，避免空箱导致 log(0) 或除零。
        pa, pb = (pa + .5) / (len(a) + .5 * len(pa)), (pb + .5) / (len(b) + .5 * len(pb))
        psi = float(np.sum((pb - pa) * np.log(pb / pa)))
        points = np.sort(np.concatenate([a, b]))
        # 此处 KS 为两个经验 CDF 的最大距离，不是显著性检验的 p-value。
        ks = float(np.max(np.abs(np.searchsorted(np.sort(a), points, side="right") / len(a)
                                 - np.searchsorted(np.sort(b), points, side="right") / len(b))))
        denominator = max(float(a.std()), abs(float(a.mean())) * 1e-6, 1e-8)
        z = min(abs(float(b.mean() - a.mean())) / denominator, 1e12)
        flagged = enough and (psi >= config["psi_threshold"] or ks >= config["ks_threshold"]
                              or z >= config["mean_z_threshold"])
        features.append({"feature": j + 1, "sensor": j // 8 + 1, "kind": KINDS[j % 8],
                         "psi": psi, "ks": ks, "mean_shift_z": z, "flagged": bool(flagged)})
    ranked = sorted(features, key=lambda item: item["psi"], reverse=True)
    # 每个传感器对应连续 8 维；PSI 排名表示分布变化程度，不表示特征重要性。
    sensors = [{"sensor": k + 1, "mean_psi": float(np.mean([f["psi"] for f in features[k*8:k*8+8]])),
                "flagged_features": sum(f["flagged"] for f in features[k*8:k*8+8])} for k in range(16)]
    return {"reference_rows": len(x), "current_rows": len(y), "sufficient_samples": enough,
            "drift_detected": any(f["flagged"] for f in features),
            "flagged_features": sum(f["flagged"] for f in features), "thresholds": dict(config),
            "top_features": ranked[:16], "sensor_ranking": sorted(sensors, key=lambda s: s["mean_psi"], reverse=True),
            "features": features, "feature_schema_source": SCHEMA_SOURCE,
            "interpretation": "按当前规则触发的分布变化提示；这些统计不能单独确定变化原因。"}


def diagnose(train, validation, metrics, drift, *, train_metrics=None):
    """结合已揭示标签与漂移统计整理弱类、混淆对和类别比例变化，结论仅作为待验证线索。"""
    per_class = metrics["per_class"]
    observed = [c for c in CLASSES if per_class[c]["support"] > 0]
    weakest = min(observed, key=lambda c: per_class[c]["f1"])
    pairs = []
    cm = np.array(metrics["confusion_matrix"])
    for i in range(6):
        for j in range(6):
            if i != j and cm[i, j]:
                pairs.append({"actual": CLASSES[i], "predicted": CLASSES[j], "count": int(cm[i, j])})
    train_counts = np.bincount(train["y"], minlength=6)
    val_counts = np.bincount(validation["y"], minlength=6)
    prior_change = .5 * float(np.abs(train_counts / train_counts.sum() - val_counts / val_counts.sum()).sum())
    fit = {} if train_metrics is None else {
        "train_macro_f1": train_metrics["macro_f1"], "validation_macro_f1": metrics["macro_f1"],
        "macro_f1_gap": train_metrics["macro_f1"] - metrics["macro_f1"],
        "weak_train_recall": train_metrics["per_class"][weakest]["recall"],
        "weak_validation_recall": per_class[weakest]["recall"],
        "acetaldehyde_train_recall": train_metrics["per_class"]["Acetaldehyde"]["recall"],
        "acetaldehyde_validation_recall": per_class["Acetaldehyde"]["recall"],
        "interpretation": "训练与验证表现差异用于区分训练欠拟合和迁移问题；不构成物理归因。"}
    return {"weakest_class": weakest, "weakest_class_metrics": per_class[weakest], "fit_comparison": fit,
            "top_confusions": sorted(pairs, key=lambda p: p["count"], reverse=True)[:6],
            "train_class_counts": dict(zip(CLASSES, map(int, train_counts))),
            "validation_class_counts": dict(zip(CLASSES, map(int, val_counts))),
            "class_prior_total_variation": prior_change,
            "drift_detected": drift["drift_detected"], "top_sensors": drift["sensor_ranking"][:4],
            "hypotheses": ["输入特征分布变化可能导致协变量漂移", "类别不均衡或训练与验证的类别比例变化",
                           "优化过程或分类决策边界不匹配"],
            "limitation": "相关性和分类混淆现象不能单独确定漂移的物理原因。"}

