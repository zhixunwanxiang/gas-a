"""本地规则、SDK 和时间批次调度器共用的候选执行步骤。"""
from pathlib import Path
import time

from .data import evaluate, load_model
from .io import read_json
from .optimize import train_candidate
from .quality import gate, assess_candidate


def execute_candidate(parent, train, validation, directory, spec, diagnosis, limits, *,
                      active_metrics, baseline_metrics, policy, validation_gate=assess_candidate, shadow=None):
    """执行候选训练、重载和评估，保留失败资源记录；提交与部署由调度器负责。

    可选的 shadow 为（数据集、活动版本指标、baseline 指标）元组，
    仅用于影子评估，不参与训练梯度或 epoch 选择。"""
    directory = Path(directory)
    started, cpu_started = time.monotonic(), time.process_time()
    trial = {"name": spec["name"], "configuration": spec, "evidence_directory": str(directory),
             "proposal_origin": spec.get("proposal_origin", "legacy_configuration")}
    try:
        limits = {**limits, 'checkpoint_gate': policy, 'checkpoint_reference': baseline_metrics,
                  'checkpoint_assessor': validation_gate}
        report = train_candidate(parent, train, validation, directory, spec, diagnosis, limits)
        trial["report"] = report
        # 选择依据来自磁盘中实际导出的 checkpoint，不直接信任训练期间的内存对象。
        exported = load_model(directory / "checkpoint.npz", train["paths"])
        measured = evaluate(exported, validation)
        if any(abs(measured[key] - report["validation_after"][key]) > 1e-12 for key in ("macro_f1", "accuracy")):
            raise ValueError("Independent checkpoint reload disagrees with training report")
        trial["metrics"] = measured
        trial["gate"] = validation_gate(measured, active_metrics, baseline_metrics, policy)
        if "robust_gate" in report:
            trial["robust_gate"] = report["robust_gate"]
            if not report["robust_gate"]["accepted"]:
                trial["gate"] = {**trial["gate"], "accepted": False, "reasons": [
                    *trial["gate"]["reasons"], *report["robust_gate"]["reasons"]]}
        if shadow is not None:
            # 当前窗口的标签只用于独立 shadow 评估，不传入 train_candidate。
            data, active, baseline = shadow
            metric = evaluate(exported, data)
            trial["shadow_metrics"] = metric
            trial["shadow_gate"] = assess_candidate(metric, active, baseline, policy, allow_absent_targets=True)
    except (ValueError, OSError, TimeoutError) as exc:
        trial["error"] = f"{type(exc).__name__}: {exc}"
        if (directory / "experiment.json").is_file():
            trial["report"] = read_json(directory / "experiment.json")
        if (directory / "failure.json").is_file():
            trial["resources"] = read_json(directory / "failure.json")["resources"]
    finally:
        # 训练失败或重载失败也计量；范围覆盖候选执行全过程，平台训练额度保持未知。
        trial["resources"] = {"wall_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started,
                              "device": "cpu", "local_gpu_training_seconds": 0, "platform_training_credits": None,
                              "new_label_requests": 0,
                              "scope": "候选训练、保存、独立重载、固定/影子评估与门槛；不含模型调用和宿主提交"}
    return trial


def candidate_resources(trial):
    """优先读取覆盖完整候选执行过程的资源计量，兼容旧版训练报告中的计量字段。"""
    return trial.get("resources") or trial.get("report", {}).get("resources", {})
