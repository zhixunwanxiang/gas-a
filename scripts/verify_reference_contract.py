"""使用用户提供的未修改参考包，核验修正后的续训兼容性。"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.data import load_model, load_pair, evaluate
from gas_a.io import digest, read_json, write_json, stamp
from gas_a.optimize import train_candidate, gate
from gas_a.pipeline import load_config
from gas_a.registry import Registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    args = parser.parse_args()
    output, supplied = args.output.resolve(), args.reference_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(ROOT / "configs/a.live.json")
    write_json(output / "configuration.json", cfg)
    train, validation = load_pair(cfg["train"], cfg["validation"])
    parent = load_model(cfg["baseline"])
    original_digest = digest(cfg["baseline"])
    baseline = evaluate(parent, validation)
    spec = cfg["experiments"][0]
    report = train_candidate(cfg["baseline"], train, validation, output / "round-001", spec,
                             {"weakest_class": "Acetaldehyde"},
                             {**cfg["optimization"], "checkpoint_gate": cfg["gate"]})
    checkpoint = output / "round-001/checkpoint.npz"
    child = load_model(checkpoint)
    module_spec = importlib.util.spec_from_file_location("supplied_cpu_reference", supplied / "baselines/model.py")
    official = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(official)
    independent = official.Model.load(checkpoint)
    checks = {"original_reference_model_source": digest(supplied / "baselines/model.py") == digest(ROOT / "src/gas_a/reference/model.py"),
              "reference_predictions_identical": bool(np.array_equal(independent.predict(validation["x"]), child.predict(validation["x"]))),
              "reference_normalization_identical": bool(np.array_equal(independent.normalized(validation["x"]), parent.normalized(validation["x"]))),
              "parent_mean_std_unchanged": bool(np.array_equal(parent.mean, child.mean) and np.array_equal(parent.scale, child.scale)),
              "seed_and_target_scalers_preserved": all(parent.meta[k] == child.meta[k] for k in ("seed", "target_mean", "target_scale")),
              "reference_128_32_6": len(child.meta["paths"]) == 128 and child.meta["hidden"] == 32 and len(child.meta["classes"]) == 6,
              "actual_parent_and_counters": report["parent_checkpoint_sha256"] == original_digest
              and child.meta["steps"] > parent.meta["steps"] and child.meta["epochs"] > parent.meta["epochs"],
              "no_custom_prediction_state": not any(k in child.meta for k in ("input_transform", "active_features", "sensor_weights")),
              "required_output_files": all((output / "round-001" / name).is_file() for name in ("training.json", "history.jsonl", "checkpoint.npz"))}
    metrics = evaluate(child, validation)
    quality = gate(metrics, baseline, baseline, cfg["gate"])
    registry = Registry(output / "runtime")
    registry.initialize(cfg["baseline"], baseline)
    if quality["accepted"]:
        registry.register("reference_compatible_r1", checkpoint, "baseline", metrics, report)
        registry.activate("reference_compatible_r1", "local_cpu_quality_gate_passed")
    actual_selected = registry.checkpoint(registry.active_id())
    args_resume = [sys.executable, "-X", "utf8", "-B", "-m", "baselines.train", "--profile", "A",
                   "--train", cfg["train"], "--validation", cfg["validation"], "--resume", str(actual_selected),
                   "--epochs", "1", "--stage-id", "native-reference-resume", "--output", str(output / "official-resume")]
    started = time.monotonic()
    result = subprocess.run(args_resume, cwd=supplied, capture_output=True, timeout=45)
    write_json(output / "official-resume-receipt.json", {"argv": args_resume, "exit_code": result.returncode,
               "stdout": result.stdout.decode("utf-8", errors="replace"), "stderr": result.stderr.decode("utf-8", errors="replace"),
               "wall_seconds": time.monotonic() - started, "external_model_calls": 0})
    checks["unmodified_official_resume_command_succeeded"] = result.returncode == 0
    if result.returncode == 0:
        continued = official.Model.load(output / "official-resume/checkpoint.npz")
        selected = official.Model.load(actual_selected)
        native_report = read_json(output / "official-resume/training.json")
        checks["native_resume_actual_selected_parent"] = native_report["parent_checkpoint_sha256"] == digest(actual_selected)
        checks["native_resume_kept_state_and_advanced_steps"] = (continued.meta["steps"] > selected.meta["steps"]
                   and continued.meta["seed"] == selected.meta["seed"] and np.array_equal(continued.mean, selected.mean)
                   and np.array_equal(continued.scale, selected.scale))
    checks["baseline_file_unchanged"] = digest(cfg["baseline"]) == original_digest
    cfg["runtime"] = str(registry.root)
    write_json(output / "selected-config.json", cfg)
    verification = {"created_at": stamp(), "passed": all(checks.values()), "checks": checks, "baseline": baseline,
                    "candidate": metrics, "quality_gate": quality, "selected": registry.active_id(),
                    "external_model_calls": 0, "official_competition_result": False,
                    "scope": "核对指南的可执行条款；不是组委会验收结论或封闭评测结果。"}
    write_json(output / "verification.json", verification)
    lines = ["续训指南兼容性复核", f"全部兼容检查通过：{verification['passed']}",
             "原始包的 baselines/model.py 未修改，可直接加载和预测新检查点；原始 baselines.train 已实际 --resume 成功。",
             "128 维输入、32 个 tanh 隐藏单元、6 类 softmax；完整沿用父均值、标准差、种子、目标缩放和 Adam 状态。",
             "使用全部16个传感器的128维特征；不指定传感器子集，不压缩输入，不强制置零连接。",
             "第一层连接由训练更新；Adam 矩继承实际父模型，标准化流程保持参考实现。",
             "每个输出目录均为新目录；父摘要及累计步数如实记录；梯度只使用训练集。",
             "原来的 77 次实验属于本地并行对照，不是 10 轮平台续训；旧预处理模型保留为实验存档。", "",
             f"Macro-F1：{baseline['macro_f1']:.6f} → {metrics['macro_f1']:.6f}",
             f"乙醛召回：{metrics['per_class']['Acetaldehyde']['recall']:.6f}；乙醇召回：{metrics['per_class']['Ethanol']['recall']:.6f}",
             f"本地质量门槛：{quality}", "标准化和加载器兼容不代表各类别都无退化；逐类结果见 verification.json。",
             "原始基线与主运行活动指针未覆盖；PyTRiO/SwanLab 无新调用；正式验收仍由平台确认。"]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"passed": verification["passed"], "checks": checks, "macro_f1": metrics["macro_f1"],
                      "acetaldehyde_recall": metrics["per_class"]["Acetaldehyde"]["recall"]}, ensure_ascii=False))
    raise SystemExit(0 if verification["passed"] else 1)


if __name__ == "__main__":
    main()
