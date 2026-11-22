"""核验实际续训产物和服务方用量，不发起新的模型调用。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from gas_a.data import load_pair, load_model, evaluate
from gas_a.io import digest, history, read_json, stamp, write_json, writer_active
from gas_a.model_records import collect_model_calls, expand_receipts
from gas_a.optimize import fit_weighted
from gas_a.pipeline import load_config
from gas_a.review import usage_totals


def planning_evidence_matches(planning, parent, train, validation):
    """独立复算业务视图，兼容旧事实格式；不依赖提示词生成器自证。"""
    evidence = planning.get("evidence", {})
    if planning.get("prompt_version") not in ("gas-a.business-planning.v1", "gas-a.configuration-planning.v2"):
        facts = evidence.get("facts", {})
        return facts.get("F", {}).get("sha256") == validation["sha256"] and len(facts.get("C", {}).get("classes", [])) == 6
    try:
        measured = {"current": evaluate(parent, validation), "fixed": evaluate(parent, validation), "train": evaluate(parent, train)}
        rows = [dict(zip(evidence["class_columns"], values)) for values in evidence["class_rows"]]
        if len(rows) != 6 or [r["gas"] for r in rows] != parent.meta["classes"]:
            return False
        for row in rows:
            for scope, metrics in measured.items():
                actual = metrics["per_class"][row["gas"]]
                for name in ("support", "recall", "f1") + (("precision",) if scope == "current" else ()):
                    if row[scope + "_" + name] != round(actual[name], 6):
                        return False
        return True
    except (ValueError, KeyError, TypeError):
        return False


def verify(config_path, expected):
    started, cpu_started = time.monotonic(), time.process_time()
    config = load_config(config_path)
    root = Path(config["runtime"])
    train, validation = load_pair(config["train"], config["validation"])
    original = read_json(ROOT / "data/official_a/training.json")
    baseline = load_model(config["baseline"], train["paths"])
    baseline_metrics = evaluate(baseline, validation)
    registry = read_json(root / "registry.json")["versions"]
    rows = history(root / "history.jsonl")
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    check("original_baseline_sha256", digest(config["baseline"]) == original["checkpoint_sha256"])
    check("original_train_sha256", train["sha256"] == original["train_sha256"])
    check("original_validation_sha256", validation["sha256"] == original["validation_sha256"])
    check("isolated_sample_ids", not set(train["ids"]) & set(validation["ids"]))
    check("expected_completed_rounds", len(rows) == expected)
    check("sequential_round_numbers", [r["round"] for r in rows] == list(range(1, expected + 1)))
    check("no_active_writer", not writer_active(root))
    selected, previous_score, known_history, provider_ids = "baseline", baseline_metrics["macro_f1"], [], []
    candidate_count = 0
    for row in rows:
        prefix = f"round_{row['round']:02d}/"
        check(prefix + "actual_selected_parent", row["parent_artifact_id"] == selected)
        check(prefix + "parent_digest", row["parent_checkpoint_sha256"] == registry[selected]["sha256"])
        check(prefix + "history_ids", row["history_ids"] == known_history)
        receipts = expand_receipts(row, root)
        attempts = row.get("selection_attempts", [])
        expected_calls = 1 + len(attempts) if attempts else (1 if row.get("selection_mode") == "single_eligible" else 2)
        check(prefix + "actual_model_call_count", len(receipts) == expected_calls)
        ids = [c["call_id"] for c in receipts]
        provider_ids.extend(ids)
        check(prefix + "real_provider_receipts", all(c["receipt_origin"] == "pytrio_openai_api_not_contest_host" and
                                                    c["call_id"] and not c["call_id"].startswith("fixture") for c in receipts))
        check(prefix + "call_ids_match", row["model_call_ids"] == ids)
        planning = json.loads(receipts[0]["messages"][0]["content"])
        if "history" in planning:
            check(prefix + "actual_prompt_history", [r["id"] for r in planning["history"]] == known_history[-5:])
        else:
            parent_for_evidence = load_model(root / registry[selected]["path"], train["paths"])
            check(prefix + "measured_planning_evidence", planning_evidence_matches(planning, parent_for_evidence, train, validation))
        check(prefix + "candidate_budget", len(row["experiments"]) <= planning["budget"]["max_experiments"])
        parent_path = root / registry[selected]["path"]
        parent = load_model(parent_path, train["paths"])
        for trial in row["experiments"]:
            label = prefix + trial["name"] + "/"
            report = trial.get("report")
            check(label + "experiment_completed", report is not None and not trial.get("error"))
            if report is None or not trial.get("host_receipt"):
                continue
            candidate_count += 1
            identity = trial["host_receipt"]["artifact_id"]
            checkpoint = root / registry[identity]["path"]
            candidate = load_model(checkpoint, train["paths"])
            directory = checkpoint.parent
            check(label + "training_file_kit", all((directory / name).is_file() for name in ("training.json", "history.jsonl", "checkpoint.npz")))
            training = read_json(directory / "training.json")
            epochs = history(directory / "history.jsonl")
            check(label + "correct_parent", report["parent_checkpoint_sha256"] == training["parent_checkpoint_sha256"] == digest(parent_path))
            check(label + "checkpoint_digest", digest(checkpoint) == report["checkpoint_sha256"] == training["checkpoint_sha256"] == registry[identity]["sha256"])
            check(label + "input_file_digests", training["train_sha256"] == train["sha256"] and training["validation_sha256"] == validation["sha256"])
            check(label + "source_digests", all(digest(directory / "source" / name if (directory / "source" / name).is_file()
                                                    else ROOT / "src/gas_a" / name) == value
                                               for name, value in training["source_sha256"].items()))
            check(label + "architecture_and_seed", all(candidate.meta[k] == parent.meta[k] for k in ("paths", "task", "profile", "classes", "hidden", "seed", "target_mean", "target_scale")))
            check(label + "parent_standardizer", np.array_equal(parent.mean, candidate.mean) and np.array_equal(parent.scale, candidate.scale))
            check(label + "initial_counters", training["initial_steps"] == parent.meta["steps"] and training["initial_epochs"] == parent.meta["epochs"])
            check(label + "increasing_counters", training["final_steps"] == candidate.meta["steps"] > parent.meta["steps"] and training["final_epochs"] == candidate.meta["epochs"] > parent.meta["epochs"])
            selection = report.get("checkpoint_selection", {})
            check(label + "epoch_count", len(epochs) == selection.get("executed_epoch_count", report["configuration"]["epochs"])
                  and 0 < len(epochs) <= report["configuration"]["epochs"])
            measured = evaluate(candidate, validation)
            check(label + "independent_exported_metrics", all(abs(measured[k] - report["validation_after"][k]) <= 1e-12 for k in ("macro_f1", "accuracy")))
            probe = load_model(parent_path, train["paths"])
            anchor = {k: v.copy() for k, v in probe.params.items()}
            selected_tensors = None
            epoch_metrics_match = True
            for offset, epoch in enumerate(epochs):
                fit_weighted(probe, train, report["configuration"], report["weak_class"], time.monotonic() + 30,
                             epoch_offset=offset, epochs_to_run=1, anchor_params=anchor)
                current = evaluate(probe, validation)
                epoch_metrics_match &= epoch["epoch"] == probe.meta["epochs"] and epoch["step"] == probe.meta["steps"]
                epoch_metrics_match &= all(abs(current[k] - epoch["validation"][k]) <= 1e-12 for k in ("macro_f1", "accuracy"))
                if probe.meta["epochs"] == candidate.meta["epochs"]:
                    selected_tensors = {group: {k: v.copy() for k, v in getattr(probe, group).items()}
                                        for group in ("params", "first", "second")}
            check(label + "per_epoch_metrics", epoch_metrics_match)
            for group in ("params", "first", "second"):
                check(label + "reproduced_" + group, selected_tensors is not None and
                      all(np.allclose(selected_tensors[group][k], getattr(candidate, group)[k], rtol=0, atol=1e-12)
                          for k in getattr(candidate, group)))
            check(label + "actual_resource_measurement", training["resources"]["wall_seconds"] >= 0 and training["resources"]["cpu_seconds"] >= 0)
        selected = row["selected_version"]
        measured = evaluate(load_model(root / registry[selected]["path"]), validation)
        check(prefix + "selected_metrics", all(abs(measured[k] - row["after"][k]) <= 1e-12 for k in ("macro_f1", "accuracy")))
        check(prefix + "static_baseline_floor", measured["macro_f1"] + 1e-12 >= baseline_metrics["macro_f1"])
        check(prefix + "previous_version_floor", measured["macro_f1"] + 1e-12 >= previous_score)
        previous_score = measured["macro_f1"]
        known_history.append(row["id"])
    check("unique_provider_call_ids", len(provider_ids) == len(set(provider_ids)))
    check("active_is_final_selection", read_json(root / "active.json")["version"] == selected)
    calls = collect_model_calls(root)
    return {"format": "gas-a.real-run-verification.v1", "created_at": stamp(), "passed": all(c["passed"] for c in checks),
            "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks), "checks": checks,
            "completed_rounds": len(rows), "candidates": candidate_count, "selected": selected,
            "baseline_macro_f1": baseline_metrics["macro_f1"], "final_macro_f1": previous_score,
            "all_attempts_costs": usage_totals(calls), "official_competition_result": False,
            "verification_resources": {"wall_seconds": time.monotonic() - started,
                                       "cpu_seconds": time.process_time() - cpu_started, "external_model_calls": 0},
            "scope": "实际公开数据续训产物及真实模型回执核验；重放检查不部署、不调用外部模型，也不计为新候选实验。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/a.live.json"))
    parser.add_argument("--expected", type=int, default=10)
    args = parser.parse_args()
    result = verify(args.config, args.expected)
    output = Path(load_config(args.config)["runtime"]) / "verification/continuation.json"
    write_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
