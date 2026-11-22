"""Paired real-provider prompt probes on frozen public-data evidence; no tuning on test labels."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from gas_a.cloud_records import Audit
from gas_a.contest import parse_object
from gas_a.data import load_model
from gas_a.io import digest, write_json
from gas_a.live_access import PyTrioChat, load_credentials
from gas_a.model_records import recorded_chat, call_path, call_reference
from gas_a.pipeline import load_config
from gas_a.planning import build_prompt, iteration_brief, mechanism_tags, resolve_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-rounds", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--after-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = Path(args.cases)
    load_credentials(ROOT / ".env.local")
    client = PyTrioChat(model="Qwen/Qwen3.8-27B", timeout=90)
    config = load_config(ROOT / "configs/a.json")
    model = load_model(config["baseline"])
    audit = Audit(out / "audit", cloud=True)
    audit.start({"mode": "paired_prompt_probe", "cases": args.case_rounds, "max_output_tokens": 768,
                 "model": client.model, "official_competition_result": False,
                 "scope": "One response per version per frozen case; behavior comparison, not a performance score."})
    successful, rows = False, []
    try:
        for file in sorted(cases.glob("case-*.json")):
            case = json.loads(file.read_text(encoding="utf-8"))
            if case["completed_rounds"] not in args.case_rounds:
                continue
            before, past, evidence = case["before_prompt"], case["past"], case["evidence"]
            after = build_prompt(config, model, config["experiments"], evidence, 2, 30,
                                 mode="sdk", index=case["completed_rounds"]+1, past=past, parent_sha=case["parent_sha256"])
            memory = iteration_brief(config, model, evidence, past, case["parent_sha256"])
            old_mechanisms = {tuple(group["mechanisms"]) for group in memory["same_parent_same_public_data"]}
            variants = [("after_v3", after)] if args.after_only else [("before_v2", before), ("after_v3", after)]
            for version, prompt in variants:
                label = f"case-{case['completed_rounds']:02d}-{version}"
                write_json(out / (label+"-prompt.json"), prompt)
                messages = [{"role": "system", "content": "本次为本地提示词对照实验，非官方评测。只依据公开训练/验证统计和已完成实验设计配置。按用户消息契约返回简短JSON，不使用隐藏标签。"},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]
                print(json.dumps({"stage": "calling_real_27b", "case": case["completed_rounds"], "version": version}), flush=True)
                receipt = recorded_chat(client, out / "model-calls", messages, max_tokens=768, temperature=0,
                    enable_thinking=False, purpose="experiment_planning",
                    context={"completed_rounds": case["completed_rounds"], "version": version})
                # record_id is carried in the returned reference from capture_call.
                references = sorted((out / "model-calls").glob("call-*.json"), key=lambda p: p.stat().st_mtime_ns)
                reference = {"file": references[-1].relative_to(out).as_posix(), "sha256": digest(references[-1])}
                audit.record("actual_provider_probe", {"case": case["completed_rounds"], "version": version,
                                                       "receipt": receipt, "record": json.loads(references[-1].read_text(encoding="utf-8"))})
                error, plan, specs = None, None, []
                try:
                    raw = parse_object(receipt["text"])
                    plan, specs = resolve_plan(raw, config, model, [], evidence, 2, past,
                                              case["parent_sha256"], weak_class="Acetaldehyde")
                except (ValueError, KeyError, TypeError) as exc:
                    error = str(exc)
                tags = [mechanism_tags(spec, model) for spec in specs]
                row = {"case": case["completed_rounds"], "version": version, "valid_first_reply": error is None,
                       "prompt_version": prompt["prompt_version"],
                       "error": error, "candidate_count": len(specs), "mechanisms": tags,
                       "distinct_mechanism_sets": len({tuple(tag) for tag in tags}),
                       "new_vs_matched_history": sum(tuple(tag) not in old_mechanisms for tag in tags),
                       "iteration_note_present": bool(plan and plan.get("iteration")), "configurations": specs,
                       "plan": plan, "call_id": receipt["call_id"], "usage": receipt.get("usage", {}),
                       "provider_seconds": receipt["seconds"], "record": reference}
                rows.append(row)
                write_json(out / (label+"-result.json"), row)
                print(json.dumps({key: row[key] for key in ("case", "version", "valid_first_reply", "candidate_count", "mechanisms", "iteration_note_present", "error")}, ensure_ascii=False), flush=True)
        summary = {"mode": "real_27b_paired_prompt_behavior_probe", "rows": rows,
                   "real_calls": len(rows), "tokens": sum(row["usage"].get("total_tokens", 0) for row in rows),
                   "one_response_per_case_version": True, "performance_improvement_proven": False,
                   "planning_source_sha256": digest(ROOT / "src/gas_a/planning.py")}
        write_json(out / "summary.json", summary)
        successful = True
    finally:
        cloud = audit.finish(successful)
        if successful:
            summary["cloud"] = cloud
            write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main()
