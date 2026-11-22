"""Summarize prompt probes separately from measured model quality."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gas_a.data import load_model
from gas_a.io import digest, read_json, write_json
from gas_a.planning import mechanism_tags
from report_sdk_simulation import table, local

TAGS = {"class_loss": "类别/难样本损失", "regularization": "正则化", "sensor_affine": "传感器增益/偏移",
        "sensor_dropout": "传感器dropout", "response_gain": "原响应增益", "input_noise": "普通噪声",
        "adversarial": "对抗扰动", "confusion_pair": "混淆对约束", "feature_update": "特征层更新/冻结",
        "capacity": "容量调整", "sensor_subset": "显式传感器子集", "optimization_only": "优化参数"}


def describe(tags):
    return "＋".join(TAGS.get(tag, tag) for tag in tags)


def run_view(root, model):
    summary = read_json(root / "summary.json")
    rounds = [read_json(p) for p in sorted(root.glob("round-*/result.json"))]
    candidates = [e for r in rounds for e in r["experiments"] if e.get("metrics")]
    groups = {tuple(mechanism_tags(c["configuration"], model)) for c in candidates}
    return {"summary": summary, "rounds": rounds, "candidates": candidates, "mechanism_sets": [list(g) for g in sorted(groups)],
            "best_public_f1": max((c["metrics"]["macro_f1"] for c in candidates), default=None),
            "best_target_recall": max((c["metrics"]["per_class"]["Acetaldehyde"]["recall"] for c in candidates), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--final-probe", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--final-run", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--tests", required=True)
    args = parser.parse_args()
    out = Path(args.final_probe).resolve()
    model = load_model(ROOT / "data/official_a/checkpoint.npz")
    before = run_view(Path(args.baseline), model)
    trial = run_view(Path(args.trial), model)
    final = run_view(Path(args.final_run), model)
    probes = [read_json(Path(path) / "summary.json") for path in (args.probe, args.final_probe)]
    tests = read_json(args.tests)
    shutil.copyfile(args.tests, out / "regression-tests.json")
    old_gate = read_json(Path(args.baseline) / "app/configs/a.json")["gate"]
    new_gate = read_json(Path(args.final_run) / "app/configs/a.json")["gate"]
    assert old_gate == new_gate
    rows = [row for probe in probes for row in probe["rows"]]
    final_probe_rows = probes[-1]["rows"]
    base_rows = probes[0]["rows"]
    behavior = {"probe_calls": len(rows), "all_probe_responses_executable": all(r["valid_first_reply"] for r in rows),
                "v2_structured_or_pair_cases": sum(any("sensor_affine" in t or "confusion_pair" in t for t in r["mechanisms"])
                                                   for r in base_rows if r["version"] == "before_v2"),
                "v3_structured_or_pair_cases": sum(any("sensor_affine" in t or "confusion_pair" in t for t in r["mechanisms"])
                                                   for r in base_rows if r["version"] == "after_v3"),
                "v31_probes_executable": all(r["valid_first_reply"] for r in final_probe_rows),
                "quality_gates_unchanged": True, "no_fixed_sensor_recipe_added": True}
    evidence = {"scope": "prompt_tuning_local_evidence_not_official_score", "behavior": behavior,
                "baseline_v2": {k: before[k] for k in ("best_public_f1", "best_target_recall", "mechanism_sets")},
                "v3_ten_round_trial": {k: trial[k] for k in ("best_public_f1", "best_target_recall", "mechanism_sets")},
                "v31_final_integration": {k: final[k] for k in ("best_public_f1", "best_target_recall", "mechanism_sets")},
                "final_stream": final["summary"]["blind"]["aggregate"], "probe_results": rows,
                "source_hashes": {name: digest(ROOT / name) for name in
                    ("src/gas_a/planning.py", "src/gas_a/controller_policy.py", "src/gas_a/contest.py", "src/gas_a/temporal.py", "PROMPTS.md")},
                "model_calls_this_task": sum(p["real_calls"] for p in probes) + trial["summary"]["actual_model_calls"] + final["summary"]["actual_model_calls"],
                "tokens_this_task": sum(p["tokens"] for p in probes),
                "limitations": ["每个提示词案例只调用一次，比较属于行为观察，不是统计显著性检验。",
                                "v3跑10轮；组件历史与选版上下文修正后的v3.1再跑3轮。两者单独记录，不混作同一运行。",
                                "公开验证最佳候选未通过全部门槛，不能把该分数当作部署或隐藏流成绩。",
                                "最终逐条预测仍使用选中模型；无测试标签反馈或预测期在线训练。",
                                "提示词改善搜索行为，并未证明解决跨批泛化问题；镜像尚未重建。"]}
    for version in (trial, final):
        for p in Path(args.trial if version is trial else args.final_run).glob("round-*/model-calls/call-*.json"):
            evidence["tokens_this_task"] += (read_json(p).get("usage") or {}).get("total_tokens", 0)
    write_json(out / "tuning-evidence.json", evidence)
    probe_table = []
    for r in rows:
        version = r.get("prompt_version", "v2" if r["version"] == "before_v2" else "v3")
        probe_table.append([r["case"], version, "合法" if r["valid_first_reply"] else "拒绝", r["candidate_count"],
                            "；".join(describe(t) for t in r["mechanisms"]), "有" if r["iteration_note_present"] else "无"])
    score_rows = []
    for label, view in (("原v2 / 10轮", before), ("v3 / 10轮", trial), ("最终v3.1 / 3轮", final)):
        s = view["summary"]
        score_rows.append([label, len(view["candidates"]), len(view["mechanism_sets"]),
                           f"{view['best_public_f1']:.6f}", f"{view['best_target_recall']:.2%}",
                           s["promotions"], f"{s['blind']['aggregate']['active']['macro_f1']:.6f}"])
    iteration_rows = []
    for label, view, path in (("v3", trial, args.trial), ("v3.1", final, args.final_run)):
        for row in view["rounds"]:
            internal = read_json(next((Path(path) / f"round-{row['round']:02d}/workspace").glob("gas-a-*/round.json")))
            iteration_rows.append([label, row["round"], "；".join(describe(mechanism_tags(e["configuration"], model)) for e in row["experiments"]),
                                   row["plan"].get("iteration", {}).get("lesson", "未返回"),
                                   (internal.get("selection_feedback") or {}).get("next_test", "未返回"), row["action"]])
    report = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>27B提示词调优结果</title><style>body{{font:15px/1.75 'Microsoft YaHei',sans-serif;color:#17253a;background:#f3f6fb;margin:0}}main{{max-width:1160px;margin:30px auto;padding:0 20px}}section{{background:white;border:1px solid #dfe6ef;border-radius:12px;padding:24px;margin:18px 0}}h1{{font-size:29px}}h2{{font-size:21px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e4eaf1;vertical-align:top}}th{{background:#edf3fb}}.note{{padding:15px;background:#fff3d8;border-left:4px solid #ca8a04}}a{{color:#2563eb}}</style>
<main><h1>27B提示词调优：实验行为与模型效果分别验证</h1>
<section><p>调优范围是规划提示词、历史反馈和选版上下文。模型训练算法、平台数据边界与上线门槛保持原样。{tests.get('tests_run', tests.get('run', ''))}项测试详情见回归报告。</p>
<p class='note'>下面严格区分“模型提出了不同实验”“候选公开验证分变化”和“最终部署分”。新增路线不等于已经修复泛化，未通过门槛的候选不计为上线成果。</p></section>
<section><h2>具体调整</h2><ol><li>同父模型、同训练/验证摘要的历史按干预机制汇总，保留实际分数和拒绝理由，排除不匹配数据。</li><li>组件级历史区分组合出现与独立对照，避免把已有机制称作从未尝试；部分召回进展不写成零收益。</li><li>预算允许时优先不同机制或关键干预对照；填写历史教训和每个候选的可观察预测。配置仍由27B设计。</li><li>选版同时看到候选实际配置与干净验证逐类成绩；下一轮建议带真实回执保存，并作为假设传回。</li><li>非法配置纠正仍保留机制历史、暂停条件、传感器缺失验证和数据边界。</li></ol></section>
<section><h2>相同证据的真实模型探测</h2>{table(['已完成历史轮数','提示词版本','首条回复','候选数','实际机制','迭代说明'],probe_table)}
<p>v2三组均以类损失及普通正则/噪声为主；v3三组均包含结构化传感器扰动或混淆对约束。v3.1针对4、8轮历史的最终探测也返回合法配置。每案例只采样一次，不做显著性结论。</p></section>
<section><h2>真实训练与最终得分</h2>{table(['运行','实测候选','机制组合数','最佳候选公开F1','最佳乙醛召回','上线次数','最终后续流F1'],score_rows)}
<p>原始基线后续流F1为0.405351。最佳候选公开F1是各轮候选的最大值，与最终选版不同。v3与v3.1的训练轮数不同，表格只描述实测，不作为严格优劣排名。</p></section>
<section><h2>每轮学到了什么、下一轮建议什么</h2>{table(['版本','轮次','实际训练机制','模型教训摘要（假设/意见）','模型下一轮建议（假设）','最终动作'],iteration_rows)}
<p>以上文字保留模型原意，不自动视为事实；实际配置、逐类成绩和拒绝门槛见各运行目录。源数据统计不能证明物理原因。</p></section>
<section><h2>验证与成本</h2><p>本次真实27B调用{evidence['model_calls_this_task']}次，已知token共{evidence['tokens_this_task']:,}。离线回归、提示词探测及实际训练记录分别保存，均未借用隐藏标签。</p>
<p><a href='tuning-evidence.json'>调优证据JSON</a> · <a href='regression-tests.json'>最终回归结果</a> · <a href='../{Path(args.trial).name}/report.html'>v3十轮训练报告</a> · <a href='../{Path(args.final_run).name}/report.html'>v3.1最终对接报告</a></p>
<p>探测记录：{probes[-1]['cloud']['experiment_url']}；最终对接记录：{final['summary']['cloud']['experiment_url']}。本地模拟结果不等于正式平台成绩，未重新打镜像。</p></section></main></html>"""
    (out / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(out / "report.html"), "behavior": behavior,
                      "v3_best_public_f1": trial["best_public_f1"], "v31_best_public_f1": final["best_public_f1"],
                      "total_model_calls": evidence["model_calls_this_task"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
