"""基于完整且未改写的实验证据生成精简报告。"""
import json
import math
from pathlib import Path
from .io import history, read_json, stamp, write_json


def metrics_brief(metrics):
    result = {key: metrics[key] for key in ("macro_f1", "accuracy") if key in metrics}
    weak = metrics.get("per_class", {}).get("Acetaldehyde")
    if weak is not None:
        result.update(acetaldehyde_recall=weak["recall"], acetaldehyde_precision=weak["precision"])
    return result


def trial_brief(trial):
    report = trial.get("report", {})
    gate = trial.get("local_gate", trial.get("gate", {}))
    resources = trial.get("resources") or report.get("resources", {})
    if not resources and report.get("training_seconds") is not None:
        resources = {"wall_seconds": report["training_seconds"], "cpu_seconds": None,
                     "scope": "旧记录中的训练及导出耗时；CPU 时长未记录"}
    return {"name": trial["name"], "configuration": report.get("configuration"),
            "metrics": metrics_brief(report.get("validation_after", {})),
            "gain": (report["validation_after"]["macro_f1"] - report["validation_before"]["macro_f1"])
                    if report else None,
            "gate": gate, "host_gate_passed": trial.get("host_gate_passed"),
            "validation_comparison": trial.get("validation_comparison"), "eligible": trial.get("eligible"),
            "checkpoint_selection": report.get("checkpoint_selection"),
            "terminal_macro_f1": report.get("terminal_validation", {}).get("macro_f1"),
            "resources": resources, "error": trial.get("error")}


def planning_context(metrics, diagnosis, historical):
    weak = diagnosis["weakest_class"]
    return {"parent_metrics": metrics_brief(metrics),
            "diagnosis": {"weakest_class": weak, "weakest_class_metrics": diagnosis["weakest_class_metrics"],
                          "top_confusions": diagnosis["top_confusions"][:3],
                          "train_class_counts": diagnosis["train_class_counts"],
                          "class_prior_total_variation": diagnosis["class_prior_total_variation"],
                          "fit_comparison": diagnosis.get("fit_comparison", {}),
                          "top_sensors": [{"sensor": s["sensor"], "mean_psi": round(s["mean_psi"], 4)}
                                          for s in diagnosis["top_sensors"][:3]]},
            "history": [{key: r[key] for key in ("id", "round", "macro_f1", "accuracy", "selected_version") if key in r}
                        for r in historical[-5:]]}


def usage_totals(calls):
    """服务方未提供的 usage 保持未知状态，失败尝试也按此处理。"""
    totals = {"calls": len(calls), "prompt_tokens_known": 0, "completion_tokens_known": 0,
              "unknown_usage_calls": 0, "unsuccessful_calls": 0}
    for call in calls:
        usage = call.get("usage") or {}
        complete = True
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if type(value) is int and value >= 0:
                totals[key + "_known"] += value
            else:
                complete = False
        totals["unknown_usage_calls"] += not complete
        totals["unsuccessful_calls"] += call.get("status", "success") != "success"
    totals["tokens_known"] = totals["prompt_tokens_known"] + totals["completion_tokens_known"]
    totals["tokens_total"] = totals["tokens_known"] if not totals["unknown_usage_calls"] else None
    return totals


def round_summary(record, calls=()):
    """从单轮证据生成结果与成本摘要，服务方未提供的用量保持未知。"""
    calls = list(calls)
    before, after = record.get("before", {}), record.get("after", {})
    baseline = record.get("static_baseline", {})
    trials = [trial_brief(t) for t in record.get("experiments", [])]
    gain = after.get("macro_f1", 0) - before.get("macro_f1", 0)
    costs = usage_totals(calls)
    plan = record.get("plan")
    if plan is None:
        for call in calls:
            if call.get("purpose") != "experiment_planning":
                continue
            try:
                reply = json.loads(call.get("reply_text") or call.get("text") or "")
            except (ValueError, TypeError):
                continue
            if isinstance(reply, dict):
                names = reply.get("experiment_names", [])
                names = names if isinstance(names, list) and all(isinstance(n, str) for n in names) else []
                plan = {"experiments": names, "reason": reply.get("reason") if isinstance(reply.get("reason"), str) else None,
                        "source": "原始模型回复"}
                break
    resources = [t["resources"] for t in trials]
    costs.update(candidate_wall_seconds_known=sum(r.get("wall_seconds") or 0 for r in resources),
                 candidate_cpu_seconds_known=sum(r.get("cpu_seconds") or 0 for r in resources),
                 candidates_with_unknown_resources=sum(r.get("wall_seconds") is None or r.get("cpu_seconds") is None for r in resources),
                 gpu_seconds_platform=None, training_credits_platform=None, new_label_requests=0)
    costs["tokens_per_f1_gain"] = costs["tokens_total"] / gain if gain > 0 and costs["tokens_total"] is not None else None
    if record.get("recoveries"):
        plan = {**record.get("plan", {}), "source": "程序恢复后的执行方案；原始回复见模型交互"}
    return {"format": "gas-a.round-review.v1", "round": record["round"], "status": "completed",
            "scope": "公开验证集开发结果；封闭评测与正式评分由平台确认",
            "before": metrics_brief(before), "after": metrics_brief(after), "gain": gain,
            "baseline": metrics_brief(baseline),
            "baseline_floor_passed": after.get("macro_f1", -1) >= baseline.get("macro_f1", math.inf),
            "selected": record.get("selected_version", record.get("selected_artifact_id")),
            "action": record.get("action"), "reason": record.get("reason"),
            "plan": plan, "selection_mode": record.get("selection_mode", "model"),
            "fallback": bool(record.get("model_plan_parse_error") or record.get("selection_parse_error")),
            "recoveries": record.get("recoveries", []),
            "costs": costs, "experiments": trials,
            "evidence": {"parent_checkpoint_sha256": record.get("parent_checkpoint_sha256"),
                         "model_call_ids": record.get("model_call_ids", []),
                         "history_ids": record.get("history_ids", [])}}


def round_text(summary):
    def number(value):
        return "未提供" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    costs = summary["costs"]
    actions = {"promote": "采用新候选", "retain": "保留当前模型", "rollback": "回滚历史模型"}
    modes = {"model": "模型选择", "model_retry": "模型纠正后选择", "program_recovery": "程序按实测合格集合恢复选择",
             "model_authorized_parent_fallback": "按模型显式授权保留父版本",
             "single_eligible": "唯一合格版本，直接选择（历史策略）", "rule_controller": "规则控制器"}
    reasons = {"macro_f1_gain_below_minimum": "Macro-F1 增益不足", "below_static_baseline": "低于静态基线",
               "accuracy_regression": "准确率下降超限", "quality_gates_passed": "全部质量门槛通过", "invalid_metrics": "指标无效"}
    plan = summary.get("plan") or {}
    plan_reason = plan.get("reason") or "原始理由见本轮下方的模型交互"
    experiment_names = [item.get("name", "未命名") if isinstance(item, dict) else str(item)
                        for item in plan.get("experiments", [])]
    if not any("\u4e00" <= c <= "\u9fff" for c in plan_reason):
        plan_reason = "原始英文理由见本轮下方的模型交互"
    decision = summary.get("reason") or "未记录"
    if not any("\u4e00" <= c <= "\u9fff" for c in decision):
        decision = {"promote": "采用通过门槛的新候选，实际增益见结果行。", "retain": "保留当前合格版本，本轮候选结果见实验行。",
                    "rollback": "恢复合格的历史版本。"}.get(summary.get("action"), "原始英文理由见本轮下方的模型交互")
    lines = [f"第 {summary['round']} 轮｜已完成｜{actions.get(summary.get('action'), summary.get('action'))}",
             f"结果：Macro-F1 {number(summary['before'].get('macro_f1'))} → {number(summary['after'].get('macro_f1'))}，增益 {summary['gain']:+.6f}",
             f"静态基线：{number(summary['baseline'].get('macro_f1'))}；静态基线下限：{'通过' if summary['baseline_floor_passed'] else '未通过'}",
             f"模型建议：{', '.join(experiment_names) or ('观察，不训练' if plan.get('action') == 'observe' else '未执行实验')}；{plan_reason}",
             f"决策：{summary.get('selected')}；方式：{modes.get(summary.get('selection_mode'), summary.get('selection_mode'))}；异常回退：{'是' if summary.get('fallback') else '否'}",
             f"结果说明：{decision}",
             f"成本：模型调用 {costs['calls']} 次，已知 Token {costs['tokens_known']}，用量不完整 {costs['unknown_usage_calls']} 次；"
             f"候选耗时已知值 {costs['candidate_wall_seconds_known']:.3f} 秒，CPU 已知值 {costs['candidate_cpu_seconds_known']:.3f} 秒，资源不完整 {costs['candidates_with_unknown_resources']} 个实验，新增标注请求 0 次",
             "平台 GPU 时长、训练额度：未提供；失败调用和未采纳实验均计入成本。"]
    if "acetaldehyde_recall" in summary["after"]:
        lines.insert(3, f"乙醛：召回 {number(summary['before'].get('acetaldehyde_recall'))} → {number(summary['after']['acetaldehyde_recall'])}；"
                       f"精确率 {number(summary['after'].get('acetaldehyde_precision'))}")
    if summary["gain"] <= 0:
        lines.append("本轮没有正向性能增益，单位增益成本不计算。")
    for trial in summary["experiments"]:
        why = "；".join(reasons.get(r, "类别召回下降超限：" + r.split(":", 1)[1] if r.startswith("class_recall_regression:") else r)
                        for r in trial["gate"].get("reasons", []))
        lines.append(f"  实验 {trial['name']}：F1={number(trial['metrics'].get('macro_f1'))}，"
                     f"增益={number(trial['gain'])}，门槛={'通过' if trial['gate'].get('accepted') else '未通过'}（{why}），异常={trial['error'] or '无'}")
    lines.append(summary["scope"])
    return "\n".join(lines) + "\n"


def write_round_review(directory, record, calls=()):
    summary = round_summary(record, calls)
    directory = Path(directory)
    write_json(directory / "round-summary.json", summary)
    (directory / "round-summary.txt").write_text(round_text(summary), encoding="utf-8")
    return summary


def export_review(source, output):
    """对原始证据只读，不调用模型或云服务。"""
    from .model_records import collect_model_calls, purpose_for, transcript, expand_receipts
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("Review destination must not contain or replace source evidence")
    rows = history(source / "history.jsonl")
    calls = collect_model_calls(source, exclude=output)
    if not calls:
        # 旧版运行可能仅在 history 中保留实际请求和回复。
        for row in rows:
            for index, receipt in enumerate(expand_receipts(row, source)):
                text = receipt.get("text") or ""
                calls.append({"record_id": receipt.get("call_id") or f"history-{row['round']}-{index}",
                              "provider_call_id": receipt.get("call_id"), "purpose": purpose_for(receipt.get("messages", [])),
                              "context": {"round": row["round"], "session": row.get("id", "").rsplit("-", 1)[-1]},
                              "request": {"messages": receipt.get("messages", [])}, "reply_text": text,
                              "response": receipt.get("response"), "usage": receipt.get("usage"),
                              "status": "success" if text.strip() else "empty_response",
                              "evidence_file": str(source / "history.jsonl")})
    summaries = []
    round_calls = []
    for row in rows:
        matching = [c for c in calls if c.get("context", {}).get("round") == row["round"]
                    and (not row.get("id") or c.get("context", {}).get("session", "") == row["id"].rsplit("-", 1)[-1])]
        round_calls.append(matching)
        summaries.append(round_summary(row, matching))
    costs = usage_totals(calls)
    costs.update(candidate_wall_seconds_known=sum(s["costs"]["candidate_wall_seconds_known"] for s in summaries),
                 candidate_cpu_seconds_known=sum(s["costs"]["candidate_cpu_seconds_known"] for s in summaries),
                 candidates_with_unknown_resources=sum(s["costs"]["candidates_with_unknown_resources"] for s in summaries),
                 gpu_seconds_platform=None, training_credits_platform=None, new_label_requests=0)
    gain = summaries[-1]["after"].get("macro_f1", 0) - summaries[0]["baseline"].get("macro_f1", 0) if summaries else None
    costs["tokens_per_f1_gain"] = costs["tokens_total"] / gain if gain and gain > 0 and costs["tokens_total"] is not None else None
    failed_sessions = []
    for event_path in sorted(source.glob("sessions/*/events.jsonl")):
        failed_sessions.extend({"session": event_path.parent.name, "error": e["payload"].get("error")}
                               for e in history(event_path) if e.get("event") == "session_failed")
    versions = read_json(source / "registry.json").get("versions", {}) if (source / "registry.json").is_file() else {}
    file_audit = []
    for row in rows:
        for trial in row.get("experiments", []):
            if not trial.get("report"):
                continue
            identity = trial.get("host_receipt", {}).get("artifact_id")
            if identity in versions:
                directory = (source / versions[identity]["path"]).parent.resolve()
            elif trial.get("evidence_directory"):
                directory = Path(trial["evidence_directory"]).resolve()
            else:
                directory = source / "rounds" / f"round-{row['round']:03d}" / trial["name"]
            expected = ("training.json", "history.jsonl", "checkpoint.npz")
            missing = [name for name in expected if not directory.is_relative_to(source) or not (directory / name).is_file()]
            file_audit.append({"round": row["round"], "experiment": trial["name"], "missing_files": missing})
    training_audit = {"scope": "仅核对续训指南三件套是否存在，不代表正式实验规范验收。",
                      "complete_candidates": sum(not item["missing_files"] for item in file_audit),
                      "incomplete_candidates": sum(bool(item["missing_files"]) for item in file_audit),
                      "details": [item for item in file_audit if item["missing_files"]]}
    report = {"format": "gas-a.scoring-review.v1", "created_at": stamp(), "rounds_completed": len(rows),
              "local_baseline_floor_passed": all(s["baseline_floor_passed"] for s in summaries) if summaries else None,
              "gain_vs_static_baseline": gain, "costs": costs, "failed_sessions": failed_sessions,
              "training_file_audit": training_audit,
              "closed_evaluation_verified": False, "official_score": None,
              "cost_scope": "源目录内全部调用尝试与实验成本；正式运行与获准重跑须使用独立输出目录，分别核算。",
              "rerun_policy": "平台原因失败后可申请一次重跑，申请及两次取高由平台处理；本程序不自动重跑正式评测。",
              "review_dimensions": {"方法设计与技术路线": "每轮问题、模型建议、实际参数与对照结果均在本报告中。",
                                    "实验规范与分析深度": "保留父检查点、数据和代码摘要、逐 epoch 记录、失败及未采纳实验；源目录为完整证据。",
                                    "安全可控与工业落地": "只采用通过本地与宿主门槛的候选，异常选择优先保留合格父模型；封闭综合门槛仍由平台验证。",
                                    "报告与答辩": "本报告统一展示逐轮结果、模型交互、成本和缺项；不生成虚构专家评分。"},
              "recording_scope": "SwanLab 推荐记录；完整原始调用、训练和评估文件保留。正式实验规范与封闭门槛需平台验收。",
              "source": str(source), "external_model_calls": 0}
    output.mkdir(parents=True, exist_ok=True)
    header = ["实验与模型调用报告", "本报告汇总全部结果、模型交互、成本、失败与记录缺项；正式评分由平台确认。",
              f"原始证据目录：{source}", "", "【总体结果与成本】",
              f"已完成轮次：{len(rows)}；已记录失败会话：{len(failed_sessions)}",
              f"相对静态基线增益：{gain}",
              f"全部调用尝试：{costs['calls']}；已知输入 Token：{costs['prompt_tokens_known']}；已知输出 Token：{costs['completion_tokens_known']}；用量缺失：{costs['unknown_usage_calls']} 次",
              f"候选耗时已知值：{costs['candidate_wall_seconds_known']:.3f} 秒；缺少资源计量的候选：{costs['candidates_with_unknown_resources']} 个",
              f"续训三件套检查：完整 {training_audit['complete_candidates']} 个候选，缺项 {training_audit['incomplete_candidates']} 个；缺项逐条列在本报告末尾。",
              "平台 GPU 时长和训练额度：未提供；程序新增标注请求：0。",
              "单位增益 Token 成本：" + (f"{costs['tokens_per_f1_gain']:.3f}" if costs["tokens_per_f1_gain"] is not None
                                      else "未计算（无正向收益或存在未知用量）"),
              "封闭评测基础门槛与正式评分：尚未由平台确认。", ""]
    sections = ["\n".join(header)]
    verification_path = source / "verification/continuation.json"
    if rows and verification_path.is_file():
        verified = read_json(verification_path)
        if verified.get("completed_rounds") == len(rows) and verified.get("selected") == summaries[-1]["selected"]:
            sections.append("【本次实际运行与验收】")
            sections.append(f"实际完成 {len(rows)} 轮、{verified.get('candidates')} 个续训候选；产物核验 {verified.get('checks_passed')}/{verified.get('checks_total')} 项通过。")
            session = rows[-1].get("id", "").rsplit("-", 1)[-1]
            run_rows = [row for row in rows if row.get("id", "").endswith("-" + session)]
            run_receipts = [receipt for row in run_rows for receipt in expand_receipts(row, source)]
            run_cost = usage_totals(run_receipts)
            sections.append(f"完成会话：{len(run_rows)} 轮，模型调用 {len(run_receipts)} 次，输入 {run_cost['prompt_tokens_known']} Token、输出 {run_cost['completion_tokens_known']} Token，共 {run_cost['tokens_known']} Token。前期探测与超时另计在总体成本中。")
            live_path = source / "live_summary.json"
            if live_path.is_file():
                live = read_json(live_path)
                sections.append(f"调优控制模型：{live.get('model')}；工业模型：原始 128→32→6 CPU NumPy MLP 的续训检查点。")
                sections.append(f"当前版本：{live['active_version']}；Macro-F1={live['metrics']['macro_f1']:.6f}，Accuracy={live['metrics']['accuracy']:.6f}。")
                weak = live["metrics"].get("per_class", {}).get("Acetaldehyde", {})
                if weak.get("recall") == 0:
                    sections.append("分析限制：乙醛召回率仍为 0，尚未解决该弱类的识别问题；不能据此宣称已完成全部漂移补偿。")
                sections.append("主实验云端地址：" + str(live.get("cloud", {}).get("experiment_url")))
            for name, title in (("cloud-readback.json", "云端读回"), ("prediction.json", "CLI 与 HTTP 预测"),
                                ("archive-readback.json", "诊断与验收归档读回")):
                path = source / "verification" / name
                if path.is_file():
                    value = read_json(path)
                    sections.append(f"{title}：{'通过' if value.get('passed') else '未通过'}；"
                                    f"核验项 {value.get('checks_passed', '-')}/{value.get('checks_total', '-')}。")
            archive_path = source / "verification/archive-cloud.json"
            if archive_path.is_file():
                sections.append("诊断、失败尝试和验收证据云端地址：" + str(read_json(archive_path)["cloud"].get("experiment_url")))
            sections.append("验收范围：本地真实端到端联调；正式封闭评测的综合指标、模型白名单、训练额度和正式评分仍由赛事平台确认。")
    sections.append("【逐轮结果与模型交互】")
    assigned = set()
    for summary, matching in zip(summaries, round_calls):
        sections.append(round_text(summary))
        for call in matching:
            assigned.add(call.get("record_id"))
            sections.append(transcript(call, full=True))
    other_calls = [call for call in calls if call.get("record_id") not in assigned]
    if other_calls:
        sections.append("【未完成轮次与其它调用：同样计入总成本】")
        sections.extend(transcript(call, full=True) for call in other_calls)
    sections.append("【失败会话与记录缺项】")
    if not failed_sessions and not training_audit["details"]:
        sections.append("未发现已记录的失败会话或续训三件套缺项。")
    for failed in failed_sessions:
        sections.append(f"失败会话 {failed['session']}：{failed['error']}")
    for item in training_audit["details"]:
        sections.append(f"第 {item['round']} 轮 / {item['experiment']}：缺少 {', '.join(item['missing_files'])}")
    sections.append("\n【评分材料与重跑说明】")
    sections.extend(f"{name}：{description}" for name, description in report["review_dimensions"].items())
    sections.extend([report["cost_scope"], report["rerun_policy"], report["recording_scope"]])
    report_path = output / "report.txt"
    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return {"report": str(report_path), "rounds": len(rows), "calls": len(calls), "external_model_calls": 0}
