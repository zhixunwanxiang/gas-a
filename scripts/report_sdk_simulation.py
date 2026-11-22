"""Export a standalone Chinese report from committed Docker simulation evidence."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gas_a.io import CLASSES, digest, read_json, write_json

CN = dict(zip(CLASSES, ("乙醇", "乙烯", "氨", "乙醛", "丙酮", "甲苯")))
ACTIONS = {"retain": "保留父模型", "promote": "采用新候选", "rollback": "回滚基线"}


def local(value):
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S")


def table(head, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join("<th>"+html.escape(str(v))+"</th>" for v in head) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join("<td>"+html.escape(str(v))+"</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def lines(labels, series, title):
    w, h, l, r, top, bottom = 920, 320, 60, 35, 40, 55
    x = lambda i: l + i * (w-l-r) / max(1, len(labels)-1)
    y = lambda v: h-bottom-v*(h-top-bottom)
    parts = [f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='{html.escape(title)}'><title>{html.escape(title)}</title>"]
    for v in (0, .25, .5, .75, 1):
        parts.append(f"<path d='M{l},{y(v)}H{w-r}' stroke='#e2e8f0'/><text x='{l-12}' y='{y(v)+4}' text-anchor='end'>{v:.2f}</text>")
    for i, label in enumerate(labels):
        parts.append(f"<text x='{x(i)}' y='{h-24}' text-anchor='middle'>{html.escape(str(label))}</text>")
    for j, (name, values, color) in enumerate(series):
        points = " ".join(f"{x(i)},{y(value)}" for i, value in enumerate(values))
        dash = " stroke-dasharray='6 4'" if j else ""
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3'{dash}/>")
        for i, value in enumerate(values):
            parts.append(f"<circle cx='{x(i)}' cy='{y(value)}' r='{5 if j else 3}' fill='{color}'><title>{html.escape(name)} / {labels[i]}: {value:.6f}</title></circle>")
        parts.append(f"<text x='{l+j*220}' y='20' fill='{color}'>{html.escape(name)}</text>")
    return "".join(parts)+"</svg>"


def export(source, previous):
    root = Path(source).resolve()
    summary = read_json(root / "summary.json")
    blind = summary["blind"]
    rounds = [read_json(p) for p in sorted(root.glob("round-*/result.json"))]
    aggregate = blind["aggregate"]
    active, base = aggregate["active"], aggregate["baseline"]
    calls = [read_json(p) for p in sorted(root.glob("round-*/model-calls/call-*.json"))]
    drill_calls = [read_json(p) for p in sorted(root.glob("drills/*/round-*/model-calls/call-*.json"))]
    prior_calls = [read_json(p) for directory in previous
                   for pattern in ('round-*/model-calls/call-*.json','drills/*/round-*/model-calls/call-*.json')
                   for p in Path(directory).glob(pattern)]
    assert len(calls) == summary["actual_model_calls"]
    tokens = lambda values: sum((v.get("usage") or {}).get("total_tokens", 0) for v in values)
    costs = {"completed_run_calls": len(calls), "completed_run_tokens": tokens(calls),
             "prior_attempt_calls": len(prior_calls), "prior_attempt_tokens": tokens(prior_calls),
             "drill_calls": len(drill_calls), "drill_tokens": tokens(drill_calls),
             "total_actual_calls": len(calls)+len(prior_calls)+len(drill_calls), "total_tokens": tokens(calls+prior_calls+drill_calls),
             "previous_attempts": [str(Path(p).resolve()) for p in previous]}
    complete = read_json(root / "blind/prediction_complete.json")
    assert complete["completed_at"] <= blind["scoring_started_at"] and complete["workers_exited"]
    public_before, public_after = rounds[0]["before"], rounds[-1]["after"]
    candidates = [e for r in rounds for e in r["experiments"] if "metrics" in e]
    best = max(candidates, key=lambda e: e["metrics"]["macro_f1"]) if candidates else None
    peak = max(int(r["resources"]["memory.peak"]) for r in rounds)
    protocol = {"completed_rounds": len(rounds), "all_acked": all(r["sdk_finish_acknowledged"] for r in rounds),
                "chunks": sum(r["chunk_count"] for r in rounds), "candidate_count": len(candidates),
                "all_continuations_verified": all(c["continuation_verified"] for r in rounds for c in r["candidates"]),
                "prediction_rows": active["rows"], "labels_after_full_stream": True,
                "no_feedback": blind["feedback_calls"] == 0, "train_peak_memory_bytes": peak,
                "source_snapshot_sha256": digest(root / "source-snapshot.json"),
                "runner_sha256": digest(ROOT / "scripts/simulate_sdk_docker.py")}
    evidence = {"scope": "local_test_not_official_platform", "costs": costs, "checks": protocol,
                "public_validation_before": public_before, "public_validation_after": public_after,
                "best_candidate": {"name": best["name"], "metrics": best["metrics"], "local_gate": best["local_gate"],
                                   "eligible": best["eligible"]} if best else None,
                "stream": aggregate, "cloud": summary.get("cloud"),
                "limitations": ["运行旧镜像的 Python/NumPy 环境，并以只读挂载覆盖最新代码和原版 SDK 1.0.4；未重新打镜像。",
                                "宿主 RPC、历史及候选回执由本地驱动实现，编号不代表赛事官方回执。",
                                "训练只用 batch1，验证只用 batch2；10 轮检查点不等于10个后续数据批次。",
                                "预测为冻结策略，没有在 batch3–batch10 中触发训练、Qwen、上线或回滚。",
                                "Windows 绑定工作目录没有强制4GiB磁盘配额；CPU、内存、临时卷限制由容器执行。",
                                "本地临时目录的介质和硬配额见summary.json；本地绑定卷的磁盘性能不代表官方设备。",
                                "单次运行不能证明统计显著提升，历史公开数据不能代替官方隐藏评测。"]}
    write_json(root / "verification.json", evidence)
    batch_rows = [[f"batch{w['batch']}", w["active"]["rows"], f"{w['baseline']['macro_f1']:.6f}",
                   f"{w['active']['macro_f1']:.6f}", f"{w['gain_macro_f1']:+.6f}",
                   f"{w['active']['accuracy']:.2%}"] for w in blind["windows"]]
    class_rows = [[CN[c], base["per_class"][c]["support"], f"{base['per_class'][c]['recall']:.2%}",
                   f"{active['per_class'][c]['recall']:.2%}", f"{active['per_class'][c]['precision']:.2%}"] for c in CLASSES]
    round_rows, timeline, candidate_rows = [], [], []
    for row in rounds:
        round_rows.append([row["round"], local(row["started_at"]), f"{row['seconds']:.1f}s", len(row["candidates"]),
                           ACTIONS.get(row["action"], row["action"]), f"{row['after']['macro_f1']:.6f}", len(row["call_ids"])])
        history_event = next(e for e in row["events"] if e["stage"] == "history_read")
        timeline.append([row["round"], "漂移监测＋诊断", local(row["started_at"]), local(history_event["time"]),
                         "容器启动至请求历史前已完成，区间时间；参考batch1，当前batch2"])
        for event in row["events"]:
            if event["stage"] == "candidate_evaluated":
                timeline.append([row["round"], "候选独立验证完成", "", local(event["time"]), event["artifact_id"]])
        for call in calls:
            if call.get("context", {}).get("round") == row["round"]:
                timeline.append([row["round"], "Qwen规划" if call["purpose"] == "experiment_planning" else "Qwen选版",
                                 local(call["started_at"]), local(call["finished_at"]), call.get("provider_call_id", "")])
        timeline.append([row["round"], "状态上传＋部署决策确认", "", local(row["completed_at"]), ACTIONS.get(row["action"], row["action"])])
        for item in row["experiments"]:
            if not item.get("metrics"):
                candidate_rows.append([row["round"], item["name"], "未测得", "未测得", "执行失败",
                                       item.get("error", "无可用测量")])
                continue
            candidate_rows.append([row["round"], item["name"], f"{item['metrics']['macro_f1']:.6f}",
                                   f"{item['metrics']['per_class']['Acetaldehyde']['recall']:.2%}",
                                   "通过" if item["eligible"] else "拒绝", "; ".join(item["local_gate"]["reasons"])])
    for name, header, rows in (("batches.csv", ["batch", "rows", "baseline_macro_f1", "selected_macro_f1", "gain", "selected_accuracy"], batch_rows),
                               ("rounds.csv", ["round", "started_shanghai", "seconds", "candidates", "action", "selected_validation_f1", "model_calls"], round_rows),
                               ("timeline.csv", ["round", "stage", "start_shanghai", "end_shanghai", "basis"], timeline),
                               ("candidates.csv", ["round", "name", "validation_f1", "acetaldehyde_recall", "eligibility", "reasons"], candidate_rows)):
        with (root / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    chart = lines([f"B{w['batch']}" for w in blind["windows"]],
                  [("选定模型", [w["active"]["macro_f1"] for w in blind["windows"]], "#2563eb"),
                   ("静态官方基线", [w["baseline"]["macro_f1"] for w in blind["windows"]], "#e87928")], "后续数据批次 Macro-F1")
    recall = lines([CN[c] for c in CLASSES],
                   [("选定模型召回率", [active["per_class"][c]["recall"] for c in CLASSES], "#2563eb"),
                    ("基线召回率", [base["per_class"][c]["recall"] for c in CLASSES], "#e87928")], "后续流各类召回率")
    same = abs(aggregate["gain_macro_f1"]) < 1e-12
    conclusion = "后续数据效果与静态基线一致，未获得改善。" if same else f"后续流 Macro-F1 相对静态基线变化 {aggregate['gain_macro_f1']:+.6f}。"
    previous_comparison = ''
    completed_previous = [Path(p) for p in previous if (Path(p)/'summary.json').is_file()
                          and list(Path(p).glob('round-*/result.json'))]
    if completed_previous:
        prior_root = completed_previous[-1]; prior_summary = read_json(prior_root/'summary.json')
        prior_rounds = [read_json(p) for p in sorted(prior_root.glob('round-*/result.json'))]
        prior_stream = prior_summary['blind']['aggregate']['active']
        comparison_rows = [
            ['原始基线',f"{public_before['macro_f1']:.6f}",f"{base['macro_f1']:.6f}",f"{base['accuracy']:.2%}"],
            [f"上轮模拟（{prior_summary['rounds']}个检查点）",f"{prior_rounds[-1]['after']['macro_f1']:.6f}",f"{prior_stream['macro_f1']:.6f}",f"{prior_stream['accuracy']:.2%}"],
            [f"本轮模拟（{summary['rounds']}个检查点）",f"{public_after['macro_f1']:.6f}",f"{active['macro_f1']:.6f}",f"{active['accuracy']:.2%}"]]
        previous_comparison = ('<section><h2>与上一轮模拟比较</h2>'
            +table(['版本','公开验证F1','后续漂移流F1','后续流准确率'],comparison_rows)
            +f"<p>相对上轮，公开验证F1变化 {public_after['macro_f1']-prior_rounds[-1]['after']['macro_f1']:+.6f}，后续流F1变化 {active['macro_f1']-prior_stream['macro_f1']:+.6f}。两次真实AI提出的配置和检查点数可能不同，不能作为单一干预的因果对照。公开验证与后续回测分开报告，后续标签未用于反调参数或规则。</p></section>")
    drift = rounds[0]["drift"]
    cloud = (summary.get("cloud") or {}).get("experiment_url")
    cloud_link = f"<a href='{html.escape(cloud)}'>SwanLab 实验记录</a>" if cloud else "本地记录"
    observe_rounds = "、".join(str(row["round"]) for row in rounds if not row["candidates"]) or "无"
    public_before, public_after = rounds[0]['before'], rounds[-1]['after']
    public_rows = [[CN[c],f"{public_before['per_class'][c]['recall']:.2%}",f"{public_after['per_class'][c]['recall']:.2%}"] for c in CLASSES]
    response_used = any(t.get('report',{}).get('architecture',{}).get('response_adapter')
                        for row in rounds for t in row.get('experiments',[]))
    response_note = ("<p>本轮新增全16传感器的相对响应分类头。迁移时新头为零，保留父权重、Adam、原标准化和计数；附加权重在公开train学习。覆盖外关闭附加修正，使用保留的MLP分支。新统计只在train首次拟合且冻结。开发CPU对照与真实27B平台模拟记录分开标识。</p>" if response_used else '')
    temporary_description = "宿主磁盘临时目录，未强制磁盘配额" if summary["limits"].get("tmp_storage") == "host_disk_bind" else "1GiB tmpfs临时目录（官方为磁盘卷）"
    body = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Gas · SDK 1.0.4 Docker 本地验证</title><style>
body{{font:15px/1.7 'Microsoft YaHei',system-ui,sans-serif;margin:0;background:#f4f7fb;color:#17253a}}main{{max-width:1120px;margin:35px auto;padding:0 24px}}h1{{font-size:30px;line-height:1.4}}h2{{font-size:21px;margin:0 0 15px}}section,.hero{{background:white;padding:26px;border:1px solid #dde5ef;border-radius:12px;margin:20px 0}}.muted{{color:#596a80}}.callout{{background:#fff4db;border-left:4px solid #d58a0b;padding:14px 18px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{padding:16px;background:#edf3fb;border-radius:8px}}.card b{{display:block;font-size:25px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 12px;text-align:left;border-bottom:1px solid #e5eaf1;vertical-align:top}}th{{background:#edf3fb;white-space:nowrap}}.scroll{{overflow:auto}}svg{{width:100%;height:auto}}svg text{{font:13px 'Microsoft YaHei',sans-serif;fill:#5d6b80}}li{{margin:7px 0}}code{{background:#edf3fb;padding:2px 5px}}a{{color:#2563eb}}details{{margin-top:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}main{{padding:0 12px}}section{{padding:18px}}}}</style>
<main><p class='muted'>SDK 1.0.4 · 真实 Linux 容器 · 最新源码只读覆盖 · 本地宿主</p>
<h1>接口完成运行，效果是否提升？</h1><div class='hero'><h2>{html.escape(conclusion)}</h2>
<p>完成 {summary['rounds']} 轮 SDK 训练检查点，真实训练 {summary['candidate_count']} 个候选，采用新候选 {summary['promotions']} 次，回滚 {summary['rollbacks']} 次。随后对 batch3–batch10 的 {active['rows']:,} 条样本逐条预测，全部提交并退出预测容器后才独立评分。</p>
<div class='cards'><div class='card'>选定模型 Macro-F1<b>{active['macro_f1']:.6f}</b></div><div class='card'>静态基线 Macro-F1<b>{base['macro_f1']:.6f}</b></div><div class='card'>选定模型准确率<b>{active['accuracy']:.2%}</b></div><div class='card'>相对增益<b>{aggregate['gain_macro_f1']:+.6f}</b></div></div></div>
<section><h2>这次代码调整与公开验证</h2>{response_note}<p>公开验证 Macro-F1：{public_before['macro_f1']:.6f} → {public_after['macro_f1']:.6f}。此处是候选选版数据，后续漂移流的独立评分见下方；两者不能混报为隐藏成绩。</p>{table(['气体','原模型验证召回','最终模型验证召回'],public_rows)}</section>
{previous_comparison}
<section><h2>测试数据与隔离方式</h2><p>batch1（{summary['train_rows']}条）对应官方公开训练文件；batch2（{summary['validation_rows']}条）对应公开验证文件，已核对原始 Dataset 的全部特征和标签与公开投影相同。梯度及拟合统计量仅使用训练集。后续8批只把128维特征送入预测容器，不挂载 DAT 文件或训练、验证文件。</p>
<p>容器使用 UID 65534、只读根文件系统、禁用网络、2核、4GiB内存、{temporary_description}。模型调用由宿主转发 PyTRiO，凭据没有进入容器。app、sdk 的实际容器文件摘要已逐项核对。训练峰值内存 {peak/1024**2:.1f} MiB。</p>
<div class='callout'>使用现有镜像的 Python 3.12/NumPy 1.26.4 环境，最新代码和原版 SDK 以只读挂载覆盖。未重新构建镜像；本结果不代表正式提交镜像已经更新，也不是赛事官方隐藏分数。</div></section>
<section><h2>后续漂移流表现</h2>{chart}{table(['批次','样本数','基线F1','选定F1','差值','选定准确率'], batch_rows)}<p class='muted'>每批得分与全流得分均直接由冻结预测计算；全流 Macro-F1 对全部样本合并后重算，非各批简单平均。口径与官方参考基线一致：计入真实或预测出现的类别。重合曲线表示两个模型输出效果相同。</p></section>
<section><h2>各气体召回率</h2>{recall}{table(['气体','样本数','基线召回','选定召回','选定精确率'], class_rows)}</section>
<section><h2>{summary['rounds']}轮检查点做了什么</h2><p>公开验证前后 Macro-F1：{public_before['macro_f1']:.6f} → {public_after['macro_f1']:.6f}。监测发现 {drift['flagged_features']}/128 维触发分布变化规则；弱类为 {html.escape(CN[rounds[0]['diagnosis']['weakest_class']])}。诊断提供相关性线索，不能单独证明物理原因。每轮读取历史，再由真实 Qwen 规划和选版。</p>
{table(['轮次','开始时间（上海）','耗时','候选数','动作','选定验证F1','Qwen调用'], round_rows)}
<p>连续3轮没有合格候选时，相同父模型与相同数据签名触发程序的暂停训练规则。本次没有提交新候选的轮次为：{observe_rounds}，详情见各轮记录。观察轮仍真实调用模型进行规划和选版。检查点序号不会自动切换到后续DAT批次。</p></section>
<section><h2>候选效果与拒绝原因</h2>{table(['轮次','实验','公开验证F1','乙醛召回','上线资格','门槛原因'], candidate_rows)}
<p class='muted'>门槛属于工程策略，以本次source-snapshot中的配置为准。当前默认：乙醛最低召回50%、最小F1增益0.002，最差模拟场景允许F1下降0.005、单场景允许0.02；满足保护条件后按主指标Macro-F1选epoch。此前零下降约束已根据公开验证收益与代价调整，未使用后续标签调规则。</p></section>
<section><h2>闭环时间与边界</h2><p>以下时间均为北京时间。监测和诊断发生在每次 train 内；训练候选通过另一 predict 容器独立重算公开验证得分。选版后分块保存状态并获取 round_ack。上线/回滚状态以实际动作记录为准；没有触发回滚时，不声称本次验证了自然回滚效果。</p>
{table(['轮次','环节','开始','完成','依据'], timeline)}
<div class='callout'>batch3–batch10 预测阶段执行冻结模型：没有在线漂移监测、Qwen规划、再训练、选版或回滚。因此本次验证的是“公开训练阶段闭环＋后续流泛化”，不能据此声称隐藏预测流上已实现动态自进化。</div>
<p>全流预测提交完成：{local(complete['completed_at'])}；独立标签评分开始：{local(blind['scoring_started_at'])}。预测反馈0次，预测期Qwen调用0次，在线更新0次。</p></section>
<section><h2>证据与调用成本</h2><p>完整运行 Qwen {costs['completed_run_calls']} 次、{costs['completed_run_tokens']:,} token；隔离演练另用 {costs['drill_calls']} 次、{costs['drill_tokens']:,} token；先前失败尝试另用 {costs['prior_attempt_calls']} 次、{costs['prior_attempt_tokens']:,} token。总计 {costs['total_actual_calls']} 次、{costs['total_tokens']:,} token。演练及失败尝试均未计入正常成绩。</p>
<p>SDK chunked-v1 分块上传 {protocol['chunks']} 次，{protocol['completed_rounds']} 次最终确认。候选均核验父摘要、起始优化器步数、实际更新步数及未重拟合的标准化参数。</p><p>{cloud_link} · <a href='summary.json'>完整结果JSON</a> · <a href='verification.json'>核验证据JSON</a> · <a href='source-snapshot.json'>源码摘要</a> · <a href='runtime-mount-proof.json'>容器权限证据</a> · <a href='timeline.csv'>时间线CSV</a> · <a href='batches.csv'>批次CSV</a> · <a href='candidates.csv'>候选CSV</a></p>
<details><summary>测试限制</summary><ul>{''.join('<li>'+html.escape(v)+'</li>' for v in evidence['limitations'])}</ul></details></section>
<p class='muted'>生成依据：{html.escape(str(root))}。原始协议、请求/回复、逐轮workspace、候选和全流预测承诺均保留在此目录。{local(summary['completed_at'])}</p></main></html>"""
    if (root/'acceptance.json').is_file():
        links = ("<section><h2>验收与复现材料</h2><p><a href='acceptance.json'>逐项验收</a> · "
                 "<a href='complete-loops.csv'>完整闭环记录</a> · <a href='reproduction.json'>数值复现</a> · "
                 "<a href='cloud-readback.json'>云端读回</a> · <a href='REPRODUCE.txt'>复现说明</a> · "
                 "<a href='reproduction-materials.zip'>源码包（不是镜像）</a></p></section>")
        body = body.replace('</main>',links+'</main>')
    (root / "report.html").write_text(body, encoding="utf-8")
    print(json.dumps({"report": str(root / "report.html"), "costs": costs, "checks": protocol,
                      "aggregate": {"macro_f1": active["macro_f1"], "accuracy": active["accuracy"],
                                    "gain": aggregate["gain_macro_f1"]}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--previous", action="append", default=[])
    args = parser.parse_args()
    export(args.source, args.previous)
