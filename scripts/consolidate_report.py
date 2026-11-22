"""把已有回放报告、图片、逐批建议和调用证据合并成原位置的单文件HTML。"""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gas_a.model_records import read_call
from gas_a.redaction import redact


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def esc(value):
    return html.escape(str(value))


def pretty(value):
    return esc(json.dumps(redact(value), ensure_ascii=False, indent=2))


def score(value):
    return "未提供" if value is None else f"{value:.6f}"


def local_time(value):
    if not value:
        return "未记录"
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S")


def table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(x)}</td>' for x in row)
            + '</tr>' for row in rows) + '</tbody></table></div>')


def call_html(call):
    names = {"experiment_planning": "规划实验", "artifact_selection": "选择版本"}
    usage = call.get("usage") or {}
    title = (f'{names.get(call.get("purpose"), call.get("purpose", "模型调用"))} · '
             f'{local_time(call.get("started_at"))} · '
             f'输入 {usage.get("prompt_tokens", "未知")} / 输出 {usage.get("completion_tokens", "未知")} Token')
    return (f'<details class="call"><summary>{esc(title)}</summary><h4>AI完整回复</h4>'
            f'<pre>{esc(call.get("reply_text") or "未取得回复；见调用状态和错误记录。")}</pre>'
            '<details><summary>发给AI的完整提示词</summary><pre>' + pretty(call.get("request"))
            + '</pre></details><details><summary>完整调用记录、回执和耗时</summary><pre>'
            + pretty(call) + '</pre></details></details>')


def planning_update_html(root):
    manifest = root / "planning-update.json"
    if not manifest.is_file():
        return "", ""
    update = read(manifest)
    probe = Path(update["probe_run"]).resolve()
    if not probe.is_relative_to(root.parents[1]):
        raise ValueError("Planning probe must be within the experiment output root")
    verification, row = read(probe / "verification.json"), read(probe / "result.json")
    calls = sorted((read_call(p) for p in (probe / "session/model_calls").glob("call-*.json")),
                   key=lambda c: c.get("started_at", ""))
    plan, trials = row.get("plan", {}), row.get("experiments", [])
    costs = verification["costs"]
    notice = ('<!-- planning-notice --><p class="lead">本页原dat3–dat10成绩来自旧菜单机制（源码5fbf984）。'
              '新版已改为AI按数据提出完整配置；下方新增的历史dat6隔离功能验证不计入原8批成绩，原部署未改变。</p><!-- /planning-notice -->')
    parts = ['<!-- planning-update --><h2>新机制：AI根据数据提出合适的组合</h2>',
             '<p>不再从固定方案名中选择，不限制最多改两项，也不要求每轮把所有能力都用上。配置被拒绝后，将原因和合法范围反馈给AI修正；通过校验才训练。</p>',
             '<p><b>这次真实AI的假设：</b>' + esc(plan.get("hypothesis")) + '</p>',
             '<p><b>提出这些调整的依据：</b>' + esc(plan.get("reason")) + '</p>',
             '<details open><summary>AI实际提出的配置</summary><pre>' + pretty(plan.get("experiments")) + '</pre></details>']
    for trial in trials:
        report = trial.get("report", {})
        parts.append('<p><b>实际执行：</b>' + esc(report.get("name", trial.get("name")))
            + '；网络结构 ' + esc(json.dumps(report.get("architecture"), ensure_ascii=False))
            + '；冻结张量 ' + esc(json.dumps(report.get("freezing", {}).get("frozen_parameters", []))) + '。</p>')
        parts.append(table(["模型", "固定验证集F1", "本批影子F1"], [
            ("父模型", score(row.get("fixed_validation_before", {}).get("macro_f1")), score(row.get("shadow_before", {}).get("macro_f1"))),
            ("AI新候选", score(report.get("validation_after", {}).get("macro_f1")), score(trial.get("shadow_metrics", {}).get("macro_f1")))]))
    parts.append('<p><b>最后选择：</b>' + esc(row.get("selection", {}).get("reason") or row.get("reason"))
                 + ' 这是配置组合的功能验证，本批影子收益不能当作未来部署收益。</p>')
    rejections = row.get("planning_rejections", [])
    if rejections:
        parts.append('<h3>拒绝与AI修正的实际记录</h3>' + table(["次数", "拒绝原因", "处理"],
            [(i+1, r["error"], "将原建议和合法范围反馈给AI，得到修正回复后再校验") for i, r in enumerate(rejections)]))
        if any(r["error"] == "Invalid concise reason" for r in rejections):
            parts.append('<p>本次首回复因理由超过提示中的80字上限触发了纠正。后续代码已放宽可读文字的长度校验，避免类似文案造成额外调用；参数、状态和事实检查保持严格。原拒绝及费用保留。</p>')
    parts.append(f'<p><b>新增实际成本：</b>{costs["calls"]}次真实调用，{costs["tokens_known"]:,} Token（包含纠正）；没有重新运行全部8批。</p>')
    url = verification.get("cloud", {}).get("experiment_url")
    if url:
        parts.append('<p>本次隔离验证的私有SwanLab：<a href="' + esc(url) + '">查看记录</a>。本地SDK已返回完成，未单独进行此次云端读回。</p>')
    parts.append('<details><summary>这次验证的完整AI对话、拒绝及修正回复</summary>' + ''.join(call_html(c) for c in calls) + '</details>')
    parts.append('<details><summary>实际执行配置和完整训练记录</summary><pre>' + pretty(row) + '</pre></details>')
    parts.append('<!-- /planning-update -->')
    return notice, ''.join(parts)


def batch_html(root, row, calls, stage, id_prefix="batch"):
    batch, plan = row["batch"], row.get("plan", {})
    trials = row.get("experiments", [])
    decision = {"promote": "采用新模型", "retain": "保留当前模型", "rollback": "恢复历史模型"}.get(
        row.get("action"), row.get("action", "查看本批决定"))
    title = f'dat{batch} · {decision} · {", ".join(t.get("name", "") for t in trials) or "没有新增训练"}'
    parts = [f'<details class="batch" id="{esc(id_prefix)}-{batch}" {"open" if batch == 3 else ""}><summary>{esc(title)}</summary>']
    parts.append('<h3>什么时候执行</h3>')
    parts.append(table(["步骤", "完成记录（北京时间）"], [(label, stage.get(key) or "本批未执行 / 无该完成记录")
        for label, key in (("漂移监测", "监测完成记录"), ("诊断归因", "诊断证据记录"),
                           ("实验规划", "规划完成记录"), ("候选训练完成", "候选完成末条记录"),
                           ("评估与选版", "评估选版完成记录"))]))
    parts.append('<p class="note">以上是完成日志的时间，不能用同秒记录推算训练耗时。历史恢复检查、自然切换和独立故障演练见本页时间线。</p>')
    origin = "AI建议（假设仍需实验验证）" if calls else "程序决定（本批没有新增AI调用）"
    parts.append(f'<h3>{origin}</h3><p><b>本轮想验证：</b>{esc(plan.get("hypothesis") or "无新增实验假设")}</p>')
    parts.append(f'<p><b>选择这个方向的理由：</b>{esc(plan.get("reason") or row.get("reason") or "未提供")}</p>')
    configurations = plan.get("experiments")
    direct = isinstance(configurations, list) and bool(configurations) and isinstance(configurations[0], dict)
    parts.append('<details><summary>' + ('AI提出的完整训练配置' if direct else 'AI建议的参数修改') + '</summary><pre>'
                 + pretty(configurations if direct else plan.get("parameter_changes", {})) + '</pre></details>')
    rejections = row.get("planning_rejections", [])
    if rejections:
        parts.append('<h3>建议被拒绝后，怎样让AI修改</h3>')
        parts.append(table(["次数", "拒绝原因", "后续处理"],
            [(i+1, r.get("error"), "反馈给AI修正" if r.get("next_action") == "ask_ai_to_revise" else "纠正预算用完，保留安全版本")
             for i, r in enumerate(rejections)]))
    parts.append('<h3>同一批数据上的成绩</h3>')
    values = [("原始基线", score(row.get("fixed_static_baseline", {}).get("macro_f1")),
               score(row.get("static_baseline", {}).get("macro_f1"))),
              ("本轮续训起点（父模型）", score(row.get("fixed_validation_before", {}).get("macro_f1")),
               score(row.get("shadow_before", row.get("operational", {})).get("macro_f1")))]
    values.extend((t["name"], score(t.get("report", {}).get("validation_after", {}).get("macro_f1")),
                   score(t.get("shadow_metrics", {}).get("macro_f1"))) for t in trials)
    parts.append(table(["模型", "原固定验证集F1", "当前这批数据F1"], values))
    parts.append('<p class="note">F1越接近1越好；同列使用同一批数据。候选的当前批成绩是标签揭示后的影子评估，部署效果从后续批次检验。</p>')
    parts.append('<h3>实际训练了什么</h3>')
    if not trials:
        parts.append(f'<p>{esc(row.get("reason") or "本批没有新增训练。")}</p>')
    for trial in trials:
        report = trial.get("report", {})
        cfg = report.get("configuration", trial.get("configuration", {}))
        evidence = Path(trial.get("evidence_directory", root)).resolve()
        training = read(evidence / "training.json") if evidence.is_relative_to(root) and (evidence / "training.json").is_file() else {}
        actual = training.get("configuration", cfg)
        freezing = report.get("freezing", {})
        labels = [("学习率", actual.get("learning_rate")), ("批大小", actual.get("batch_size")),
                  ("计划训练轮数", actual.get("requested_epochs", cfg.get("epochs"))),
                  ("实际执行轮数", report.get("checkpoint_selection", {}).get("executed_epoch_count")),
                  ("最终保留的累计训练轮数", report.get("final_epochs")),
                  ("隐藏层结构", report.get("architecture")), ("冻结参数", freezing.get("frozen_parameters", [])),
                  ("架构迁移", "已执行" if (report.get("architecture_migration") or {}).get("performed") else "本候选未改变架构")]
        source = "程序参照（不是AI提出）" if trial.get("proposal_origin") == "program_reference_control" else trial.get("proposal_origin")
        parts.append(f'<h4>{esc(trial["name"])}</h4>' + (f'<p>配置来源：{esc(source)}</p>' if source else '') + table(["项目", "实际值"],
            [(key, json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value if value is not None else "未提供")
             for key, value in labels]))
        parts.append('<details><summary>完整实际配置与逐轮训练证据</summary><pre>' + pretty(training or report) + '</pre></details>')
    reason = row.get("selection", {}).get("reason") or row.get("reason") or "未提供"
    parts.append(f'<h3>最后为什么这样选</h3><p>{esc(reason)}</p><p><b>后续使用版本：</b>{esc(row.get("selected_version", "未提供"))}</p>')
    parts.append('<h3>本批完整AI对话</h3>' + (''.join(call_html(c) for c in calls) if calls else '<p>本批没有新增AI调用。</p>'))
    parts.append('<details><summary>本批完整实验记录</summary><pre>' + pretty(row) + '</pre></details></details>')
    return ''.join(parts)


def consolidate(run):
    root = Path(run).resolve()
    target = root / "closed-loop-report.html"
    original = target.read_text(encoding="utf-8-sig")
    analysis = read(root / "analysis.json")
    rows = [read(path) for path in sorted((root / "windows").glob("batch*/result.json"))]
    calls = [read_call(path) for path in (root / "session/model_calls").glob("call-*.json")]
    calls.sort(key=lambda c: (c.get("started_at", ""), c.get("record_id", "")))
    stages = {int(str(s["批次"]).replace("dat", "")): s for s in analysis.get("timeline", [])}
    if [r["batch"] for r in rows] != list(range(3, 11)):
        raise ValueError("Expected complete DAT3-DAT10 records")
    section = '<h2>每批建议、实际训练与选版</h2><p>点击批次展开。建议、实际执行、成绩和完整对话已放在同一处；原始模型文字仍按待验证假设阅读。</p>'
    section += '<p class="batch-links">' + ' '.join(f'<a href="#batch-{r["batch"]}">dat{r["batch"]}</a>' for r in rows) + '</p>'
    for row in rows:
        group = [c for c in calls if c.get("context", {}).get("batch") == row["batch"]]
        section += batch_html(root, row, group, stages.get(row["batch"], {}))
    warning = re.search(r'<p>模型文字不等于执行事实：.*?</p>', original, re.S)
    if warning:
        section += warning.group()
    pattern = r'<h2(?:\s[^>]*)?>(?:27B原始假设与实测后选择|每批建议、实际训练与选版)</h2>.*?(?=<h2|</html>)'
    content, count = re.subn(pattern, lambda _: section, original, count=1, flags=re.S)
    if count != 1:
        raise ValueError("Cannot locate the existing AI advice section")
    content = re.sub(r'<!-- consolidated-nav -->.*?<!-- /consolidated-nav -->', '', content, flags=re.S)
    content = re.sub(r'<!-- consolidated-evidence -->.*?<!-- /consolidated-evidence -->', '', content, flags=re.S)
    content = re.sub(r'<!-- planning-notice -->.*?<!-- /planning-notice -->', '', content, flags=re.S)
    content = re.sub(r'<!-- planning-update -->.*?<!-- /planning-update -->', '', content, flags=re.S)
    content = re.sub(r'<!-- optimization-update -->.*?<!-- /optimization-update -->', '', content, flags=re.S)
    notice, update = planning_update_html(root)
    if update:
        content = re.sub(r'<title>.*?</title>', '<title>气体模型实验与AI配置组合</title>', content, count=1, flags=re.S)
        content = re.sub(r'<h1[^>]*>.*?</h1>', '<h1>气体模型实验结果与AI调优记录</h1>' + notice, content, count=1, flags=re.S)
        content = re.sub(r'(?=<h2)', lambda _: update, content, count=1)
    optimization_update = root / "optimization-update.html"
    if optimization_update.is_file():
        fragment = optimization_update.read_text(encoding="utf-8-sig")
        if '<script' in fragment.lower() or '<!-- optimization-update -->' in fragment:
            raise ValueError("Optimization annex must be a static HTML fragment without wrapper markers")
        content = re.sub(r'(<h1[^>]*>.*?</h1>)', lambda m: m.group() + '<!-- optimization-update -->' + fragment
                         + '<!-- /optimization-update -->', content, count=1, flags=re.S)
        content = content.replace('本轮最新真实27B', '此前菜单真实27B（历史）')
        content = content.replace("<p class='lead'>真实闭环运行完成", "<p class='lead'>历史菜单回放：真实闭环运行完成")
    annex = '<!-- consolidated-evidence --><h2>完整核验资料</h2><p>原文件的内容已嵌入本页，展开即可查看。</p>'
    for file, label in (("analysis.json", "总体分析与对比"), ("independent-verification.json", "预测和训练独立核验"),
                        ("service-verification.json", "服务预测核验"), ("cloud-readback.json", "SwanLab读回核验"),
                        ("offline-tests.json", "离线检查"), ("rollback_drill/result.json", "独立故障回滚演练")):
        if (root / file).is_file():
            annex += '<details><summary>' + esc(label) + '</summary><pre>' + pretty(read(root / file)) + '</pre></details>'
    annex += '<!-- /consolidated-evidence -->'
    content = content.replace('</html>', annex + '</html>')
    content = re.sub(r'<p>文件：analysis\.json.*?</p>', '<p>图表、逐批记录、AI完整对话和核验资料已合并在本页；底层原始文件继续保留用于追溯。</p>', content, flags=re.S)
    assets = {}
    def inline_image(match):
        value = match.group(2)
        if value.startswith("data:image/"):
            return match.group()
        path = (root / value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Image must be an existing asset within this run")
        raw = path.read_bytes()
        assets[value] = hashlib.sha256(raw).hexdigest()
        mime = "image/svg+xml" if path.suffix == ".svg" else "image/png"
        return match.group(1) + 'data:' + mime + ';base64,' + base64.b64encode(raw).decode() + match.group(3)
    content = re.sub(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', inline_image, content)
    headings = []
    def heading(match):
        index = len(headings) + 1
        headings.append((f"part-{index}", re.sub('<[^>]+>', '', match.group(1))))
        return f'<h2 id="part-{index}">{match.group(1)}</h2>'
    content = re.sub(r'<h2(?:\s[^>]*)?>(.*?)</h2>', heading, content)
    nav = '<!-- consolidated-nav --><nav class="report-nav" aria-label="报告目录"><strong>一页查看全部结果</strong>'
    nav += ''.join(f'<a href="#{key}">{label}</a>' for key, label in headings) + '</nav><!-- /consolidated-nav -->'
    content = re.sub(r'(<h1[^>]*>.*?</h1>)', lambda m: m.group() + nav, content, count=1, flags=re.S)
    css = '''<style id="consolidated-style">html{scroll-behavior:smooth;scroll-padding-top:100px}*{box-sizing:border-box}
.report-nav{position:sticky;top:0;z-index:5;display:flex;gap:8px;align-items:center;overflow:auto;padding:12px 0;background:#f7f9fc;border-bottom:1px solid #cedae7}
.report-nav a,.batch-links a{display:inline-block;padding:5px 10px;border-radius:6px;background:#e5edf6;color:#174e78;text-decoration:none;white-space:nowrap}
.report-nav strong{white-space:nowrap;font-size:14px}.batch{margin:12px 0;background:white;border:1px solid #dbe3ed;border-radius:9px;padding:16px}
summary{cursor:pointer;font-weight:600}h3{font-size:19px;margin:24px 0 10px}h4{font-size:16px}pre{font:13px/1.65 Consolas,"Microsoft YaHei",monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:650px;overflow:auto;background:#f3f6fa;padding:14px;border-radius:6px}
.note{font-size:13px;color:#52677b}.batch .scroll{max-width:100%}.batch table th{min-width:100px}img{display:block;max-width:100%;height:auto}
@media(max-width:700px){body{margin:16px auto;padding:0 12px}h1{font-size:23px}h2{font-size:22px}.batch{padding:12px}.report-nav{position:static}table{font-size:12px}}
</style>'''
    content = re.sub(r'<style id="consolidated-style">.*?</style>', '', content, flags=re.S)
    content = content.replace('</style>', '</style>' + css, 1)
    if '<meta name="viewport"' not in content:
        content = content.replace('<meta charset="utf-8">', '<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    if len(content.encode()) > 25 * 1024 * 1024:
        raise ValueError("Consolidated report unexpectedly exceeds 25 MiB")
    images = re.findall(r'<img\b[^>]*src=["\']([^"\']+)', content)
    assert len(images) >= 4 and all(s.startswith('data:image/') for s in images)
    assert all(f'id="batch-{r["batch"]}"' in content for r in rows)
    assert all(esc(c.get('reply_text') or '') in content for c in calls)
    backup = root / "_archive/closed-loop-report.before-consolidation.html"
    if not backup.exists():
        backup.parent.mkdir(exist_ok=True)
        backup.write_text(original, encoding="utf-8")
    temporary = target.with_suffix('.html.tmp')
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(target)
    result = {"path": str(target), "bytes": target.stat().st_size, "batches": len(rows),
              "historical_model_calls": len(calls), "embedded_figures": len(images), "figure_sha256": assets,
              "raw_advice_preserved": True, "external_model_calls": 0}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    consolidate(parser.parse_args().run)
