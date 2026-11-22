"""持久化审计记录，并在显式启用时同步至 SwanLab 云端。"""
import importlib
import json
import math
import os
import uuid
from pathlib import Path
from .io import append_jsonl, read_json, write_json, write_jsonl, history, stamp
from .redaction import redact


# 显示规则仅影响图表；全部事件仍写入 events.jsonl 和
# audit/details，包括不适合在默认看板展示的字段。
DASHBOARD_METRICS = {
    "performance/macro_f1": "模型效果",
    "performance/accuracy": "模型效果",
    "performance/acetaldehyde_recall": "模型效果",
    "performance/acetaldehyde_precision": "模型效果",
    "performance/baseline_macro_f1": "模型效果",
    "performance/gain": "模型效果",
    "cost/calls": "资源成本",
    "cost/tokens_known": "资源成本",
    "cost/unknown_usage_calls": "资源成本",
    "cost/candidate_wall_seconds_known": "资源成本",
    "cost/candidate_cpu_seconds_known": "资源成本",
    "epoch/loss": "训练曲线",
    "review/summary": "每轮摘要",
}


def dashboard_summary(summary):
    """生成简短的云端展示内容，完整摘要仍保留在审计事件中。"""
    def number(value, digits=6):
        return "未提供" if value is None else f"{value:.{digits}f}"

    actions = {"promote": "采用新候选", "retain": "保留当前模型", "rollback": "回滚历史模型"}
    costs = summary["costs"]
    return "\n".join([
        f"第 {summary['round']} 轮｜{actions.get(summary.get('action'), '决策见完整记录')}",
        f"Macro-F1：{number(summary['before'].get('macro_f1'))} → {number(summary['after'].get('macro_f1'))}"
        f"（{summary['gain']:+.6f}）；准确率：{number(summary['after'].get('accuracy'))}",
        f"乙醛召回：{number(summary['after'].get('acetaldehyde_recall'))}；精确率：{number(summary['after'].get('acetaldehyde_precision'))}；"
        f"静态基线下限：{'通过' if summary['baseline_floor_passed'] else '未通过'}；"
        f"异常回退：{'是' if summary.get('fallback') else '否'}",
        f"选定版本：{summary.get('selected') or '未记录'}",
        f"调用 {costs['calls']} 次；已知 Token {costs['tokens_known']}（用量缺失 {costs['unknown_usage_calls']} 次）；"
        f"候选耗时 {number(costs['candidate_wall_seconds_known'], 3)} 秒；CPU {number(costs['candidate_cpu_seconds_known'], 3)} 秒"
        f"（资源缺失 {costs['candidates_with_unknown_resources']} 项）",
        "公开验证集结果。完整请求、回复、候选和决策依据：隐藏图表中的 audit/details。",
    ])


def scalar_metrics(value, prefix=""):
    result = {}
    if not isinstance(value, dict):
        return result
    for key, item in value.items():
        name = prefix + str(key)
        if isinstance(item, dict):
            result.update(scalar_metrics(item, name + "/"))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
            result[name] = item
    return result


class Audit:
    def __init__(self, directory, cloud=True, module=None):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cloud = cloud
        self.module = module
        self.run = None
        self.step = 0
        self.pending = []
        self.receipt = {"requested_cloud": cloud, "status": "not_started", "experiment_url": None}

    def start(self, config):
        self.record("session_start", config, upload=False)
        if not self.cloud:
            self.receipt["status"] = "local_only"
            return
        if not os.environ.get("SWANLAB_API_KEY"):
            raise RuntimeError("Missing SWANLAB_API_KEY. Register at swanlab.cn, then run live-configure")
        # SwanLab 0.10 将这些环境变量保留给 JSON 设置对象。
        # 兼容早期按字面值填写的别名，转换后再交给 SDK。
        for old, new in (("SWANLAB_PROJECT", "GAS_A_SWANLAB_PROJECT"), ("SWANLAB_WORKSPACE", "GAS_A_SWANLAB_WORKSPACE")):
            if old in os.environ:
                os.environ.setdefault(new, os.environ.pop(old))
        try:
            self.module = self.module or importlib.import_module("swanlab")
            self.module.login(api_key=os.environ["SWANLAB_API_KEY"], save=False)
            self.run = self.module.init(
                project=os.environ.get("GAS_A_SWANLAB_PROJECT", "gas-a"),
                workspace=os.environ.get("GAS_A_SWANLAB_WORKSPACE") or None,
                name="gas-a-" + self.root.name, config=redact(config),
                mode="online", on_online_error="raise", public=False,
                log_dir=str(self.root / "swanlog"),
                settings=self.module.Settings(interactive=False),
            )
            if getattr(self.run, "mode", "online") not in ("online", "cloud"):
                raise RuntimeError("SwanLab unexpectedly entered an offline mode")
            url = getattr(self.run, "url", None)
            if not url:  # 兼容仍通过旧版 public 接口提供 URL 的 SDK。
                url = self.run.public.cloud.experiment_url
            if not isinstance(url, str) or not url.startswith("https://"):
                raise RuntimeError("SwanLab did not return a cloud experiment URL")
            # 在首条日志之前定义展示规则；其余指标和完整审计文本
            # 仍会上传，默认隐藏以减少看板拥挤。
            self.run.define_metric("*", hidden=True)
            for key, section in DASHBOARD_METRICS.items():
                self.run.define_metric(key, section_name=section, hidden=False)
            self.receipt.update(status="online_created", experiment_url=url)
            write_json(self.root / "cloud.json", self.receipt)
            self.record("cloud_connected", self.receipt)
        except ImportError:
            raise RuntimeError("Install live dependencies first: python -m pip install -e .[live]") from None
        except Exception as exc:
            self.receipt.update(status="initialization_failed", error=redact(str(exc))[:400])
            write_json(self.root / "cloud.json", self.receipt)
            raise RuntimeError("SwanLab initialization failed: " + str(redact(str(exc)))[:400]) from None

    def record(self, kind, payload, upload=True):
        """先追加本地审计事件，再按需上传云端；长记录分块保留，展示隐藏不影响证据完整性。"""
        self.step += 1
        event = {"event_id": uuid.uuid4().hex, "time": stamp(), "event": kind, "payload": redact(payload)}
        append_jsonl(self.root / "events.jsonl", event)  # 任何网络操作之前先持久化到本地。
        if upload and self.run is not None:
            self._send_or_queue(event, kind, payload)
        return event

    def record_model_call(self, reference, root):
        """本地事件保存引用，云端接收一次完整的原始记录。"""
        from .model_records import resolve_call
        resolve_call(reference, root)
        record = read_json(Path(root) / reference["file"])
        event = self.record("model_call", reference, upload=False)
        if self.run is not None:
            self._send_or_queue({**event, "payload": {"reference": reference, "record": record}}, "model_call", reference)
        return event

    def _send_or_queue(self, event, kind, payload):
        if not self.pending:
            for attempt in range(2):
                try:
                    self._upload(event, kind, payload)
                    if self.receipt["status"] == "upload_error":
                        self.receipt.update(status="online_created", last_upload_error=self.receipt.pop("error", None))
                    return
                except RuntimeError:
                    if not attempt:
                        self.receipt["upload_retries"] = self.receipt.get("upload_retries", 0) + 1
        self.pending.append({"event": redact(event), "kind": kind, "payload": redact(payload), "step": self.step})
        write_jsonl(self.root / "cloud-outbox.jsonl", self.pending)
        self.receipt.update(status="upload_pending", pending_uploads=len(self.pending))
        write_json(self.root / "cloud.json", self.receipt)

    def _upload(self, event, kind, payload, step=None):
        if self.run is not None:
            try:
                text = json.dumps(event, ensure_ascii=False, allow_nan=False)
                # 较长的实验和模型记录按块拆分，保留完整内容。
                chunks = [text[i:i+12000] for i in range(0, len(text), 12000)]
                values = {}
                if kind == "round_summary":
                    performance = {**payload["after"], "baseline_macro_f1": payload["baseline"].get("macro_f1"),
                                   "gain": payload["gain"]}
                    values.update(scalar_metrics({"performance": performance, "cost": payload["costs"]}))
                    values["review/summary"] = self.module.Text(dashboard_summary(payload), caption=f"第 {payload['round']} 轮摘要")
                elif kind == "temporal_summary":
                    performance = {**payload["operational"], "baseline_macro_f1": payload["static_baseline"]["macro_f1"],
                                   "gain": payload["gain"]}
                    values.update(scalar_metrics({"performance": performance, "cost": payload["costs"]}))
                    text = (f"batch{payload['batch']}｜更新前实际预测｜版本 {payload['version']}\n"
                            f"Macro-F1 {performance['macro_f1']:.6f}；静态基线 {performance['baseline_macro_f1']:.6f}；"
                            f"增益 {performance['gain']:+.6f}\n"
                            f"乙醛召回 {performance.get('acetaldehyde_recall', 0):.4f}；"
                            f"后续版本 {payload['selected']}；回滚 {payload['rollback']}\n"
                            f"{payload['status']}｜{payload['reason']}\n"
                            "候选影子评估与完整模型交互见 audit/details；本批更新前成绩保持固定。")
                    values["review/summary"] = self.module.Text(text, caption=f"batch{payload['batch']} 时间窗口")
                elif kind == "candidate_epoch":
                    values.update(scalar_metrics({"epoch": {key: payload[key] for key in ("loss", "steps", "epoch") if key in payload}}))
                # hidden 仅控制展示，不过滤或截断记录。
                values["audit/details"] = [self.module.Text(chunk, caption=f"{event['event_id']}:{i+1}/{len(chunks)}") for i, chunk in enumerate(chunks)]
                self.run.log(values, step=self.step if step is None else step)
            except Exception as exc:
                self.receipt.update(status="upload_error", error=redact(str(exc))[:400])
                write_json(self.root / "cloud.json", self.receipt)
                raise RuntimeError("Cloud logging failed; local audit is retained for sync-records") from None

    def finish(self, succeeded):
        if self.run is not None:
            while self.pending:
                item = self.pending[0]
                try:
                    self._upload(item["event"], item["kind"], item["payload"], step=item["step"])
                except RuntimeError:
                    self.receipt.update(status="upload_pending", pending_uploads=len(self.pending))
                    break
                self.pending.pop(0)
                write_jsonl(self.root / "cloud-outbox.jsonl", self.pending)
            failed_before = bool(self.pending) or self.receipt["status"] == "initialization_failed"
            try:
                self.module.finish(state="success" if succeeded and not failed_before else "crashed",
                                   error=None if succeeded and not failed_before else "Local failure or incomplete cloud upload; inspect local audit/outbox")
                self.receipt["status"] = "sdk_finish_returned" if not failed_before else self.receipt["status"]
            except Exception as exc:
                self.receipt.update(status="finish_pending", error=redact(str(exc))[:400])
                write_json(self.root / "cloud.json", self.receipt)
            finally:
                self.run = None
        self.receipt["session_succeeded"] = succeeded
        self.receipt["pending_uploads"] = len(self.pending)
        self.receipt["upload_complete"] = (self.receipt["status"] == "sdk_finish_returned") if self.cloud else None
        self.receipt["verification"] = "Open the experiment URL to verify cloud records; a URL alone is not proof of complete upload"
        write_json(self.root / "cloud.json", self.receipt)
        return self.receipt


def import_records(source, audit):
    source = Path(source)
    if source.is_dir() and (source / "cloud-outbox.jsonl").is_file():
        pending = history(source / "cloud-outbox.jsonl")
        if pending:
            for item in pending:
                audit.record("outbox_replay", {"source": str(source), "original_event_id": item["event"]["event_id"]}, upload=False)
                audit._send_or_queue(item["event"], item["kind"], item["payload"])
            return len(pending)
    if source.is_dir():
        source = source / ("events.jsonl" if (source / "events.jsonl").exists() else "history.jsonl")
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(source.read_text(encoding="utf-8"))
        records = value.get("rounds", [value]) if isinstance(value, dict) else value
    if not isinstance(records, list) or not records:
        raise ValueError("No actual experiment records found")
    for record in records:
        audit.record("historical_import", {"source": str(source), "imported_at": stamp(), "original_record": record})
    return len(records)
