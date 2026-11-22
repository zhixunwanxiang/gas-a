"""成对记录模型请求与回复，并对凭据脱敏；失败调用也保留。"""
import hashlib
import json
from pathlib import Path
import time
import uuid
from .io import append_jsonl, read_json, stamp, write_json, digest
from .redaction import redact


PURPOSES = {
    "experiment_planning": "选择候选实验",
    "artifact_selection": "选择最终模型版本",
    "connectivity_probe": "模型连通性探测",
    "model_call": "模型调用",
}


def purpose_for(messages):
    for message in messages:
        try:
            content = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(content, dict):
            if "allowed_experiments" in content or "search_space" in content:
                return "experiment_planning"
            if "eligible_artifacts" in content:
                return "artifact_selection"
    return "model_call"


def safe_record(value, secrets=()):
    value = redact(value)
    if isinstance(value, dict):
        return {key: safe_record(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_record(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def _reply_fields(result):
    def field(name, default=None):
        return result.get(name, default) if isinstance(result, dict) else getattr(result, name, default)
    response = field("response")
    response_map = response if isinstance(response, dict) else {}
    text = field("text")
    if text is None:
        try:
            text = response_map["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            text = None
    return {"provider_call_id": field("call_id") or response_map.get("id"),
            "reply_text": text, "response": response,
            "usage": field("usage", response_map.get("usage", {})),
            "receipt_origin": field("receipt_origin"),
            "http_status": field("http_status")}


def _filename(record):
    # 导入日志中的 ID 不得直接用作文件系统路径。
    return "call-" + hashlib.sha256(record["record_id"].encode("utf-8")).hexdigest()[:24]


def _expanded_content(value):
    """展开消息 content 内的 JSON 以便阅读，保留全部字段。"""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "content" and isinstance(item, str):
                try:
                    decoded = json.loads(item)
                except ValueError:
                    decoded = None
                if isinstance(decoded, (dict, list)):
                    item = decoded
            result[key] = _expanded_content(item)
        return result
    if isinstance(value, list):
        return [_expanded_content(item) for item in value]
    return value


def transcript(record, evidence_file=None, *, full=False):
    """生成便于阅读的视图，配套 JSON 保留完整请求和响应。"""
    context = record.get("context", {})
    request = record["request"]
    messages = request.get("body", request.get("captured_event_payload", request)).get("messages", [])
    def short(value, limit=700):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return text if len(text) <= limit else text[:limit] + "…（摘要截断，完整内容见原始 JSON）"
    lines = ["=" * 48,
             f"第 {context.get('round', '未记录')} 轮｜{PURPOSES.get(record.get('purpose'), '模型调用')}｜{record['status']}",
             f"调用：{record.get('provider_call_id') or '未取得服务方调用ID'}；时间（UTC）：{record.get('started_at')}"]
    if full:
        pretty = lambda value: json.dumps(_expanded_content(value), ensure_ascii=False, indent=2, allow_nan=False)
        lines.extend(["说明：以下完整展示已捕获字段，不截断。content 内的 JSON 字符串仅在展示时展开，原始记录保持不变。",
                      "【发送给模型的完整请求数据】", pretty(request),
                      "【模型返回的完整数据】", pretty(record.get("response"))])
        reply = record.get("reply_text")
        captured_reply = _reply_fields({"response": record.get("response")})["reply_text"]
        if reply is not None and reply != captured_reply:
            # 某些宿主回执将回复放在 response 外部，也需要保留。
            lines.extend(["【单独捕获的模型回复正文】", pretty({"content": reply})])
        if record.get("response") is None and not reply:
            lines.append("未取得服务方返回数据；上面的 null 表示缺失，不能视为成功回复。")
        if record.get("usage") != _reply_fields({"response": record.get("response")})["usage"]:
            lines.extend(["【单独捕获的用量】", pretty(record.get("usage"))])
        if record.get("error"):
            lines.extend(["【完整错误记录】", pretty(record["error"])])
        location = evidence_file or record.get("evidence_file") or _filename(record) + ".json"
        lines.append("原始记录：" + str(location))
        return "\n".join(lines) + "\n"
    for message in messages:
        content = message.get("content", "")
        try:
            body = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            body = None
        if isinstance(body, dict) and "request" in body:
            lines.append("问题：" + short(body["request"]))
            if "parent_metrics" in body:
                m = body["parent_metrics"]
                lines.append(f"当前表现：Macro-F1={m.get('macro_f1')}，Accuracy={m.get('accuracy')}")
            if "diagnosis" in body:
                d = body["diagnosis"]
                lines.append(f"诊断：弱类={d.get('weakest_class')}；主要混淆={short(d.get('top_confusions', [])[:2], 250)}")
            if "allowed_experiments" in body:
                lines.append("允许方案：" + short(body["allowed_experiments"]))
            if "eligible_artifacts" in body:
                lines.append("合格版本：" + "、".join(body["eligible_artifacts"]))
        else:
            lines.append("问题：" + short(content))
    lines.append("模型回复：" + short(record.get("reply_text") or "未取得非空回复，见状态与原始记录"))
    usage = record.get("usage") or {}
    lines.append(f"实际 Token：输入 {usage.get('prompt_tokens', '未提供')}，输出 {usage.get('completion_tokens', '未提供')}")
    if record.get("error"):
        lines.append("异常：" + short(record["error"], 250))
    location = evidence_file or record.get("evidence_file") or _filename(record) + ".json"
    lines.append("完整请求和响应：" + str(location) + "（简表未重复展开）")
    return "\n".join(lines) + "\n"


def _save(directory, record):
    directory = Path(directory)
    stored = dict(record, format="gas-a.model-call.v2")
    # 原始 response 已包含的正文在读取时提取，避免重复保存。
    if stored.get("reply_text") == _reply_fields({"response": stored.get("response")})["reply_text"]:
        stored.pop("reply_text", None)
    write_json(directory / (_filename(record) + ".json"), stored)


def call_path(directory, record_id):
    return Path(directory) / (_filename({"record_id": record_id}) + ".json")


def read_call(path):
    """兼容读取 v1/v2 调用记录，按需从原始 response 提取回复正文。"""
    record = read_json(path)
    if record.get("format") not in ("gas-a.model-call.v1", "gas-a.model-call.v2"):
        raise ValueError("Not a model-call record")
    if "reply_text" not in record:
        record["reply_text"] = _reply_fields({"response": record.get("response")})["reply_text"]
    record["evidence_file"] = str(Path(path).resolve())
    return record


def call_reference(path, root):
    """用相对路径、SHA-256 摘要和调用元数据引用完整原始记录，避免重复存储正文。"""
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Model-call reference is outside its record root")
    record = read_call(path)
    reference = {key: record.get(key) for key in ("record_id", "purpose", "context", "status", "usage",
                  "provider_call_id", "receipt_origin", "http_status", "started_at", "finished_at", "seconds")}
    return {**reference, "format": "gas-a.model-call-ref.v1", "file": path.relative_to(root).as_posix(),
            "sha256": digest(path), "reply_summary": (record.get("reply_text") or "")[:160]}


def resolve_call(reference, root):
    """展开引用前验证路径范围、文件摘要和 record ID，拒绝越界或已变更的记录。"""
    if reference.get("format") != "gas-a.model-call-ref.v1":
        raise ValueError("Not a model-call reference")
    root = Path(root).resolve()
    declared = Path(reference["file"])
    target = root / declared
    path = target.resolve()
    # 引用展开时同时核验位置和内容；报告不能静默读取越界或已经改写的原始记录。
    if declared.is_absolute() or not path.is_relative_to(root) or target.is_symlink() or digest(path) != reference["sha256"]:
        raise ValueError("Model-call reference path or hash mismatch")
    record = read_call(path)
    if record["record_id"] != reference["record_id"]:
        raise ValueError("Model-call reference ID mismatch")
    return record


def expand_receipts(row, root):
    """展开新版记录引用，同时兼容旧版直接内嵌的服务方回执。"""
    result = []
    for receipt in row.get("provider_receipts", []):
        if receipt.get("format") != "gas-a.model-call-ref.v1":
            result.append(receipt)
            continue
        record = resolve_call(receipt, root)
        request = record["request"]
        result.append({"call_id": record["provider_call_id"], "text": record.get("reply_text"),
                       "response": record.get("response"), "usage": record.get("usage"),
                       "messages": request.get("body", request).get("messages", []),
                       "receipt_origin": record.get("receipt_origin"), "record_id": record["record_id"]})
    return result


def receipt_reference(receipt, directory):
    if getattr(receipt, "record_ref", None) is not None:
        return receipt.record_ref
    for path in (Path(directory) / "model_calls").glob("call-*.json"):
        if read_json(path).get("provider_call_id") == receipt.call_id:
            return call_reference(path, directory)
    raise ValueError("Missing canonical record for model receipt")


def _index(directory, record):
    entry = {key: record.get(key) for key in ("record_id", "purpose", "context", "status", "provider_call_id", "usage", "started_at", "finished_at")}
    entry["file"] = _filename(record) + ".json"
    append_jsonl(Path(directory) / "index.jsonl", entry)


def capture_call(directory, request, invoke, *, context=None, purpose="model_call", record_id=None, secrets=()):
    """调用前先持久化请求；失败时先记录证据，再继续抛出异常。"""
    record = {"format": "gas-a.model-call.v1", "record_id": record_id or uuid.uuid4().hex,
              "purpose": purpose, "context": context or {}, "started_at": stamp(), "finished_at": None,
              "status": "pending", "request": request, "provider_call_id": None,
              "reply_text": None, "response": None, "usage": None, "error": None}
    record = safe_record(record, secrets)
    # 调用前写入 pending，即使进程中断，也能区分已尝试请求与已取得响应。
    _save(directory, record)
    started = time.monotonic()
    try:
        result = invoke()
    except Exception as exc:
        record.update(status="error", response=getattr(exc, "response", None),
                      provider_call_id=getattr(exc, "request_id", None),
                      http_status=getattr(exc, "http_status", None),
                      error={"type": type(exc).__name__, "message": str(exc)})
        if isinstance(record["response"], dict):
            response = record["response"]
            record["provider_call_id"] = response.get("id") or record["provider_call_id"]
            record["usage"] = response.get("usage")
            record["reply_text"] = _reply_fields({"response": response})["reply_text"]
        record.update(finished_at=stamp(), seconds=time.monotonic() - started)
        record = safe_record(record, secrets)
        _save(directory, record)
        _index(directory, record)
        raise
    record.update(_reply_fields(result))
    record.update(status="success" if isinstance(record.get("reply_text"), str) and record["reply_text"].strip() else "empty_response",
                  finished_at=stamp(), seconds=time.monotonic() - started)
    record = safe_record(record, secrets)
    _save(directory, record)
    _index(directory, record)
    return result


def recorded_chat(client, directory, messages, *, max_tokens=768, temperature=0,
                  enable_thinking=None, context=None, purpose=None, record_id=None):
    # 直连 HTTP 客户端仅供本地使用，不进入赛事镜像。
    """记录本地 HTTP 客户端实际发送的请求参数，并成对保留服务方响应。"""
    from .live_access import BASE_URL, chat_payload
    body = chat_payload(client.model, messages, max_tokens, temperature, enable_thinking)
    options = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return capture_call(directory, {"transport": "pytrio_http",
                        "endpoint": getattr(client, "base_url", BASE_URL) + "/chat/completions", "body": body},
                        lambda: client.complete(messages, max_tokens=max_tokens, temperature=temperature, **options),
                        context=context, purpose=purpose or purpose_for(messages), record_id=record_id,
                        secrets=(getattr(client, "api_key", None),))


def recorded_round_model(rnd, directory, messages, *, max_tokens, temperature, purpose):
    # LocalRound 已记录实际 HTTP 请求体，此处避免重复记录同次调用。
    """记录宿主 Round 模型调用；LocalRound 已有真实请求记录时复用其引用。"""
    if getattr(rnd, "records_model_calls", False):
        return rnd.model(messages, max_tokens=max_tokens, temperature=temperature)
    return capture_call(Path(directory) / "model_calls",
                        {"transport": "contest_host_rpc", "model": getattr(rnd, "model_name", None),
                         "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                         "capture_note": "参与者发送的完整消息；宿主额外注入的官方提示词由宿主管理。"},
                        lambda: rnd.model(messages, max_tokens=max_tokens, temperature=temperature),
                        context={"round": rnd.round_index, "origin": "contest_host_rpc"}, purpose=purpose)


def _event_records(path):
    pending, records = [], []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            kind, payload = event.get("event"), event.get("payload", {})
            if kind == "model_call" and payload.get("format") == "gas-a.model-call-ref.v1":
                for root in (path.parent, *path.parent.parents):
                    if (root / payload["file"]).is_file():
                        records.append(resolve_call(payload, root))
                        break
                else:
                    raise ValueError("Canonical model-call record is missing: " + payload["record_id"])
            elif kind == "model_request":
                rid = payload.get("record_id") or event.get("event_id")
                rid = rid or hashlib.sha256((str(path.resolve()) + ":" + str(number)).encode()).hexdigest()
                record = {"format": "gas-a.model-call.v1", "record_id": rid,
                          "purpose": payload.get("purpose") or purpose_for(payload.get("messages", [])),
                          "context": {"session": path.parent.name, "round": payload.get("round"), "sequence": len(records) + 1},
                          "started_at": event.get("time"), "finished_at": None, "status": "pending",
                          "request": {"captured_event_payload": payload,
                                      "capture_note": "来自已有model_request审计；未记录的请求参数不补填。"},
                          "provider_call_id": None, "reply_text": None, "response": None, "usage": None,
                          "error": None, "source": {"path": str(path.resolve()), "line": number}}
                pending.append(record)
                records.append(record)
            elif kind in ("model_response", "model_response_invalid", "model_error"):
                match = next((r for r in pending if payload.get("record_id") == r["record_id"]), None)
                if match is None:
                    match = next((r for r in pending if r["context"]["round"] == payload.get("round")), None)
                if match is None:
                    continue
                match.update(_reply_fields(payload))
                match.update(finished_at=event.get("time"), seconds=payload.get("seconds"),
                             status="success" if payload.get("text") else "empty_response")
                if kind != "model_response":
                    match.update(status="error", error={"message": payload.get("error"), "event": kind})
                pending.remove(match)
            elif kind == "session_failed":
                for record in pending:
                    record.update(status="error", finished_at=event.get("time"),
                                  error={"message": payload.get("error"),
                                         "note": "原审计未保存该调用的服务方回复正文，无法补回。"})
                pending.clear()
    return records


def collect_model_calls(source, exclude=None):
    """汇总并去重已保存的调用和旧版事件，包括失败尝试。"""
    source = Path(source).resolve()
    exclude = Path(exclude).resolve() if exclude is not None else None
    if not source.exists():
        raise ValueError("Model-call source does not exist: " + str(source))
    records = {}
    candidates = [source] if source.is_file() and source.suffix == ".json" else source.rglob("call-*.json") if source.is_dir() else []
    for path in sorted(candidates):
        if exclude is not None and path.is_relative_to(exclude):
            continue
        record = read_json(path)
        if record.get("format") in ("gas-a.model-call.v1", "gas-a.model-call.v2"):
            record = read_call(path)
            records[record["record_id"]] = record
    events = [source] if source.is_file() and source.name == "events.jsonl" else source.rglob("events.jsonl") if source.is_dir() else []
    for path in sorted(events):
        if exclude is not None and path.is_relative_to(exclude):
            continue
        for record in _event_records(path):
            records.setdefault(record["record_id"], record)
    ordered = sorted((safe_record(r) for r in records.values()), key=lambda r: (
        r.get("started_at") or "", r.get("context", {}).get("session") or "",
        r.get("context", {}).get("round") or 0, r.get("context", {}).get("sequence") or 0, r["record_id"]))
    return ordered


def export_model_calls(source, output):
    """导出引用索引和可读对话，不复制已有的原始记录文件。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("Export destination must not contain or replace the source")
    ordered = collect_model_calls(source, exclude=output)
    if not ordered:
        raise ValueError("No recorded model requests were found")
    output.mkdir(parents=True, exist_ok=True)
    references = []
    for record in ordered:
        if record.get("evidence_file"):
            references.append(call_reference(record["evidence_file"], source if source.is_dir() else source.parent))
        else:
            # 旧版仅有事件的记录保持原位，以原始事件位置为准。
            references.append({key: record.get(key) for key in ("record_id", "status", "source", "provider_call_id", "purpose")})
    (output / "model_calls.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in references), encoding="utf-8")
    (output / "conversations.txt").write_text("大模型请求与回复汇总（依据实际记录；时间为UTC）\n\n" + "\n".join(transcript(r, full=True) for r in ordered), encoding="utf-8")
    summary = {"calls": len(ordered), "status_counts": {status: sum(r["status"] == status for r in ordered)
               for status in ("success", "empty_response", "error", "pending")},
               "source": str(source), "directory": str(output), "transcript": str(output / "conversations.txt"),
               "reference_root": str(source if source.is_dir() else source.parent),
               "index": str(output / "model_calls.jsonl"), "external_model_calls": 0}
    write_json(output / "summary.json", summary)
    return summary
