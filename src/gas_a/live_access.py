"""本地开发使用的 PyTRiO 接入；赛事镜像仍通过宿主 RPC 调用模型。"""
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from .redaction import redact

ALLOWED = {"PYTRIO_API_KEY", "PYTRIO_MODEL", "SWANLAB_API_KEY", "GAS_A_SWANLAB_PROJECT", "GAS_A_SWANLAB_WORKSPACE", "SWANLAB_PROJECT", "SWANLAB_WORKSPACE"}
ALIASES = {"SWANLAB_PROJECT": "GAS_A_SWANLAB_PROJECT", "SWANLAB_WORKSPACE": "GAS_A_SWANLAB_WORKSPACE"}
BASE_URL = "https://pytrio.com/api/openai/v1"


class ProviderCallError(RuntimeError):
    def __init__(self, message, response=None, http_status=None, request_id=None):
        super().__init__(message)
        self.response = response
        self.http_status = http_status
        self.request_id = request_id


class InvalidProviderResponse(ProviderCallError):
    pass


def chat_payload(model, messages, max_tokens=768, temperature=0, enable_thinking=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    if enable_thinking is not None:
        if type(enable_thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean")
        body["enable_thinking"] = enable_thinking
    return body


def load_credentials(path):
    """按字面值读取 KEY=value 行，不执行 Shell 代码或进行变量插值。"""
    path = Path(path)
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in ALLOWED:
            raise ValueError(f"Unsupported credential entry at line {number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        name = ALIASES.get(name, name)
        if value and not os.environ.get(name):
            os.environ[name] = value


class PyTrioChat:
    def __init__(self, model=None, api_key=None, timeout=90, base_url=BASE_URL):
        self.model = model or os.environ.get("PYTRIO_MODEL", "").strip()
        self.api_key = api_key or os.environ.get("PYTRIO_API_KEY", "").strip()
        if not self.model or not self.api_key:
            raise ValueError("Set PYTRIO_MODEL and PYTRIO_API_KEY in .env.local; use the platform-approved model identity")
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    def complete(self, messages, max_tokens=768, temperature=0, enable_thinking=None):
        started = time.monotonic()
        body = chat_payload(self.model, messages, max_tokens, temperature, enable_thinking)
        request = Request(self.base_url + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        raw, http_status, request_id = None, None, None
        try:
            with urlopen(request, timeout=self.timeout) as reply:
                request_id = reply.headers.get("x-request-id")
                http_status = getattr(reply, "status", None)
                raw = reply.read().decode("utf-8", errors="replace")
                response = json.loads(raw)
        except HTTPError as exc:
            status = exc.code
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            raw = exc.read().decode("utf-8", errors="replace")
            exc.close()
            try:
                error_response = json.loads(raw)
            except ValueError:
                error_response = raw
            if status in (402, 409) or "billing_insufficient_balance" in raw:
                raise ProviderCallError("PyTRiO balance is insufficient or overdue; resolve workspace billing before retrying", error_response, status, request_id) from None
            if status in (401, 403):
                raise ProviderCallError("PyTRiO authentication/model access failed; check the API Key and model permissions", error_response, status, request_id) from None
            raise ProviderCallError(f"PyTRiO HTTP {status}; {redact(raw)[:400]}", error_response, status, request_id) from None
        except (URLError, TimeoutError) as exc:
            raise ProviderCallError("PyTRiO connection failed: " + str(redact(str(exc)))[:300]) from None
        except (ValueError, UnicodeError):
            raise InvalidProviderResponse("PyTRiO returned an invalid JSON response", raw, http_status, request_id) from None
        try:
            text = response["choices"][0]["message"].get("content")
            identifier = response.get("id") or request_id
            if not isinstance(identifier, str) or not identifier or text is not None and not isinstance(text, str):
                raise ValueError("invalid content or missing provider receipt")
        except (KeyError, IndexError, TypeError, ValueError):
            raise InvalidProviderResponse("PyTRiO response lacks a valid message or an actual provider completion/request ID", response, http_status, request_id) from None
        return {"call_id": identifier, "text": text or "", "response": response,
                "usage": response.get("usage", {}), "seconds": time.monotonic() - started,
                "receipt_origin": "pytrio_openai_api_not_contest_host", "http_status": http_status}
