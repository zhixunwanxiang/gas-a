"""宿主 RPC 审计与本地开发工具共用的凭据脱敏逻辑。

本模块不包含客户端、服务端点或网络依赖。"""
import os
import re


def redact(value):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if not isinstance(item, bool) and item is not None and re.search(r"api.?key|password|authorization|secret|access.?token", str(key), re.I)
                else redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for key in ("PYTRIO_API_KEY", "SWANLAB_API_KEY"):
            secret = os.environ.get(key)
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value
