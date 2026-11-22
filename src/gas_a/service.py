"""本地轻量推理服务，每次请求均读取通过原子替换维护的活动版本指针。"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from .registry import Registry
from .data import load_model
from .io import lock


class ActivePredictor:
    """按活动版本和摘要缓存模型，每次请求仍检查 checkpoint 完整性。"""
    def __init__(self, runtime):
        self.registry = Registry(runtime)
        self.key = None
        self.model = None

    def load_active(self):
        """正常读取不持写锁；损坏时在运行锁内恢复完整基线，全部不可用则明确拒绝。"""
        try:
            version = self.registry.active_id()
            checkpoint = self.registry.checkpoint(version)  # 即使命中缓存，也检查文件完整性。
            key = (version, self.registry.entries()[version]["sha256"])
            if key != self.key:
                self.model = load_model(checkpoint)
                self.key = key
        except (OSError, ValueError, KeyError) as exc:
            try:
                with lock(self.registry.root):
                    self.registry.recover_activation()
                    self.registry.checkpoint("baseline")
                    self.registry.activate("baseline", "http_integrity_recovery:" + str(exc))
                    version = "baseline"
                    checkpoint = self.registry.checkpoint(version)
                    self.model = load_model(checkpoint)
                    self.key = (version, self.registry.entries()[version]["sha256"])
            except (OSError, ValueError, KeyError, RuntimeError) as recovery_error:
                raise RuntimeError("Active model unavailable and safe recovery failed: " + str(recovery_error)) from recovery_error
        return version

    def predict(self, inputs):
        """解析当前活动指针，必要时恢复和更新缓存，再返回固定六类概率。"""
        from .cli import prediction
        version = self.load_active()
        return {"version": version, "outputs": prediction(self.model, inputs)}


def make_server(runtime, host="127.0.0.1", port=8080):
    """创建本地 HTTP 服务：GET /health 检查模型可用性，POST /predict 执行受限输入预测。"""
    predictor = ActivePredictor(runtime)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "unknown route"})
            try:
                version = predictor.load_active()
                self.respond(200, {"status": "ok", "version": version})
            except (ValueError, OSError, KeyError, RuntimeError) as exc:
                self.respond(503, {"status": "unavailable", "error": str(exc)})

        def do_POST(self):
            if self.path != "/predict":
                return self.respond(404, {"error": "unknown route"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Request body must be 1..65536 bytes")
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict) or set(request) != {"inputs"}:
                    raise ValueError("Expected {inputs: {features: [...]}}")
                self.respond(200, predictor.predict(request["inputs"]))
            except (ValueError, KeyError) as exc:
                self.respond(400, {"error": str(exc)})
            except (OSError, RuntimeError) as exc:
                self.respond(503, {"error": str(exc)})

    return HTTPServer((host, port), Handler)


def serve(runtime, host="127.0.0.1", port=8080):
    server = make_server(runtime, host, port)
    server.timeout = 30
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
