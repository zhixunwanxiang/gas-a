"""共享测试替身；不调用真实模型或云服务。"""
import base64
import hashlib
import json
from types import SimpleNamespace


class ChunkedUploads:
    """SDK 1.0.4 离线宿主上传存储；校验清单、偏移、摘要及每轮预留预算。"""
    limits = {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": 512 * 1024 * 1024,
              "state_bytes": 512 * 1024 * 1024, "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
              "state_upload_bytes_per_round": 512 * 1024 * 1024, "upload_chunk_bytes": 64 * 1024}

    def __init__(self):
        self.uploads = {}
        self.reserved = {"candidate": 0, "state": 0}
        self.chunk_count = 0

    def handle(self, frame):
        request = frame["type"]
        if request == "upload_begin_request":
            kind, files = frame["kind"], frame["files"]
            if kind not in self.reserved or len(files) > 64:
                raise AssertionError("Invalid upload kind or file count")
            size = sum(item["size_bytes"] for item in files)
            per_upload = self.limits["artifact_bytes" if kind == "candidate" else "state_bytes"]
            if size > per_upload or self.reserved[kind] + size > self.limits[kind + "_upload_bytes_per_round"]:
                raise AssertionError("Upload budget exceeded")
            if len({item["path"] for item in files}) != len(files) or any("data_base64" in item for item in files):
                raise AssertionError("Expected unique manifest paths without inline file bytes")
            self.reserved[kind] += size
            identity = f"upload-{len(self.uploads) + 1}"
            self.uploads[identity] = {"kind": kind, "manifest": files,
                                      "data": {item["path"]: bytearray() for item in files}, "finished": False}
            return {"type": "upload_begin_response", "upload_id": identity}
        upload = self.uploads[frame["upload_id"]]
        if request == "upload_chunk_request":
            data = upload["data"][frame["path"]]
            chunk = base64.b64decode(frame["data_base64"], validate=True)
            declared = next(item for item in upload["manifest"] if item["path"] == frame["path"])
            if (upload["finished"] or frame["offset"] != len(data)
                    or len(chunk) > self.limits["upload_chunk_bytes"] or len(data) + len(chunk) > declared["size_bytes"]):
                raise AssertionError("Invalid upload chunk")
            data.extend(chunk)
            self.chunk_count += 1
            return {"type": "upload_chunk_response", "offset": len(data)}
        if request != "upload_finish_request":
            raise AssertionError("Unknown upload frame")
        for item in upload["manifest"]:
            data = upload["data"][item["path"]]
            if len(data) != item["size_bytes"] or "sha256:" + hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise AssertionError("Incomplete or changed upload")
        upload["finished"] = True
        return {"type": "upload_finish_response", "upload_id": frame["upload_id"]}

    def files(self, frame, kind):
        if "files" in frame:
            raise AssertionError("SDK 1.0.4 candidate/state frames must reference upload_id")
        upload = self.uploads[frame["upload_id"]]
        if not upload["finished"] or upload["kind"] != kind:
            raise AssertionError("Upload is not committed for this operation")
        # 仅测试宿主内部转换，线上帧中没有整份内联产物。
        return [{"path": name, "data_base64": base64.b64encode(data).decode()}
                for name, data in upload["data"].items()]


def legacy_config(*args, **kwargs):
    """原菜单协议的回归明确使用兼容模式；新默认组合协议由test_planning覆盖。"""
    from gas_a.pipeline import load_config
    result = load_config(*args, **kwargs)
    result.setdefault("controller", {})["planning_mode"] = "menu"
    result["gate"]["reject_zero_target_recall"] = False  # 历史配置的弱类下限仅作预警。
    result["gate"].pop("max_parent_fixed_f1_drop", None)
    return result


def fixture_plan(prompt, names=None, *, action="experiment"):
    if names is None:
        names = [s["name"] for s in prompt["allowed_experiments"][:prompt["budget"]["max_experiments"]]]
    if not names:
        action = "observe"
    return {"action": action, "experiment_names": [] if action == "observe" else names,
            "fallback": "retain_parent",
            "evidence_ids": ["C", "F"], "hypothesis": "可能存在迁移差异，离线测试验证候选收益。",
            "success_condition": "configured_quality_gates",
            "stop_condition": "observe_then_reassess" if action == "observe" else "reject_if_gates_fail",
            "claims": [], "reason": "明确标记的离线测试方案。"}


class CloudFixture:
    def __init__(self, fail_init=False):
        self.records, self.finished, self.fail_init = [], 0, fail_init
        self.definitions = []

    def Settings(self, **kwargs):
        return kwargs

    def login(self, **kwargs):
        if kwargs["save"]:
            raise AssertionError("Credentials must not be persisted by SDK login")

    def init(self, **kwargs):
        if self.fail_init:
            raise RuntimeError("fixture cloud unavailable")
        if kwargs["mode"] != "online" or kwargs["on_online_error"] != "raise" or kwargs["public"]:
            raise AssertionError("Must explicitly request private online logging without fallback")
        return SimpleNamespace(mode="online", url="https://fixture.invalid/experiments/offline-test",
                               define_metric=lambda key, **options: self.definitions.append((key, options)),
                               log=lambda data, step: self.records.append((data, step)))

    def Text(self, data, caption=None):
        return {"text": data, "caption": caption}

    def finish(self, **kwargs):
        self.finished += 1


class ModelFixture:
    model = "offline-fixture-only"

    def __init__(self, invalid_selection=False):
        self.calls, self.invalid_selection = 0, invalid_selection

    def complete(self, messages, **kwargs):
        self.calls += 1
        prompt = json.loads(messages[0]["content"])
        if "allowed_experiments" in prompt:
            answer = fixture_plan(prompt)
        else:
            eligible = prompt["eligible_artifacts"]
            selected = max(eligible, key=lambda k: eligible[k].get("host", eligible[k]["local"])["macro_f1"])
            answer = {"selected_artifact_id": "not-eligible" if self.invalid_selection else selected, "reason": "fixture comparison"}
        text = json.dumps(answer)
        return {"call_id": f"fixture-call-{self.calls}", "text": text,
                "response": {"choices": [{"message": {"content": text}}]},
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}, "seconds": .001,
                "receipt_origin": "offline_test_fixture"}


class ReplayModelFixture:
    model = "temporal-offline-test-fixture"

    def __init__(self, invalid=False):
        self.requests, self.invalid = [], invalid

    def complete(self, messages, **options):
        prompt = json.loads(messages[-1]["content"])
        self.requests.append(prompt)
        if "allowed_experiments" in prompt:
            result = fixture_plan(prompt, ["outside-menu" if self.invalid else prompt["allowed_experiments"][0]["name"]])
        else:
            result = {"selected_artifact_id": next(iter(prompt["eligible_artifacts"])), "reason": "retain fixture"}
        text = json.dumps(result)
        return {"call_id": f"temporal-fixture-{len(self.requests)}", "text": text,
                "response": {"choices": [{"message": {"content": text}}]},
                "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "seconds": .001,
                "receipt_origin": "explicit_offline_test_fixture"}
