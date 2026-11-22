"""联网模式的 CLI 分发、凭据配置、连通性检查和审计记录管理。"""
import getpass
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid
from .pipeline import load_config
from .live_access import ALLOWED, load_credentials, PyTrioChat, redact
from .cloud_records import Audit, import_records
from .live import run_live
from .model_records import export_model_calls, recorded_chat


def configure(path, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise ValueError("Configuration already exists; edit .env.local or explicitly use --replace")
    if replace:
        load_credentials(path)
    print("API Keys are entered with hidden input and saved only in this local file.")
    values = {}
    for key in ("PYTRIO_API_KEY", "PYTRIO_MODEL", "SWANLAB_API_KEY", "GAS_A_SWANLAB_PROJECT", "GAS_A_SWANLAB_WORKSPACE"):
        if key.endswith("API_KEY"):
            value = getpass.getpass(key + " (blank keeps current value): ") or os.environ.get(key, "")
        else:
            default = os.environ.get(key, "gas-a" if key == "GAS_A_SWANLAB_PROJECT" else "")
            value = input(key + (f" [{default}]" if default else "") + ": ").strip() or default
        if any(c in value for c in "\r\n\0"):
            raise ValueError("Configuration values must be single-line literals")
        values[key] = value
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write("# Local credentials. Excluded from git, Docker context and deliverables.\n")
        output.write("\n".join(f"{key}={value}" for key, value in values.items()) + "\n")
    return {"saved_to": str(path.resolve()), "configured": {key: bool(value) for key, value in values.items()}}


def dispatch(args):
    if args.command == "live-configure":
        result = configure(args.env_file, args.replace)
    else:
        load_credentials(args.env_file)
        config_path = "configs/a.live.json" if args.config == "configs/a.json" else args.config
        config = load_config(config_path)
        if args.command == "doctor":
            import numpy as np
            present = {key: bool(os.environ.get(key, "").strip()) for key in ALLOWED}
            pytrio_ready = present["PYTRIO_API_KEY"] and present["PYTRIO_MODEL"]
            cloud_ready = present["SWANLAB_API_KEY"] and bool(importlib.util.find_spec("swanlab"))
            result = {"pytrio_configuration_ready": pytrio_ready, "cloud_configuration_ready": cloud_ready,
                      "credentials_present": present, "swanlab_installed": bool(importlib.util.find_spec("swanlab")),
                      "python": sys.version.split()[0], "numpy": np.__version__,
                      "pinned_numpy_match": np.__version__ == "1.26.4", "network_checked": False,
                      "note": "Presence checks do not verify keys, model permissions, billing or cloud connectivity"}
            if args.probe_model:
                record_directory = Path(config["runtime"]) / "probes" / uuid.uuid4().hex[:10] / "model_calls"
                receipt = recorded_chat(PyTrioChat(), record_directory,
                                        [{"role": "user", "content": "Reply with OK only."}],
                                        max_tokens=32, purpose="connectivity_probe")
                probe = {key: receipt[key] for key in ("call_id", "usage", "seconds", "receipt_origin")}
                probe["record_directory"] = str(record_directory)
                result.update(network_checked=True, pytrio_probe=probe)
                from .io import append_jsonl
                append_jsonl(Path(config["runtime"]) / "model_probes.jsonl", probe)
            print(json.dumps(redact(result), ensure_ascii=False, indent=2))
            return 0 if pytrio_ready and cloud_ready else 2
        elif args.command == "export-model-calls":
            result = export_model_calls(args.source or config["runtime"],
                                        args.output or Path(config["runtime"]) / "model_calls_archive")
        elif args.command == "agent-run":
            result = run_live(config, rounds=args.rounds, cloud=not args.local_records)
        elif args.command == "temporal-run":
            from .temporal import run_temporal
            from .adaptive import RulePlanner
            config["runtime"] = str(Path(args.output).resolve())
            config["prior_attempts"] = [str(Path(path).resolve()) for path in args.prior_attempt]
            rule = args.controller == "rule"
            result = run_temporal(config, args.dataset_directory, cloud=False if rule else not args.local_records,
                                  client=RulePlanner() if rule else None)
        elif args.command == "sync-records":
            directory = Path(config["runtime"]) / "imports" / uuid.uuid4().hex[:10]
            audit = Audit(directory, cloud=True)
            succeeded = False
            try:
                audit.start({"mode": "historical_import", "source": str(Path(args.source).resolve()),
                             "does_not_reclassify_rule_runs_as_llm_runs": True})
                count = import_records(args.source, audit)
                succeeded = True
            finally:
                receipt = audit.finish(succeeded)
            result = {"imported_records": count, "cloud": receipt}
        else:
            raise ValueError("Unknown live command")
    display = result
    if args.command == "temporal-run":
        aggregate = result["aggregate_frozen_operational"]
        display = {"完成": result["success"], "窗口数": result["finished_windows"],
                   "Macro-F1": aggregate.get("operational", {}).get("macro_f1"),
                   "静态基线": aggregate.get("static_baseline", {}).get("macro_f1"),
                   "调用次数": result["costs"]["calls"], "已知Token": result["costs"]["tokens_known"],
                   "报告": result["report"], "SwanLab": (result.get("cloud") or {}).get("experiment_url")}
        display["云端上传状态"] = (result.get("cloud") or {}).get("status")
        display["待补传事件"] = (result.get("cloud") or {}).get("pending_uploads", 0)
        display["已恢复窗口"] = result.get("recovered_windows", 0)
    elif args.command == "agent-run":
        display = {"活动版本": result["active_version"], "累计完成轮数": result["completed_rounds"],
                   "Macro-F1": result["metrics"]["macro_f1"], "模型": result["model"],
                   "记录目录": result["audit_directory"], "SwanLab": result["cloud"].get("experiment_url")}
        display["云端上传状态"] = result["cloud"].get("status")
        display["待补传事件"] = result["cloud"].get("pending_uploads", 0)
    print(json.dumps(redact(display), ensure_ascii=False, indent=2))
    if args.command == "temporal-run" and not result.get("success", False):
        return 2
    return 0
