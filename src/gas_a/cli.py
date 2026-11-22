"""任务 A 的 CLI 入口：分发训练、监测、评估、推理、回滚和记录管理命令。"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from .data import load_pair, load_data, load_unlabelled, load_model, evaluate
from .pipeline import load_config, run
from .registry import Registry
from .monitor import monitor, diagnose
from .io import lock, CLASSES


def prediction(model, inputs):
    """校验单条 128 维输入，输出六类概率；最高分并列时使用确定性的微小扰动打破平局。"""
    if not isinstance(inputs, dict) or set(inputs) != {"features"}:
        raise ValueError("Only inputs.features is accepted")
    values = inputs["features"]
    if not isinstance(values, list) or len(values) != 128 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("Exactly 128 numeric features are required")
    x = np.asarray([values], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite features")
    if model.meta.get("classes") != CLASSES:
        raise ValueError("Task A requires the fixed six gas classes in their original order")
    result = np.asarray(model.predict(x), dtype=np.float64)
    if (result.shape != (1, 6) or not np.isfinite(result).all() or np.any(result < 0)
            or not np.allclose(result.sum(axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError("Task A prediction must contain six finite probabilities summing to one")
    scores = result[0].copy()
    top = int(scores.argmax())
    if np.sum(scores == scores[top]) > 1:
        scores[top] += 1e-12
        scores /= scores.sum()
    return [{"label": label, "score": float(score)} for label, score in zip(model.meta["classes"], scores)]


def main(argv=None):
    """解析 CLI 参数并分发命令；联网能力仅在相关子命令中延迟导入。"""
    parser = argparse.ArgumentParser(description="A-task gas classifier lifecycle")
    parser.add_argument("--config", default="configs/a.json")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run"); p.add_argument("--rounds", type=int)
    sub.add_parser("evaluate"); sub.add_parser("status")
    p = sub.add_parser("monitor"); p.add_argument("--current", help="Optional unlabelled JSONL feature window")
    p = sub.add_parser("predict"); p.add_argument("--checkpoint")
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8080)
    p = sub.add_parser("blind-evaluate", help="Feature-only subprocesses; score labels only after the entire prediction stream ends")
    p.add_argument("--checkpoint", required=True); p.add_argument("--baseline")
    p.add_argument("--dataset-directory", required=True); p.add_argument("--output", required=True)
    p.add_argument("--row-timeout", type=float, default=10)
    p = sub.add_parser("rollback"); p.add_argument("--version", default="baseline"); p.add_argument("--reason", required=True)
    p = sub.add_parser("live-configure"); p.add_argument("--env-file", default=".env.local"); p.add_argument("--replace", action="store_true")
    p = sub.add_parser("doctor"); p.add_argument("--env-file", default=".env.local"); p.add_argument("--probe-model", action="store_true", help="One actual billed PyTRiO request")
    p = sub.add_parser("agent-run"); p.add_argument("--env-file", default=".env.local"); p.add_argument("--rounds", type=int, default=1); p.add_argument("--local-records", action="store_true", help="Explicitly use local audit only; does not create cloud records")
    p = sub.add_parser("temporal-run", help="Chronological public DAT replay with real PyTRiO and SwanLab; original training set only")
    p.add_argument("--env-file", default=".env.local")
    p.add_argument("--dataset-directory", required=True)
    p.add_argument("--output", required=True, help="Fresh directory for this complete replay")
    p.add_argument("--local-records", action="store_true")
    p.add_argument("--prior-attempt", action="append", default=[], help="Previous attempt directory; count its model/training costs")
    p.add_argument("--controller", choices=("pytrio", "rule"), default="pytrio", help="rule is CPU screening with no external model/cloud calls")
    p = sub.add_parser("sync-records"); p.add_argument("--env-file", default=".env.local"); p.add_argument("--source", required=True)
    p = sub.add_parser("export-model-calls"); p.add_argument("--env-file", default=".env.local"); p.add_argument("--source"); p.add_argument("--output")
    p = sub.add_parser("review-run", help="Export concise scoring evidence without model or cloud calls"); p.add_argument("--source"); p.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "review-run":
            from .review import export_review
            cfg = load_config(args.config)
            source = Path(args.source or cfg["runtime"])
            print(json.dumps(export_review(source, args.output or source / "review"), ensure_ascii=False, indent=2))
            return 0
        if args.command in ("live-configure", "doctor", "agent-run", "temporal-run", "sync-records", "export-model-calls"):
            from .live_commands import dispatch
            return dispatch(args)
        cfg = load_config(args.config)
        registry = Registry(cfg["runtime"])
        if args.command == "blind-evaluate":
            from .blind import run_blind
            summary = run_blind(args.checkpoint, args.baseline or cfg["baseline"], args.dataset_directory,
                                args.output, timeout=args.row_timeout)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        elif args.command == "run":
            run(cfg, args.rounds)
        elif args.command == "status":
            active = registry.active_id()
            print(json.dumps({"active": active, "checkpoint": str(registry.checkpoint(active)), "versions": registry.entries()}, ensure_ascii=False))
        elif args.command == "predict":
            if args.checkpoint:
                model = load_model(args.checkpoint)
            else:
                from .service import ActivePredictor
                live = ActivePredictor(cfg["runtime"])
            for line in sys.stdin:
                if not line.strip():
                    continue
                request = json.loads(line)
                if not isinstance(request, dict) or set(request) - {"task", "inputs"} or request.get("task", "classification") != "classification":
                    raise ValueError("Prediction request is {task: classification, inputs: ...}")
                response = {"outputs": prediction(model, request["inputs"])} if args.checkpoint else live.predict(request["inputs"])
                print(json.dumps(response), flush=True)
        elif args.command == "serve":
            from .service import serve
            serve(cfg["runtime"], args.host, args.port)
        elif args.command in ("evaluate", "monitor"):
            if args.command == "monitor" and args.current:
                train = load_data(cfg["train"])
                current = load_unlabelled(args.current)
                print(json.dumps({"monitor": monitor(train["x"], current, cfg["monitor"]),
                                  "diagnosis": {"status": "labels_required_for_error_attribution"}}, ensure_ascii=False, indent=2))
                return 0
            train, val = load_pair(cfg["train"], cfg["validation"])
            cp = registry.checkpoint(registry.active_id()) if (Path(cfg["runtime"]) / "active.json").exists() else cfg["baseline"]
            metric = evaluate(load_model(cp, train["paths"]), val)
            if args.command == "evaluate":
                print(json.dumps(metric, ensure_ascii=False, indent=2))
            else:
                drift = monitor(train["x"], val["x"], cfg["monitor"])
                print(json.dumps({"monitor": drift, "diagnosis": diagnose(train, val, metric, drift)}, ensure_ascii=False, indent=2))
        elif args.command == "rollback":
            with lock(cfg["runtime"]):
                # 手动回滚也要在当前 validation 上检查相对静态 baseline 的质量下限。
                train, val = load_pair(cfg["train"], cfg["validation"])
                candidate = evaluate(load_model(registry.checkpoint(args.version), train["paths"]), val)
                baseline = evaluate(load_model(registry.checkpoint("baseline"), train["paths"]), val)
                if candidate["macro_f1"] + 1e-12 < baseline["macro_f1"]:
                    raise ValueError("Rollback target is below the static baseline on current validation")
                registry.activate(args.version, "manual_rollback:" + args.reason)
                print(json.dumps({"active": args.version, "metrics": candidate}))
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"gas-a: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
