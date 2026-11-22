"""Independent evidence checks, numeric training reproduction and submission inventory.

Only public train/validation are used for reconstruction. No provider model calls.
"""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
from fixtures import ChunkedUploads
from gas_a.io import digest, read_json, write_json
from gas_a.model_records import purpose_for
from gas_a.live_access import load_credentials


def check_round(directory):
    from gas_a.contest import parse_object, sdk_text
    result, start = read_json(directory / "result.json"), read_json(directory / "start.json")
    uploads, waiting, histories, measurements, calls, types = ChunkedUploads(), None, None, {}, [], []
    finish, ack, chunks = None, False, 0
    for line in (directory / "protocol-duplex.jsonl").read_text(encoding="utf-8").splitlines():
        wire = json.loads(line)
        frame = wire["frame"]
        kind = frame["type"]
        types.append(kind)
        assert not ack, "Frames after final acknowledgement"
        if wire["direction"] == "participant_to_host":
            assert waiting is None
            if kind == "round_complete":
                finish = frame["record"]
            else:
                assert kind != "round_failed"
                waiting = frame
                if kind.startswith("upload_"):
                    expected = uploads.handle(frame)
                    waiting = {**frame, "_expected": expected}
                    chunks += kind == "upload_chunk_request"
        else:
            if kind == "round_start":
                assert frame == start
                continue
            if kind == "round_ack":
                assert finish and frame["accepted"] is True
                ack = True
                continue
            assert waiting and frame["id"] == waiting["id"]
            expected_kind = waiting["type"].removesuffix("_request") + "_response"
            assert kind == expected_kind
            if "_expected" in waiting:
                assert {k: v for k, v in frame.items() if k != "id"} == waiting["_expected"]
            elif kind == "history_response":
                histories = [item["id"] for item in frame["records"]]
            elif kind == "model_response":
                calls.append((frame["call_id"], purpose_for(waiting["messages"])))
            elif kind == "candidate_response":
                hypothesis = result["plan"]["hypothesis"]
                test = result["plan"].get("iteration", {}).get("tests", {}).get(waiting["configuration"]["name"])
                if test:
                    hypothesis += "\n该候选检验：" + test
                assert waiting["hypothesis"] == sdk_text(hypothesis)
                uploaded = uploads.uploads[waiting["upload_id"]]
                assert uploaded["finished"] and uploaded["kind"] == "candidate" and "files" not in waiting
                candidate = directory / "artifacts" / frame["artifact_id"]
                for name, data in uploaded["data"].items():
                    assert (candidate / name).read_bytes() == data
                measurements[frame["artifact_id"]] = frame["summary"]["metrics"]
            elif kind == "state_response":
                uploaded = uploads.uploads[waiting["upload_id"]]
                assert uploaded["finished"] and uploaded["kind"] == "state" and "files" not in waiting
                for name, data in uploaded["data"].items():
                    assert (directory / "saved-state" / name).read_bytes() == data
            waiting = None
    assert ack and finish and histories is not None and waiting is None
    assert finish["history_ids"] == histories and finish["model_call_ids"] == [c[0] for c in calls]
    assert result["finish_record"] == finish
    assert len({c[0] for c in calls}) == len(calls) and len(calls) >= 1
    records = [read_json(p) for p in (directory / "model-calls").glob("call-*.json")]
    assert {r["provider_call_id"] for r in records} == {c[0] for c in calls}
    for record in records:
        request = record["request"]["body"]
        assert request["model"] == start["model_name"] and request["max_tokens"] <= start["max_output_tokens"]
        assert request["messages"][0]["role"] == "system"
        assert sum(m.get("content", "").count(start["prompt"]) for m in request["messages"]) == 1
        assert record["status"] == "success" and record["receipt_origin"] == "pytrio_openai_api_not_contest_host"
    for candidate in result["candidates"]:
        assert candidate["metrics"] == measurements[candidate["artifact_id"]]
        assert candidate["continuation_verified"] and candidate["final_steps"] > candidate["initial_steps"]
    choice = read_json(next((directory / "workspace").glob("gas-a-*/model-choice.json")))
    measured_decision = choice["selection_mode"] in ("model", "model_retry")
    if measured_decision:
        receipt = next(r for r in records if r["provider_call_id"] == choice["call_id"])
        answer = parse_object(receipt["response"]["choices"][0]["message"]["content"])
        assert answer["selected_artifact_id"] == finish["selected_artifact_id"]
        assert sdk_text(answer["reason"]) == finish["reason"]
    contains_experiment = bool(measurements) and any(c[1] == "experiment_planning" for c in calls)
    based_on_results = bool(measurements) and any(c[1] == "artifact_selection" for c in calls)
    strict_complete = bool(histories) and contains_experiment and based_on_results and measured_decision and ack
    return {"round": result["round"], "history_ids": histories, "candidate_ids": list(measurements),
            "model_call_ids": [c[0] for c in calls], "decision": result["action"], "selected": result["selected"],
            "acknowledged": ack, "complete_with_nonempty_history": strict_complete, "chunks": chunks,
            "drift_detected": result["drift"]["drift_detected"], "reason": result["reason"],
            "started_at": result["started_at"], "completed_at": result["completed_at"]}


def check_data_boundary(root):
    checked = 0
    files = [*root.glob("round-*/mounts.json"), *root.glob("round-*/*-mounts.json"), *root.glob("blind/*-mounts.json")]
    for file in files:
        args = read_json(file)["args"]
        assert args[args.index("--network")+1] == "none" and "--read-only" in args
        mounts = {}
        for i, piece in enumerate(args):
            if piece != "--mount":
                continue
            options = args[i+1].split(",")
            values = dict(p.split("=", 1) for p in options if "=" in p)
            mounts[values["target"]] = (Path(values["source"]).resolve(), "readonly" in options)
        if args[-1] == "train":
            assert set(mounts) <= {"/app", "/sdk", "/tmp", "/workspace", "/data", "/previous", "/state"}
            assert mounts["/data"] == ((root / "public").resolve(), True)
            assert mounts["/previous"][1] is True
            if "/state" in mounts:
                assert mounts["/state"][1] is True
        else:
            assert args[-1] == "predict" and set(mounts) == {"/app", "/sdk", "/tmp", "/model"}
            assert mounts["/model"][1] is True
        assert mounts["/app"] == ((root / "app").resolve(), True)
        assert mounts["/sdk"] == ((root / "sdk").resolve(), True)
        for i, piece in enumerate(args):
            if piece == "--env":
                assert not any(key in args[i+1].split("=", 1)[0].upper() for key in ("API_KEY", "TOKEN", "SECRET"))
        checked += 1
    complete = read_json(root / "blind/prediction_complete.json")
    score = read_json(root / "blind/summary.json")
    assert complete["workers_exited"] and complete["completed_at"] <= score["scoring_started_at"]
    assert len(complete["commits"]) == 8 and score["feedback_calls"] == score["model_calls_in_prediction"] == score["online_updates"] == 0
    for item in complete["commits"]:
        assert digest(root / "blind" / item["predictions"]) == item["predictions_sha256"]
    return {"passed": True, "mount_commands_checked": checked, "all_predictions_committed_before_scoring": True,
            "no_data_mount_in_prediction": True, "credentials_not_passed_to_containers": True}


REPRODUCE = r'''
import json,time
from pathlib import Path
import numpy as np
from gas_a.data import load_data,load_model
from gas_a.optimize import fit_weighted
began=time.monotonic();cpu=time.process_time()
report=json.loads(Path('/expected/training.json').read_text())
parent=Path('/expected/migrated_parent.npz')
if not parent.is_file():parent=Path('/previous/checkpoint.npz')
actual=load_model(parent);expected=load_model('/expected/checkpoint.npz')
train=load_data('/data/train.jsonl',actual.meta['paths'])
count=report['final_epochs']-report['initial_epochs']
# Preserve the ORIGINAL epoch horizon for cosine learning-rate progress.
fit_weighted(actual,train,report['configuration'],report['weak_class'],time.monotonic()+180,epochs_to_run=count)
groups={};equal=True;close=True;maximum=0.
for name in ('params','first','second'):
 values={}
 for key,value in getattr(expected,name).items():
  got=getattr(actual,name)[key];difference=float(np.max(np.abs(got-value)))
  exact=bool(np.array_equal(got,value));within=bool(np.allclose(got,value,rtol=1e-10,atol=1e-12))
  equal &= exact;close &= within;maximum=max(maximum,difference)
  values[key]={'array_equal':exact,'within_tolerance':within,'maximum_abs_error':difference}
 groups[name]=values
scaler=bool(np.array_equal(actual.mean,expected.mean) and np.array_equal(actual.scale,expected.scale))
for key in ('response_mean','response_scale','response_precision'):
 if hasattr(expected,key):
  exact=bool(np.array_equal(getattr(actual,key),getattr(expected,key)))
  scaler &= exact
  groups[key]={'array_equal':exact}
meta=actual.meta==expected.meta
result={'passed':bool(close and scaler and meta),'all_arrays_exact':equal,'scaler_exact':scaler,'metadata_exact':meta,
        'selected_epochs_replayed':count,'original_requested_epochs':report['configuration']['epochs'],
        'maximum_abs_error':maximum,'rtol':1e-10,'atol':1e-12,'groups':groups,
        'wall_seconds':time.monotonic()-began,'cpu_seconds':time.process_time()-cpu,
        'gradient_data':'authorized public train only','model_calls':0}
Path('/workspace/reproduction.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''


def reproduce(root, docker):
    summary = read_json(root / "summary.json")
    candidates = [p.parent for p in sorted(root.glob("round-*/artifacts/*/training.json"))]
    parents = {digest(root / "baseline/checkpoint.npz"): root / "baseline"}
    parents.update({digest(p / "checkpoint.npz"): p for p in candidates})
    rows = []
    for index, directory in enumerate(candidates):
        training = read_json(directory / "training.json")
        output = root / "reproduction" / directory.name
        output.mkdir(parents=True, exist_ok=True)
        command = [docker, "run", "--rm", "--network", "none", "--read-only", "--user", "65534:65534",
                   "--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--cap-drop", "ALL",
                   "--security-opt", "no-new-privileges", "--env", "PYTHONPATH=/app/src:/app:/sdk",
                   "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OPENBLAS_NUM_THREADS=1",
                   "--env", "OMP_NUM_THREADS=1", "--env", "MKL_NUM_THREADS=1"]
        for source, target, writable in ((root / "app", "/app", False), (root / "sdk", "/sdk", False),
                                          (root / "public", "/data", False), (directory, "/expected", False),
                                          (parents[training["parent_checkpoint_sha256"]], "/previous", False),
                                          (output, "/workspace", True)):
            command += ["--mount", f"type=bind,source={source.resolve()},target={target}" + ("" if writable else ",readonly")]
        command += ["--entrypoint", "python3", summary["runtime_image_id"], "-B", "-c", REPRODUCE]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=210)
        (output / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Training reproduction failed: " + directory.name + " / " + result.stderr[-800:])
        detail = json.loads(result.stdout)
        rows.append({"candidate": directory.name, **detail})
        print(json.dumps({"stage": "numeric_reproduction", "candidate": directory.name, "passed": detail["passed"],
                          "all_arrays_exact": detail["all_arrays_exact"], "maximum_abs_error": detail["maximum_abs_error"]}), flush=True)
    outcome = {"passed": bool(rows) and all(r["passed"] for r in rows), "candidates": rows,
               "all_arrays_exact": all(r["all_arrays_exact"] for r in rows), "real_model_calls": 0,
               "scope": "Replay selected gradient updates using saved configurations, parents, Adam, seeds and original cosine horizon; no new tuning."}
    write_json(root / "reproduction.json", outcome)
    return outcome


def read_cloud(root):
    load_credentials(ROOT / ".env.local")
    from swanlab import Api
    summary = read_json(root / "summary.json")
    url = summary["cloud"]["experiment_url"]
    parts = url.split("/", 3)[3].split("/")
    remote = Api(api_key=os.environ["SWANLAB_API_KEY"]).run("/".join((parts[0].lstrip("@"), parts[1], parts[3])))
    measurements = remote.metrics(keys=["performance/macro_f1", "cost/calls"], all=True)
    actual = {r["key"]: [v["value"] for v in r.get("metrics", [])] for r in measurements.get("list", [])}
    expected = [json.loads(line)["payload"] for line in (root / "audit/events.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "round_summary"]
    fields = {"performance/macro_f1": [r["after"]["macro_f1"] for r in expected],
              "cost/calls": [r["costs"]["calls"] for r in expected]}
    checks = {key: len(actual.get(key, [])) == len(values) and
                   all(abs(a-b) < 1e-10 for a, b in zip(actual.get(key, []), values)) for key, values in fields.items()}
    result = {"passed": all(checks.values()), "checks": checks, "remote_state": remote.state,
              "records": len(expected), "experiment_url": url, "metrics": measurements,
              "scope": "Authenticated scalar readback, including the separately labelled drill; no normative logging audit."}
    write_json(root / "cloud-readback.json", result)
    return result


def bundle(root):
    target = root / "reproduction-materials.zip"
    selected = set()
    for directory in ("src", "configs", "scripts", "tests", "vendor", "data/official_a"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                selected.add(path)
    for name in ("Dockerfile", ".dockerignore", "pyproject.toml", "harness.py", "contest_submission.py", "README.md", "LIVE_GUIDE.md", "PROMPTS.md"):
        selected.add(ROOT / name)
    entries = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(ROOT).as_posix()
            assert not any(part.startswith(".env") for part in path.parts)
            archive.write(path, "gas/" + relative)
            entries.append({"path": "gas/" + relative, "sha256": digest(path), "size_bytes": path.stat().st_size})
    result = {"archive": target.name, "sha256": digest(target), "files": entries,
              "private_keys_included": False, "git_or_venv_included": False, "raw_batch3_10_included": False,
              "image_included": False, "includes_public_training_data": True,
              "note": "Source reproduction archive, not a Docker submission image. Dockerfile does not copy training data into the image."}
    write_json(root / "materials-manifest.json", result)
    instructions = f"""Gas SDK 1.0.4 本地验收复现

本目录是本地模拟证据，不是官方成绩。复现源码见 reproduction-materials.zip；不包含密钥、虚拟环境或新镜像。
压缩包包含官方公开训练、验证数据和官方基线，仅用于源码复现。Dockerfile 不把这些数据复制进镜像。

1. 解压 gas/，使用 Python 3.12 创建虚拟环境；在gas目录执行：
   python -m pip install numpy==1.26.4 -e ".[live]" vendor/contest-harness-sdk-1.0.4.tar.gz
   live依赖供本地宿主记录SwanLab；运行容器只需NumPy及官方SDK，密钥不进入容器。
2. 离线回归：设置 PYTHONPATH=src;tests，运行 python -B -m unittest discover -s tests -v。
3. 原 SDK 离线测试：解压 vendor 中原包，在 SDK 根目录运行
   python -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
4. 真实模拟需使用已有 Linux Python/NumPy 运行镜像。创建本地 .env.local，填写你自己的 PYTRIO_API_KEY、SWANLAB_API_KEY。
   python scripts/simulate_sdk_docker.py --docker <docker.exe> --image <已有运行镜像> --dataset <原始Dataset目录> --output <新的空目录> --rounds {read_json(root / 'summary.json')['rounds']} --rollback-drill
   该命令调用真实模型并可能产生费用。真实API密钥不进入容器。既有运行镜像只提供Python/NumPy；源码与SDK将被只读覆盖。
5. 生成图表：python scripts/report_sdk_simulation.py --source <运行输出目录>
6. 验收与数值复现：python scripts/verify_platform_simulation.py --source <运行输出目录> --docker <docker.exe>

本次已保存每轮双向SDK协议、实际模型请求和回复、候选配置、父模型摘要、训练记录、逐条预测结果及决策。
模型回复来自外部服务，重新请求不保证文本完全相同；reproduction.json 验证的是重放已记录配置的数值训练结果。
正式测试仍需：重新构建包含最新代码及SDK的提交镜像，在官方宿主运行并取得正式成绩。
基线比较只能说明本地同数据上的得分关系，不能代替官方隐藏评测。运行路径：{root}
"""
    (root / "REPRODUCE.txt").write_text(instructions, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--docker", required=True)
    args = parser.parse_args()
    root = Path(args.source).resolve()
    summary = read_json(root / "summary.json")
    rows = [check_round(p.parent) for p in sorted(root.glob("round-*/result.json"))]
    assert len(rows) == summary["rounds"]
    complete = [r for r in rows if r["complete_with_nonempty_history"]]
    drill_directory = root / "drills/rollback"
    drill_protocol = check_round(drill_directory / "round-02") if drill_directory.is_dir() else None
    boundary = check_data_boundary(root)
    numeric = reproduce(root, args.docker)
    try:
        cloud = read_cloud(root)
    except Exception as exc:
        cloud = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "scope": "Readback unavailable; not proof of upload failure."}
        write_json(root / "cloud-readback.json", cloud)
    material = bundle(root)
    gain = summary["blind"]["aggregate"]["gain_macro_f1"]
    checks = {
        "unattended_local_run_completed": summary["success"] and all(r["acknowledged"] for r in rows),
        "three_history_experiment_decision_records": len(complete) >= 3,
        "full_duplex_protocol_and_actual_model_receipts": True,
        "local_score_not_below_same_official_baseline": gain >= -1e-12,
        "local_score_strictly_better_than_baseline": gain > 1e-12,
        "numeric_training_reproduction": numeric["passed"],
        "isolated_injected_rollback": bool((summary.get("rollback_drill") or {}).get("passed")) and bool(drill_protocol),
        "cloud_scalar_readback": cloud["passed"],
        "formal_submission_materials_complete": False,
        "official_platform_score_verified": False,
        "natural_promotion_observed": summary["promotions"] > 0,
        "natural_rollback_observed": summary["rollbacks"] > 0,
        "hidden_labels_used_by_training_or_decision": False,
    }
    result = {"scope": "local_sdk104_simulation_not_official_acceptance", "checks": checks,
              "complete_records_count_excluding_empty_history": len(complete), "complete_record_rounds": [r["round"] for r in complete],
              "round_evidence": rows, "normal_aggregate": summary["blind"]["aggregate"],
              "data_boundary_checks": boundary,
              "rollback_drill_excluded_from_score": summary["rollback_drill"],
              "materials": {"source_archive": material["archive"], "pending": ["New Docker submission image", "Official-host run and score"]},
              "boundaries": ["Only official public train/validation mounted during train; only selected model during prediction.",
                             "Batch3--10 DAT targets are read only by the independent scorer after both prediction containers exit.",
                             "No hidden-label feedback interface or test-label tuning is used. Actual official hidden data is unavailable.",
                             "Image source overlay validates current code, not the bytes of an updated submission image.",
                             "Diagnostic attribution is a set of hypotheses, not proof of physical causes.",
                             "No normal promotion means end-to-end natural improvement/deployment was not demonstrated.",
                             "Rollback is an explicitly injected recovery drill, not a naturally observed test-stream rollback."]}
    write_json(root / "acceptance.json", result)
    with (root / "complete-loops.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "history_ids", "candidate_ids", "model_call_ids", "decision", "selected", "reason"])
        for row in complete:
            writer.writerow([row["round"], ";".join(row["history_ids"]), ";".join(row["candidate_ids"]),
                             ";".join(row["model_call_ids"]), row["decision"], row["selected"], row["reason"]])
    # Add a concise acceptance panel to the detailed charts without altering the evidence.
    table_rows = [
        ("无人干预运行", "通过" if checks["unattended_local_run_completed"] else "未通过", f"{len(rows)}轮自动结束并收到宿主确认"),
        ("至少3次完整记录", "通过" if len(complete) >= 3 else "未通过", f"严格统计非空历史＋真实训练＋实测后决策，共{len(complete)}次：{result['complete_record_rounds']}"),
        ("本地不低于基线", "通过" if gain >= -1e-12 else "未通过", f"Macro-F1差值 {gain:+.6f}；{'相等，没有提升' if abs(gain)<1e-12 else '以本地同数据比较为准'}"),
        ("实际新版本上线", "已观察" if checks["natural_promotion_observed"] else "未发生", f"正常运行采用新候选{summary['promotions']}次；保留基线不能当作上线收益"),
        ("回滚能力", "演练通过" if checks["isolated_injected_rollback"] else "演练未通过", "隔离副本注入故障后选回基线，收益不计正常成绩"),
        ("模型数值复现", "通过" if numeric["passed"] else "未通过", f"重放{len(numeric['candidates'])}个候选，数组完全相同：{numeric['all_arrays_exact']}"),
        ("云端记录读回", "通过" if cloud["passed"] else "未确认", "已登录读回标量；这不是实验记录规范审核"),
        ("正式材料齐全", "待完成", "已备源码、配置、证据和复现说明；最新提交镜像及官方成绩仍缺失"),
        ("数据使用边界", "本地隔离核验", "训练只用公开train/validation，预测无标签，评分在全流预测退出后进行"),
    ]
    panel = "<section><h2>平台模拟验收结果</h2><div class='scroll'><table><thead><tr><th>验收项</th><th>结果</th><th>依据</th></tr></thead><tbody>"
    panel += "".join("<tr>"+"".join("<td>"+html.escape(str(v))+"</td>" for v in row)+"</tr>" for row in table_rows)
    panel += "</tbody></table></div><p><a href='acceptance.json'>逐项验收JSON</a> · <a href='complete-loops.csv'>完整闭环记录CSV</a> · <a href='reproduction.json'>数值复现</a> · <a href='cloud-readback.json'>云端读回</a> · <a href='REPRODUCE.txt'>复现说明</a> · <a href='reproduction-materials.zip'>源码复现包（不是镜像）</a></p></section>"
    page = root / "report.html"
    body = page.read_text(encoding="utf-8")
    body = body.replace("<section><h2>测试数据与隔离方式</h2>", panel+"<section><h2>测试数据与隔离方式</h2>", 1)
    page.write_text(body, encoding="utf-8")
    print(json.dumps({"checks": checks, "complete_record_rounds": result["complete_record_rounds"],
                      "report": str(page), "source_archive": str(root/material["archive"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
