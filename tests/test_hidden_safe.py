"""No-label prediction isolation, actual parent continuation and robustness gates."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from gas_a import robustness
from gas_a.blind import Predictor, feature_rows, run_blind
from gas_a.candidates import execute_candidate
from gas_a.data import evaluate, load_model, load_pair
from gas_a.io import digest, read_json
from gas_a.live import run_live
from gas_a.optimize import bounded_experiment, fit_weighted, train_candidate
from gas_a.pipeline import load_config

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT/"data/official_a/checkpoint.npz"


def public_data():
    return load_pair(ROOT/"data/official_a/public/train.jsonl", ROOT/"data/official_a/public/validation.jsonl")


def dat(directory, target=1):
    _, validation = public_data()
    directory.mkdir(exist_ok=True)
    for batch in (3, 4):
        lines = [str(target)+" "+" ".join(f"{j+1}:{value:.17g}" for j,value in enumerate(row))
                 for row in validation["x"][:3]]
        (directory/f"batch{batch}.dat").write_text("\n".join(lines)+"\n",encoding="utf-8")


class HiddenSafeTests(unittest.TestCase):
    def test_validation_stress_covers_every_sensor_without_refitting_scaler(self):
        _, validation = public_data()
        model=load_model(CHECKPOINT)
        original=validation["x"].copy();mean=model.mean.copy();scale=model.scale.copy()
        suite=list(robustness.validation_suite(model, validation))
        self.assertEqual(len(suite),16)
        for i,(name,data) in enumerate(suite):
            self.assertEqual(name,f"sensor_{i+1:02d}_missing")
            np.testing.assert_array_equal(data["x"][:,i*8:(i+1)*8],np.broadcast_to(mean[i*8:(i+1)*8],(len(original),8)))
            np.testing.assert_array_equal(data["x"][:,:i*8],original[:,:i*8])
            np.testing.assert_array_equal(data["x"][:,(i+1)*8:],original[:,(i+1)*8:])
            self.assertIs(data["y"],validation["y"])
        np.testing.assert_array_equal(model.mean,mean);np.testing.assert_array_equal(model.scale,scale)
        np.testing.assert_array_equal(validation["x"],original)

    def test_all_clean_fraction_replays_parent_adam_and_leaves_scaler_exact(self):
        train,_=public_data()
        left,right=load_model(CHECKPOINT),load_model(CHECKPOINT)
        spec={"name":"clean_control","learning_rate":.001,"epochs":2,"balance":0.,"weak_boost":1.,"l2":0.,"batch_size":32}
        fit_weighted(left,train,spec,"Acetaldehyde",time.monotonic()+20)
        fit_weighted(right,train,{**spec,"input_noise":.2,"sensor_dropout":.1,"response_gain":.2,"clean_fraction":1.},"Acetaldehyde",time.monotonic()+20)
        for group in ("params","first","second"):
            for key,value in getattr(left,group).items():np.testing.assert_array_equal(value,getattr(right,group)[key])
        np.testing.assert_array_equal(left.mean,right.mean);np.testing.assert_array_equal(left.scale,right.scale)
        with self.assertRaises(ValueError):bounded_experiment({**spec,"clean_fraction":1.01},{"max_epochs":32})

    def test_robust_best_epoch_exports_actual_parent_state_and_measured_suite(self):
        train,validation=public_data()
        spec={"name":"robust_test","learning_rate":.001,"epochs":3,"balance":0.,"weak_boost":1.,"l2":0.,
              "batch_size":32,"input_noise":.1,"clean_fraction":.5}
        limits={"max_epochs":32,"max_seconds_per_candidate":20,"select_best_epoch":True,
                "robustness":{"enabled":True,"max_scenario_f1_drop":.02}}
        with tempfile.TemporaryDirectory() as tmp:
            report=train_candidate(CHECKPOINT,train,validation,Path(tmp)/"candidate",spec,{"weakest_class":"Acetaldehyde"},limits)
            child=load_model(Path(tmp)/"candidate/checkpoint.npz");parent=load_model(CHECKPOINT)
            actual=robustness.evaluate_suite(child,tuple(robustness.validation_suite(parent,validation)))
            self.assertEqual(actual,report["robust_validation_after"])
            self.assertEqual(report["parent_checkpoint_sha256"],digest(CHECKPOINT))
            self.assertGreater(child.meta["steps"],parent.meta["steps"])
            np.testing.assert_array_equal(child.mean,parent.mean);np.testing.assert_array_equal(child.scale,parent.scale)
            selected=next(r for r in report["epoch_history"] if r["epoch"]==child.meta["epochs"])
            self.assertEqual(actual,selected["robust_validation"])

    def test_stress_rejection_blocks_otherwise_accepted_clean_candidate(self):
        train,validation=public_data();metric=evaluate(load_model(CHECKPOINT),validation)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shutil.copyfile(CHECKPOINT,root/"checkpoint.npz")
            report={"validation_after":metric,"robust_gate":{"accepted":False,"reasons":["robust_worst_case_regression"]}}
            with patch("gas_a.candidates.train_candidate",return_value=report):
                trial=execute_candidate(CHECKPOINT,train,validation,root,{"name":"bad_robustness"},{},{},
                    active_metrics=metric,baseline_metrics=metric,policy={},validation_gate=lambda *a:{"accepted":True,"reasons":[]})
            self.assertFalse(trial["gate"]["accepted"])
            self.assertIn("robust_worst_case_regression",trial["gate"]["reasons"])

    def test_blind_predictions_ignore_targets_and_scoring_follows_complete_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";dat(data,1)
            a=run_blind(CHECKPOINT,CHECKPOINT,data,root/"a",(3,4))
            complete=read_json(root/"a/prediction_complete.json")
            self.assertLessEqual(complete["completed_at"],a["scoring_started_at"])
            self.assertTrue(complete["workers_exited"])
            dat(data,6)
            b=run_blind(CHECKPOINT,CHECKPOINT,data,root/"b",(3,4))
            self.assertEqual(a["aggregate"]["gain_macro_f1"],0)
            for batch in (3,4):
                with np.load(root/f"a/windows/batch{batch:02d}/predictions.npz") as x,np.load(root/f"b/windows/batch{batch:02d}/predictions.npz") as y:
                    np.testing.assert_array_equal(x["active"],y["active"])
                    np.testing.assert_array_equal(x["active"],x["baseline"])
            self.assertNotEqual(a["aggregate"]["active"]["accuracy"],b["aggregate"]["active"]["accuracy"])

    def test_nonlabel_token_does_not_block_prediction_but_fails_later_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";dat(data,"unavailable")
            self.assertEqual(len(list(feature_rows(data/"batch3.dat"))),3)
            with self.assertRaises(ValueError):run_blind(CHECKPOINT,CHECKPOINT,data,root/"out",(3,4))
            complete=read_json(root/"out/prediction_complete.json")
            self.assertTrue(complete["workers_exited"]);self.assertEqual(len(complete["commits"]),2)

    def test_prediction_restarts_and_input_order_do_not_require_feedback(self):
        _,validation=public_data();x=validation["x"][:3].tolist()
        signature=digest(CHECKPOINT)
        with Predictor(CHECKPOINT) as worker:a=[worker.predict(row) for row in x]
        with Predictor(CHECKPOINT) as worker:b=[worker.predict(row) for row in reversed(x)]
        np.testing.assert_array_equal(a,list(reversed(b)))
        self.assertEqual(signature,digest(CHECKPOINT))

    def test_qwen_planning_and_selection_receive_only_authorized_robustness_evidence(self):
        class Model:
            model="explicit_offline_qwen_interface_fixture"
            def __init__(self):self.requests=[]
            def complete(self,messages,**kwargs):
                prompt=json.loads(messages[-1]["content"]);self.requests.append(prompt)
                if "search_space" in prompt:
                    reply={"action":"experiment","hypothesis":"测试训练扰动能否改善公开验证稳健性。","reason":"授权验证证据。","fallback":"retain_parent",
                           "experiments":[{"name":"noise_mix","learning_rate":.001,"batch_size":32,"epochs":2,"input_noise":.1,"clean_fraction":.5}]}
                else:reply={"selected_artifact_id":prompt["comparison"]["measured_preference"],"reason":"按同口径实测与程序门槛选择。"}
                return {"call_id":f"test-{len(self.requests)}","text":json.dumps(reply),"usage":{"prompt_tokens":1,"completion_tokens":1},
                        "response":{},"seconds":0.,"receipt_origin":"explicit_offline_fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config(ROOT/"configs/a.live.json");cfg["runtime"]=str(Path(tmp)/"live")
            client=Model();run_live(cfg,rounds=1,cloud=False,client=client)
            self.assertEqual(len(client.requests),2)
            plan,selection=client.requests
            self.assertEqual(len(plan["robust_validation"]["cases"]),16)
            self.assertIn("hidden",plan["robust_validation"]["scope"])
            self.assertIn("prediction_contract",plan)
            self.assertIsNotNone(selection["experiments"][0]["robust_validation"])
            self.assertEqual(selection["comparison"]["scope"],"host_current_validation")


if __name__ == "__main__":
    unittest.main()
