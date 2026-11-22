"""真实发送模板的业务信息、简短输出及历史断言恢复。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util

from gas_a import adaptive
from gas_a.controller_policy import build_planning_prompt, bind_plan, validate_plan, measured_evidence
from gas_a.data import load_pair, load_model, evaluate
from gas_a.io import history, read_json
from gas_a.monitor import monitor, diagnose
from fixtures import legacy_config as load_config
from gas_a.live import run_live
from gas_a.temporal import run_temporal
from fixtures import ModelFixture, ReplayModelFixture

ROOT = Path(__file__).resolve().parents[1]


class ShortReply(ModelFixture):
    def complete(self, messages, **options):
        result = super().complete(messages, **options)
        prompt = json.loads(messages[-1]["content"])
        if "allowed_experiments" in prompt:
            reply = json.loads(result["text"])
            result["text"] = json.dumps({k: reply[k] for k in ("action", "experiment_names", "hypothesis", "reason")})
            result["response"]["choices"][0]["message"]["content"] = result["text"]
        return result


class PromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=load_config(ROOT/"configs/a.adaptive.json")
        cls.train,cls.val=load_pair(cls.cfg["train"],cls.cfg["validation"])
        model=load_model(cls.cfg["baseline"])
        cls.metric=evaluate(model,cls.val)
        drift=monitor(cls.train["x"],cls.val["x"],cls.cfg["monitor"])
        diagnosis=diagnose(cls.train,cls.val,cls.metric,drift)
        cls.evidence=measured_evidence(cls.metric,cls.metric,evaluate(model,cls.train),diagnosis,drift,[],
            "parent",[s["name"] for s in cls.cfg["experiments"]],validation_sha=cls.val["sha256"])

    def test_business_prompt_keeps_class_fit_confusion_and_hard_soft_targets(self):
        original=copy.deepcopy(self.evidence)
        prompt=build_planning_prompt(self.cfg["experiments"][:2],self.evidence,adaptive.policy(self.cfg),2,30,
            mode="temporal",index=6,parameter_bounds=adaptive.parameter_bounds())
        self.assertEqual(self.evidence,original)
        rows=[dict(zip(prompt["evidence"]["class_columns"],r)) for r in prompt["evidence"]["class_rows"]]
        for row,old in zip(rows,self.evidence["facts"]["C"]["classes"]):
            self.assertEqual(row["current_support"],old[0])
            self.assertEqual(row["current_recall"],old[1])
            self.assertEqual(row["current_precision"],old[2])
            self.assertEqual(row["current_f1"],old[3])
        self.assertEqual(prompt["evidence"]["top_confusions"],self.evidence["facts"]["D"]["top_confusions"])
        self.assertEqual(prompt["decision_context"]["recall_targets"],[{"gas":"Acetaldehyde","minimum":.5,"kind":"soft"}])
        self.assertNotIn("claims",prompt["plan_contract"])
        self.assertNotIn("evidence_ids",prompt["plan_contract"])
        self.assertNotIn("success_condition",prompt["plan_contract"])
        self.assertNotIn("quality_gate",prompt)
        self.assertNotIn(self.val["sha256"],json.dumps(prompt))
        self.assertEqual(prompt["permitted_parameter_changes"]["bounds"]["learning_rate"],[1e-5,.003])

    def test_four_field_reply_is_bound_without_inventing_history_or_changing_reason(self):
        raw={"action":"experiment","experiment_names":["gentle"],"hypothesis":"可能需要温和更新，检验弱类召回。","reason":"当前弱类F1低，先做小步续训参照。"}
        original=copy.deepcopy(raw);bound=bind_plan(raw)
        validate_plan(bound,self.cfg["experiments"],2,self.evidence)
        self.assertEqual(raw,original)
        self.assertEqual(bound["reason"],raw["reason"])
        self.assertEqual(bound["hypothesis"],raw["hypothesis"])
        self.assertEqual(bound["claims"],[])
        self.assertEqual(bound["evidence_ids"],["C","F"])
        bad=bind_plan({**raw,"reason":"该方案历史最佳。"})
        with self.assertRaisesRegex(ValueError,"Historical ranking"):
            validate_plan(bad,self.cfg["experiments"],2,self.evidence)

    def test_verifier_recomputes_business_table_and_rejects_changed_metric(self):
        module_spec=importlib.util.spec_from_file_location("prompt_verifier",ROOT/"scripts/verify_live_run.py")
        verifier=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(verifier)
        prompt=build_planning_prompt(self.cfg["experiments"][:2],self.evidence,self.cfg["gate"],2,30,mode="sdk",index=1)
        parent=load_model(self.cfg["baseline"])
        self.assertTrue(verifier.planning_evidence_matches(prompt,parent,self.train,self.val))
        prompt["evidence"]["class_rows"][0][3]+=.1
        self.assertFalse(verifier.planning_evidence_matches(prompt,parent,self.train,self.val))

    def test_sdk_path_accepts_short_business_reply_without_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=copy.deepcopy(self.cfg);cfg["runtime"]=str(Path(tmp)/"live")
            client=ShortReply();run_live(cfg,cloud=False,client=client)
            row=history(Path(cfg["runtime"])/"history.jsonl")[0]
            self.assertIsNone(row["model_plan_parse_error"])
            self.assertEqual(row["recoveries"],[])
            self.assertEqual(client.calls,2)
            self.assertTrue(row["experiments"])

    def test_temporal_wire_has_one_business_template_and_accepts_short_reply(self):
        class Replay(ReplayModelFixture):
            def complete(self,messages,**options):
                result=super().complete(messages,**options)
                prompt=json.loads(messages[-1]["content"])
                if "allowed_experiments" in prompt:
                    self.test_prompt=prompt
                    reply=json.loads(result["text"])
                    result["text"]=json.dumps({k:reply[k] for k in ("action","experiment_names","hypothesis","reason")})
                    result["response"]["choices"][0]["message"]["content"]=result["text"]
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=copy.deepcopy(self.cfg);cfg["runtime"]=str(root/"result")
            cfg["experiments"]=[{**cfg["experiments"][0],"epochs":1}]
            for batch,offset in ((3,0),(4,40)):
                rows=[str(int(y)+1)+" "+" ".join(f"{i+1}:{v:.17g}" for i,v in enumerate(x))
                      for x,y in zip(self.val["x"][offset:offset+40],self.val["y"][offset:offset+40])]
                (root/f"batch{batch}.dat").write_text("\n".join(rows)+"\n",encoding="utf-8")
            client=Replay();summary=run_temporal(cfg,root,(3,4),cloud=False,client=client)
            self.assertTrue(summary["success"])
            self.assertEqual(summary["controller_failed_windows"],0)
            self.assertEqual(summary["controller_recovered_windows"],0)
            self.assertEqual(client.test_prompt["prompt_version"],"gas-a.business-planning.v1")
            self.assertNotIn("recent_attempts",client.test_prompt)
            self.assertNotIn("claims",client.test_prompt["plan_contract"])
            self.assertTrue(read_json(Path(cfg["runtime"])/"windows/batch03/result.json")["experiments"])


if __name__=="__main__":unittest.main()
