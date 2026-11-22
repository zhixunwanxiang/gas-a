"""Regressions for the four reviewed admission, history and preflight failures."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from gas_a.architecture import migrate_parent
from gas_a.candidates import assess_candidate, execute_candidate
from gas_a.controller_policy import measured_evidence
from gas_a.data import evaluate, load_model, load_pair, score_predictions
from gas_a.io import digest
from gas_a.monitor import monitor, diagnose
from gas_a.optimize import fit_weighted, gate
from gas_a.pipeline import load_config, ordered_specs
from gas_a.planning import configurations, resolve_plan

ROOT = Path(__file__).resolve().parents[1]


def measured_counts(counts):
    """Consistent six-class metrics from a synthetic confusion matrix, not scores."""
    labels = np.repeat(np.arange(6), 100)
    predicted = labels.copy()
    for label, count in enumerate(counts):
        ids = np.flatnonzero(labels == label)
        predicted[ids[count:]] = (label + 1) % 6
    return score_predictions(np.eye(6)[predicted], labels)


class AuditFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT/'configs/a.json')
        cls.train, cls.val = load_pair(cls.cfg['train'], cls.cfg['validation'])
        cls.parent = load_model(cls.cfg['baseline']); cls.sha = digest(cls.cfg['baseline'])
        cls.spec = dict(name='boundary',learning_rate=.0005,batch_size=32,epochs=2,balance=0.,weak_boost=1.,l2=0.)
        cls.static = measured_counts([100,70,70,70,70,70])
        cls.active = measured_counts([90,80,80,80,80,80])
        metric = evaluate(cls.parent, cls.val)
        drift = monitor(cls.train['x'],cls.val['x'],cls.cfg['monitor'])
        diag = diagnose(cls.train,cls.val,metric,drift)
        cls.evidence = measured_evidence(metric,metric,evaluate(cls.parent,cls.train),diag,drift,[],cls.sha,[],
            validation_sha=cls.val['sha256'],training_sha=cls.train['sha256'])

    def test_default_sdk_rule_honors_configured_target_recall_drop(self):
        candidate = measured_counts([95,95,95,70,95,95])
        for assessor in (gate, assess_candidate):
            decision=assessor(candidate,self.active,self.static,self.cfg['gate'])
            self.assertFalse(decision['accepted'])
            self.assertIn('target_recall_regression:Acetaldehyde',decision['reasons'])
            # It is still an editable engineering setting.
            relaxed={**self.cfg['gate'],'target_recall_max_drop':.10}
            self.assertTrue(assessor(candidate,self.active,self.static,relaxed)['accepted'])

    def check_epochs(self, metrics, reason):
        initial=self.parent.meta['epochs'];original=evaluate
        def measured(model,data):
            return copy.deepcopy(metrics[model.meta['epochs']-initial]) if data is self.val else original(model,data)
        with tempfile.TemporaryDirectory() as tmp,patch('gas_a.optimize.evaluate',side_effect=measured),patch('gas_a.candidates.evaluate',side_effect=measured):
            folder=Path(tmp)/'candidate'
            trial=execute_candidate(self.cfg['baseline'],self.train,self.val,folder,self.spec,{'weakest_class':'Acetaldehyde'},
                {**self.cfg['optimization'],'robustness':{'enabled':False}},active_metrics=self.active,
                baseline_metrics=self.static,policy=self.cfg['gate'])
            self.assertNotIn('error',trial);self.assertTrue(trial['gate']['accepted'])
            report=trial['report'];self.assertEqual(report['final_epochs'],initial+1)
            first,last=report['epoch_history'];self.assertTrue(first['checkpoint_eligible']);self.assertFalse(last['checkpoint_eligible'])
            self.assertIn(reason,last['quality_gate']['reasons'])
            self.assertEqual(report['checkpoint_selection']['reference_metrics'],self.static)
            actual=load_model(folder/'checkpoint.npz');replay=load_model(self.cfg['baseline'])
            fit_weighted(replay,self.train,self.spec,'Acetaldehyde',time.monotonic()+20,epochs_to_run=1)
            for group in ('params','first','second'):
                for name,value in getattr(actual,group).items():np.testing.assert_array_equal(value,getattr(replay,group)[name])

    def test_epoch_selection_uses_original_baseline_class_protection(self):
        self.check_epochs({0:self.active,1:measured_counts([86,88,88,88,88,88]),2:measured_counts([80,98,98,98,98,98])},
                          'class_recall_regression:Ethanol')

    def test_epoch_selection_uses_the_same_target_drop_rule_as_release(self):
        self.check_epochs({0:self.active,1:measured_counts([95,95,95,80,95,95]),2:measured_counts([99,99,99,70,99,99])},
                          'target_recall_regression:Acetaldehyde')

    def history(self, train_sha=None, val_sha=None):
        return [{'round':1,'parent_checkpoint_sha256':self.sha,'train_sha256':train_sha,'validation_sha256':val_sha,
                 'experiments':[{'configuration':self.spec}]}]

    def resolve(self, past, evidence=None):
        return resolve_plan({'action':'experiment','experiments':[self.spec],'hypothesis':'检验授权数据中的分类错误。',
                             'reason':'依据公开验证设计实验。'},self.cfg,self.parent,[],copy.deepcopy(evidence or self.evidence),
                            2,past,self.sha,weak_class='Acetaldehyde')

    def test_duplicate_scope_requires_identical_parent_and_both_public_data_hashes(self):
        past=self.history(self.train['sha256'],self.val['sha256'])
        with self.assertRaisesRegex(ValueError,'already attempted'):self.resolve(past)
        for changes in ({'T':'new_train'},{'F':'new_validation'},{'T':None},{'F':None}):
            evidence=copy.deepcopy(self.evidence)
            for key,value in changes.items():evidence['facts'][key]['sha256']=value
            _,specs=self.resolve(past,evidence);self.assertEqual(len(specs),1)
        _,specs=self.resolve(self.history());self.assertEqual(len(specs),1)

    def test_rule_menu_reuses_configuration_when_public_data_changes(self):
        scope=dict(train_sha=self.train['sha256'],validation_sha=self.val['sha256'])
        fresh=ordered_specs(self.cfg,[],self.sha,model=self.parent,**scope)
        past=[{'parent_checkpoint_sha256':self.sha,'train_sha256':self.train['sha256'],
               'validation_sha256':self.val['sha256'],'experiments':[{'configuration':s}for s in fresh]}]
        repeated=ordered_specs(self.cfg,past,self.sha,model=self.parent,**scope)
        self.assertNotEqual([s['learning_rate'] for s in fresh],[s['learning_rate']for s in repeated])
        changed=ordered_specs(self.cfg,past,self.sha,model=self.parent,train_sha='new_train',validation_sha=self.val['sha256'])
        self.assertEqual(fresh,changed)

    def test_response_refit_freeze_conflicts_are_rejected_before_execution(self):
        spec={**self.spec,'response_adapter':{'ridge':10.,'strength':16.,'guard_multiplier':1.5}}
        for extra in ({'freeze':'features'},{'frozen_parameters':['wa']},{'frozen_parameters':['ba']}):
            with self.subTest(extra=extra),self.assertRaisesRegex(ValueError,'frozen response'):
                configurations({'action':'experiment','experiments':[{**spec,**extra}]},self.cfg,self.parent,[],self.sha,2)
        valid={**spec,'freeze':'first_hidden'}
        specs=configurations({'action':'experiment','experiments':[valid]},self.cfg,self.parent,[],self.sha,2)
        model,_=migrate_parent(self.parent,specs[0]['architecture'],self.sha)
        original=copy.deepcopy(model)
        fit_weighted(model,self.train,specs[0],'Acetaldehyde',time.monotonic()+20,epochs_to_run=1)
        for group in ('params','first','second'):
            for name in ('w1','b1'):np.testing.assert_array_equal(getattr(model,group)[name],getattr(original,group)[name])
        self.assertTrue(np.any(model.params['wa']))


if __name__=='__main__':unittest.main()
