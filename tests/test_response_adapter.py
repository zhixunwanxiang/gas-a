"""Validate continuation, train-only fitting, guarding and actual predictive benefit."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from gas_a import response, robustness
from gas_a.architecture import migrate_parent, shape
from gas_a.data import load_model, load_pair, evaluate
from gas_a.io import digest, read_json
from gas_a.optimize import bounded_experiment, fit_weighted, train_candidate, gate, classification_delta
from gas_a.pipeline import load_config
from gas_a.planning import configurations, search_space

ROOT = Path(__file__).resolve().parents[1]


class ResponseAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT/'configs/a.json')
        cls.train, cls.val = load_pair(cls.cfg['train'], cls.cfg['validation'])
        cls.sha = digest(cls.cfg['baseline'])
        cls.spec = dict(name='relative_response', learning_rate=.0001, epochs=4, batch_size=32,
                        balance=.5, weak_boost=1., l2=0.,
                        response_adapter=dict(ridge=10., strength=16., guard_multiplier=1.5))

    def parent(self): return load_model(self.cfg['baseline'])

    def migrated(self): return migrate_parent(self.parent(), {'response_adapter': True}, self.sha)[0]

    def fitted(self):
        m = self.migrated(); response.fit_head(m, self.train, self.spec['response_adapter']); return m

    def test_zero_head_migration_preserves_function_and_all_parent_state(self):
        parent = self.parent(); model = self.migrated()
        for group in ('params', 'first', 'second'):
            for name, value in getattr(parent, group).items():
                np.testing.assert_array_equal(getattr(model, group)[name], value)
            for name in ('wa', 'ba'): self.assertFalse(np.any(getattr(model, group)[name]))
        for key in ('seed', 'steps', 'epochs'): self.assertEqual(model.meta[key], parent.meta[key])
        np.testing.assert_array_equal(model.mean, parent.mean); np.testing.assert_array_equal(model.scale, parent.scale)
        np.testing.assert_array_equal(model.predict(self.val['x']), parent.predict(self.val['x']))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'parent.npz'; model.save(path)
            np.testing.assert_array_equal(load_model(path).predict(self.val['x']), parent.predict(self.val['x']))

    def test_fitted_head_can_widen_without_losing_auxiliary_state(self):
        parent=self.fitted()
        fit_weighted(parent,self.train,{**self.spec,'epochs':1},'Acetaldehyde',time.monotonic()+30)
        model,_=migrate_parent(parent,{'hidden':64,'hidden_layers':2},self.sha)
        np.testing.assert_allclose(model.predict(self.val['x']),parent.predict(self.val['x']),rtol=0,atol=1e-12)
        for group in ('params','first','second'):
            for name,value in getattr(parent,group).items():
                np.testing.assert_array_equal(getattr(model,group)[name][tuple(slice(0,s) for s in value.shape)],value)
        for name in response.ARRAYS:np.testing.assert_array_equal(getattr(model,name),getattr(parent,name))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'wide.npz';model.save(p);loaded=load_model(p)
            np.testing.assert_array_equal(loaded.predict(self.val['x']),model.predict(self.val['x']))
            fit_weighted(loaded,self.train,{k:v for k,v in self.spec.items() if k!='response_adapter'},'Acetaldehyde',time.monotonic()+30,epochs_to_run=1)
            self.assertGreater(loaded.meta['steps'],parent.meta['steps'])

    def test_only_train_fits_new_statistics_and_ridge_preserves_adam(self):
        model=self.migrated(); old=copy.deepcopy(model)
        response.fit_head(model, self.train, self.spec['response_adapter'])
        np.testing.assert_array_equal(model.response_mean, response.shape_features(self.train['x']).mean(axis=0))
        for group in ('first', 'second'):
            for name,value in getattr(old,group).items(): np.testing.assert_array_equal(getattr(model,group)[name],value)
        for name in ('w1','b1','w2','b2'): np.testing.assert_array_equal(model.params[name],old.params[name])
        self.assertEqual(model.meta['steps'],old.meta['steps'])
        baseline_stats=[getattr(model,n).copy() for n in response.ARRAYS]
        old_head=model.params['wa'].copy()
        response.fit_head(model, self.train, {**self.spec['response_adapter'], 'ridge':20.})
        for name,value in zip(response.ARRAYS,baseline_stats): np.testing.assert_array_equal(getattr(model,name),value)
        self.assertFalse(np.array_equal(old_head,model.params['wa']))
        self.assertEqual(model.meta['response_statistics_train_sha256'],self.train['sha256'])

    def test_forward_backward_matches_finite_difference_for_base_and_head(self):
        model=self.fitted(); x=model.normalized(self.train['x'][:8]); y=self.train['y'][:8]
        _,logits=model.forward(x); p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        _,delta=classification_delta(p,y,np.ones(len(y))); grads,_=model.backward(x,delta)
        def loss():
            _,z=model.forward(x);return float(np.mean(np.log(np.exp(z-z.max(axis=1,keepdims=True)).sum(axis=1))+z.max(axis=1)-z[np.arange(len(y)),y]))
        for name,index in [('w1',(4,3)),('b1',(3,)),('w2',(3,2)),('b2',(2,)),('wa',(4,2)),('ba',(2,))]:
            value=model.params[name][index];epsilon=1e-5
            model.params[name][index]=value+epsilon;plus=loss()
            model.params[name][index]=value-epsilon;minus=loss();model.params[name][index]=value
            self.assertAlmostEqual(grads[name][index],(plus-minus)/(2*epsilon),places=7)

    def test_guard_is_per_row_frozen_and_outside_coverage_uses_backbone(self):
        model=self.fitted(); x=self.val['x']; before=copy.deepcopy(model)
        z=model.normalized(x); outside=response.guard(model,z[:,128:])[:,0]==0
        self.assertTrue(np.any(outside));self.assertTrue(np.any(~outside))
        hidden,_=model._forward_cache(z[outside]); expected=hidden@model.params['w2']+model.params['b2']
        _,actual=model.forward(z[outside]);np.testing.assert_array_equal(actual,expected)
        scores=model.predict(x);indices=np.array([0,13,200,700])
        np.testing.assert_allclose(model.predict(x[indices]),scores[indices],rtol=0,atol=1e-12)
        for name in response.ARRAYS: np.testing.assert_array_equal(getattr(model,name),getattr(before,name))
        self.assertEqual(model.meta,before.meta)

    def test_configuration_and_unsupported_combinations_fail_before_training(self):
        resolved=bounded_experiment(self.spec,self.cfg['optimization'])
        self.assertEqual(resolved['architecture'],{'response_adapter':True})
        for extra in ({'kept_sensors':[3,13,14,15,16]}, {'sensor_dropout':.1}, {'input_noise':.1}, {'adversarial_epsilon':.1}):
            model=self.migrated();before=copy.deepcopy(model)
            with self.assertRaises(ValueError): fit_weighted(model,self.train,{**resolved,**extra},'Acetaldehyde',time.monotonic()+30)
            self.assertEqual(model.meta,before.meta)
            for name in model.params: np.testing.assert_array_equal(model.params[name],before.params[name])
        for invalid in ({'ridge':0,'strength':16,'guard_multiplier':1.5},{'ridge':10,'strength':True,'guard_multiplier':1.5}):
            with self.assertRaises(ValueError): bounded_experiment({**self.spec,'response_adapter':invalid},self.cfg['optimization'])

    def test_ai_configuration_resolves_the_real_migration(self):
        raw={'action':'experiment','experiments':[self.spec]}
        specs=configurations(raw,self.cfg,self.parent(),[],self.sha,2,'Acetaldehyde')
        self.assertEqual(specs[0]['architecture'],{'response_adapter':True})
        self.assertEqual(specs[0]['proposal_origin'],'ai_composed_configuration')

    def test_existing_head_only_exposes_compatible_augmentation_and_can_preserve_architecture(self):
        model=self.fitted();space=search_space(self.cfg,model)
        self.assertEqual(set(space['augmentation']),{'response_gain','response_sensor_gain','clean_fraction'})
        self.assertNotIn('kept_sensors',space)
        spec={**self.spec,'architecture':{'response_adapter':True}};spec.pop('response_adapter')
        rows=configurations({'action':'experiment','experiments':[spec]},self.cfg,model,[],self.sha,2,'Acetaldehyde')
        self.assertEqual(rows[0]['architecture'],{'response_adapter':True})
        with self.assertRaisesRegex(ValueError,'explicit training'):
            configurations({'action':'experiment','experiments':[spec]},self.cfg,self.parent(),[],self.sha,2,'Acetaldehyde')

    def test_risk_allowance_is_configurable_and_still_rejects_larger_losses(self):
        parent={'worst_macro_f1':.70,'mean_macro_f1':.75,'cases':{'a':{'macro_f1':.70},'b':{'macro_f1':.80}}}
        current=copy.deepcopy(parent);current['worst_macro_f1']=.6981;current['mean_macro_f1']=.749
        current['cases']['a']['macro_f1']=.6981
        policy=robustness.settings(self.cfg['optimization'])
        self.assertTrue(robustness.gate(current,parent,policy)['accepted'])
        self.assertFalse(robustness.gate(current,parent,{**policy,'max_worst_f1_drop':0})['accepted'])
        current['worst_macro_f1']=.69
        self.assertFalse(robustness.gate(current,parent,policy)['accepted'])
        current['worst_macro_f1']=.6981;current['cases']['b']['macro_f1']=.77
        self.assertIn('robust_scenario_regression:b',robustness.gate(current,parent,policy)['reasons'])

    def test_within_risk_budget_checkpoint_ranking_follows_primary_metric(self):
        initial=self.parent().meta['epochs'];real_evaluate=evaluate
        def measured(model,data):
            metric=real_evaluate(model,data)
            if data is self.val:metric['macro_f1']={0:.7,1:.72,2:.74}[model.meta['epochs']-initial]
            return metric
        def stress(model,suite):
            value=.70-.001*(model.meta['epochs']-initial)
            return {'worst_macro_f1':value,'mean_macro_f1':value,'cases':{'sensor_01_missing':{'macro_f1':value}}}
        spec={**self.cfg['experiments'][0],'epochs':2}
        with tempfile.TemporaryDirectory() as tmp,patch('gas_a.optimize.evaluate',side_effect=measured),patch('gas_a.robustness.evaluate_suite',side_effect=stress):
            for ranking,chosen in [('macro_f1',2),('worst_case',1)]:
                limits={**self.cfg['optimization'],'robustness':{'enabled':True,'max_worst_f1_drop':.005,'max_scenario_f1_drop':.02,'ranking':ranking}}
                r=train_candidate(self.cfg['baseline'],self.train,self.val,Path(tmp)/ranking,spec,{'weakest_class':'Acetaldehyde'},limits)
                self.assertEqual(r['final_epochs'],initial+chosen)
                self.assertTrue(r['robust_gate']['accepted'])

    def test_export_passes_unchanged_quality_gates_and_reproduces_selected_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'trial'
            report=train_candidate(self.cfg['baseline'],self.train,self.val,folder,self.spec,{'weakest_class':'Acetaldehyde'},
                {**self.cfg['optimization'],'checkpoint_gate':self.cfg['gate']})
            metric=report['validation_after']
            self.assertGreater(metric['macro_f1'],.85)
            self.assertGreaterEqual(metric['per_class']['Acetaldehyde']['recall'],.5)
            self.assertTrue(gate(metric,report['validation_before'],report['validation_before'],self.cfg['gate'])['accepted'])
            self.assertTrue(report['robust_gate']['accepted'])
            actual=load_model(folder/'checkpoint.npz');replay=load_model(folder/'migrated_parent.npz')
            fit_weighted(replay,self.train,report['configuration'],'Acetaldehyde',time.monotonic()+30,
                         epochs_to_run=report['final_epochs']-report['initial_epochs'])
            for group in ('params','first','second'):
                for name,value in getattr(actual,group).items():np.testing.assert_array_equal(getattr(replay,group)[name],value)
            for name in response.ARRAYS:np.testing.assert_array_equal(getattr(actual,name),getattr(replay,name))
            self.assertEqual(actual.meta,replay.meta)
            training=read_json(folder/'training.json')
            self.assertEqual(training['configuration']['optimizer'],'ridge_increment_then_Adam')
            self.assertEqual(training['configuration']['response_adapter'],self.spec['response_adapter'])
            recorded=load_model(folder/'migrated_parent.npz')
            fit_weighted(recorded,self.train,training['configuration'],'Acetaldehyde',time.monotonic()+30,
                         epochs_to_run=report['final_epochs']-report['initial_epochs'])
            for name,value in actual.params.items():np.testing.assert_array_equal(recorded.params[name],value)
            # The ACTUAL selected child, not a fresh baseline, is the next starting point.
            next_spec={**self.spec,'name':'resume_response','epochs':1};next_spec.pop('response_adapter')
            second=train_candidate(folder/'checkpoint.npz',self.train,self.val,Path(tmp)/'resume',next_spec,
                {'weakest_class':'Acetaldehyde'},self.cfg['optimization'])
            self.assertEqual(second['parent_checkpoint_sha256'],digest(folder/'checkpoint.npz'))
            self.assertGreater(second['final_steps'],report['final_steps'])
            self.assertNotIn('response_adapter',read_json(Path(tmp)/'resume/training.json')['configuration'])
            np.testing.assert_array_equal(load_model(Path(tmp)/'resume/checkpoint.npz').response_mean,actual.response_mean)


if __name__ == '__main__': unittest.main()
