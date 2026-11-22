"""Reproduce public-data developer controls; not platform history or AI decisions."""
import argparse
import os
from pathlib import Path
import sys

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gas_a.data import load_pair
from gas_a.io import digest, write_json
from gas_a.optimize import train_candidate, gate
from gas_a.pipeline import load_config
from gas_a.planning import CONTROL_SOURCES
from gas_a import robustness


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/a.json');train,val=load_pair(cfg['train'],cfg['validation'])
    rows=[]
    for multiplier in (1.,1.5,2.):
        name='response_guard_'+str(multiplier).replace('.','_')
        spec=dict(name=name,learning_rate=.0001,batch_size=32,epochs=4,balance=.5,weak_boost=1.,l2=0.,
                  response_adapter=dict(ridge=10.,strength=16.,guard_multiplier=multiplier))
        r=train_candidate(cfg['baseline'],train,val,out/name,spec,{'weakest_class':'Acetaldehyde'},
                          {**cfg['optimization'],'checkpoint_gate':cfg['gate']})
        rows.append({'configuration':spec,'macro_f1':r['validation_after']['macro_f1'],
                     'recalls':{k:v['recall'] for k,v in r['validation_after']['per_class'].items()},
                     'clean_gate':gate(r['validation_after'],r['validation_before'],r['validation_before'],cfg['gate']),
                     'robust_gate':r['robust_gate'],'robust_worst_f1':r['robust_validation_after']['worst_macro_f1'],
                     'checkpoint_sha256':r['checkpoint_sha256'],'report_sha256':digest(out/name/'training.json')})
    summary={'scope':'developer_CPU_controls_on_public_data; NOT_platform_rounds; NOT_AI_decisions',
             'parent_checkpoint_sha256':digest(cfg['baseline']),'train_sha256':train['sha256'],'validation_sha256':val['sha256'],
             'source_sha256':{n:digest(ROOT/'src/gas_a'/n) for n in CONTROL_SOURCES},
             'quality_gate':cfg['gate'],'robustness_policy':robustness.settings(cfg['optimization']),
             'real_model_calls':0,'future_data_used':False,'rows':rows,
             'reading':'仅为开发期同父同公开数据的实测先验；参数可由AI修改，须重新训练并由宿主测量后选版；不能冒充本轮成绩或平台history_ids。'}
    write_json(out/'summary.json',summary);write_json(ROOT/'configs/evidence/response_controls.json',summary)
    print([(r['configuration']['response_adapter']['guard_multiplier'],r['macro_f1'],r['clean_gate']['accepted'],r['robust_gate']['accepted']) for r in rows])


if __name__=='__main__':main()
