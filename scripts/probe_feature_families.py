"""Public-data feature diagnostics, not deployable models or SDK experiments."""
import argparse
import os
from pathlib import Path
import sys
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
from gas_a.data import load_pair,load_model,score_predictions
from gas_a.pipeline import load_config
from gas_a.io import write_json
from probe_generalization import response_shape,kinetic_shape


def relative(raw):
    source=raw.reshape(-1,16,8);out=source.copy();delta=source[:,:,1]-1
    out[:,:,0]=delta;out[:,:,1]=np.sign(delta)*np.log1p(np.abs(delta))
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        ratio=np.divide(source[:,:,2:],np.abs(source[:,:,:1]),out=np.zeros_like(source[:,:,2:]),where=np.abs(source[:,:,:1])>0)
        out[:,:,2:]=ratio*np.abs(delta[:,:,None])/np.asarray([.001,.01,.1,.001,.01,.1])
    return np.clip(out,-100,100).reshape(-1,128)


def unit(x):
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)


def softmax(x):
    value=np.exp(x-x.max(axis=1,keepdims=True));return value/value.sum(axis=1,keepdims=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/a.json');train,val=load_pair(cfg['train'],cfg['validation']);parent=load_model(cfg['baseline'])
    transforms={'raw':lambda x:x,'shape':response_shape,'kinetic':kinetic_shape,'relative_baseline':relative,
                'signed_log':lambda x:np.sign(x)*np.log1p(np.abs(x))}
    labels=train['y'];rows=[]
    for name,function in transforms.items():
        full_x,full_y=function(train['x']),function(val['x'])
        groups=[('all',np.arange(128))]+[(f'kind_{i}',np.arange(i,128,8)) for i in range(8)]
        for kind,cols in groups:
            x,y=full_x[:,cols],full_y[:,cols]
            for scaling in ('standard','unit'):
                if scaling=='standard':
                    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-12]=1
                    xx,yy=(x-mean)/scale,(y-mean)/scale
                else:xx,yy=unit(x),unit(y)
                centres=np.asarray([xx[labels==i].mean(axis=0) for i in range(6)])
                logits= -(np.sum(yy**2,axis=1,keepdims=True)+np.sum(centres**2,axis=1)[None,:]-2*yy@centres.T)
                metric=score_predictions(softmax(logits),val['y'])
                rows.append({'transform':name,'family':kind,'scaling':scaling,'classifier':'centroid','macro_f1':metric['macro_f1'],'target_recall':metric['per_class']['Acetaldehyde']['recall']})
                if scaling=='standard':
                    xx=np.column_stack([xx,np.ones(len(xx))]);yy=np.column_stack([yy,np.ones(len(yy))])
                    weights=len(labels)/(6*np.bincount(labels,minlength=6)[labels])
                    targets=np.eye(6)[labels]
                    for ridge in (.01,1.,100.):
                        matrix=xx.T@(xx*weights[:,None])+ridge*np.eye(xx.shape[1])
                        coeff=np.linalg.solve(matrix,xx.T@(targets*weights[:,None]))
                        metric=score_predictions(softmax(yy@coeff),val['y'])
                        rows.append({'transform':name,'family':kind,'scaling':scaling,'classifier':'ridge','ridge':ridge,'macro_f1':metric['macro_f1'],'target_recall':metric['per_class']['Acetaldehyde']['recall']})
    rows.sort(key=lambda r:(r['macro_f1'],r['target_recall']),reverse=True)
    write_json(out/'summary.json',{'research_only':True,'provider_calls':0,'future_data_used':False,'gradient_data':'public train only','evaluation_data':'public validation only','rows':rows})
    import json
    print(json.dumps({'top_f1':rows[:8],'top_target':sorted(rows,key=lambda r:r['target_recall'],reverse=True)[:5]},ensure_ascii=False))


if __name__=='__main__':main()
