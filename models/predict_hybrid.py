"""Frozen conditional hybrids, loaded from JSON/NPZ without pickle or API calls."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from .predict import Predictor,BITS
from .model_archive import load_archive

HERE=Path(__file__).resolve().parent/'hybrid_checkpoints'

class HybridPredictor(Predictor):
    def __init__(self,fold=1,category='three_selected'):
        manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
        options=[b for b in manifest['examples'] if b['fold']==fold and b['category']==category]
        if len(options)!=1:raise ValueError('Available: fold1 three_selected/LR_selected, fold2 three_selected')
        entry=options[0];self.bundle=load_archive(HERE/entry['meta']);self.fold=fold;self.kind=category;self.config={};self.members=[]
        self.columns=self.bundle['spec']['columns']
        for member in entry['members']:
            pre=object.__new__(Predictor);folder=HERE/member['preprocessor']
            pre.base=dict(np.load(folder/'base_preprocessor.npz',allow_pickle=False));pre.extra=dict(np.load(folder/'extra_preprocessor.npz',allow_pickle=False))
            models={int(j):load_archive(HERE/path) for j,path in member['models'].items()};self.members.append((pre,models))

    def joint(self,raw):
        raw=np.atleast_2d(np.asarray(raw,float))
        if raw.shape[1]!=19 or not len(raw) or np.isinf(raw).any():raise ValueError('Expected nonempty19-column finite-or-missing inputs')
        predictions=[]
        for pre,models in self.members:
            x=pre.transform(raw).astype('float32');pp=[]
            for j in self.columns:
                b=models[j]
                pp.append(np.column_stack([m.predict_proba(x)[:,1] for m in b['model']]) if j==3 else b['model'].predict(x))
            predictions.append(np.stack(pp,1))
        p=np.mean(predictions,0);meta=self.bundle['model'];q=meta.predict(p,raw) if self.bundle['spec']['dynamic'] else meta.predict(p)
        q=np.clip(np.asarray(q,float),1e-8,1-1e-8)
        joint=np.prod(np.where(BITS[None,:,:]>0,q[:,None,:],1-q[:,None,:]),axis=2)
        return softmax(np.log(np.clip(joint,1e-12,1))/self.bundle['temperature'],axis=1)

    def predict(self,state):
        result=super().predict(state)
        result.update(source_fold=self.fold,combiner=self.bundle['spec']['kind'],base_path=self.bundle['base_path'],
                      includes_separate_LR_expert=3 in self.columns,base_model_evaluations=len(self.columns)*len(self.members),external_model_calls=0,
                      note='Single-fold research checkpoint; comparison uses five-fold OOF. No diagnostic or screening cutoff. living_alone means single-person household.')
        return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--fold',type=int,default=1);a.add_argument('--category',default='three_selected');args=a.parse_args()
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):result=HybridPredictor(args.fold,args.category).predict(json.loads(Path(args.input).read_text(encoding='utf-8')))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
