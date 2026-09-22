"""Portable inference for actual improved fold1 checkpoints; no pickle/API key.

These are single-fold models, not a full-data refit or an ensemble. Published
macro AP is five-fold OOF, not the in-sample score of these example checkpoints.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax
from .predict import FEATURES,Predictor,BITS
from .native_candidates import physics_module
from .vortex_improved import ImprovedVortex
from .physical_improved import ImprovedPhysical
from .portia_improved import ImprovedPortia

HERE=Path(__file__).resolve().parent/'improved_checkpoints'

class ImprovedPredictor:
    def __init__(self,name='vortex'):
        if name not in ['vortex','physical','portia']:raise ValueError('Unknown model')
        self.name=name;self.meta=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))[name]
        arrays=dict(np.load(HERE/f'{name}.npz',allow_pickle=False))
        self.pre=object.__new__(Predictor)
        self.pre.base=dict(np.load(HERE/'base_preprocessor.npz',allow_pickle=False));self.pre.extra=dict(np.load(HERE/'extra_preprocessor.npz',allow_pickle=False))
        if name=='vortex':
            import torch
            self.model=ImprovedVortex(**self.meta['constructor'])
            self.model.load_state_dict({k:torch.tensor(v) for k,v in arrays.items()});self.model.eval()
        else:
            self.model=object.__new__(ImprovedPhysical if name=='physical' else ImprovedPortia)
            self.model.__dict__.update(self.meta['attributes'])
            self.model.__dict__.update({k[5:]:v for k,v in arrays.items() if k.startswith('attr.')})
            if name=='physical':self.model.medium=physics_module().Medium(**{k[7:]:v for k,v in arrays.items() if k.startswith('medium.')})
            self.coef=arrays['readout.coef'];self.bias=arrays['readout.bias']

    def joint(self,raw):
        x=self.pre.transform(np.atleast_2d(raw)).astype('float32')
        if self.name=='vortex':
            import torch
            with torch.inference_mode():p=torch.sigmoid(self.model(torch.tensor(x))).numpy()
        else:
            features=self.model._features(x) if self.name=='physical' else self.model.transform(x)
            p=expit(features@self.coef.T+self.bias)
        p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
        joint=np.prod(np.where(BITS[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2)
        return softmax(np.log(np.clip(joint,1e-12,1))/self.meta['temperature'],axis=1)

    def predict(self,state):
        if not isinstance(state,dict) or set(state)-set(FEATURES):raise ValueError('Only documented noninvasive features allowed')
        if state.get('age') is None or not 19<=state['age']<=39 or state.get('sex') not in [1,2]:raise ValueError('Age19-39 and sex1/2 required')
        if all(state.get(k) is None for k in ['HE_BMI','HE_wc','WHtR']):raise ValueError('At least one body measurement required')
        for k in ['HE_BMI','HE_wc','WHtR']:
            if state.get(k) is not None and not state[k]>0:raise ValueError('Body measurements must be positive')
        if state.get('walking_days') is not None and state['walking_days'] not in range(8):raise ValueError('Walking days must be0..7')
        if state.get('sedentary_hours') is not None and not 0<=state['sedentary_hours']<=24:raise ValueError('Sedentary hours must be0..24')
        raw=np.array([np.nan if state.get(k) is None else state[k] for k in FEATURES],float)
        joint=self.joint(raw)[0];p=joint@BITS
        return {'model':self.name,'fold':1,'component_probabilities':dict(zip(['glucose','bp','tg','low_hdl'],p.tolist())),
                'any_abnormality_probability':float(joint[1:].sum()),'expected_abnormality_count':float(p.sum()),
                'missing_inputs':[k for k,v in zip(FEATURES,raw) if np.isnan(v)],'research_only':True,
                'note':'Probability comparison only; no diagnostic or screening cutoff selected. Single fold checkpoint; reported scores use pooled5fold OOF.'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['vortex','physical','portia'],default='vortex');ap.add_argument('--input',required=True);a=ap.parse_args()
    print(json.dumps(ImprovedPredictor(a.model).predict(json.loads(Path(a.input).read_text(encoding='utf-8'))),ensure_ascii=False,indent=2,allow_nan=False))
