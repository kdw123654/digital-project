"""Portable NumPy inference for the two retained research models. No network calls."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax

HERE=Path(__file__).resolve().parent
BITS=((np.arange(16)[:,None]>>np.arange(4))&1).astype(float)
NUMERIC=['age','HE_BMI','HE_wc','WHtR']
CATEGORIES={'sex':[1,2],'incm':[1,2,3,4],'edu':[1,2,3,4],'sm_presnt':[0,1],'dr_month':[0,1],'pa_aerobic':[0,1]}
EXTRA_NUM=['sedentary_hours','walking_days']
EXTRA_CAT={'stress_level':[1,2,3,4],'employed':[0,1],'alcohol_frequency':[0,1,2,3,4,5],
           'alcohol_amount':[0,1,2,3,4,5],'strength_days':[0,1,2,3,4,5],'family_history':[0,1],'living_alone':[0,1]}
FEATURES=NUMERIC+list(CATEGORIES)+EXTRA_NUM+list(EXTRA_CAT)

class Predictor:
    def __init__(self,model='nn'):
        if model not in ['nn','lr','sparse']:raise ValueError('Choose nn, lr or sparse')
        self.kind=model;self.base=dict(np.load(HERE/'preprocessing_base.npz',allow_pickle=False))
        self.extra=dict(np.load(HERE/'preprocessing_extra.npz',allow_pickle=False))
        self.params=dict(np.load(HERE/f'{model}.npz',allow_pickle=False))
        self.config=json.loads((HERE/'config.json').read_text(encoding='utf-8'))[model]
        if model=='nn':
            self.fused=np.zeros((12,72),dtype='float32')
            self.bias=np.concatenate([self.params[n+'.bias'] for n in ['body','context','behavior']])
            for k,n in enumerate(['body','context','behavior']):self.fused[4*k:4*k+4,self.params[n+'_indices']]=self.params[n+'.weight']
        if model=='sparse':
            import importlib.util,torch
            spec=importlib.util.spec_from_file_location('adaptive_structure',HERE/'adaptive_structures.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.reliability=module.reliability
            self.model=module.AdaptiveNet(72,self.config['groups'],'sparse_reliable')
            # Restore saved topology, whose edge count depends on the training seed.
            self.model.edges=torch.tensor(self.params['edges'],dtype=torch.long)
            self.model.edge_weights=torch.nn.Parameter(torch.zeros(self.params['edge_weights'].shape))
            self.model.load_state_dict({k:torch.tensor(v) for k,v in self.params.items()});self.model.eval()

    def transform(self,x):
        x=np.atleast_2d(np.asarray(x,float))
        if x.shape[1]!=19 or np.isinf(x).any():raise ValueError('Expected 19 finite-or-missing features')
        n=x[:,:4];missing=np.isnan(n);z=(np.where(missing,self.base['median'],n)-self.base['mean'])/self.base['scale']
        blocks=[z,missing.astype(float)]
        for j,values in enumerate(CATEGORIES.values()):
            c=x[:,4+j]
            if not (np.isnan(c)|np.isin(c,values)).all():raise ValueError('Unknown base category')
            blocks.extend([(c[:,None]==np.asarray(values[1:])).astype(float),np.isnan(c)[:,None].astype(float)])
        blocks.extend([np.maximum(z[:,:,None]-self.base['knots'][None,:,:],0).reshape(len(x),12),z*(x[:,4]==1)[:,None]])
        extra=x[:,10:12];miss=np.isnan(extra)
        blocks.extend([(np.where(miss,self.extra['median'],extra)-self.extra['mean'])/self.extra['scale'],miss.astype(float)])
        for j,values in enumerate(EXTRA_CAT.values()):
            c=x[:,12+j]
            if not (np.isnan(c)|np.isin(c,values)).all():raise ValueError('Unknown extra category')
            blocks.extend([(c[:,None]==np.asarray(values[1:])).astype(float),np.isnan(c)[:,None].astype(float)])
        return np.concatenate(blocks,axis=1)

    def joint(self,x):
        raw=np.atleast_2d(np.asarray(x,float));x=self.transform(raw)
        if self.kind=='sparse':
            import torch
            packed=np.column_stack([x,self.reliability(raw)]).astype('float32')
            with torch.inference_mode():p=self.model(torch.tensor(packed)).exp().numpy()
            return softmax(np.log(np.clip(p,1e-12,1))/self.config['temperature'],axis=1)
        if self.kind=='nn':
            x=x.astype('float32');h=np.tanh(x@self.fused.T+self.bias);a,c,b=h[:,:4],h[:,4:8],h[:,8:]
            logits=np.concatenate([x,a,c,b,a*c,a*b],axis=1)@self.params['output.weight'].T+self.params['output.bias']
            risk=expit(logits[:,0]/self.config['temperature'])
            conditional=softmax(logits[:,1:]@BITS[1:].T.astype('float32')/self.config['pattern_temperature'],axis=1)
            return np.column_stack([1-risk,risk[:,None]*conditional])
        marginal=np.clip(expit(x@self.params['coef'].T+self.params['bias']),1e-8,1-1e-8)
        joint=np.prod(np.where(BITS[None,:,:]>0,marginal[:,None,:],1-marginal[:,None,:]),axis=2)
        return softmax(np.log(np.clip(joint,1e-12,1))/self.config['temperature'],axis=1)

    def predict(self,state):
        if not isinstance(state,dict) or set(state)-set(FEATURES):raise ValueError('Use the documented feature names only')
        if state.get('age') is None or not 19<=state['age']<=39 or state.get('sex') not in [1,2]:raise ValueError('Age19-39 and sex1/2 are required')
        if all(state.get(k) is None for k in ['HE_BMI','HE_wc','WHtR']):raise ValueError('At least one body measurement is required')
        for k in ['HE_BMI','HE_wc','WHtR']:
            if state.get(k) is not None and not state[k]>0:raise ValueError('Body measurements must be positive')
        if state.get('walking_days') is not None and state['walking_days'] not in range(8):raise ValueError('Walking days must be0..7')
        if state.get('sedentary_hours') is not None and not 0<=state['sedentary_hours']<=24:raise ValueError('Sedentary hours must be0..24')
        x=[np.nan if state.get(k) is None else state[k] for k in FEATURES]
        joint=self.joint(x)[0];marginal=joint@BITS;risk=float(joint[1:].sum())
        result={'model':self.kind,'component_probabilities':dict(zip(['glucose','bp','tg','low_hdl'],marginal.tolist())),
                'any_abnormality_probability':risk,'expected_abnormality_count':float(marginal.sum()),
                'missing_inputs':[k for k,v in zip(FEATURES,x) if np.isnan(v)],
                'research_only':True,'note':'Development sensitivity95 is not a guarantee or a diagnosis. living_alone means single-person household.'}
        if 'screen_cut95' in self.config:result.update({'screen_positive_at_development_95_policy':bool(risk>=self.config['screen_cut95']),'cutoff':self.config['screen_cut95']})
        else:result['note']='Experimental sparse candidate; probability comparison only. No screening cutoff selected.'
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['nn','lr','sparse'],default='nn');p.add_argument('--input',required=True);a=p.parse_args()
    print(json.dumps(Predictor(a.model).predict(json.loads(Path(a.input).read_text(encoding='utf-8'))),ensure_ascii=False,indent=2,allow_nan=False))
