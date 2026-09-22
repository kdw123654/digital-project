"""Local single-pass clinical multitask research inference, no external calls."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .predict import Predictor,FEATURES
from .clinical_multitask import build,TargetTransform,MonotoneCalibration,scores,risk_from_components,TARGETS

HERE=Path(__file__).resolve().parent/'clinical_checkpoints'

class ClinicalPredictor(Predictor):
    def __init__(self,model='EPF_joint',device='cpu'):
        self.metadata=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
        if model not in self.metadata['models']:raise ValueError('Unknown clinical model')
        self.name=model;self.device=device;self.spec=self.metadata['models'][model]
        self.base=dict(np.load(HERE/'base_preprocessor.npz',allow_pickle=False));self.extra=dict(np.load(HERE/'extra_preprocessor.npz',allow_pickle=False))
        stats=dict(np.load(HERE/'target_transform.npz',allow_pickle=False));self.target=TargetTransform();self.target.mean=stats['mean'];self.target.scale=stats['scale']
        self.calibration=MonotoneCalibration();self.calibration.coefficients=self.spec['calibration']
        if model in ['LR','Ridge']:
            q=dict(np.load(HERE/'linear.npz',allow_pickle=False));self.coef=q['coef'];self.bias=q['bias'];self.model=None
        else:
            self.model=build(self.spec['family'],72,42);q=dict(np.load(HERE/(model+'.npz'),allow_pickle=False));self.model.load_state_dict({k:torch.tensor(v) for k,v in q.items()});self.model.to(device).eval()
    def infer(self,raw):
        raw=np.atleast_2d(np.asarray(raw,float));x=self.transform(raw).astype('float32')
        if self.model is None:output=x@self.coef.T+self.bias
        else:
            with torch.inference_mode():output=self.model(torch.tensor(x,device=self.device)).cpu().numpy()
        probability=self.calibration.predict(scores(output,self.spec['mode'],self.target,raw[:,4]))
        values=None if self.spec['mode']=='classification' else self.target.inverse(output[:,4:])
        return probability,values
    def predict(self,state):
        if not isinstance(state,dict) or set(state)-set(FEATURES):raise ValueError('Only documented non-invasive features are accepted')
        if state.get('age') is None or not 19<=state['age']<=39 or state.get('sex') not in [1,2]:raise ValueError('Age19-39 and sex1/2 required')
        if all(state.get(k) is None for k in ['HE_BMI','HE_wc','WHtR']):raise ValueError('At least one body measurement required')
        for k in ['HE_BMI','HE_wc','WHtR']:
            if state.get(k) is not None and not state[k]>0:raise ValueError('Positive body measurements required')
        if state.get('walking_days') is not None and state['walking_days'] not in range(8):raise ValueError('Walking days0..7 required')
        if state.get('sedentary_hours') is not None and not 0<=state['sedentary_hours']<=24:raise ValueError('Sedentary hours0..24 required')
        raw=np.array([np.nan if state.get(k) is None else state[k] for k in FEATURES]);p,values=self.infer(raw);risk=float(risk_from_components(p)[0])
        result={'version':'clinical_multitask_v18','model':self.name,'component_probabilities':dict(zip(['glucose','bp','tg','low_hdl'],p[0].tolist())),'any_abnormality_probability':risk,'screening':{target:{'cutoff':cut,'positive':bool(risk>=cut)} for target,cut in self.spec['cutoffs'].items()},'research_only':True,'source_fold':1,'source_seed':42,'single_shared_field':self.name.startswith('EPF'),'external_model_calls':0,'note':'Development sensitivity targets are not test guarantees. Continuous estimates are auxiliary research predictions, not measured values. Classification and regression heads are not forced to agree. living_alone is single-person household.'}
        if values is not None:result['estimated_measurements']=dict(zip(['glucose_mg_dL','sbp_mmHg','dbp_mmHg','tg_mg_dL','hdl_mg_dL'],values[0].tolist()))
        return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',default='EPF_joint');a.add_argument('--input',required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=a.parse_args();torch.set_num_threads(1)
    print(json.dumps(ClinicalPredictor(args.model,args.device).predict(json.loads(Path(args.input).read_text(encoding='utf-8'))),indent=2,ensure_ascii=False,allow_nan=False))
