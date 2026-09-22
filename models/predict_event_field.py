"""Portable inference for the single event-plastic field; no pickle or API key."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from scipy.special import softmax
from .predict import Predictor,BITS
from .event_plastic_field import EventPlasticField

HERE=Path(__file__).resolve().parent/'event_field_checkpoint'

class EventFieldPredictor(Predictor):
    def __init__(self,device='cpu'):
        meta=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'));self.kind='event_plastic_field';self.device=device;self.meta=meta
        self.base=dict(np.load(HERE/'base_preprocessor.npz',allow_pickle=False));self.extra=dict(np.load(HERE/'extra_preprocessor.npz',allow_pickle=False))
        self.model=EventPlasticField(**meta['constructor']);arrays=dict(np.load(HERE/'weights.npz',allow_pickle=False));self.model.load_state_dict({k:torch.tensor(v) for k,v in arrays.items()});self.model.to(device);self.model.eval()
        self.config={'temperature':meta['temperature'],'screen_cut95':meta['cutoff']}
    def joint(self,raw):
        x=self.transform(np.atleast_2d(raw)).astype('float32')
        with torch.inference_mode():logits=self.model(torch.tensor(x,device=self.device)).cpu().numpy()
        p=np.clip(np.asarray(1/(1+np.exp(-np.clip(logits,-60,60))),float),1e-8,1-1e-8)
        joint=np.prod(np.where(BITS[None]>0,p[:,None],1-p[:,None]),axis=2)
        return softmax(np.log(np.clip(joint,1e-12,1))/self.config['temperature'],axis=1)
    def predict(self,state):
        result=super().predict(state)
        result.update(source_fold=1,source_seed=42,single_shared_field=True,external_model_calls=0,
                      note='Single-fold research model. Development sensitivity95 is not a test guarantee or diagnosis. living_alone is single-person household.')
        return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=a.parse_args();torch.set_num_threads(1)
    print(json.dumps(EventFieldPredictor(args.device).predict(json.loads(Path(args.input).read_text(encoding='utf-8'))),ensure_ascii=False,indent=2,allow_nan=False))
