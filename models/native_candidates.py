"""Family-specific mechanisms, explicitly distinguishing originals from proposals."""
import importlib.util,sys,types
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VORTEX_SOURCE=Path(__file__).resolve().parent/'native_sources/vortex'
FREE_SOURCE=Path(__file__).resolve().parent/'native_sources/free_v01'

def vortex_model():
    # Original code, input/task dimensions adapted; no edits to source project.
    for p in [VORTEX_SOURCE,VORTEX_SOURCE/'model']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    from vortexnet import VortexNet
    return VortexNet(input_dim=72,latent_dim=128,hidden_dim=64,num_steps=4,num_tasks=4,is_classification=True,superluminal_reg=True)

def physics_module():
    name='knhanes_free_physics'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,FREE_SOURCE/'physics.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

class PhysicalReadout:
    """Original FREE v0.1 medium/integrator + classification readout.

    Static rows each start from zero; this is not the latest v0.2 open medium.
    No gradient or label updates to the physical medium. Only readout is fitted.
    """
    def __init__(self,steps=16,C=.1):self.steps=steps;self.C=C
    def transform(self,x):
        p=physics_module();sim=p.Simulator(self.medium,dt=.04,seed=42,batch=len(x))
        drive=np.tanh((x-self.input_mean)/self.input_scale)
        for _ in range(self.steps):sim.step(drive,temperature=0.)
        return np.column_stack([x,sim.x,sim.v])
    def fit(self,x,y,w):
        p=physics_module();self.medium=p.Medium.random(n=24,input_dim=x.shape[1],seed=42)
        self.medium.drive/=np.sqrt(x.shape[1]);self.input_mean=x.mean(0);self.input_scale=x.std(0);self.input_scale[self.input_scale<1e-8]=1
        self.scaler=StandardScaler();features=self.scaler.fit_transform(self.transform(x))
        self.readout=[LogisticRegression(C=self.C,max_iter=3000).fit(features,y[:,j],sample_weight=w) for j in range(4)]
        return self
    def predict(self,x):
        f=self.scaler.transform(self.transform(x))
        return np.column_stack([m.predict_proba(f)[:,1] for m in self.readout])

class PortiaSTDP:
    """New implementation of the Portia design document, NOT an existing connectome.

    Fixed input drive, LIF state/reset, E/I recurrent wiring, local STDP on
    excitatory contacts and rate homeostasis. Labels train only linear readout.
    """
    def __init__(self,plasticity=.001,C=.1,epochs=4):self.plasticity=plasticity;self.C=C;self.epochs=epochs
    def encode(self,x,learn=False,w=None):
        z=np.tanh((x-self.input_mean)/self.input_scale);drive=.4/(1+np.exp(-(z@self.input_weights)))+.05
        n=len(x);v=np.zeros((n,24));spike=np.zeros_like(v);trace=np.zeros_like(v);total=np.zeros_like(v)
        meanw=np.ones(n)/n if w is None else w/w.sum();rate=0.
        for _ in range(24):
            v=.9*v+drive+.15*spike@self.recurrent.T
            new=(v>=self.threshold).astype(float);v=np.where(new>0,0,v)
            if learn:
                # Old traces exclude same-step co-spikes; row weights are design weights.
                ltp=(new*meanw[:,None]).T@trace
                ltd=(trace*meanw[:,None]).T@new
                delta=self.plasticity*(ltp-1.05*ltd)
                excit=self.mask&(self.sign[None,:]>0)
                self.recurrent[excit]=np.clip(self.recurrent[excit]+delta[excit],0,.25)
                self.threshold=np.clip(self.threshold+.01*((new*meanw[:,None]).sum(0)-.15),.7,1.4)
            trace=.9*trace+new;spike=new;total+=new
        return np.column_stack([x,total/24,v])
    def fit(self,x,y,w):
        rng=np.random.default_rng(42);self.input_mean=x.mean(0);self.input_scale=x.std(0);self.input_scale[self.input_scale<1e-8]=1
        self.input_weights=rng.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],24))
        self.mask=(rng.random((24,24))<.2)&~np.eye(24,dtype=bool);self.sign=np.where(np.arange(24)<19,1.,-1.)
        self.recurrent=rng.uniform(.02,.1,(24,24))*self.mask*self.sign[None,:];self.threshold=np.ones(24)
        initial=self.recurrent.copy()
        for _ in range(self.epochs):self.encode(x,learn=True,w=w)
        self.plasticity_change_norm=float(np.linalg.norm(self.recurrent-initial))
        features=self.encode(x);self.mean_firing=float(features[:,72:96].mean())
        self.scaler=StandardScaler().fit(features);f=self.scaler.transform(features)
        self.readout=[LogisticRegression(C=self.C,max_iter=3000).fit(f,y[:,j],sample_weight=w) for j in range(4)]
        return self
    def predict(self,x):
        f=self.scaler.transform(self.encode(x));return np.column_stack([m.predict_proba(f)[:,1] for m in self.readout])
