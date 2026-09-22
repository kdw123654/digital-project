"""Input-quality/profile-conditioned combinations trained on OOF scores only."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from .hybrid_ensembles import _fit_arrays,_probabilities

class InputContext:
    def __init__(self,mode='quality'):
        if mode not in ['quality','profile']:raise ValueError('Unknown context')
        self.mode=mode
    def raw(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=19 or np.isinf(x).any():raise ValueError('Expected19 raw features, NaN allowed')
        groups=[[1,2,3],[0,4,5,6,13,17,18],[7,8,9,10,11,12,14,15,16]]
        missing=np.column_stack([np.isnan(x[:,g]).mean(1) for g in groups])
        return missing if self.mode=='quality' else np.column_stack([x[:,0],x[:,4]-1,x[:,1],x[:,3],missing])
    def fit(self,x,w):
        z=self.raw(x);self.median=np.array([np.median(z[np.isfinite(z[:,j]),j]) if np.isfinite(z[:,j]).any() else 0. for j in range(z.shape[1])])
        z=np.where(np.isnan(z),self.median,z);self.mean=np.average(z,axis=0,weights=w);self.scale=np.sqrt(np.average((z-self.mean)**2,axis=0,weights=w));self.scale[self.scale<1e-6]=1
        return self
    def transform(self,x):
        z=self.raw(x);return np.clip((np.where(np.isnan(z),self.median,z)-self.mean)/self.scale,-6,6)

class ContextualLogitStack:
    """Score×context interactions; coefficients are not mixture percentages."""
    def __init__(self,C=.1,mode='quality',interaction_scale=.25):self.C=C;self.mode=mode;self.interaction_scale=interaction_scale
    def features(self,p,raw):
        p=_probabilities(p,self.experts);q=logit(np.clip(p,1e-7,1-1e-7)).reshape(len(p),-1)
        z=(q-self.mean)/self.scale;c=self.context.transform(raw)
        return np.column_stack([z,c*self.interaction_scale,(z[:,:,None]*c[:,None,:]).reshape(len(z),-1)*self.interaction_scale])
    def fit(self,p,raw,y,w):
        p,y,w=_fit_arrays(p,y,w);self.experts=p.shape[1];self.context=InputContext(self.mode).fit(raw,w)
        q=logit(np.clip(p,1e-7,1-1e-7)).reshape(len(p),-1);self.mean=np.average(q,axis=0,weights=w);self.scale=np.sqrt(np.average((q-self.mean)**2,axis=0,weights=w));self.scale[self.scale<1e-6]=1
        z=self.features(p,raw);self.models=[LogisticRegression(C=self.C,max_iter=3000).fit(z,y[:,j],sample_weight=w) for j in range(4)]
        return self
    def predict(self,p,raw):
        z=self.features(p,raw);return np.column_stack([m.predict_proba(z)[:,1] for m in self.models])

class ContextualMixture:
    """Per-label softmax expert weights plus explicit conditional calibration.

    With one expert this becomes the matched conditional-calibration control.
    The gate depends on input context, not outcomes or participant order.
    """
    def __init__(self,regularization=.1,mode='quality',max_iter=250):self.regularization=regularization;self.mode=mode;self.max_iter=max_iter
    def _unpack(self,v):
        cut=self.dim*4*self.experts
        return v[:cut].reshape(self.dim,4,self.experts),v[cut:cut+4],v[cut+4:cut+8],v[cut+8:].reshape(self.dim-1,4)
    def _forward(self,v,p,c):
        theta,logs,bias,shift=self._unpack(v)
        gates=torch.softmax(torch.einsum('nd,dke->nke',torch.cat([torch.ones_like(c[:,:1]),c],1),theta),dim=2)
        mixture=(gates*p.transpose(1,2)).sum(2).clamp(1e-7,1-1e-7)
        z=torch.exp(logs.clamp(-2,2))*torch.logit(mixture)+bias+c@shift
        return z,gates
    def fit(self,p,raw,y,w):
        p,y,w=_fit_arrays(p,y,w);self.experts=p.shape[1];self.context=InputContext(self.mode).fit(raw,w);c=self.context.transform(raw);self.dim=c.shape[1]+1
        tp=torch.tensor(p,dtype=torch.float64);tc=torch.tensor(c,dtype=torch.float64);ty=torch.tensor(y,dtype=torch.float64);tw=torch.tensor(w/w.sum(),dtype=torch.float64)
        initial=np.zeros(self.dim*4*self.experts+8+(self.dim-1)*4)
        def objective(v):
            t=torch.tensor(v,dtype=torch.float64,requires_grad=True);z,g=self._forward(t,tp,tc);theta,logs,bias,shift=self._unpack(t)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,ty,reduction='none').mean(1)*tw).sum()
            loss=loss+self.regularization*(theta.square().mean()+shift.square().mean()+logs.square().mean())
            loss.backward();return float(loss.detach()),t.grad.numpy()
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'ftol':1e-11,'gtol':1e-6})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise RuntimeError('Invalid mixture optimization')
        if not result.success and np.max(np.abs(result.jac))>1e-4:raise RuntimeError('Mixture did not converge')
        self.parameters=result.x.copy();self.diagnostics={'iterations':int(result.nit),'objective':float(result.fun),'max_gradient':float(np.max(np.abs(result.jac))),'success':bool(result.success)}
        return self
    def predict(self,p,raw):
        p=_probabilities(p,self.experts);c=self.context.transform(raw)
        with torch.inference_mode():
            z,_=self._forward(torch.tensor(self.parameters),torch.tensor(p.copy()),torch.tensor(c));return torch.sigmoid(z).numpy()
    def gating_weights(self,p,raw):
        p=_probabilities(p,self.experts);c=self.context.transform(raw)
        with torch.inference_mode():
            _,g=self._forward(torch.tensor(self.parameters),torch.tensor(p.copy()),torch.tensor(c));return g.numpy()
