"""Continuous measurements and screening share a single EPF representation.

All measurements in TARGETS are training targets only. No measured blood pressure
or laboratory value is accepted by the input contract.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.nn import functional as F
from .event_field_runtime import FastEventPlasticField
from .event_field_baselines import MatchedResidualMLP

TARGETS=['HE_glu','HE_sbp','HE_dbp','HE_TG','HE_HDL_st2']
MODES=['classification','regression','joint']
FAMILIES=['EPF','MLP']
NEAR_BANDS=np.array([10.,10.,5.,30.,5.])

class TargetTransform:
    def fit(self,values,weights):
        z=self.forward_raw(values);self.mean=np.average(z,axis=0,weights=weights)
        self.scale=np.sqrt(np.average((z-self.mean)**2,axis=0,weights=weights))
        if (self.scale<=0).any():raise ValueError('Degenerate fit targets')
        return self
    @staticmethod
    def forward_raw(values):
        a=np.array(values,dtype=float,copy=True)
        if a.ndim!=2 or a.shape[1]!=5 or not np.isfinite(a).all() or (a<=0).any():raise ValueError('Five complete positive target measurements required')
        a[:,3]=np.log1p(a[:,3]);return a
    def transform(self,values):return (self.forward_raw(values)-self.mean)/self.scale
    def inverse(self,z):
        a=np.asarray(z)*self.scale+self.mean;a=np.array(a,copy=True)
        a[:,3]=np.expm1(a[:,3])
        if not np.isfinite(a).all():raise FloatingPointError('Nonfinite inverse target prediction')
        return a
    def cutoffs(self,sex):
        sex=np.asarray(sex)
        if not np.isin(sex,[1,2]).all():raise ValueError('Known sex required for HDL cutoff')
        return np.column_stack([np.full(len(sex),100.),np.full(len(sex),130.),np.full(len(sex),85.),np.full(len(sex),150.),np.where(sex==1,40.,50.)])
    def regression_scores(self,prediction,sex):
        delta=np.asarray(prediction)-self.transform(self.cutoffs(sex))
        return np.column_stack([delta[:,0],np.maximum(delta[:,1],delta[:,2]),delta[:,3],-delta[:,4]])

def build(family,input_dim,seed=42):
    field=FastEventPlasticField(input_dim,9,seed=seed)
    if family=='EPF':return field
    if family=='MLP':return MatchedResidualMLP(input_dim,9,sum(p.numel() for p in field.parameters()),seed)
    raise ValueError('Unknown family')

def loss_parts(output,labels,numbers):
    return F.binary_cross_entropy_with_logits(output[:,:4],labels,reduction='none').mean(1),(output[:,4:]-numbers).square().mean(1)

def objective(class_loss,reg_loss,mode,normalizers):
    c=class_loss/normalizers[0];r=reg_loss/normalizers[1]
    if mode=='classification':return c
    if mode=='regression':return r
    if mode=='joint':return .5*(c+r)
    raise ValueError('Unknown objective')

def scores(output,mode,transform,sex):
    return transform.regression_scores(output[:,4:],sex) if mode=='regression' else output[:,:4]

class MonotoneCalibration:
    """Identical 8-parameter calibration opportunity for every candidate.

Fixed weak slope shrinkage; fit only on the separate calibration role. No
additional joint temperature or threshold selection on these calibration rows.
"""
    def fit(self,scores,labels,weights):
        x=np.asarray(scores,float);y=np.asarray(labels,float);w=np.asarray(weights,float);w=w/w.sum();self.coefficients=[]
        for j in range(4):
            if len(np.unique(y[w>0,j]))!=2:raise ValueError('Calibration requires both classes')
            def fun(ab):
                a,b=ab;z=a*x[:,j]+b;res=expit(z)-y[:,j]
                value=np.sum(w*(np.logaddexp(0,z)-y[:,j]*z))+.005*(a-1)**2
                grad=np.array([np.sum(w*res*x[:,j])+.01*(a-1),np.sum(w*res)])
                return value,grad
            opt=minimize(fun,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(0.,20.),(-20.,20.)],options={'maxiter':1000,'ftol':1e-12})
            if not opt.success:raise RuntimeError('Calibration optimization failed: '+str(opt.message))
            self.coefficients.append(opt.x.tolist())
        return self
    def predict(self,scores):
        ab=np.asarray(self.coefficients);return expit(np.asarray(scores)*ab[:,0]+ab[:,1])

def risk_from_components(probabilities):return 1-np.prod(1-np.asarray(probabilities),axis=1)
