"""Exact Vortex training routine used in the family-specific comparison.

Arguments dy/val are research frames with joint_target and wt_itvex columns.
va is [clean_validation, masked1, masked2, masked3]. No test inputs allowed.
"""
import copy
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .native_candidates import vortex_model
BITS=((np.arange(16)[:,None]>>np.arange(4))&1).astype(np.float32)
def weights(d):
    w=d.wt_itvex.to_numpy(dtype=float)
    return w/w.mean()
def product_joint(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8)
    return np.prod(np.where(BITS[None,:,:]>0,p[:,None,:],1-p[:,None,:]),axis=2)
def ap(d,p):
    y=BITS[d.joint_target.to_numpy()];m=p@BITS;w=weights(d)
    return float(np.mean([average_precision_score(y[:,j],m[:,j],sample_weight=w) for j in range(4)]))

def raw_predict(b,x):
    if b['family']=='Vortex_original_code':
        b['model'].eval();outputs=[]
        with torch.inference_mode():
            for start in range(0,len(x),512):outputs.append(torch.sigmoid(b['model'](torch.tensor(x[start:start+512],dtype=torch.float32))).numpy())
        return product_joint(np.concatenate(outputs))
    return product_joint(b['model'].predict(x))

def fit_vortex(trial,x,dy,va,val):
    torch.manual_seed(42);rng=np.random.default_rng(42);m=vortex_model();y=BITS[dy.joint_target.to_numpy()];w=weights(dy)
    with torch.no_grad():m.prediction_head.net[-1].bias.copy_(torch.tensor(np.log(np.average(y,axis=0,weights=w)/(1-np.average(y,axis=0,weights=w))),dtype=torch.float32))
    b={'family':'Vortex_original_code','model':m,'trial':trial}
    def score():
        values=[ap(val,raw_predict(b,a)) for a in va];return float(.5*(values[0]+np.mean(values[1:])))
    best=score();state=copy.deepcopy(m.state_dict());epochbest=0;stale=0;history=[{'epoch':0,'validation_score':best}]
    opt=torch.optim.AdamW(m.parameters(),lr=[.001,.0003][trial],weight_decay=1e-5)
    for epoch in range(30):
        m.train();order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),512):
            ix=order[start:start+512];xx=torch.tensor(x[ix],dtype=torch.float32);yy=torch.tensor(y[ix]);ww=torch.tensor(w[ix],dtype=torch.float32)
            opt.zero_grad();z=m(xx);loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,yy,reduction='none').mean(1)*ww).sum()/ww.sum()
            if not torch.isfinite(loss):raise RuntimeError('Vortex nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
        value=score();history.append({'epoch':epoch+1,'validation_score':value,'train_loss':float(np.mean(losses))})
        if value>best+1e-6:best=value;state=copy.deepcopy(m.state_dict());epochbest=epoch+1;stale=0
        else:stale+=1
        if stale>=8:break
    m.load_state_dict(state);m.eval();return b|{'validation_score':best,'best_epoch':epochbest,'epochs_run':epoch+1,'history':history,'parameters':sum(p.numel() for p in m.parameters())}
