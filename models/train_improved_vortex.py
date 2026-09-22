"""Actual family-specific Vortex training used in the improvement experiment.

Inputs x/dy are inner fit rows only; va/val are inner validation only.
Return includes the best validation checkpoint and mechanism gradient checks.
"""
import copy
import numpy as np
import torch
from .vortex_improved import ImprovedVortex
from .train_vortex import BITS,weights,product_joint,ap

def raw_predict(b,x,ablate=False):
    family=b['family'];m=b['model']
    if family=='Vortex_improved':
        m.eval();values=[]
        with torch.inference_mode():
            for i in range(0,len(x),512):
                tx=torch.tensor(x[i:i+512],dtype=torch.float32)
                z=m.raw_head(tx) if ablate else m(tx)
                values.append(torch.sigmoid(z).numpy())
        return product_joint(np.concatenate(values))
    if family=='LR_tuned':return product_joint(np.column_stack([a.predict_proba(x)[:,1] for a in m]))
    if ablate:
        features=m._features(x) if family=='Physical_improved' else m.transform(x)
        features[:,72:]=0
        marginal=np.column_stack([a.predict_proba(features)[:,1] for a in m.readout])
    else:marginal=m.predict(x)
    return product_joint(marginal)

def fit_vortex(config,x,dy,val,va):
    torch.manual_seed(42);rng=np.random.default_rng(42);m=ImprovedVortex(dropout=.15)
    y=BITS[dy.joint_target.to_numpy()];w=weights(dy);prev=np.average(y,axis=0,weights=w)
    with torch.no_grad():m.raw_head.weight.zero_();m.raw_head.bias.copy_(torch.tensor(np.log(prev/(1-prev)),dtype=torch.float32))
    b={'family':'Vortex_improved','model':m,'config':config}
    def score():
        values=[ap(val,raw_predict(b,a)) for a in va]
        return float(.5*(values[0]+np.mean(values[1:])))
    best=score();state=copy.deepcopy(m.state_dict());epochbest=0;stale=0;history=[{'epoch':0,'validation_score':best}]
    heads=[];dynamics=[]
    for name,p in m.named_parameters():(heads if any(name.startswith(k) for k in ['raw_head','wave_head','topology_head']) else dynamics).append(p)
    opt=torch.optim.AdamW([{'params':heads,'lr':config['lr']},{'params':dynamics,'lr':.3*config['lr']}],weight_decay=config['weight_decay'])
    for epoch in range(100):
        m.train();order=rng.permutation(len(x));losses=[]
        for start in range(0,len(x),512):
            ix=order[start:start+512];tx=torch.tensor(x[ix],dtype=torch.float32);yy=torch.tensor(y[ix]);ww=torch.tensor(w[ix],dtype=torch.float32)
            opt.zero_grad();loss=(torch.nn.functional.binary_cross_entropy_with_logits(m(tx),yy,reduction='none').mean(1)*ww).sum()/ww.sum()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite Vortex loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(float(loss.detach()))
        score_now=score();history.append({'epoch':epoch+1,'validation_score':score_now,'train_loss':float(np.mean(losses))})
        if score_now>best+1e-6:best=score_now;state=copy.deepcopy(m.state_dict());epochbest=epoch+1;stale=0
        else:stale+=1
        if stale>=20:break
    m.load_state_dict(state);m.eval();m.zero_grad()
    proxy=m.topology_head(m.feature_views(torch.tensor(x[:64],dtype=torch.float32))['proxy']);proxy.square().mean().backward()
    diagnostic={'topology_phase_gradient_norm':float(m.phase.weight.grad.norm()),'topology_unitary_gradient_norm':float(m.unitary_evolution.W.grad.norm()),
                'parameters':sum(p.numel() for p in m.parameters()),'time_steps':m.time_steps.detach().tolist()};m.zero_grad()
    return b|{'validation_score':best,'best_epoch':epochbest,'epochs_run':epoch+1,'history':history,'diagnostics':diagnostic}
