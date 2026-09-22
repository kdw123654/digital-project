"""Executed clinical fitting loop. Inputs must already be split and fit-transformed.

Use the protocol in CLINICAL_MULTITASK.md. This helper accepts arrays directly;
it does not download or publish private KNHANES records.
"""
import os,copy,time,json
from pathlib import Path
import numpy as np,torch
from .clinical_multitask import build,loss_parts,objective
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def load_bundle(path):return torch.load(path,map_location='cpu',weights_only=False)


def save(path,record):
    temporary=Path(str(path)+'.tmp');torch.save(record,temporary);os.replace(temporary,path)

def numpy_predict(model,x,batch=2048):
    device=next(model.parameters()).device;model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):values.append(model(torch.tensor(x[start:start+batch],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(values)

def fit(family,mode,config,seed,x,y,numbers,w,vx,vy,vnumbers,vw,coef,bias,normalizers,folder):
    device='cuda' if torch.cuda.is_available() else 'cpu';model=build(family,x.shape[1],seed).to(device);model.set_linear(coef,bias)
    initial={k:p.detach().clone() for k,p in model.named_parameters()};opt=torch.optim.AdamW(model.parameters(),**config)
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=7,threshold=1e-4,threshold_mode='abs',min_lr=1e-5)
    data=[torch.tensor(a,dtype=torch.float32,device=device) for a in [x,y,numbers,w]];xx,yy,nn,ww=data
    def validate():
        prediction=numpy_predict(model,vx);c=np.mean(np.logaddexp(0,prediction[:,:4])-vy*prediction[:,:4],axis=1);r=np.mean((prediction[:,4:]-vnumbers)**2,axis=1)
        return float(np.average(objective(c,r,mode,normalizers),weights=vw)),float(np.average(c,weights=vw)),float(np.average(r,weights=vw))
    best,c0,r0=validate();initial_val=best;best_state=copy.deepcopy(model.state_dict());best_epoch=0;significant=best;stale=0;history=[];gradients={};rng=np.random.default_rng(seed);reason='budget_limit';start_epoch=1
    resume=folder/f'{family}_{mode}_lr{config["lr"]}_resume.pt'
    elapsed=0.
    if resume.exists():
        q=load_bundle(resume);model.load_state_dict(q['current']);best_state=q['best_state'];opt.load_state_dict(q['optimizer']);schedule.load_state_dict(q['scheduler']);rng.bit_generator.state=q['rng'];best=q['best'];best_epoch=q['best_epoch'];significant=q['significant'];stale=q['stale'];history=q['history'];gradients=q['gradients'];start_epoch=q['epoch']+1;elapsed=q['elapsed']
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for epoch in range(start_epoch,321):
        model.train();order=rng.permutation(len(x));total=np.zeros(3);count=0.;clipped=0;steps=0
        for begin in range(0,len(x),1024):
            ii=order[begin:begin+1024];out=model(xx[ii]);cl,rl=loss_parts(out,yy[ii],nn[ii]);loss=(objective(cl,rl,mode,normalizers)*ww[ii]).sum()/ww[ii].sum()
            opt.zero_grad();loss.backward();items=[(k,p) for k,p in model.named_parameters() if p.grad is not None]
            norms=torch.stack([p.grad.detach().norm() for _,p in items]).cpu().numpy()
            for (k,_),value in zip(items,norms):gradients[k]=max(gradients.get(k,0.),float(value))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);clipped+=bool(norm>5);opt.step();steps+=1
            amount=float(ww[ii].sum());total+=np.array([float(loss.detach()),float((cl.detach()*ww[ii]).sum()/ww[ii].sum()),float((rl.detach()*ww[ii]).sum()/ww[ii].sum())])*amount;count+=amount
        score,cv,rv=validate();schedule.step(score)
        if score<best-1e-8:best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if score<significant-1e-4:significant=score;stale=0
        else:stale+=1
        row={'epoch':epoch,'train_objective':total[0]/count,'train_BCE':total[1]/count,'train_MSE':total[2]/count,'validation_objective':score,'validation_BCE':cv,'validation_MSE':rv,'lr':opt.param_groups[0]['lr'],'clipped_fraction':clipped/max(steps,1)};history.append(row)
        dump(folder/'progress.json',{'family':family,'mode':mode,'seed':seed,**row})
        if epoch%32==0:save(resume,{'current':model.state_dict(),'best_state':best_state,'optimizer':opt.state_dict(),'scheduler':schedule.state_dict(),'rng':rng.bit_generator.state,'best':best,'best_epoch':best_epoch,'significant':significant,'stale':stale,'history':history,'gradients':gradients,'epoch':epoch,'elapsed':elapsed+time.perf_counter()-started})
        if epoch>=32 and stale>=24 and opt.param_groups[0]['lr']<=config['lr']/4:reason='validation_plateau';break
    model.load_state_dict(best_state);model.eval();changes={k:float((p-initial[k]).detach().norm()) for k,p in model.named_parameters()};cosine=None
    if mode=='joint':
        shared=[p for k,p in model.named_parameters() if (k.startswith('drive.') or k in ['h_real','h_imag'])] if family=='EPF' else list(model.readout[0].parameters())
        cl,rl=loss_parts(model(xx[:256]),yy[:256],nn[:256]);a=torch.autograd.grad(cl.mean(),shared,retain_graph=True);b=torch.autograd.grad(rl.mean(),shared)
        a=torch.cat([v.flatten() for v in a]);b=torch.cat([v.flatten() for v in b]);den=a.norm()*b.norm();cosine=float((a@b/den).detach()) if den>0 else None
    result={'family':family,'mode':mode,'seed':seed,'input_dim':x.shape[1],'config':config,'initial_validation_objective':initial_val,'initial_validation_BCE':c0,'initial_validation_MSE':r0,'validation_objective':best,'best_epoch':best_epoch,'epochs':epoch,'stop_reason':reason,'normalizers':normalizers.tolist(),'parameters':sum(p.numel() for p in model.parameters()),'fit_seconds':elapsed+time.perf_counter()-started,'peak_cuda_MB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,'history':history,'max_gradient_norms':gradients,'parameter_changes':changes,'shared_task_gradient_cosine':cosine,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
    model.cpu();return result
