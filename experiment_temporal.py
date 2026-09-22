"""Reproduce the aligned v17 synthetic experiment locally; no API or health data.

This runner uses the executed equations, selection and training loop. Test data
are generated after model selection. Exact bitwise results can depend on device
and PyTorch version. Default output is ignored by Git.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from models.event_field_runtime import FastEventPlasticField
from models.event_field_baselines import MatchedResidualMLP,HistoryGRU

FAMILIES=['field','MLP','field_no_plasticity','field_no_quartic']
CONFIGS=[{'lr':.001,'weight_decay':.0001},{'lr':.003,'weight_decay':.0001}]
OUT=Path('runs/event_field_temporal')
def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def flatten(x):return np.asarray(x[:,64:]).reshape(-1,x.shape[-1])
def load_bundle(path):return torch.load(path,map_location='cpu',weights_only=False)
def make_streams(seed,n,length=512):
    u=np.random.default_rng(seed).uniform(-1,1,(n,length,1));y=np.zeros((n,length,5))
    for j,delay in enumerate([1,4,16,32]):y[:,delay:,j]=u[:,:-delay,0]
    for t in range(32,length):y[:,t,4]=u[:,t-3,0]*u[:,t-12,0]+u[:,t-1,0]*u[:,t-32,0]
    return u,y


def lag_windows(u,width=64):
    padded=np.pad(u[:,:,0],((0,0),(width-1,0)))
    return np.lib.stride_tricks.sliding_window_view(padded,width,axis=1).copy()


def build(family,input_dim,output_dim,seed):
    reference=FastEventPlasticField(input_dim,output_dim,seed=seed)
    count=sum(p.numel() for p in reference.parameters())
    if family=='MLP':return MatchedResidualMLP(input_dim,output_dim,count,seed)
    if family=='GRU_attention':return HistoryGRU(input_dim,output_dim,count,seed)
    if family=='field_no_plasticity':return FastEventPlasticField(input_dim,output_dim,seed=seed,plasticity=False)
    if family=='field_no_quartic':return FastEventPlasticField(input_dim,output_dim,seed=seed,quartic=False)
    return reference


def numpy_predict(model,x,batch=2048):
    device=next(model.parameters()).device;model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):values.append(model(torch.tensor(x[start:start+batch],dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(values)


def fit(family,config,seed,x,y,w,vx,vy,vw,coef,bias,task,folder):
    if task!='temporal':raise ValueError('This corrected runner is for temporal comparison only')
    device='cuda' if torch.cuda.is_available() else 'cpu';model=build(family,x.shape[1],y.shape[1],seed).to(device);model.set_linear(coef,bias)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    xx,yy,ww=[torch.tensor(a,dtype=torch.float32,device=device) for a in [x,y,w]]
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    # Both the scheduler and patience now use an absolute standardized-MSE
    # tolerance, avoiding endless relative improvements near a zero noise floor.
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=7,threshold=1e-5,threshold_mode='abs',min_lr=1e-5)
    rng=np.random.default_rng(seed);history=[];gradients={};gradient_steps={};initial={k:v.detach().clone() for k,v in model.named_parameters()}
    def validation():
        p=numpy_predict(model,vx);return float(np.average(np.mean((p-vy)**2,axis=1),weights=vw))
    best=validation();initial_score=best;best_state=copy.deepcopy(model.state_dict());best_epoch=0;significant_best=best;stale=0;reason='budget_limit';start_epoch=1
    resume=folder/f'{family}_lr{config["lr"]}_resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False);model.load_state_dict(saved['current']);best_state=saved['best'];optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);rng.bit_generator.state=saved['rng']
        best=saved['best_loss'];best_epoch=saved['best_epoch'];significant_best=saved['significant_best'];stale=saved['stale'];history=saved['history'];gradients=saved['gradients'];gradient_steps=saved['gradient_steps'];start_epoch=saved['epoch']+1
    started=time.perf_counter();parameters=list(model.named_parameters())
    for epoch in range(start_epoch,321):
        model.train();total=0.;normalizer=0.;clip_count=0;steps=0;order=rng.permutation(len(x))
        for begin in range(0,len(x),1024):
            ii=order[begin:begin+1024];output=model(xx[ii]);row=(output-yy[ii]).square().mean(1);loss=(row*ww[ii]).sum()/ww[ii].sum();optimizer.zero_grad();loss.backward()
            eligible=[(n,p) for n,p in parameters if p.grad is not None]
            values=torch.stack([p.grad.detach().norm() for _,p in eligible]).cpu().numpy()
            for (name,_),value in zip(eligible,values):
                gradients[name]=max(gradients.get(name,0.),float(value))
                if value>1e-10:gradient_steps[name]=gradient_steps.get(name,0)+1
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);clip_count+=bool(norm>5);optimizer.step();steps+=1
            amount=float(ww[ii].sum());total+=float(loss.detach())*amount;normalizer+=amount
        score=validation();scheduler.step(score)
        if score<best-1e-8:best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict())
        if score<significant_best-1e-5:significant_best=score;stale=0
        else:stale+=1
        row={'epoch':epoch,'train_loss':total/normalizer,'validation_loss':score,'lr':optimizer.param_groups[0]['lr'],'clipped_fraction':clip_count/max(steps,1),'best_epoch':best_epoch};history.append(row)
        dump(folder/'progress.json',{'family':family,'seed':seed,'config':config,**row})
        if epoch%16==0:
            temporary=Path(str(resume)+'.tmp');torch.save({'current':model.state_dict(),'best':best_state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng.bit_generator.state,'best_loss':best,'best_epoch':best_epoch,'significant_best':significant_best,'stale':stale,'history':history,'gradients':gradients,'gradient_steps':gradient_steps,'epoch':epoch},temporary);os.replace(temporary,resume)
        if epoch>=32 and stale>=24 and optimizer.param_groups[0]['lr']<=config['lr']/4:reason='validation_plateau_after_lr_reductions';break
    model.load_state_dict(best_state);model.eval();changes={k:float((p-initial[k]).detach().norm()) for k,p in model.named_parameters()}
    result={'family':family,'seed':seed,'input_dim':x.shape[1],'output_dim':y.shape[1],'config':config,'initial_validation_loss':initial_score,'validation_loss':best,'best_epoch':best_epoch,'epochs':epoch,'stop_reason':reason,'history':history,'max_gradient_norms':gradients,'nonzero_gradient_steps':gradient_steps,'selected_parameter_changes':changes,'parameters':sum(p.numel() for p in model.parameters()),'fit_seconds':time.perf_counter()-started,'device':device,'peak_cuda_MB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,'runtime':'grouped solve_ex status checks; absolute plateau tolerance1e-5'}
    if family.startswith('field'):
        with torch.inference_mode():_,diag=model(xx[:256],True)
        result['field_diagnostics']={k:float(v) for k,v in diag.items() if k!='field_features'}
    result['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()};model.cpu();return result


def restore(bundle,device='cpu'):
    model=build(bundle['family'],bundle['input_dim'],bundle['output_dim'],bundle['seed']);model.load_state_dict(bundle['state_dict']);model.to(device);model.eval();return model


def select_train(family,seed,x,y,w,vx,vy,vw,coef,bias,task,folder,forced_config=None):
    path=folder/f'{family}.pt'
    if path.exists():return load_bundle(path)
    candidates=[]
    for i,config in enumerate(CONFIGS if forced_config is None else [forced_config]):
        candidate_path=folder/f'{family}_trial{i}.pt'
        if candidate_path.exists():b=load_bundle(candidate_path)
        else:
            b=fit(family,config,seed,x,y,w,vx,vy,vw,coef,bias,task,folder);torch.save(b,candidate_path)
        candidates.append(b)
    best=min(candidates,key=lambda b:b['validation_loss']);torch.save(best,path)
    dump(folder/f'{family}_trials.json',[{k:v for k,v in b.items() if k not in ['state_dict','history','max_gradient_norms','selected_parameter_changes','nonzero_gradient_steps']} for b in candidates])
    return best


def temporal(seed):
    folder=OUT/'temporal'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():return
    u,y=make_streams(271828,100);vu,vy=make_streams(314159,20);x=flatten(lag_windows(u));vx=flatten(lag_windows(vu));y=flatten(y);vy=flatten(vy)
    mean=x.mean(0);scale=x.std(0);x=(x-mean)/scale;vx=(vx-mean)/scale;ym=y.mean(0);ys=y.std(0);y=(y-ym)/ys;vy=(vy-ym)/ys
    linear=Ridge(alpha=.1).fit(x,y);selected={}
    for family in FAMILIES+['GRU_attention']:
        forced=selected['field']['config'] if family in ['field_no_plasticity','field_no_quartic'] else None
        selected[family]=select_train(family,seed,x,y,np.ones(len(x)),vx,vy,np.ones(len(vx)),linear.coef_,linear.intercept_,'temporal',folder,forced)
    dump(folder/'FROZEN_SELECTION.json',{k:{a:b[a] for a in ['config','initial_validation_loss','validation_loss','best_epoch','epochs','stop_reason','parameters']} for k,b in selected.items()})
    # New test streams, generated after selection; old v16 test data are not reused.
    tu,ty=make_streams(161804,20);tx=(flatten(lag_windows(tu))-mean)/scale;truth=flatten(ty);predictions={};results={}
    for family,b in selected.items():
        model=restore(b,'cuda' if torch.cuda.is_available() else 'cpu');pred=numpy_predict(model,tx)*ys+ym;model.cpu()
        predictions[family]=pred.reshape(20,448,5);nmse=np.mean((pred-truth)**2,0)/np.var(truth,0)
        results[family]={'NMSE':nmse.tolist(),'parameters':b['parameters'],'initial_validation_loss':b['initial_validation_loss'],'validation_loss':b['validation_loss'],'epochs':b['epochs'],'best_epoch':b['best_epoch'],'stop_reason':b['stop_reason'],'fit_seconds':b['fit_seconds'],'peak_cuda_MB':b['peak_cuda_MB']}
    np.savez_compressed(folder/'private_predictions.npz',**predictions);np.savez(folder/'normalization.npz',x_mean=mean,x_scale=scale,y_mean=ym,y_scale=ys)
    dump(folder/'DONE.json',{'task':'temporal','seed':seed,'models':results,'test_seed':161804});print('New field temporal',seed,'completed',flush=True)

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44]);parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=Path(__file__).resolve().parent
    files=['experiment_temporal.py','models/event_plastic_field.py','models/event_field_baselines.py','models/event_field_runtime.py']
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files};freeze=OUT/'TRAINING_SOURCES.json'
    if freeze.exists() and json.loads(freeze.read_text())!=hashes:raise RuntimeError('Code changed: use a separate output directory')
    dump(freeze,hashes)
    with threadpool_limits(limits=1):
        for seed in [args.seed] if args.seed is not None else [42,43,44]:temporal(seed)

if __name__=='__main__':main()
