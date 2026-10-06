"""EPF-centred 2020-2024 leave-one-year-out study with full development refits.

Historical v28/v30/v31 sources, data and results remain read-only. The same
corrected 15 inputs and five exclusive groups are used by every condition.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix

from .peer_comparison_v31 import (FEATURES, NUMERIC, CATEGORICAL, CLASS_NAMES,
    M16_TO_5, CONDITIONS, JOINT, SEEDS, PeerEncoder, peer_features, role,
    configurations, metrics, predict, sha_file, write_json)
from .v30_models import (Trained, Shared, build_shared, _optimizer, _ce,
    _joint_loss, JOINT_OPTIONS, MAX_EPOCHS, PATIENCE, CLIP_NORM)
from .v30_splits import Split, inner_split, leakage_check
from .v30_metrics import PSUBootstrap, FastAP, ci, bootstrap_p
from .clinical_v23_distribution import JointTargetTransform
from .event_field_v21 import MLPControlV21
from .event_field_v23 import V23JointModel, BODY_INDICES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/epf_year_v32_202024'
BASE = ROOT/'artifacts/peer_comparison_v31_202024'
YEARS = tuple(range(2020,2025))
THREADS = 2


def year_splits(frame):
    out=[];years=frame.survey_year.to_numpy(int)
    for year in YEARS:
        development=np.flatnonzero(years!=year);held=np.flatnonzero(years==year)
        fit,val=inner_split(frame,development,f'LOYO32_{year}')
        s=Split(f'LOYO_{year}','Y',fit,val,held,tuple(y for y in YEARS if y!=year),(year,))
        leakage_check(frame,s)
        assert not np.isin(years[np.r_[fit,val]],year).any()
        assert np.array_equal(np.sort(np.r_[fit,val]),development)
        for ix in (fit,val,held):
            if np.bincount(frame.group5.to_numpy(int)[ix],minlength=5).min()<3:
                raise ValueError(f'Insufficient class support for {year}')
        out.append(s)
    assert np.array_equal(np.sort(np.concatenate([s.heldout for s in out])),np.arange(len(frame)))
    return out


def prepare():
    if (OUT/'PROTOCOL.json').exists():return json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
    if not (BASE/'PROTOCOL.json').exists():
        # Fresh checkout: build the corrected cohort from licensed local SAVs.
        # No historical v31 model training is invoked.
        from .peer_comparison_v31 import prepare as prepare_base
        prepare_base()
    old=json.loads((BASE/'PROTOCOL.json').read_text(encoding='utf-8'))
    for path,digest in old['source_sha256'].items():
        assert sha_file(ROOT/path)==digest,f'Raw source changed: {path}'
    frame=pd.read_parquet(BASE/'private/cohort.parquet')
    raw=peer_features(frame);original=pd.read_parquet(BASE/'private/predictors.parquet')
    pd.testing.assert_frame_equal(raw,original)
    assert set(frame.survey_year)==set(YEARS) and len(frame)==4508
    assert np.array_equal(frame.group5.to_numpy(int),M16_TO_5[frame.state16.to_numpy(int)].argmax(1))
    splits=year_splits(frame)
    private=OUT/'private';private.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(private/'cohort.parquet');raw.to_parquet(private/'predictors.parquet')
    units=pd.read_parquet(BASE/'private/units.parquet');units.to_parquet(private/'units.parquet')
    joblib.dump(splits,private/'splits.joblib')
    record=[]
    for s in splits:
        roles={name:{'n':len(ix),'years':sorted(frame.survey_year.iloc[ix].unique().astype(int).tolist()),
                    'class_counts':np.bincount(frame.group5.iloc[ix],minlength=5).tolist()} for name,ix in s.roles().items()}
        record.append({'test_year':s.heldout_years[0],'roles':roles,'leakage':leakage_check(frame,s)})
    protocol={**{k:old[k] for k in ['age','features','classes','cohort','weight','source_sha256']},
       'study':'EPF-centred leave-one-year-out v32','years':list(YEARS),'conditions':list(CONDITIONS),
       'predictor_count':15,'encoded_columns':45,'split':'Each year held out once; other four years development',
       'inner_split':'20% of PSUs per development year for validation; no test-year rows',
       'tuning':'Eight settings per model, seed42, same weighted validation five-class macro AP',
       'final_refit':'All four development years; fit encoder/anchors again; selected tree count or neural epoch/LR schedule frozen before held-out predictions',
       'seeds':list(SEEDS),'primary_metric':'Arithmetic mean of five test-year weighted macro AP values',
       'primary_contrast':'jEPF minus LGBM, mean test-year macro AP','auxiliary_structure_contrast':'jEPF minus jMLP',
       'bootstrap':'2000 Rao-Wu within-year/stratum full-PSU draws; fixed fitted predictions, no refit',
       'shap':'EPF ensemble five-class probability output; permutation SHAP on 15 raw inputs; weighted development-only background; same-year held-out explanations',
       'splits':record,'code_sha256':{p:sha_file(ROOT/p) for p in ['metabolic/epf_year_v32.py','metabolic/peer_comparison_v31.py','metabolic/v30_models.py','metabolic/event_field_v21.py','metabolic/event_field_v23.py','metabolic/clinical_v23_distribution.py']},
       'limitations':['Retrospective internal year-group validation on previously analysed KNHANES; not prospective external validation',
          'Joint models use additional continuous measurement supervision; direct-vs-joint contrast is not a pure architecture effect',
          'Mean year AP and pooled OOF AP are different estimands','SHAP contributions are associations in predictions, not causal intervention effects',
          'Strict eligibility and nutrition participation produce n4508, not the previous n6588 cohort'],
       'base_protocol_sha256':sha_file(BASE/'PROTOCOL.json')}
    write_json(OUT/'PROTOCOL.json',protocol)
    print('PREPARED',len(frame),'five held-out years',flush=True)
    return protocol


def refit_shared(full, selected):
    """Refit anchor statistics on all development rows, with frozen hyperparameters."""
    s=Shared();c=selected.lr_anchor['C'];a=selected.ridge['alpha']
    lr=LogisticRegression(C=c,max_iter=3000,tol=1e-6).fit(full.x,full.group,sample_weight=full.w)
    s.lr_anchor={'C':c,'coef':lr.coef_.copy(),'bias':lr.intercept_.copy()}
    s.transform=JointTargetTransform().fit(full.numbers,full.raw_w)
    s.fit_z=s.transform.transform(full.numbers)
    ridge=Ridge(alpha=a).fit(full.x,s.fit_z,sample_weight=full.w)
    residual=s.fit_z-ridge.predict(full.x);residual-=np.average(residual,axis=0,weights=full.w)
    cov=(residual*full.w[:,None]).T@residual/full.w.sum();cov=(cov+cov.T)/2
    assert np.linalg.eigvalsh(cov).min()>0
    s.ridge={'alpha':a,'coef':ridge.coef_.copy(),'bias':ridge.intercept_.copy(),'residual_cov':cov}
    return s


def tensor_model(name,seed,fit,shared,device):
    torch.manual_seed(seed)
    if name=='MLP':
        model=MLPControlV21(fit.x.shape[1],5,hidden_layers=1,head_width=64,normalization='layer',
              dropout=.1,seed=seed,initialization='linear_anchor',use_linear_skip=True,numeric_embedding=False).to(device)
        model.set_linear(torch.as_tensor(shared.lr_anchor['coef'],dtype=torch.float32,device=device),
                         torch.as_tensor(shared.lr_anchor['bias'],dtype=torch.float32,device=device))
        model.initialize_on_fit(torch.as_tensor(fit.x,dtype=torch.float32,device=device))
        with torch.no_grad():model.gate_logits.fill_(math.log(.3/.7))
    else:
        family,options=JOINT_OPTIONS[name]
        model=V23JointModel(family,'bio',fit.x.shape[1],seed=seed,**options,dropout=.1,gate_init=.3).to(device)
        r=shared.ridge;model.initialize_on_fit(fit.x,r['coef'],r['bias'],r['residual_cov'],BODY_INDICES)
    return model


def train_tensor(name,config,seed,fit,val,shared,device,schedule=None):
    model=tensor_model(name,seed,fit,shared,device)
    x=torch.as_tensor(fit.x,dtype=torch.float32,device=device);w=torch.as_tensor(fit.w,dtype=torch.float64,device=device)
    target=torch.as_tensor(fit.group if name=='MLP' else shared.fit_z,device=device)
    if val is not None:
        vx=torch.as_tensor(val.x,dtype=torch.float32,device=device);vw=torch.as_tensor(val.w,dtype=torch.float64,device=device)
        vy=torch.as_tensor(val.group if name=='MLP' else shared.val_z,device=device)
    optimizer=_optimizer(model,config['lr'],config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=10)
    rng=np.random.default_rng(seed);best=math.inf;stale=0;best_state=None;best_epoch=0;trace=[];clipped=0;steps=0
    max_epoch=len(schedule) if schedule is not None else MAX_EPOCHS
    for epoch in range(1,max_epoch+1):
        if schedule is not None:
            for group,lr in zip(optimizer.param_groups,schedule[epoch-1],strict=True):group['lr']=lr
        trace.append([g['lr'] for g in optimizer.param_groups]);model.train()
        for ix in np.array_split(rng.permutation(len(x)),max(1,len(x)//config['batch'])):
            loss=_ce(model(x[ix]),target[ix],w[ix]) if name=='MLP' else _joint_loss(model.forward_parts(x[ix]),target[ix],w[ix])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM)
            clipped+=float(norm)>CLIP_NORM;steps+=1;optimizer.step()
        if val is None:continue
        model.eval()
        with torch.no_grad():
            if name=='MLP':score=float(_ce(model(vx),vy,vw))
            else:
                from .clinical_v23_distribution import gaussian_nll
                q=model.forward_parts(vx);score=float((gaussian_nll(q['mu'],q['diag_scale'],q['loading'],vy,reduction='none')*vw).mean())
        if not math.isfinite(score):raise FloatingPointError('Nonfinite validation loss')
        scheduler.step(score)
        if score<best-1e-4:
            best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=PATIENCE:break
    if val is not None:
        if best_state is None:raise RuntimeError('No validation checkpoint')
        model.load_state_dict(best_state)
    else:best_epoch=max_epoch
    model.eval()
    meta={'best_epoch':best_epoch,'epochs_run':epoch,'lr_schedule':trace[:best_epoch],
          'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'gradient_clip_fraction':clipped/max(steps,1),'optimizer_steps':steps,'training_rows':len(x)}
    if val is not None:meta['best_validation_loss']=best
    return Trained(name,config,seed,model,meta,shared.transform if name in JOINT else None)


def train(name,config,seed,fit,val,shared,final_meta=None):
    start=time.perf_counter();full=val is None
    if name=='LR':model=LogisticRegression(C=config['C'],max_iter=3000,tol=1e-6).fit(fit.x,fit.group,sample_weight=fit.w);meta={'parameters':model.coef_.size+model.intercept_.size}
    elif name=='RF':
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier(n_estimators=300,n_jobs=THREADS,random_state=seed,**config).fit(fit.x,fit.group,sample_weight=fit.w)
        meta={'parameters':sum(t.tree_.node_count for t in model.estimators_),'parameter_unit':'tree nodes'}
    elif name=='LGBM':
        import lightgbm as lgb
        model=lgb.LGBMClassifier(objective='multiclass',num_class=5,learning_rate=.03,
              n_estimators=final_meta['best_iteration'] if full else 2000,random_state=seed,n_jobs=THREADS,
              verbose=-1,subsample=.8,subsample_freq=1,colsample_bytree=.8,deterministic=True,force_row_wise=True,**config)
        options={} if full else {'eval_set':[(val.x,val.group)],'eval_sample_weight':[val.w],'callbacks':[lgb.early_stopping(100,verbose=False)]}
        model.fit(fit.x,fit.group,sample_weight=fit.w,**options)
        meta={'best_iteration':final_meta['best_iteration'] if full else int(model.best_iteration_),
              'parameters':sum(t['num_leaves'] for t in model.booster_.dump_model()['tree_info']),'parameter_unit':'tree leaves'}
    elif name=='XGB':
        from xgboost import XGBClassifier
        model=XGBClassifier(objective='multi:softprob',num_class=5,n_estimators=final_meta['best_iteration'] if full else 1500,
              learning_rate=.03,eval_metric='mlogloss',early_stopping_rounds=None if full else 80,random_state=seed,
              n_jobs=THREADS,subsample=.8,colsample_bytree=.8,tree_method='hist',**config)
        options={} if full else {'eval_set':[(val.x,val.group)],'sample_weight_eval_set':[val.w],'verbose':False}
        model.fit(fit.x,fit.group,sample_weight=fit.w,**options)
        meta={'best_iteration':final_meta['best_iteration'] if full else int(model.best_iteration)+1,
              'parameters':int(len(model.get_booster().trees_to_dataframe())),'parameter_unit':'tree nodes'}
    else:
        t=train_tensor(name,config,seed,fit,val,shared,torch.device('cpu'),None if not full else final_meta['lr_schedule'])
        t.meta.update(fit_seconds=time.perf_counter()-start,full_refit=full);return t
    meta.update(fit_seconds=time.perf_counter()-start,training_rows=len(fit.x),full_refit=full)
    return Trained(name,config,seed,model,meta)


def prepare_year(year):
    folder=OUT/'private'/str(year);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'context.joblib').exists():return
    f=pd.read_parquet(OUT/'private/cohort.parquet');raw=pd.read_parquet(OUT/'private/predictors.parquet')
    s=next(s for s in joblib.load(OUT/'private/splits.joblib') if s.heldout_years[0]==year)
    encoder=PeerEncoder().fit(raw.iloc[s.fit],f.wt_tot.to_numpy()[s.fit]);x=encoder.transform(raw)
    fit,val=role(f,x,s.fit),role(f,x,s.validation);shared=build_shared(CONDITIONS,fit,val)
    dev=np.sort(np.r_[s.fit,s.validation]);final_encoder=PeerEncoder().fit(raw.iloc[dev],f.wt_tot.to_numpy()[dev]);fx=final_encoder.transform(raw)
    full=role(f,fx,dev);final_shared=refit_shared(full,shared)
    joblib.dump({'split':s,'encoder':encoder,'final_encoder':final_encoder,'fit':fit,'val':val,
                 'shared':shared,'full':full,'final_shared':final_shared},folder/'context.joblib',compress=3)
    print('CONTEXT',year,'fit',len(fit.x),'val',len(val.x),'full',len(full.x),flush=True)


def run(year,name):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    job=OUT/'private'/str(year)/name;job.mkdir(parents=True,exist_ok=True)
    if (job/'DONE.json').exists():print('REUSED',year,name,flush=True);return
    ctx=joblib.load(job.parent/'context.joblib');fit,val,shared=ctx['fit'],ctx['val'],ctx['shared'];trials=[]
    for i,config in enumerate(configurations(name)):
        path=job/f'trial{i}.json';mp=job/f'trial{i}_seed42.joblib'
        if path.exists():
            result=json.loads(path.read_text(encoding='utf-8'));assert sha_file(mp)==result['checkpoint_sha256']
        else:
            t=train(name,config,42,fit,val,shared);p=predict(t,val.x,val.sex)
            joblib.dump(t,mp,compress=3)
            result={'config_id':i,'config':config,'validation':metrics(val.group,p,val.raw_w),'meta':t.meta,'checkpoint_sha256':sha_file(mp)}
            write_json(path,result);print('TRIAL',year,name,i,round(result['validation']['macro_ap'],5),flush=True)
        trials.append(result)
    best=max(trials,key=lambda r:(r['validation']['macro_ap'],-r['config_id']))
    freeze={'test_year':year,'condition':name,'config_id':best['config_id'],'config':best['config'],'test_used_for_selection':False,
            'inner_validation_macro_ap':best['validation']['macro_ap'],'development_years':[y for y in YEARS if y!=year]}
    seeds=(42,) if name=='LR' else SEEDS;selection_meta={}
    for seed in seeds:
        if seed==42:selection_meta[seed]=best['meta']
        elif name=='RF':selection_meta[seed]=best['meta']
        else:
            path=job/f'selected_inner_seed{seed}.joblib'
            if path.exists():t=joblib.load(path)
            else:t=train(name,best['config'],seed,fit,val,shared);joblib.dump(t,path,compress=3)
            selection_meta[seed]=t.meta
    freeze['selected_epoch_or_tree_count']={str(s):selection_meta[s].get('best_epoch',selection_meta[s].get('best_iteration')) for s in seeds}
    write_json(job/'SELECTION_FREEZE.json',freeze)
    models=[]
    for seed in seeds:
        path=job/f'final_seed{seed}.joblib'
        if path.exists():t=joblib.load(path)
        else:
            t=train(name,best['config'],seed,ctx['full'],None,ctx['final_shared'],selection_meta[seed]);joblib.dump(t,path,compress=3)
        assert t.meta['training_rows']==len(ctx['full'].x)
        models.append(t)
    # Held-out labels/probabilities are evaluated only after settings and all refits.
    f=pd.read_parquet(OUT/'private/cohort.parquet');raw=pd.read_parquet(OUT/'private/predictors.parquet')
    test=role(f,ctx['final_encoder'].transform(raw),ctx['split'].heldout)
    ps=[];times=[]
    for t in models:
        start=time.perf_counter();ps.append(predict(t,test.x,test.sex));times.append(time.perf_counter()-start)
    mean=np.mean(ps,axis=0);score=metrics(test.group,mean,test.raw_w)
    np.savez_compressed(job/'heldout.npz',index=test.index,p5=mean,seed_p5=np.stack(ps),seeds=seeds)
    parity=float(np.max(np.abs(predict(joblib.load(job/'final_seed42.joblib'),test.x[:8],test.sex[:8])-ps[0][:8])))
    assert parity<1e-10
    write_json(job/'DONE.json',{**freeze,'metrics':score,'seeds':list(seeds),'meta':[t.meta for t in models],
        'predict_seconds':times,'reload_error':parity,'prediction_sha256':sha_file(job/'heldout.npz'),
        'checkpoints':{f'final_seed{s}.joblib':sha_file(job/f'final_seed{s}.joblib') for s in seeds}})
    print('DONE',year,name,round(score['macro_ap'],6),flush=True)


def evaluate(replicates=2000):
    f=pd.read_parquet(OUT/'private/cohort.parquet');units=pd.read_parquet(OUT/'private/units.parquet')
    y=f.group5.to_numpy(int);w=f.wt_tot.to_numpy(float);years=f.survey_year.to_numpy(int)
    boot=PSUBootstrap(units,f,replicates,seed=20261006);weights=boot.weights(w)
    results={};draws={};predictions={};seed_scores={};costs={}
    for name in CONDITIONS:
        p=np.full((len(f),5),np.nan);seedp=np.full((3,len(f),5),np.nan);annual={};metas=[]
        for year in YEARS:
            job=OUT/'private'/str(year)/name;done=json.loads((job/'DONE.json').read_text(encoding='utf-8'))
            assert sha_file(job/'heldout.npz')==done['prediction_sha256']
            a=np.load(job/'heldout.npz');idx=a['index'];p[idx]=a['p5'];annual[str(year)]=done['metrics'];metas.extend(done['meta'])
            for s in range(3):seedp[s,idx]=a['seed_p5'][min(s,len(a['seed_p5'])-1)]
        pooled=metrics(y,p,w);mean=float(np.mean([annual[str(t)]['macro_ap'] for t in YEARS]));predictions[name]=p
        ap_by_year=[]
        for year in YEARS:
            ix=np.flatnonzero(years==year);fs=[FastAP((y[ix]==k).astype(int),p[ix,k]) for k in range(5)]
            ap_by_year.append(np.array([np.mean([a(bw[ix]) for a in fs]) for bw in weights]))
        d=np.mean(ap_by_year,axis=0);draws[name]=d
        cm=confusion_matrix(y,p.argmax(1),labels=np.arange(5),sample_weight=w);cm/=cm.sum(1,keepdims=True)
        results[name]={'mean_year_macro_ap':mean,'mean_year_macro_ap_ci95':ci(d),'annual':annual,'pooled_oof':pooled,
                       'weighted_normalized_confusion':cm.tolist(),'mean_year_macro_auc':float(np.mean([annual[str(t)]['macro_auc'] for t in YEARS]))}
        seed_scores[name]=[float(np.mean([metrics(y[years==t],sp[years==t],w[years==t])['macro_ap'] for t in YEARS])) for sp in seedp]
        costs[name]={'parameter_median':float(np.median([m['parameters'] for m in metas])),
                     'parameter_unit':metas[0].get('parameter_unit','trainable parameters'),
                     'refit_seconds_median':float(np.median([m['fit_seconds'] for m in metas]))}
        print('EVALUATED',name,round(mean,6),flush=True)
    contrasts={}
    for name in CONDITIONS:
        if name=='jEPF':continue
        delta=draws['jEPF']-draws[name]
        contrasts[name]={'epf_minus_other':results['jEPF']['mean_year_macro_ap']-results[name]['mean_year_macro_ap'],
                        'ci95':ci(delta),'p_boot':bootstrap_p(delta),'years_epf_higher':int(sum(results['jEPF']['annual'][str(t)]['macro_ap']>results[name]['annual'][str(t)]['macro_ap'] for t in YEARS))}
    np.savez_compressed(OUT/'private/oof_predictions.npz',**predictions)
    report={'protocol':json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8')),'results':results,
            'epf_contrasts':contrasts,'seed_mean_year_macro_ap':seed_scores,'costs':costs,
            'bootstrap':{'replicates':replicates,'units':boot.n_psu,'strata':boot.n_strata} if hasattr(boot,'n_psu') else {'replicates':replicates},
            'interpretation':'Primary mean test-year five-group AP; fixed-prediction PSU uncertainty; no pre-declared superiority conclusion'}
    write_json(OUT/'REPORT.json',report)
    rows=[{'model':name,'mean_year_macro_ap':v['mean_year_macro_ap'],'ci_low':v['mean_year_macro_ap_ci95'][0],
           'ci_high':v['mean_year_macro_ap_ci95'][1],'pooled_macro_ap':v['pooled_oof']['macro_ap'],
           'mean_year_macro_auc':v['mean_year_macro_auc'],'pooled_macro_f1':v['pooled_oof']['macro_f1']} for name,v in results.items()]
    pd.DataFrame(rows).to_csv(OUT/'MODEL_COMPARISON.csv',index=False)
    pd.DataFrame([{'model':n,'year':int(t),**{k:v[k] for k in ['macro_ap','macro_auc','macro_f1','nll','brier']}} for n,r in results.items() for t,v in r['annual'].items()]).to_csv(OUT/'YEAR_COMPARISON.csv',index=False)
    write_json(OUT/'EVALUATION_DONE.json',{'report_sha256':sha_file(OUT/'REPORT.json'),'jobs':40,'heldout_once':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','context','train','evaluate']);p.add_argument('--year',type=int);p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--replicates',type=int,default=2000)
    a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='context':
        for year in YEARS:prepare_year(year)
    elif a.command=='train':run(a.year,a.condition)
    else:evaluate(a.replicates)


if __name__=='__main__':main()
