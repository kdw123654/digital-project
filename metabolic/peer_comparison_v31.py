"""Model-only, corrected peer-input benchmark. Historical studies are read-only.

The peer notebook selects 29 source columns, but uses 15 predictors. Its five
classes count glucose, BP, and the union of TG/low-HDL as three risk domains.
EPF retains the existing V23 joint model; joint MLP/Linear control the output
design. No variable-importance or factor-declaration study is run here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, balanced_accuracy_score

from .clinical_v23_distribution import joint_probabilities
from .event_field_v21 import MLPControlV21
from .event_field_v23 import V23JointModel, BODY_INDICES
from .v30_data import build_cohort, discover_years, BITS16, TARGETS5
from .v30_models import (Role, build_shared, _train_loop, _optimizer, _ce,
                         _train_torch_joint, Trained, configs as old_configs)
from .v30_metrics import PSUBootstrap, FastAP, ci, bootstrap_p
from .v30_splits import period_splits, leakage_check

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'peer_comparison_v31_202024'
CONDITIONS = ('LR', 'RF', 'XGB', 'LGBM', 'MLP', 'jEPF', 'jMLP', 'jLinear')
JOINT = ('jEPF', 'jMLP', 'jLinear')
SEEDS = (42, 43, 44)
CLASS_NAMES = ('Normal', 'High_BP', 'High_Glu', 'Dyslipid', 'Complex')
NUMERIC = ('age', 'HE_BMI', 'HE_wc', 'WHtR', 'sedentary_hours')
CATEGORICAL = {'sex': (1, 2), 'incm': (1, 2, 3, 4), 'edu': (1, 2, 3, 4),
               'sm_presnt': (0, 1), 'dr_month': (0, 1), 'pa_aerobic': (0, 1),
               'living_alone': (0, 1), 'solo_dinner': (0, 1, 2),
               'weight_gain': (0, 1), 'no_brush_bed': (0, 1)}
FEATURES = tuple(NUMERIC) + tuple(CATEGORICAL)
M16_TO_5 = np.zeros((16, 5))
for _s, (_g, _b, _t, _h) in enumerate(BITS16):
    _l = int(bool(_t or _h)); _n = int(_g + _b + _l)
    _k = 0 if _n == 0 else (4 if _n >= 2 else (1 if _b else (2 if _g else 3)))
    M16_TO_5[_s, _k] = 1.


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def peer_features(frame):
    """Correct the two reversed peer encodings; unknowns stay missing."""
    out = pd.DataFrame(index=frame.index)
    for col in ('age', 'HE_BMI', 'HE_wc', 'sex', 'incm', 'edu',
                'sm_presnt', 'dr_month', 'pa_aerobic'):
        out[col] = pd.to_numeric(frame[col], errors='coerce')
    out['WHtR'] = frame.HE_wc / frame.HE_ht.where(frame.HE_ht > 0)
    for col in ('HE_BMI', 'HE_wc', 'WHtR'):
        out[col] = out[col].where(out[col] > 0)
    out['living_alone'] = frame.cfam.map({1: 1., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0.})
    out['sedentary_hours'] = frame.BE8_1.where(frame.BE8_1.between(0, 24))
    # L_DN_TO: 1=together, 2=alone, 3=not applicable (<=2 dinners/week).
    out['solo_dinner'] = frame.L_DN_TO.map({1: 0., 2: 1., 3: 2.})
    # BO1_1: 1=unchanged, 2=decreased, 3=increased.
    out['weight_gain'] = frame.BO1_1.map({1: 0., 2: 0., 3: 1.})
    out['no_brush_bed'] = frame.BM1_8.map({0: 1., 1: 0.})
    for col, levels in CATEGORICAL.items():
        out[col] = out[col].where(out[col].isin(levels))
    return out[list(FEATURES)].replace([np.inf, -np.inf], np.nan)


class PeerEncoder:
    """Fit-only weighted centering; preserves the existing EPF body positions."""
    def fit(self, frame, weights):
        a = frame[list(NUMERIC)].to_numpy(float)
        self.mean, self.scale = [], []
        for j in range(a.shape[1]):
            ok = np.isfinite(a[:, j])
            if not ok.any(): raise ValueError(f'No fit observations: {NUMERIC[j]}')
            m = float(np.average(a[ok, j], weights=weights[ok]))
            s = float(np.sqrt(np.average((a[ok, j] - m) ** 2, weights=weights[ok])))
            self.mean.append(m); self.scale.append(max(s, 1e-4))
        self.mean, self.scale = np.array(self.mean), np.array(self.scale)
        self.columns = list(NUMERIC[:4]) + [f'{c}:missing' for c in NUMERIC[:4]] + ['sex==2']
        self.columns += ['sex==1', 'sex:missing', 'sedentary_hours', 'sedentary_hours:missing']
        for col, levels in CATEGORICAL.items():
            if col == 'sex': continue
            self.columns += [f'{col}=={v}' for v in levels] + [f'{col}:missing']
        assert tuple(self.columns[i] for i in BODY_INDICES) == ('age', 'HE_BMI', 'HE_wc', 'WHtR', 'sex==2')
        return self

    def transform(self, frame):
        a = frame[list(NUMERIC)].to_numpy(float); missing = ~np.isfinite(a)
        z = np.where(missing, 0., (a - self.mean) / self.scale)
        sex = frame.sex.to_numpy(float)
        columns = [z[:, :4], missing[:, :4], (sex == 2)[:, None],
                   (sex == 1)[:, None], (~np.isfinite(sex))[:, None], z[:, 4:5], missing[:, 4:5]]
        for col, levels in CATEGORICAL.items():
            if col == 'sex': continue
            v = frame[col].to_numpy(float)
            columns += [np.column_stack([v == level for level in levels]), (~np.isfinite(v))[:, None]]
        x = np.concatenate(columns, axis=1).astype(np.float32)
        if x.shape[1] != len(self.columns) or not np.isfinite(x).all(): raise ValueError('Invalid encoded predictors')
        return x


def metrics(y, p, w):
    if p.shape != (len(y), 5) or not np.isfinite(p).all() or (p < -1e-10).any() or not np.allclose(p.sum(1), 1, atol=1e-7):
        raise ValueError('Invalid five-class probabilities')
    onehot = np.eye(5)[y]; rows = []
    for k, name in enumerate(CLASS_NAMES):
        target = y == k
        if target.sum() == 0 or target.sum() == len(y): raise ValueError(f'Unsupported class {name}')
        rows.append({'class': name, 'n': int(target.sum()), 'prevalence': float(np.average(target, weights=w)),
                     'ap': float(average_precision_score(target, p[:, k], sample_weight=w)),
                     'auc': float(roc_auc_score(target, p[:, k], sample_weight=w))})
    pred = p.argmax(1)
    return {'macro_ap': float(np.mean([r['ap'] for r in rows])), 'macro_auc': float(np.mean([r['auc'] for r in rows])),
            'macro_f1': float(f1_score(y, pred, average='macro', sample_weight=w)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred, sample_weight=w)),
            'nll': float(np.average(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)), weights=w)),
            'brier': float(np.average(((onehot - p) ** 2).sum(1), weights=w)), 'classes': rows}


def prepare():
    if (OUT / 'PROTOCOL.json').exists(): return json.loads((OUT / 'PROTOCOL.json').read_text(encoding='utf-8'))
    sources = {y: p for y, p in discover_years().items() if y in range(2020, 2025) and p is not None}
    if len(sources) != 5: raise ValueError('All five 2020-2024 SAV files are required')
    needed = ['HE_ht', 'HE_BMI', 'HE_wc', 'cfam', 'incm', 'edu', 'sm_presnt', 'dr_month', 'pa_aerobic',
              'BE8_1', 'L_DN_TO', 'BO1_1', 'BM1_8']
    cohort, flow, units = build_cohort(sources, needed)
    n_original = len(cohort)
    cohort = cohort[cohort.age.between(20, 39)].copy()
    n_young = len(cohort)
    # The dinner-companionship question belongs to the nutrition survey.
    cohort = cohort[cohort.wt_tot.gt(0)].reset_index(drop=True)
    cohort['group5'] = M16_TO_5[cohort.state16.to_numpy(int)].argmax(1)
    raw = peer_features(cohort)
    splits, _ = period_splits(cohort, list(range(2020, 2025)))
    splits = [s for s in splits if s.family == 'P']
    if len(splits) != 5: raise ValueError('Expected five grouped outer folds')
    for s in splits:
        leakage_check(cohort, s)
        for indices in (s.fit, s.validation, s.heldout):
            if np.bincount(cohort.group5.to_numpy()[indices], minlength=5).min() < 3:
                raise ValueError('Insufficient five-class support in a role')
    private = OUT / 'private'; private.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(private / 'cohort.parquet'); raw.to_parquet(private / 'predictors.parquet')
    units.to_parquet(private / 'units.parquet'); joblib.dump(splits, private / 'splits.joblib')
    protocol = {'study': 'model-only peer-aligned comparison v31', 'years': list(range(2020, 2025)),
        'age': [20, 39], 'source_column_count_in_peer': 29, 'predictor_count': len(FEATURES), 'features': list(FEATURES),
        'classes': list(CLASS_NAMES), 'state_to_class5': M16_TO_5.argmax(1).tolist(), 'conditions': list(CONDITIONS),
        'weight': 'wt_tot; nutrition participants only; mean-one weights for training',
        'cohort': {'strict_19_39': n_original, 'strict_20_39': n_young, 'nutrition_20_39': len(cohort),
                   'by_year': {str(y): int(n) for y,n in cohort.groupby('survey_year').size().items()},
                   'by_class': {name: int((cohort.group5 == k).sum()) for k,name in enumerate(CLASS_NAMES)}},
        'split': '5 year-balanced PSU-group outer folds; inner PSU fit/validation; same people for all models',
        'tuning': '8 settings per condition, selection seed 42; select by weighted validation macro AP',
        'final_seeds': list(SEEDS), 'primary_contrast': 'jEPF minus jMLP weighted five-class macro AP',
        'checkpoint_selection': 'direct MLP CE; joint Gaussian NLL; tree validation logloss early stopping',
        'limitations': ['Internal post-hoc comparison on previously examined KNHANES; not independent external validation',
                       'Joint models additionally learn continuous measurements; direct-vs-joint differences are not pure architecture effects',
                       'Single-seed tuning; three final seeds; fixed-prediction bootstrap does not include refitting uncertainty',
                       'Corrected feature coding and strict unaware/fasting/pregnancy cohort differ from peer notebook',
                       'No factor discovery, SHAP, causal or clinical-benefit claims'],
        'source_sha256': {str(p.relative_to(ROOT)): sha_file(p) for p in sources.values()},
        'code_sha256': sha_file(__file__), 'peer_commit': '56728e7d27e76cc901bdc17f41d2fefd4a0b3ebd'}
    write_json(OUT / 'PROTOCOL.json', protocol)
    flow.to_csv(private / 'cohort_flow.csv', index=False, encoding='utf-8-sig')
    print('PREPARED', json.dumps(protocol['cohort']), flush=True)
    return protocol


def role(frame, x, indices):
    w = frame.wt_tot.to_numpy(float)[indices]
    return Role(x[indices], frame.group5.to_numpy(int)[indices], w/w.mean(), w,
                frame[list(TARGETS5)].to_numpy(float)[indices], frame.sex.to_numpy(int)[indices],
                frame.state16.to_numpy(int)[indices], frame[['elevated_glucose','elevated_bp','elevated_tg','low_hdl']].to_numpy(int)[indices], indices)


def configurations(condition):
    if condition == 'XGB':
        return [{'max_depth': d, 'min_child_weight': w, 'reg_lambda': l}
                for d in (2, 4) for w in (10, 30) for l in (1., 10.)]
    return old_configs(condition)


def train_one(condition, config, seed, fit, val, shared, device):
    started = time.perf_counter(); meta = {}
    if condition == 'LR':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=config['C'], max_iter=3000, tol=1e-6)
        model.fit(fit.x, fit.group, sample_weight=fit.w)
        meta['parameters'] = int(model.coef_.size + model.intercept_.size)
    elif condition == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, n_jobs=6, random_state=seed, **config)
        model.fit(fit.x, fit.group, sample_weight=fit.w)
        meta.update(parameters=int(sum(t.tree_.node_count for t in model.estimators_)), parameter_unit='tree nodes')
    elif condition == 'LGBM':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(objective='multiclass', num_class=5, learning_rate=.03, n_estimators=2000,
             random_state=seed, n_jobs=6, verbose=-1, subsample=.8, subsample_freq=1, colsample_bytree=.8,
             deterministic=True, force_row_wise=True, **config)
        model.fit(fit.x, fit.group, sample_weight=fit.w, eval_set=[(val.x,val.group)], eval_sample_weight=[val.w],
                  callbacks=[lgb.early_stopping(100,verbose=False)])
        meta.update(best_iteration=int(model.best_iteration_), parameters=int(sum(t['num_leaves'] for t in model.booster_.dump_model()['tree_info'])), parameter_unit='tree leaves')
    elif condition == 'XGB':
        from xgboost import XGBClassifier
        model = XGBClassifier(objective='multi:softprob', num_class=5, n_estimators=1500, learning_rate=.03,
            eval_metric='mlogloss', early_stopping_rounds=80, random_state=seed, n_jobs=6,
            subsample=.8, colsample_bytree=.8, tree_method='hist', **config)
        model.fit(fit.x, fit.group, sample_weight=fit.w, eval_set=[(val.x,val.group)], sample_weight_eval_set=[val.w], verbose=False)
        meta.update(best_iteration=int(model.best_iteration), parameters=int(model.get_booster().trees_to_dataframe().shape[0]), parameter_unit='tree nodes')
    elif condition == 'MLP':
        torch.manual_seed(seed)
        model = MLPControlV21(fit.x.shape[1], 5, hidden_layers=1, head_width=64, normalization='layer', dropout=.1,
            seed=seed, initialization='linear_anchor', use_linear_skip=True, residual_gate='trainable').to(device)
        with torch.no_grad(): model.gate_logits.fill_(math.log(.3/.7))
        model.set_linear(torch.as_tensor(shared.lr_anchor['coef'],dtype=torch.float32),torch.as_tensor(shared.lr_anchor['bias'],dtype=torch.float32))
        model.initialize_on_fit(torch.as_tensor(fit.x,dtype=torch.float32,device=device))
        x=torch.as_tensor(fit.x,device=device); y=torch.as_tensor(fit.group,device=device); w=torch.as_tensor(fit.w,device=device)
        vx=torch.as_tensor(val.x,device=device); vy=torch.as_tensor(val.group,device=device); vw=torch.as_tensor(val.w,device=device)
        meta=_train_loop(model,config,seed,len(x),lambda ix:_ce(model(x[ix]),y[ix],w[ix]),lambda:_ce(model(vx),vy,vw))
    else:
        model,meta=_train_torch_joint(condition,config,seed,fit,val,shared,device)
    meta['fit_seconds']=time.perf_counter()-started
    return Trained(condition,config,seed,model,meta,shared.transform if condition in JOINT else None)


def predict(trained, x, sex):
    if trained.condition in JOINT:
        out=trained.predict(x,sex,audited=True)
        return out['p16'] @ M16_TO_5
    if trained.condition != 'MLP': return trained.model.predict_proba(x)
    trained.model.eval()
    with torch.no_grad():
        dev=next(trained.model.parameters()).device
        return torch.softmax(trained.model(torch.as_tensor(x,device=dev)).double(),dim=1).cpu().numpy()


def save_model(trained,path):
    joblib.dump(trained,path,compress=3)


def run(condition=None,fold=None):
    prepare(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    private=OUT/'private'; frame=pd.read_parquet(private/'cohort.parquet'); raw=pd.read_parquet(private/'predictors.parquet')
    splits=joblib.load(private/'splits.joblib'); device=torch.device('cpu')
    for fold_id,split in enumerate(splits,1):
        if fold is not None and fold_id != fold: continue
        folder=private/f'fold{fold_id}'; folder.mkdir(parents=True,exist_ok=True)
        enc_path=folder/'encoder.joblib'
        if enc_path.exists(): encoder=joblib.load(enc_path)
        else:
            encoder=PeerEncoder().fit(raw.iloc[split.fit],frame.wt_tot.to_numpy()[split.fit]); joblib.dump(encoder,enc_path)
        x=encoder.transform(raw); fit=role(frame,x,split.fit); val=role(frame,x,split.validation)
        shared_path=folder/'shared.joblib'
        if shared_path.exists(): shared=joblib.load(shared_path)
        else:
            shared=build_shared(['MLP','jEPF','jMLP','jLinear'],fit,val); joblib.dump(shared,shared_path)
        for name in CONDITIONS:
            if condition is not None and name != condition: continue
            job=folder/name; job.mkdir(parents=True,exist_ok=True)
            done=job/'DONE.json'
            if done.exists():
                record=json.loads(done.read_text(encoding='utf-8'))
                if sha_file(job/'heldout.npz') != record['prediction_sha256']: raise RuntimeError('Resume prediction hash mismatch')
                print('REUSED',fold_id,name,flush=True); continue
            trials=[]
            for config_id,config in enumerate(configurations(name)):
                trial=job/f'config{config_id}.json'; model_path=job/f'config{config_id}_seed42.joblib'
                if trial.exists():
                    result=json.loads(trial.read_text(encoding='utf-8'))
                    if sha_file(model_path)!=result['checkpoint_sha256']: raise RuntimeError('Resume model hash mismatch')
                else:
                    trained=train_one(name,config,42,fit,val,shared,device)
                    p=predict(trained,val.x,val.sex)
                    result={'config_id':config_id,'config':config,'validation':metrics(val.group,p,val.raw_w),'meta':trained.meta}
                    save_model(trained,model_path); result['checkpoint_sha256']=sha_file(model_path); write_json(trial,result)
                    print('TRIAL',fold_id,name,config_id,round(result['validation']['macro_ap'],5),flush=True)
                trials.append(result)
            chosen=max(trials,key=lambda r:(r['validation']['macro_ap'],-r['config_id']))
            freeze={'condition':name,'fold':fold_id,'selected_config':chosen['config_id'],'config':chosen['config'],
                    'selection_validation_macro_ap':chosen['validation']['macro_ap'],'test_used_for_selection':False}
            write_json(job/'SELECTION_FREEZE.json',freeze)
            seeds=(42,) if name=='LR' else SEEDS
            models=[]
            for seed in seeds:
                path=job/f'config{chosen["config_id"]}_seed{seed}.joblib'
                if path.exists(): trained=joblib.load(path)
                else:
                    trained=train_one(name,chosen['config'],seed,fit,val,shared,device); save_model(trained,path)
                models.append(trained)
            # The held-out role is opened only after configuration freeze and final fits.
            test=role(frame,x,split.heldout); predictions=[]; times=[]
            for trained in models:
                started=time.perf_counter(); p=predict(trained,test.x,test.sex); times.append(time.perf_counter()-started)
                metrics(test.group,p,test.raw_w); predictions.append(p)
            mean=np.mean(predictions,axis=0)
            np.savez_compressed(job/'heldout.npz',index=test.index,p5=mean,seed_p5=np.stack(predictions),seeds=seeds)
            # Verify saved model prediction parity on a fixed small slice.
            loaded=joblib.load(job/f'config{chosen["config_id"]}_seed42.joblib')
            parity=float(np.abs(predict(loaded,test.x[:8],test.sex[:8])-predictions[0][:8]).max())
            if parity>1e-10: raise RuntimeError(f'Reload prediction mismatch {parity}')
            write_json(done,{**freeze,'metrics':metrics(test.group,mean,test.raw_w),'seeds':list(seeds),
                'meta':[m.meta for m in models],'predict_seconds':times,'reload_max_error':parity,
                'prediction_sha256':sha_file(job/'heldout.npz'),'protocol_sha256':sha_file(OUT/'PROTOCOL.json'),
                'checkpoints':{f'config{chosen["config_id"]}_seed{s}.joblib':sha_file(job/f'config{chosen["config_id"]}_seed{s}.joblib') for s in seeds}})
            print('DONE',fold_id,name,round(metrics(test.group,mean,test.raw_w)['macro_ap'],6),flush=True)


def evaluate(replicates=2000):
    private=OUT/'private'; frame=pd.read_parquet(private/'cohort.parquet'); units=pd.read_parquet(private/'units.parquet')
    splits=joblib.load(private/'splits.joblib'); y=frame.group5.to_numpy(int); w=frame.wt_tot.to_numpy(float)
    predictions={}; rows={}; seed_scores={}; costs={}
    boot=PSUBootstrap(units,frame,replicates,seed=20261001); draws=boot.weights(w); ap_draws={}
    for name in CONDITIONS:
        p=np.full((len(frame),5),np.nan); seed_p=np.full((3,len(frame),5),np.nan); metas=[]; elapsed=[]; predict_elapsed=[]
        for fold_id,split in enumerate(splits,1):
            job=private/f'fold{fold_id}'/name
            if not (job/'DONE.json').exists(): raise RuntimeError(f'Incomplete {fold_id}/{name}')
            done=json.loads((job/'DONE.json').read_text(encoding='utf-8')); a=np.load(job/'heldout.npz')
            p[a['index']]=a['p5']
            for j in range(3): seed_p[j,a['index']]=a['seed_p5'][min(j,len(a['seed_p5'])-1)]
            metas+=done['meta']; elapsed += [m['fit_seconds'] for m in done['meta']]; predict_elapsed += done['predict_seconds']
        rows[name]=metrics(y,p,w); predictions[name]=p
        fast=[FastAP((y==k).astype(int),p[:,k]) for k in range(5)]
        dd=np.array([np.mean([f(bw) for f in fast]) for bw in draws]); ap_draws[name]=dd
        rows[name]['macro_ap_ci95']=ci(dd)
        seed_scores[name]=[metrics(y,q,w)['macro_ap'] for q in seed_p] if name!='LR' else [rows[name]['macro_ap']]
        params=[m['parameters'] for m in metas if 'parameters' in m]
        costs[name]={'parameter_median':float(np.median(params)),'parameter_unit':metas[0].get('parameter_unit','trainable'),
                     'selected_fit_seconds_median':float(np.median(elapsed)),'predict_seconds_median':float(np.median(predict_elapsed)),
                     'note':'CPU; single seed selected fits, excluding shared anchor preparation; prediction includes p16 audit for joint models'}
    comparisons={}
    for other in CONDITIONS:
        if other=='jEPF': continue
        d=ap_draws['jEPF']-ap_draws[other]
        comparisons[other]={'epf_minus_other':rows['jEPF']['macro_ap']-rows[other]['macro_ap'],'ci95':ci(d),'p_boot':bootstrap_p(d)}
    protocol=json.loads((OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
    report={'protocol':protocol,'metrics':rows,'epf_comparisons':comparisons,'seed_macro_ap':seed_scores,
            'costs':costs,'bootstrap':boot.audit,'primary_comparator':'jMLP',
            'interpretation':'Conditional paired PSU intervals on fixed predictions; other comparisons exploratory; no clinical benefit or cause claims'}
    write_json(OUT/'REPORT.json',report)
    table=pd.DataFrame([{ 'Model':name, **{k:v for k,v in m.items() if k in ('macro_ap','macro_auc','macro_f1','balanced_accuracy','nll','brier')},
                         'AP_CI_low':m['macro_ap_ci95'][0], 'AP_CI_high':m['macro_ap_ci95'][1]} for name,m in rows.items()])
    table.to_csv(OUT/'MODEL_COMPARISON.csv',index=False,encoding='utf-8-sig')
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5)); names=list(CONDITIONS)
    values=np.array([rows[n]['macro_ap'] for n in names]); bounds=np.array([rows[n]['macro_ap_ci95'] for n in names])
    ax.errorbar(values,np.arange(len(names)),xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='o',capsize=3,color='#087F8C')
    ax.set_yticks(np.arange(len(names)),names); ax.invert_yaxis(); ax.set_xlabel('Weighted five-class macro AP (95% PSU interval)')
    ax.grid(axis='x',alpha=.25); ax.set_title('2020-2024 / 15 non-invasive predictors / 5 PSU folds')
    fig.tight_layout(); fig.savefig(OUT/'model_comparison.png',dpi=180); plt.close(fig)
    markdown=['# v31 모델 비교 결과','',f'2020–2024년, 20–39세 미인지 코호트 중 영양조사 참여자 {len(frame):,}명. 실제 비침습 입력 15개, 배타적 5군.',
              '', '29개는 원자료 선택 열 수이며 검사값·ID·진단 문항은 예측 입력으로 쓰지 않았다. 혼밥/체중 증가 코드를 수정하고 전처리는 fit에서만 적합했다.',
              '', '| 모델 | macro AP | 95% 구간 | macro AUROC | macro F1 |', '|---|---:|---|---:|---:|']
    for name,m in rows.items(): markdown.append(f'| {name} | {m["macro_ap"]:.4f} | [{m["macro_ap_ci95"][0]:.4f}, {m["macro_ap_ci95"][1]:.4f}] | {m["macro_auc"]:.4f} | {m["macro_f1"]:.4f} |')
    primary=comparisons['jMLP']; markdown += ['',f'주 비교 EPF−공동분포 MLP: {primary["epf_minus_other"]:+.4f}, 95% 구간 [{primary["ci95"][0]:+.4f}, {primary["ci95"][1]:+.4f}].',
        '', '공동분포 모델은 연속 검사값을 추가 학습 정보로 사용한다. EPF와 같은 출력·손실을 쓰는 jMLP/jLinear를 통해 구조 효과를 구분한다.',
        '', '튜닝은 모든 모델당 8설정·seed42, 확률적 최종 모델은 seed42–44 평균이다. LR은 결정론적 한 모델이다. 동일 PSU 분할·동일 평가 대상·조사 가중치를 사용했다.',
        '', '공개 노트북의 원래 점수와 직접 비교할 수 없다: 라벨/특징 코드는 같게 정리했으나 미인지·공복·임신·가중치·PSU 분할을 엄격히 적용했다.',
        '', '기존에 관찰한 KNHANES를 재사용한 내부 비교다. 구간은 고정 예측에 대한 표본 불확실성이고 재학습을 포함하지 않는다. 원인·특이요인·임상 선별 효용은 평가하지 않았다.']
    (OUT/'REPORT.md').write_text('\n'.join(markdown)+'\n',encoding='utf-8')
    print(table.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','train','evaluate'])
    parser.add_argument('--condition',choices=CONDITIONS); parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='train': run(args.condition,args.fold)
    else: evaluate()


if __name__=='__main__':
    # Persist an importable class path even when launched using python -m.
    import sys
    sys.modules['metabolic.peer_comparison_v31'] = sys.modules[__name__]
    PeerEncoder.__module__ = 'metabolic.peer_comparison_v31'
    main()
