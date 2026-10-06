"""Probability-scale permutation SHAP for the actual refitted EPF ensembles.

Vectorised antithetic permutation paths use weighted development-only donors.
Each complete path is additive. This is Monte Carlo interventional SHAP, not
TreeSHAP, a causal effect estimate, or a physiological state interpretation.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .epf_year_v32 import OUT, YEARS
from .peer_comparison_v31 import FEATURES, CLASS_NAMES, M16_TO_5, predict, sha_file, write_json
from .v30_models import p16_fast
from .epf_shap_math_v32 import shapley_fold

PERMUTATIONS=64
PER_CLASS=40
BEHAVIOUR=('sm_presnt','dr_month','pa_aerobic','sedentary_hours','solo_dinner','weight_gain','no_brush_bed')


class EPFFunction:
    def __init__(self,year,order=64,device=None):
        self.ctx=joblib.load(OUT/'private'/str(year)/'context.joblib')
        self.models=[joblib.load(OUT/'private'/str(year)/'jEPF'/f'final_seed{s}.joblib') for s in (42,43,44)]
        self.encoder=self.ctx['final_encoder'];self.order=order
        self.device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        for m in self.models:m.model.to(self.device).eval()

    def __call__(self,raw):
        frame=pd.DataFrame(np.asarray(raw,float),columns=FEATURES)
        x=self.encoder.transform(frame);sex=frame.sex.to_numpy(int);outs=[]
        for model in self.models:
            parts=[]
            for start in range(0,len(x),4096):
                xx=x[start:start+4096];ss=sex[start:start+4096]
                mu,dg,ld=model.joint_parameters(xx)
                pp=p16_fast(mu,dg,ld,ss,model.transform,order=self.order,device=str(self.device),chunk=4096)
                parts.append(pp@M16_TO_5)
            outs.append(np.concatenate(parts))
        return np.stack(outs)

    def reference(self,raw):
        frame=pd.DataFrame(raw,columns=FEATURES);x=self.encoder.transform(frame);sex=frame.sex.to_numpy(int)
        return np.stack([predict(m,x,sex) for m in self.models])


def explain(year):
    folder=OUT/'private'/str(year)/'shap';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():print('REUSED_SHAP',year,flush=True);return
    f=pd.read_parquet(OUT/'private/cohort.parquet');raw=pd.read_parquet(OUT/'private/predictors.parquet').to_numpy(float)
    fn=EPFFunction(year);split=fn.ctx['split'];development=fn.ctx['full'].index
    assert year not in set(f.survey_year.iloc[development])
    rng=np.random.default_rng(3200+year);chosen=[];inclusion=[]
    for k in range(5):
        candidates=split.heldout[f.group5.to_numpy()[split.heldout]==k];n=min(PER_CLASS,len(candidates))
        chosen.extend(rng.choice(candidates,n,replace=False));inclusion.extend([n/len(candidates)]*n)
    rows=np.array(chosen,int);inclusion=np.array(inclusion);order=np.argsort(rows);rows=rows[order];inclusion=inclusion[order]
    n,g=len(rows),len(FEATURES);half=PERMUTATIONS//2
    permutations=np.stack([np.stack([rng.permutation(g) for _ in range(half)]) for _ in rows])
    permutations=np.concatenate([permutations,permutations[:,:,::-1]],axis=1)
    ww=f.wt_tot.to_numpy()[development]
    bg_half=rng.choice(development,size=(n,half),p=ww/ww.sum());background=np.concatenate([bg_half,bg_half],axis=1)
    assert not np.isin(f.survey_year.to_numpy()[background],year).any()
    # Audit original rows, training donors and mixed rows before the large pass.
    probe=np.concatenate([raw[rows[:8]],raw[bg_half[:8,0]],np.where(rng.random((16,g))<.5,raw[rows[:16]],raw[bg_half[:16,0]])])
    reference=fn.reference(probe);maxerror=float(np.abs(fn(probe)-reference).max())
    if maxerror>1e-4:
        fn.order=128;maxerror=float(np.abs(fn(probe)-reference).max())
    if maxerror>1e-4:
        fn.order=256;maxerror=float(np.abs(fn(probe)-reference).max())
    if maxerror>1e-4:raise FloatingPointError(f'SHAP prediction integration error {maxerror}')
    start=time.perf_counter();result=shapley_fold(fn,raw,rows,permutations,background,chunk_rows=4)
    assert result['additivity_max_abs_error']<1e-7
    # Store a standard SHAP Explanation object for the ensemble, not surrogate
    # LightGBM explanations. The permutation calculation itself is vectorised.
    import shap
    phi=result['phi'].mean(0);base=result['f_background_mean'].mean(0);fx=result['fx'].mean(0)
    explanation=shap.Explanation(values=phi,base_values=base,data=raw[rows],feature_names=list(FEATURES),output_names=list(CLASS_NAMES))
    joblib.dump(explanation,folder/'epf_explanation.joblib',compress=3)
    weight=f.wt_tot.to_numpy()[rows]/inclusion
    np.savez_compressed(folder/'values.npz',index=rows,phi=result['phi'],mc_var=result['mc_var'],fx=result['fx'],
                        base=result['f_background_mean'],weight=weight,inclusion=inclusion,raw=raw[rows])
    # Official SHAP explainer audit of the same EPF callable on a tiny fixed
    # subset. Same output units; numerical estimates may differ by MC sampling.
    official=shap.PermutationExplainer(lambda a:fn(a).mean(0),raw[bg_half[0,:4]],seed=year,feature_names=list(FEATURES))
    checked=official(raw[rows[:1]],max_evals=31*2,batch_size=256,silent=True)
    official_error=float(np.max(np.abs(checked.values.sum(1)+checked.base_values-fn(raw[rows[:1]]).mean(0))))
    assert official_error<1e-6
    meta={'year':year,'n_explained':n,'permutations':PERMUTATIONS,'antithetic_pairs':half,'players':list(FEATURES),
          'output':'EPF seed42-44 ensemble five-class probabilities','development_background_only':True,
          'sampling':'Up to40 random held-out rows per class, inverse-inclusion survey weights for aggregation',
          'quadrature_order':fn.order,'audited_prediction_max_error':maxerror,'additivity_max_error':result['additivity_max_abs_error'],
          'official_shap_callable_additivity_error':official_error,'shap_version':shap.__version__,
          'seconds':time.perf_counter()-start,'values_sha256':sha_file(folder/'values.npz')}
    write_json(folder/'DONE.json',meta)
    print('SHAP_DONE',year,n,'rows',round(meta['seconds'],1),'seconds',flush=True)


def aggregate():
    annual=[];directions=[];metadata=[]
    for year in YEARS:
        folder=OUT/'private'/str(year)/'shap';meta=json.loads((folder/'DONE.json').read_text(encoding='utf-8'))
        assert sha_file(folder/'values.npz')==meta['values_sha256'];metadata.append(meta)
        a=np.load(folder/'values.npz');phi=a['phi'].mean(0);w=a['weight'];raw=a['raw'];ctx=joblib.load(OUT/'private'/str(year)/'context.joblib')
        importance=np.average(np.abs(phi),axis=0,weights=w)
        for j,feature in enumerate(FEATURES):
            for k,name in enumerate(CLASS_NAMES):annual.append({'year':year,'feature':feature,'class':name,'mean_abs_shap':float(importance[j,k])})
            if feature not in BEHAVIOUR:continue
            if feature=='sedentary_hours':
                threshold=float(ctx['final_encoder'].mean[list(('age','HE_BMI','HE_wc','WHtR','sedentary_hours')).index(feature)])
                high=raw[:,j]>=threshold;low=raw[:,j]<threshold
            else:
                threshold=1.;high=raw[:,j]==1;low=raw[:,j]==0
            if high.sum()<5 or low.sum()<5:continue
            for k,name in enumerate(CLASS_NAMES):
                difference=np.average(phi[high,j,k],weights=w[high])-np.average(phi[low,j,k],weights=w[low])
                directions.append({'year':year,'feature':feature,'class':name,'mean_shap_difference':float(difference),
                                   'high_definition':'above development weighted mean' if feature=='sedentary_hours' else 'coded1 versus coded0',
                                   'threshold':threshold,'n_high':int(high.sum()),'n_low':int(low.sum())})
    ann=pd.DataFrame(annual);ann.to_csv(OUT/'SHAP_YEAR_CLASS.csv',index=False)
    byclass=ann.groupby(['feature','class'],sort=False).mean_abs_shap.mean().reset_index();byclass.to_csv(OUT/'SHAP_CLASS.csv',index=False)
    globalimp=byclass.groupby('feature').mean_abs_shap.mean().sort_values(ascending=False).reset_index()
    yearglobal=ann.groupby(['year','feature']).mean_abs_shap.mean().reset_index()
    spans=yearglobal.groupby('feature').mean_abs_shap.agg(['min','max']).reset_index();globalimp=globalimp.merge(spans,on='feature')
    globalimp.to_csv(OUT/'SHAP_GLOBAL.csv',index=False)
    d=pd.DataFrame(directions);d.to_csv(OUT/'SHAP_BEHAVIOUR_YEAR.csv',index=False)
    candidate=[]
    for (feature,name),v in d.groupby(['feature','class'],sort=False):
        vals=v.mean_shap_difference.to_numpy();positive=int((vals>0).sum());negative=int((vals<0).sum())
        candidate.append({'feature':feature,'class':name,'mean_shap_difference':float(vals.mean()),'years_supported':len(vals),
                          'positive_years':positive,'negative_years':negative,'same_sign_4_of_5':len(vals)==5 and max(positive,negative)>=4})
    write_json(OUT/'SHAP_REPORT.json',{'method':'Weighted-background antithetic permutation SHAP on the real EPF ensemble; standard SHAP Explanation storage and official callable audit',
       'annual':metadata,'global_importance':globalimp.to_dict('records'),'behaviour_patterns':candidate,
       'interpretation':'Descriptive prediction contributions. Four-of-five direction consistency is a reporting rule, not a significance test or a causal claim. Correlated BMI/waist/WHtR can redistribute attribution; sex also changes the HDL label threshold.'})
    print('SHAP_AGGREGATED',sum(m['n_explained'] for m in metadata),'held-out explanations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['explain','aggregate']);p.add_argument('--year',type=int);args=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if args.command=='aggregate':aggregate()
    else:explain(args.year)
