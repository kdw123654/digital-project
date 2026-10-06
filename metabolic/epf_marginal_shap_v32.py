"""Supplementary original-flag SHAP to distinguish subtype competition from risk.

The primary five-class predictions and SHAP are unchanged. Negative contribution
to a mutually exclusive *single* subtype does not imply lower marginal risk.
"""
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch

from .epf_shap_v32 import EPFFunction, PERMUTATIONS, BEHAVIOUR
from .epf_year_v32 import OUT, YEARS
from .peer_comparison_v31 import FEATURES, write_json, sha_file
from .v30_data import BITS16, FLAGS
from .v30_models import p16_fast
from .epf_shap_math_v32 import shapley_fold


class FlagFunction(EPFFunction):
    def __call__(self,raw):
        frame=pd.DataFrame(np.asarray(raw,float),columns=FEATURES);x=self.encoder.transform(frame);sex=frame.sex.to_numpy(int);out=[]
        for model in self.models:
            parts=[]
            for start in range(0,len(x),4096):
                mu,dg,ld=model.joint_parameters(x[start:start+4096])
                p=p16_fast(mu,dg,ld,sex[start:start+4096],model.transform,order=self.order,device=str(self.device),chunk=4096)
                parts.append(p@BITS16.astype(float))
            out.append(np.concatenate(parts))
        return np.stack(out)

    def reference(self,raw):
        frame=pd.DataFrame(raw,columns=FEATURES);x=self.encoder.transform(frame);sex=frame.sex.to_numpy(int)
        return np.stack([m.predict(x,sex,audited=True)['p16']@BITS16.astype(float) for m in self.models])


def run():
    f=pd.read_parquet(OUT/'private/cohort.parquet');raw=pd.read_parquet(OUT/'private/predictors.parquet').to_numpy(float)
    records=[];audit=[]
    for year in YEARS:
        folder=OUT/'private'/str(year)/'shap_marginal';folder.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter();fn=FlagFunction(year);ctx=fn.ctx;dev=ctx['full'].index
        old=np.load(OUT/'private'/str(year)/'shap/values.npz');rows=old['index'];w=old['weight'];n,g=len(rows),len(FEATURES)
        # Same selected rows; new, fixed antithetic donor/permutation draws.
        rng=np.random.default_rng(9300+year);half=PERMUTATIONS//2
        pp=np.stack([np.stack([rng.permutation(g) for _ in range(half)]) for _ in rows]);pp=np.concatenate([pp,pp[:,:,::-1]],axis=1)
        ww=f.wt_tot.to_numpy()[dev];bg=rng.choice(dev,size=(n,half),p=ww/ww.sum());bg=np.concatenate([bg,bg],axis=1)
        probe=raw[rows[:12]];ref=fn.reference(probe);error=float(np.abs(fn(probe)-ref).max())
        if error>1e-4:fn.order=128;error=float(np.abs(fn(probe)-ref).max())
        assert error<=1e-4
        result=shapley_fold(fn,raw,rows,pp,bg,chunk_rows=4);phi=result['phi'].mean(0)
        assert result['additivity_max_abs_error']<1e-7
        np.savez_compressed(folder/'values.npz',index=rows,phi=result['phi'],weight=w)
        for j,name in enumerate(FEATURES):
            if name not in BEHAVIOUR:continue
            threshold=fn.encoder.mean[4] if name=='sedentary_hours' else 1.
            high=raw[rows,j]>=threshold if name=='sedentary_hours' else raw[rows,j]==1
            low=raw[rows,j]<threshold if name=='sedentary_hours' else raw[rows,j]==0
            if min(high.sum(),low.sum())<5:continue
            for k,flag in enumerate(FLAGS):
                delta=np.average(phi[high,j,k],weights=w[high])-np.average(phi[low,j,k],weights=w[low])
                records.append({'year':year,'feature':name,'flag':flag,'mean_shap_difference':float(delta)})
        meta={'year':year,'n':n,'reference_error':error,'additivity_error':result['additivity_max_abs_error'],'seconds':time.perf_counter()-start}
        audit.append(meta);write_json(folder/'DONE.json',meta);print('MARGINAL_SHAP_DONE',year,round(meta['seconds'],1),flush=True)
    rows=pd.DataFrame(records);rows.to_csv(OUT/'SHAP_MARGINAL_YEAR.csv',index=False)
    summary=[]
    for (feature,flag),v in rows.groupby(['feature','flag']):
        d=v.mean_shap_difference.to_numpy();summary.append({'feature':feature,'flag':flag,'mean_shap_difference':float(d.mean()),
            'positive_years':int((d>0).sum()),'negative_years':int((d<0).sum()),'n_years':len(d)})
    pd.DataFrame(summary).to_csv(OUT/'SHAP_MARGINAL_DIRECTION.csv',index=False)
    write_json(OUT/'SHAP_MARGINAL_REPORT.json',{'purpose':'Supplementary original-flag explanations for subtype-competition interpretation; not additional primary model comparisons',
         'audit':audit,'patterns':summary,'interpretation':'Conditional prediction associations, not intervention effects. Flags overlap; five subtypes are mutually exclusive.'})


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1);run()
