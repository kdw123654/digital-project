"""Checkpointed process scheduler; no shared job writers, no row-level logging."""
from pathlib import Path
import concurrent.futures
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/epf_year_v32_202024'
PY=Path(sys.executable)
ORDER=('jEPF','jMLP','MLP','LGBM','XGB','RF','jLinear','LR')
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
jobs=[(year,name) for name in ORDER for year in range(2020,2025)]


def run(job):
    year,name=job;folder=OUT/'private'/str(year)/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():return year,name,0,'reused'
    with (folder/'run.log').open('a',encoding='utf-8') as log:
        p=subprocess.Popen([str(PY),'-m','metabolic.epf_year_v32','train','--year',str(year),'--condition',name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        (folder/'PROCESS.json').write_text(json.dumps({'pid':p.pid,'parent_pid':os.getpid(),'year':year,'condition':name}),encoding='utf-8')
        code=p.wait()
    return year,name,code,'finished'


if __name__=='__main__':
    start=time.time();finished=[];failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            year,name,code,state=future.result();finished.append([year,name])
            if code:failed.append([year,name,code])
            progress={'scheduler_pid':os.getpid(),'finished':finished,'failed':failed,'total':len(jobs),'elapsed_seconds':time.time()-start}
            temp=OUT/'PROGRESS.tmp';temp.write_text(json.dumps(progress,indent=2),encoding='utf-8');temp.replace(OUT/'PROGRESS.json')
            print('COMPLETE',len(finished),'/',len(jobs),year,name,'exit',code,flush=True)
    if failed:raise SystemExit('Some jobs failed: '+str(failed))
    print('TRAINING_COMPLETE',round(time.time()-start,1),'seconds',flush=True)
