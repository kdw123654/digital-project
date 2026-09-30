"""Regenerate manuscript tables and a figure from public aggregate results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
ORDER = ('LR','RF','XGB','LGBM','MLP','jLinear','jMLP','jEPF')


def load_summary():
    data=json.loads((PAPER/'results/comparison_summary.json').read_text(encoding='utf-8'))
    assert data['years']==list(range(2020,2025)) and data['predictor_count']==15
    assert data['cohort']['nutrition_20_39']==4508 and len(data['metrics'])==8
    for key in ORDER:
        m=data['metrics'][key]
        assert 0<=m['macro_ap']<=1 and m['macro_ap_ci95'][0]<=m['macro_ap_ci95'][1]
    return data


def result_table(data=None):
    data=load_summary() if data is None else data
    rows=[]
    for key in ORDER:
        m=data['metrics'][key]
        rows.append({'Model':key,'Description':data['model_labels'][key],
          'macro_AP':m['macro_ap'],'AP_CI_low':m['macro_ap_ci95'][0],'AP_CI_high':m['macro_ap_ci95'][1],
          'macro_AUROC':m['macro_auc'],'macro_F1':m['macro_f1'],'balanced_accuracy':m['balanced_accuracy'],
          'NLL':m['nll'],'Brier':m['brier']})
    return pd.DataFrame(rows)


def render():
    data=load_summary(); table=result_table(data)
    lines=['# Model comparison table','','| Model | macro AP | 95% PSU interval | macro AUROC | macro F1 |',
           '|---|---:|---|---:|---:|']
    for row in table.itertuples():
        lines.append(f'| {row.Model} | {row.macro_AP:.4f} | [{row.AP_CI_low:.4f}, {row.AP_CI_high:.4f}] | {row.macro_AUROC:.4f} | {row.macro_F1:.4f} |')
    (PAPER/'results/model_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    values=table.macro_AP.to_numpy(); bounds=table[['AP_CI_low','AP_CI_high']].to_numpy()
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.errorbar(values,np.arange(8),xerr=np.stack([values-bounds[:,0],bounds[:,1]-values]),fmt='o',capsize=3,color='#087F8C')
    ax.set_yticks(np.arange(8),ORDER); ax.invert_yaxis(); ax.grid(axis='x',alpha=.25)
    ax.set_xlabel('Weighted five-class macro AP (95% PSU interval)')
    ax.set_title('2020-2024 / 15 non-invasive predictors / 5 PSU folds')
    fig.tight_layout(); fig.savefig(PAPER/'results/model_comparison.png',dpi=180); plt.close(fig)
    print(table.round(5).to_string(index=False))


if __name__=='__main__':
    render()
