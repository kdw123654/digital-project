"""Read completed aggregate tables; no training or individual data access."""
from pathlib import Path
import hashlib,json
import pandas as pd

root=Path(__file__).resolve().parents[1]
folder=root/'paper/results/v32'
done=json.loads((folder/'COMPLETION.json').read_text(encoding='utf-8'))
assert hashlib.sha256((folder/'REPORT.json').read_bytes()).hexdigest()==done['report_sha256']
table=pd.read_csv(folder/'MODEL_COMPARISON.csv').sort_values('mean_year_macro_ap',ascending=False)
print(table.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
for k in range(1,6):assert (root/'paper/figures'/f'figure{k}.png').exists()
print('Verified completed aggregate report and five R figures. No refitting performed.')
