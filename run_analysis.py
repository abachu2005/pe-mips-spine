from pathlib import Path
import json, sys, numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
import did
with np.load(ROOT/'inputs/stack.npz',allow_pickle=False) as z:
    s={k:z[k] for k in z.files}
s['counts']=json.loads((ROOT/'inputs/counts.json').read_text())
r={'primary':did.pooled(s),'event_study':did.event_study(s)}
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results/analysis_results.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r['primary'],indent=2))
