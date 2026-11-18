from pathlib import Path
import json,math
R=Path(__file__).resolve().parent
checked=0
def compare(a,b,path=''):
    global checked
    if isinstance(a,dict):
        for k,v in a.items():
            if k in {'note','specifications'}:continue
            assert k in b,(path,k)
            compare(v,b[k],path+'/'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(v,w) in enumerate(zip(a,b)):compare(v,w,path+'/'+str(i))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert math.isclose(a,b,rel_tol=0,abs_tol=1e-9),(path,a,b)
        checked+=1
    else:assert a==b,(path,a,b)
for src,dst in [('expected/analysis_results.json','results/analysis_results.json'),('expected/supporting/supplemental_results.json','results/supporting/supplemental_results.json')]:
    compare(json.loads((R/src).read_text()),json.loads((R/dst).read_text()))
print(f'Verified {checked} numerical entries at absolute tolerance 1e-9')
