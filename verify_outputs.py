from pathlib import Path
import json,csv,math
p=Path(__file__).resolve().parent;expected=p/"expected";actual=p/"study/results"
def leaves(x,prefix=""):
 if isinstance(x,dict):
  for k,v in x.items():yield from leaves(v,prefix+"/"+k)
 elif isinstance(x,list):
  for k,v in enumerate(x):yield from leaves(v,prefix+"/"+str(k))
 elif isinstance(x,(int,float)) and not isinstance(x,bool):yield prefix,x
count=0
for e,a in [(expected/"reference_results.json",actual/"analysis_results.json"),(expected/"supporting/supporting_results.json",actual/"supporting/supporting_results.json")]:
 ref=dict(leaves(json.loads(e.read_text())));got=dict(leaves(json.loads(a.read_text())))
 assert all(math.isclose(v,got[k],abs_tol=1e-9,rel_tol=1e-9) for k,v in ref.items())
 count+=len(ref)
files=list(expected.rglob("*.csv"))
for f in files:
 aa=list(csv.DictReader(f.open()));bb=list(csv.DictReader((actual/f.relative_to(expected)).open()));assert len(aa)==len(bb)
 for x,y in zip(aa,bb):
  assert x.keys()==y.keys()
  for k,v in x.items():
   try:ok=math.isclose(float(v),float(y[k]),abs_tol=1e-9,rel_tol=1e-9)
   except ValueError:ok=v==y[k]
   assert ok,(f.name,k)
print(f"Passed {count} numerical result entries and {len(files)} aggregate tables.")
