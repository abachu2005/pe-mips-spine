from pathlib import Path
import json,csv,math
p=Path(__file__).resolve().parent;expected=p/"expected";actual=p/"study/results"
def leaves(x,prefix=""):
 if isinstance(x,dict):
  for k,v in x.items():yield from leaves(v,prefix+"/"+k)
 elif isinstance(x,list):
  for k,v in enumerate(x):yield from leaves(v,prefix+"/"+str(k))
 elif isinstance(x,(int,float)) and not isinstance(x,bool):yield prefix,x
ref=json.loads((expected/"reference_results.json").read_text());got=json.loads((actual/"analysis_results.json").read_text());a=dict(leaves(ref));b=dict(leaves(got))
assert all(math.isclose(v,b[k],abs_tol=1e-9,rel_tol=1e-9) for k,v in a.items())
for f in expected.glob("*.csv"):
 aa=list(csv.DictReader(f.open()));bb=list(csv.DictReader((actual/f.name).open()));assert len(aa)==len(bb)
 for x,y in zip(aa,bb):
  assert x.keys()==y.keys()
  for k,v in x.items():
   try:ok=math.isclose(float(v),float(y[k]),abs_tol=1e-9,rel_tol=1e-9)
   except ValueError:ok=v==y[k]
   assert ok,(f.name,k)
print(f"Passed {len(a)} numerical result entries and 7 aggregate tables.")
