"""Secondary outcomes, specialty estimates, balance, and robustness in the operative spine cohort."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,sys,math
import numpy as np
PACKAGE_ROOT=Path(__file__).resolve().parent
ROOT=PACKAGE_ROOT/'study';sys.path[:0]=[str(ROOT/'analysis'),str(ROOT/'scripts')]
import did
OUT=ROOT/'results/supporting';OUT.mkdir(parents=True,exist_ok=True)
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 raw={(r['npi'],int(r['year'])):r for r in csv.DictReader(did.PANEL.open())}
 records=defaultdict(dict)
 for (n,t),r in raw.items():
  adj=did.num(r['payment_adjustment']);records[n][t]={k:did.num(r[k]) for k in ['final_score','quality_score']}
  records[n][t].update(positive_adjustment=None if adj is None else 100.*(adj>0),individual=r['participation'].strip().lower()=='individual',hardship=r['extreme_hardship'].strip().lower()=='true')
 exposed,pool=did.load_people();billing=defaultdict(list)
 for r in csv.DictReader((ROOT/'data/spine_services.csv').open()):billing[r['Rndrng_NPI']].append((int(r['year']),r['Rndrng_Prvdr_State_Abrvtn'],r['Rndrng_Prvdr_Type']))
 def prior(n,e):return [r for r in billing[n] if r[0]<e]
 def specialty(n,e):return 'neurosurgery' if any(r[2]=='Neurosurgery' for r in prior(n,e)) else 'orthopedic_spine'
 for p in exposed:p['specialty']=specialty(p['npi'],p['event'])
 cache={}
 def controls(state,event):
  if event not in cache:
   by=defaultdict(list)
   for p in pool:
    rr=prior(p['npi'],event)
    if not rr:continue
    states=Counter(r[1] for r in rr);home=min(states,key=lambda s:(-states[s],s))
    by[home].append(dict(p,state=home,specialty=specialty(p['npi'],event)))
   cache[event]=by
  return cache[event][state]
 def run(people=exposed,outcome='final_score',opts=None):
  op={'controls_fn':controls,**(opts or {})}
  s=did.build_stack(people,pool,records,outcome,op)
  if s is None:return None
  try:return did.pooled(s)
  except (ValueError,np.linalg.LinAlgError) as e:return {'status':'not_estimable','reason':str(e)}
 primary=did.build_stack(exposed,pool,records,'final_score',{'controls_fn':controls});result={'primary':did.pooled(primary),'secondary':{},'specialty':{},'robustness':{},'parent_omission':{},'complete_followup':{}}
 assert primary['counts']['exposed_surgeons']==115
 assert math.isclose(result['primary']['estimate']['estimate'],1.356222818015417,abs_tol=1e-12)
 def check_support(s):
  for cell,n in set(zip(s['cell'],s['npi'])):assert prior(n,int(cell.split(':')[1]))
 check_support(primary)
 for o in ['quality_score','positive_adjustment']:
  result['secondary'][o]=run(outcome=o);print(o,result['secondary'][o]['estimate'],flush=True)
 for label in ['neurosurgery','orthopedic_spine']:
  fn=lambda state,event,label=label:[c for c in controls(state,event) if c['specialty']==label]
  result['specialty'][label]=run([p for p in exposed if p['specialty']==label],opts={'controls_fn':fn});print(label,result['specialty'][label]['estimate'],flush=True)
 robust=result['robustness']
 robust['practice_mso_only']=run([p for p in exposed if p.get('connection')!='operator_professional_group'])
 robust['exclude_2017']=run(opts={'drop_years':{2017}})
 robust['exclude_pandemic_hardship']=run(opts={'drop_years':{2020,2021},'drop_hardship':True})
 robust['individual_reporting_only']=run(opts={'individual_only':True})
 robust['preacquisition_affiliation_only']=run([p for p in exposed if p['affiliation_basis'] in ('pre_deal_cms','pre_deal_roster')])
 placebo=[dict(p,event=p['event']-2,before=p['event']) for p in exposed if prior(p['npi'],p['event']-2)]
 robust['placebo_two_years_earlier']=run(placebo,opts={'controls_fn':lambda s,e:[dict(c,before=e+2) for c in controls(s,e)]})
 def notyet(state,event):
  out=[]
  for p in exposed:
   if p['event']<event+2:continue
   rr=prior(p['npi'],event)
   if not rr:continue
   states=Counter(r[1] for r in rr);home=min(states,key=lambda s:(-states[s],s))
   if home==state:out.append(dict(p,state=home,before=p['event'],specialty=specialty(p['npi'],event)))
  return out
 robust['not_yet_acquired_comparators']=run(opts={'controls_fn':notyet})
 for k,v in robust.items():print(k,v.get('estimate',v) if v else None,flush=True)
 for parent in sorted({p['parent'] for p in exposed if p['npi'] in set(primary['npi'][primary['treated']==1])}):
  result['parent_omission'][parent]=run([p for p in exposed if p['parent']!=parent])
 def reweight(s):
  units=set(zip(s['cell'],s['npi'],s['treated']));nt=Counter(c for c,n,d in units if d);nc=Counter(c for c,n,d in units if not d);cells=set(nt)&set(nc)
  s={k:v[np.isin(s['cell'],list(cells))] for k,v in s.items() if isinstance(v,np.ndarray)}
  units=set(zip(s['cell'],s['npi'],s['treated']));nt=Counter(c for c,n,d in units if d);nc=Counter(c for c,n,d in units if not d);a,b=sum(nt.values()),sum(nc.values())
  s['weight']=np.array([1. if d else (nt[c]/a)/(nc[c]/b) for c,d in zip(s['cell'],s['treated'])]);s['counts']={'exposed_surgeons':len(set(s['npi'][s['treated']==1])),'exposed_practices':len(set(s['cluster'][s['treated']==1])),'comparison_surgeons':len(set(s['npi'][s['treated']==0])),'comparison_cell_units':b,'cells':len(cells),'observations':len(s['y'])}
  return s
 for h in (3,4):
  ks=set(range(-1,h+1));seen=defaultdict(set)
  for c,n,k in zip(primary['cell'],primary['npi'],primary['k']):seen[(c,n)].add(int(k))
  good={u for u,v in seen.items() if ks.issubset(v)};m=np.array([(c,n) in good and int(k) in ks for c,n,k in zip(primary['cell'],primary['npi'],primary['k'])]);s=reweight(did.subset(primary,m));check_support(s)
  late=did.subset(s,(s['k']==-1)|(s['k']>=3));late['counts']=dict(s['counts'],observations=int(len(late['y'])))
  result['complete_followup'][str(h)]={'pooled':did.pooled(s),'event_study':did.event_study(s),'late':did.pooled(late)}
 # Last observed preacquisition characteristics, weighted over surgeon-cohort units.
 units={}
 for c,n,t,k,d,w in zip(primary['cell'],primary['npi'],primary['year'],primary['k'],primary['treated'],primary['weight']):
  if k<0 and ((c,n) not in units or t>units[(c,n)]['year']):units[(c,n)]={'year':int(t),'d':int(d),'w':float(w)}
 labels=['Neurosurgeon, %','Orthopedic surgeon, %','MIPS final score','Positive payment adjustment, %','Group reporting, %','APM reporting, %','Individual reporting, %','Practice size, clinicians','Small practice, %','Rural, %','Years in Medicare','Medicare patients']
 vals=defaultdict(lambda:{1:([],[]),0:([],[])})
 for (cell,n),u in units.items():
  r=raw[(n,u['year'])];event=int(cell.split(':')[1]);part=r['participation'].strip().lower()
  f=lambda k:float(r[k]) if r.get(k) not in ('',None) else np.nan
  b=lambda k:100.*(r[k].strip().lower()=='true') if r.get(k) not in ('',None) else np.nan
  values=[100.*(specialty(n,event)=='neurosurgery'),100.*(specialty(n,event)=='orthopedic_spine'),f('final_score'),100.*(f('payment_adjustment')>0) if r['payment_adjustment'] else np.nan,100.*(part in ('group','virtual group','subgroup')) if part else np.nan,100.*('apm' in part) if part else np.nan,100.*(part=='individual') if part else np.nan,f('practice_size'),b('small_practice'),b('rural'),f('years_in_medicare'),f('medicare_patients')]
  for label,v in zip(labels,values):a,w=vals[label][u['d']];a.append(v);w.append(u['w'])
 table=[]
 for label in labels:
  row={'characteristic':label};stats=[]
  for name,d in [('acquired',1),('comparison',0)]:
   v,w=map(lambda x:np.array(x,float),vals[label][d]);ok=np.isfinite(v);v,w=v[ok],w[ok];mean=float(np.average(v,weights=w));var=float(np.average((v-mean)**2,weights=w));stats.append((mean,var));row.update({name+'_mean':mean,name+'_sd':math.sqrt(var),name+'_available':int(ok.sum()),name+'_missing':int((~ok).sum())})
  den=math.sqrt((stats[0][1]+stats[1][1])/2);row['smd']=(stats[0][0]-stats[1][0])/den if den else None;table.append(row)
 write('baseline_balance.csv',table)
 def rows(block):
  return [{'analysis':k,**v['counts'],**{z:v['estimate'][z] for z in ['estimate','se','df','p']},'ci_low':v['estimate']['ci'][0],'ci_high':v['estimate']['ci'][1]} for k,v in block.items() if v and 'estimate' in v]
 for key in ['secondary','specialty','robustness','parent_omission']:write(key+'.csv',rows(result[key]))
 write('complete_followup.csv',[{'horizon':h,'contrast':k,**v['counts'],**{z:v['estimate'][z] for z in ['estimate','se','df','p']},'ci_low':v['estimate']['ci'][0],'ci_high':v['estimate']['ci'][1]} for h,b in result['complete_followup'].items() for k,v in b.items() if k!='event_study'])
 (OUT/'supporting_results.json').write_text(json.dumps(result,indent=2)+'\n')
 assert all(v['estimate']['df']==v['counts']['exposed_practices']-1 for b in [result['secondary'],result['specialty'],result['robustness'],result['parent_omission']] for v in b.values() if v and 'estimate' in v)
 print('DONE',OUT,flush=True)
if __name__=='__main__':main()
