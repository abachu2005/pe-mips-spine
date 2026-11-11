"""Primary spine cohort final-score model and event-study coefficients."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import os
import sys
import subprocess
import hashlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT/"study"
os.environ["P4_SCOPE"]="v2"
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"analysis")]
import did


def write(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    code_file=PACKAGE_ROOT/"config/procedure_codes.json"
    included_codes=set(json.loads(code_file.read_text())["codes"])
    for row in csv.DictReader((ROOT/"data/spine_services.csv").open()):
        if row["HCPCS_Cd"] not in included_codes:
            raise RuntimeError("Spine frame contains a procedure outside the operative code list")
        if row["HCPCS_Cd"]=="0375T" and int(row["year"])>=2020:
            raise RuntimeError("Historical code 0375T cannot qualify in 2020 or later")
    subprocess.run([sys.executable,"-B","scripts/build_cohort.py"],cwd=ROOT,env=os.environ,check=True)
    records=did.load_panel()
    exposed,pool=did.load_people()
    billing=defaultdict(list)
    for r in csv.DictReader((ROOT/"data/spine_services.csv").open()):
        billing[r["Rndrng_NPI"]].append(r)
    def prior(npi,event):
        return [r for r in billing[npi] if int(r["year"])<event]
    def controls(state,event):
        people=[]
        for p in pool:
            rows=prior(p["npi"],event)
            if not rows:continue
            states=Counter(r["Rndrng_Prvdr_State_Abrvtn"] for r in rows)
            home=min(states,key=lambda s:(-states[s],s))
            if home==state:
                people.append(dict(p,state=home))
        return people
    for p in exposed:
        rows=prior(p["npi"],p["event"])
        assert rows
        p["specialty"]="neurosurgery" if any(r["Rndrng_Prvdr_Type"]=="Neurosurgery" for r in rows) else "orthopedic_spine"
    stack=did.build_stack(exposed,pool,records,"final_score",{"controls_fn":controls})
    if stack is None:raise RuntimeError("No eligible spine-only analytic support")
    assert not (set(stack["npi"][stack["treated"]==1]) & set(stack["npi"][stack["treated"]==0]))
    assert np.isfinite(stack["y"]).all() and np.isfinite(stack["weight"]).all()
    assert (stack["weight"]>0).all()
    result={"definition":"Neurosurgery/orthopedic surgery with a reported spine operative-code row in a calendar year strictly before acquisition year; comparisons assessed at each cell's event year.",
            "analysis":"Primary weighted stacked difference-in-differences model of final score with event-time interactions.",
            "primary":did.pooled(stack),"event_study":did.event_study(stack),
            "selection_limit":"Spine-service baseline ascertainment spans 2013–2022. Suppressed low-volume billing cannot qualify. Some clinical spine surgeons are unascertained; MIPS final-score performance data span 2017–2024.",
            "prestate_rule":"Comparators: modal state in qualifying pre-event spine service rows, lexicographic tie break. Acquired: transaction state.",
            "cohort_scope":"Spine-only; no specialty-alone neurosurgeon admission."}
    out=ROOT/"results";out.mkdir(parents=True,exist_ok=True)
    (out/"analysis_results.json").write_text(json.dumps(result,indent=2)+"\n")
    write(out/"primary_estimate.csv",[dict(stack["counts"],estimate=result["primary"]["estimate"]["estimate"],se=result["primary"]["estimate"]["se"],df=result["primary"]["estimate"]["df"],ci_low=float(result["primary"]["estimate"]["ci"][0]),ci_high=float(result["primary"]["estimate"]["ci"][1]),p=result["primary"]["estimate"]["p"])])
    es=[{"k":-1,"estimate":0,"ci_low":0,"ci_high":0}]
    es += [{"k":int(k),"estimate":v["estimate"],"ci_low":v["ci"][0],"ci_high":v["ci"][1]} for k,v in result["event_study"]["coefficients"].items()]
    write(out/"event_study.csv",es)
    curve=[]
    for k in sorted(set(stack["k"])):
        row={"k":int(k)}
        for label,d in (("exposed",1),("comparison",0)):
            m=(stack["k"]==k)&(stack["treated"]==d)
            row[label+"_mean"]=float(np.average(stack["y"][m],weights=stack["weight"][m]))
            row[label+"_surgeons"]=len(set(stack["npi"][m]))
            row[label+"_parents_or_clusters"]=len(set(stack["cluster"][m]))
        curve.append(row)
    write(out/"trajectories_support.csv",curve)
    weight_audit=[]
    for cell in sorted(set(stack["cell"])):
        m=stack["cell"]==cell
        nt=len(set(stack["npi"][m&(stack["treated"]==1)]))
        nc=len(set(stack["npi"][m&(stack["treated"]==0)]))
        ws=stack["weight"][m&(stack["treated"]==0)]
        assert np.allclose(ws,ws[0])
        weight_audit.append({"cell":cell,"treated_units":nt,"comparison_units":nc,
                             "comparison_weight":float(ws[0])})
    write(out/"cell_weights.csv",weight_audit)
    ledger=[]
    selected=set(stack["npi"][stack["treated"]==1])
    for p in exposed:
        pre=prior(p["npi"],p["event"])
        obs=records.get(p["npi"],{})
        ys=[t for t,v in obs.items() if -4<=t-p["event"]<=4 and v["final_score"] is not None]
        reason="included" if p["npi"] in selected else (
            "no_mips_panel" if not obs else
            "no_pre_score" if not any(t<p["event"] for t in ys) else
            "no_post_score" if not any(t>p["event"] for t in ys) else "no_eligible_comparison_cell")
        ledger.append({"npi":p["npi"],"parent":p["parent"],"event":p["event"],"specialty":p["specialty"],
                       "first_spine_year":min(int(r["year"]) for r in pre),
                       "pre_spine_codes":";".join(sorted({r["HCPCS_Cd"] for r in pre})),
                       "status":reason})
    write(out/"treated_eligibility.csv",ledger)
    for cell,npi,d in set(zip(stack["cell"],stack["npi"],stack["treated"])):
        event=int(cell.split(":")[1])
        assert prior(npi,event)
    write(out/"analytic_unit_eligibility.csv",[
        {"cell":cell,"npi":n,"treated":int(d),"first_spine_year":min(int(r["year"]) for r in prior(n,int(cell.split(":")[1])))}
        for cell,n,d in sorted(set(zip(stack["cell"],stack["npi"],stack["treated"])))])
    raw={(r["npi"],int(r["year"])):r for r in csv.DictReader(did.PANEL.open())}
    baseline={}
    for cell,n,t,k,d,w in zip(stack["cell"],stack["npi"],stack["year"],stack["k"],stack["treated"],stack["weight"]):
        if k < 0 and ((cell,n) not in baseline or t > baseline[(cell,n)][0]):
            baseline[(cell,n)]=(int(t),int(d),float(w))
    descriptive=[]
    for field in ("final_score","practice_size","years_in_medicare"):
        row={"characteristic":field}
        for name,d in (("acquired",1),("comparison",0)):
            vals=[];weights=[]
            for (cell,n),(t,group,w) in baseline.items():
                text=raw.get((n,t),{}).get(field,"")
                if group==d and text:
                    vals.append(float(text));weights.append(w)
            row[name+"_weighted_mean"]=float(np.average(vals,weights=weights)) if vals else None
            row[name+"_available_cell_units"]=len(vals)
        descriptive.append(row)
    write(out/"baseline_characteristics.csv",descriptive)
    mix=[]
    for name,d in (("acquired",1),("comparison",0)):
        sub=stack["npi"][stack["treated"]==d]
        mix.append({"group":name,"distinct_surgeons":len(set(sub)),
                    "cell_units":len(set(zip(stack["cell"][stack["treated"]==d],sub)))})
    write(out/"distinct_and_reused_units.csv",mix)
    flow=json.loads((ROOT/"cohort/flow.json").read_text())
    stages=[
        {"stage":"operative_spine_frame","surgeons":flow["operative_spine_frame_surgeons"]},
        {"stage":"eligible_acquired","surgeons":len(exposed)},
        {"stage":"linked_excluded_from_comparators","surgeons":flow["frame_excluded_as_linked"]},
        {"stage":"unlinked_comparison_pool_before_cell_eligibility","surgeons":len(pool)},
        {"stage":"primary_acquired","surgeons":stack["counts"]["exposed_surgeons"]},
        {"stage":"primary_distinct_comparators","surgeons":stack["counts"]["comparison_surgeons"]},
    ]
    write(out/"cohort_flow.csv",stages)
    (out/"verification.json").write_text(json.dumps({
        "every_analytic_unit_has_pre_event_spine_billing":True,
        "t_df_equals_acquired_parents_minus_one":result["primary"]["estimate"]["df"]==stack["counts"]["exposed_practices"]-1,
        "analytic_unit_count":len(set(zip(stack["cell"],stack["npi"]))),
        "stack_rows":len(stack["y"]),"pooled_rows_excluding_zero":int(sum(stack["k"]!=0)),
        "treated_status_counts":dict(Counter(r["status"] for r in ledger)),

    },indent=2)+"\n")
    print(json.dumps(result["primary"],indent=2))


if __name__=="__main__":
    main()
