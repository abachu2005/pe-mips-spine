"""Secondary outcomes, specialty models, robustness, and baseline balance."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,sys
import numpy as np
SECONDARY={"quality_score":"Quality", "positive_adjustment":"Positive adjustment", "pi_score":"Promoting interoperability", "ia_score":"Improvement activities", "cost_score":"Cost"}
PLAN={}
def csvout(path,data):
    keys=list(dict.fromkeys(k for p in data for k in p))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def run(root):
    sys.path.insert(0, str(root / "analysis"))
    import did
    context = json.loads((root / "inputs/supplemental_context.json").read_text())
    saved = json.loads((root / "expected/analysis_results.json").read_text())
    outdir = root / "results/supporting"
    outdir.mkdir(parents=True,exist_ok=True)
    exposed, pool = context["exposed"], context["pool"]
    raw = {(p["npi"], int(p["year"])): p for p in context["raw_panel"]}
    billing = context["billing_summary"]
    records = defaultdict(dict)

    def num(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except (ValueError, TypeError):
            return None

    for (n, t), p in raw.items():
        adj = num(p["payment_adjustment"])
        records[n][t] = {
            **{k: num(p[k]) for k in ["final_score", "quality_score", "pi_score", "ia_score", "cost_score"]},
            "positive_adjustment": None if adj is None else 100.0 * (adj > 0),
            "individual": p["participation"].strip().lower() == "individual",
            "hardship": p["extreme_hardship"].strip().lower() == "true",
        }

    def pre_info(n, event):
        pre = [p for p in billing.get(n, []) if p[0] < event]
        if not pre:
            return None, None
        states = Counter()
        for t, state, specialty, count in pre:
            states[state] += count
        state = min(states, key=lambda s: (-states[s], s))
        specialty = "neurosurgery" if any(p[2] == "Neurosurgery" for p in pre) else "orthopedic_spine"
        return state, specialty

    for p in exposed:
        p["event"] = int(p["deal_date"][:4])
        p["cluster"] = "practice:" + p["parent"]
        p["specialty"] = pre_info(p["npi"], p["event"])[1]
    ctl_cache = {}

    def controls_for(candidates):
        ids = tuple(p["npi"] for p in candidates)
        def ctl(state, event):
            key = (ids, state, event)
            if key not in ctl_cache:
                ctl_cache[key] = [dict(p, state=state, specialty=pre_info(p["npi"], event)[1])
                                  for p in candidates if pre_info(p["npi"], event)[0] == state]
            return ctl_cache[key]
        return ctl

    base_ctl = controls_for(pool)
    stacks = {}

    def calculate(label, exp=exposed, ctl=pool, outcome="final_score", opts=None):
        options = {"controls_fn": controls_for(ctl), **(opts or {})}
        s = did.build_stack(exp, ctl, records, outcome, options)
        if s is None:
            return {"status": "not_estimable", "reason": "No paired-score cell with treated and comparison support."}
        if s["counts"]["exposed_practices"] < 2:
            return {"status": "not_estimable", "reason": "Fewer than two exposed parent clusters.", "counts": s["counts"]}
        try:
            result = did.pooled(s)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {"status": "not_estimable", "reason": str(exc), "counts": s["counts"]}
        stacks[label] = s
        return {"status": "estimated", **result}

    result = {"specifications": PLAN, "primary": calculate("primary"),
              "secondary": {}, "specialty": {}, "robustness": {}}
    s = stacks["primary"]
    assert s["counts"] == saved["primary"]["counts"]
    for key in ("estimate", "se", "p"):
        assert np.isclose(result["primary"]["estimate"][key], saved["primary"]["estimate"][key], atol=1e-9)
    for outcome in SECONDARY:
        result["secondary"][outcome] = calculate(outcome, outcome=outcome)
    for specialty in ("neurosurgery", "orthopedic_spine"):
        exp = [p for p in exposed if p["specialty"] == specialty]
        def specialty_ctl(state, event, spec=specialty):
            return [p for p in base_ctl(state, event) if p["specialty"] == spec]
        result["specialty"][specialty] = calculate(specialty, exp=exp,
                                                  opts={"controls_fn": specialty_ctl})
    # Interaction tests compare exposed specialty effects within one pooled model.
    # This does not claim a difference from observing separate subgroup P values.
    ps = did.subset(s, s["k"] != 0)
    n_spec = {p["npi"]: p["specialty"] for p in exposed}
    dpost = ((ps["treated"] == 1) & (ps["k"] > 0)).astype(float)
    ns = np.array([n_spec.get(n) == "neurosurgery" for n in ps["npi"]], float)
    m = did.model(ps, [dpost, dpost * ns])
    result["specialty_interaction"] = {
        "neurosurgery_minus_orthopedic": did.summarize(m, 1),
        "note": "Exploratory pooled interaction; all 24 treated clusters determine t df. Subgroup estimates use specialty-matched controls and are different estimands.",
    }
    rb = result["robustness"]
    rb["exclude_2017"] = calculate("exclude_2017", opts={"drop_years": {2017}})
    rb["exclude_2020_2021_and_hardship"] = calculate("exclude_2020_2021_and_hardship",
                                                   opts={"drop_years": {2020, 2021}, "drop_hardship": True})
    rb["individual_reporting_only"] = calculate("individual_reporting_only", opts={"individual_only": True})
    # Source-level affiliation coding is retained; dated paper1 rosters are predeal verified.
    verified_bases = {"pre_deal_cms", "latest_predeal_cms", "pre_deal_roster",
                      "existing_paper1_dated_roster"}
    rb["predeal_verified_affiliation_only"] = calculate(
        "predeal_verified_affiliation_only",
        exp=[p for p in exposed if p["affiliation_basis"] in verified_bases])
    rb["affiliation_index_practices_only"] = calculate("affiliation_index_practices_only",
                                          exp=[p for p in exposed if p["source"] == "paper1"])
    rb["practice_mso_only"] = calculate("practice_mso_only",
        exp=[p for p in exposed if p["connection"] == "practice_or_linked_mso"])
    rb["practice_mso_only"]["note"] = "Excludes nine operator-linked professional-group surgeons; retains direct practice/linked-MSO acquisitions."
    placebo_exp = [dict(p, event=p["event"]-2, before=p["event"]) for p in exposed]
    def placebo_ctl(state, event):
        return [dict(p, before=event+2) for p in base_ctl(state, event)]
    rb["placebo_event_minus_2"] = calculate("placebo_event_minus_2", exp=placebo_exp,
                                           opts={"controls_fn": placebo_ctl})
    # Prevent parent-level treatment contamination and own-observation comparisons.
    def not_yet(state, event):
        current_parents = {p["cluster"] for p in exposed if p["state"] == state and p["event"] == event}
        return [dict(p, before=p["event"]) for p in exposed
                if pre_info(p["npi"], event)[0] == state and p["event"] >= event+2
                and p["cluster"] not in current_parents]
    rb["not_yet_treated_comparators"] = calculate("not_yet_treated_comparators",
                                                 opts={"controls_fn": not_yet})
    def fixed(label, required):
        new_records = defaultdict(dict)
        # Truncate to exactly common years; each unit must possess every required score.
        def complete(n, event):
            return all(records.get(n, {}).get(event+k, {}).get("final_score") is not None for k in required)
        # did.build_stack only uses these selected records for final_score.
        exp = [p for p in exposed if complete(p["npi"], p["event"])]
        def ctl(state, event):
            return [p for p in base_ctl(state, event) if complete(p["npi"], event)]
        # opts drop_years cannot express cell-relative years; use separate event-specific
        # records via a temporary usable wrapper, restoring it even on failure.
        original_usable = did.usable
        def usable(years, outcome, event, opts):
            return original_usable({t: v for t, v in years.items() if t-event in required},
                                   outcome, event, opts)
        did.usable = usable
        try:
            answer = calculate(label, exp=exp, opts={"controls_fn": ctl})
        finally:
            did.usable = original_usable
        answer["required_event_years"] = required
        return answer
    rb["fixed_composition_k_minus1_plus1_plus2"] = fixed("fixed_composition_k_minus1_plus1_plus2", [-1, 1, 2])
    rb["fixed_composition_k_minus1_plus1_to_plus4"] = fixed("fixed_composition_k_minus1_plus1_to_plus4", [-1, 1, 2, 3, 4])
    for label, ks in [("post_k1_k2", [1, 2]), ("post_k3_k4", [3, 4])]:
        original_usable = did.usable
        def usable(years, outcome, event, opts, ks=ks):
            return original_usable({t: v for t, v in years.items() if t < event or t-event in ks},
                                   outcome, event, opts)
        did.usable = usable
        try:
            rb[label] = calculate(label)
        finally:
            did.usable = original_usable
    rb["urology_window_2017_2020"] = calculate("urology_window_2017_2020",
                                             opts={"drop_years": {2021, 2022, 2023, 2024}})
    loo = {}
    for parent in sorted({p["parent"] for p in exposed}):
        loo[parent] = calculate("loo_" + parent, exp=[p for p in exposed if p["parent"] != parent])
    result["leave_one_parent_out"] = loo
    rb["leave_one_parent_out"] = {"status": "range_only",
        "min": min(p["estimate"]["estimate"] for p in loo.values() if p["status"] == "estimated"),
        "max": max(p["estimate"]["estimate"] for p in loo.values() if p["status"] == "estimated"),
        "note": "All 24 parent exclusions reported, not a hypothesis-test family."}

    def holm(block):
        items = sorted([(key, v["estimate"]["p"]) for key, v in block.items()
                        if v.get("status") == "estimated" and v["estimate"]["p"] is not None],
                       key=lambda x: x[1])
        previous = 0
        for i, (key, p) in enumerate(items):
            previous = max(previous, min(1.0, p * (len(items)-i)))
            block[key]["p_holm_within_family"] = previous
    for family in ("secondary", "specialty", "robustness"):
        block = result[family]
        holm(block)
    result.pop("specifications", None)
    dump(outdir / "supplemental_results.json", result)
    table = []
    for family in ("primary", "secondary", "specialty", "robustness", "leave_one_parent_out"):
        block = {"primary": result["primary"]} if family == "primary" else result[family]
        for key, value in block.items():
            e, c = value.get("estimate", {}), value.get("counts", {})
            table.append({"family": family, "analysis": key, "status": value["status"],
                          "estimate": e.get("estimate", ""), "ci_low": e.get("ci", ["", ""])[0],
                          "ci_high": e.get("ci", ["", ""])[1], "se": e.get("se", ""),
                          "p_nominal": e.get("p", ""), "p_holm_within_family": value.get("p_holm_within_family", ""),
                          "df": e.get("df", ""), "treated_parent_clusters": e.get("treated_clusters", ""),
                          "comparison_surgeons": c.get("comparison_surgeons", ""),
                          "exposed_surgeons": c.get("exposed_surgeons", ""),
                          "cells": c.get("cells", ""), "observations": c.get("observations", ""),
                          "reason_or_note": value.get("reason", value.get("note", ""))})
    csvout(outdir / "all_estimates.csv", table)
    for family in ("secondary", "specialty", "robustness", "leave_one_parent_out"):
        csvout(outdir / (family + ".csv"), [p for p in table if p["family"] == family])
    # Outcome-specific inclusion/attrition among the frozen primary exposed cohort.
    eligibility = []
    for label, outcome in [("primary", "final_score")] + [(k, k) for k in SECONDARY]:
        present = set(stacks[label]["npi"][stacks[label]["treated"] == 1]) if label in stacks else set()
        for p in exposed:
            obs = {t: v for t, v in records[p["npi"]].items()
                   if -4 <= t-p["event"] <= 4 and v[outcome] is not None}
            pre, post = any(t < p["event"] for t in obs), any(t > p["event"] for t in obs)
            status = "included" if p["npi"] in present else ("no_pre_score" if not pre else
                     "no_post_score" if not post else "no_comparison_cell_support")
            eligibility.append({"outcome": outcome, "npi": p["npi"], "parent": p["parent"],
                                "status": status, "pre_years": ";".join(str(t) for t in sorted(obs) if t < p["event"]),
                                "post_years": ";".join(str(t) for t in sorted(obs) if t > p["event"])})
    csvout(outdir / "secondary_eligibility.csv", eligibility)
    # Baseline table: latest observed pre-acquisition primary score within each cell.
    units = {}
    for cell, n, t, k, d, w in zip(s["cell"], s["npi"], s["year"], s["k"], s["treated"], s["weight"]):
        if k < 0:
            u = units.setdefault((cell, n), {"year": int(t), "treated": int(d), "weight": float(w)})
            u["year"] = max(u["year"], int(t))
    def baseline(n, u):
        p = raw[(n, u["year"])]
        event = int(next(cell for cell, nn in units if nn == n and units[(cell, nn)] is u).split(":")[1])
        adj = num(p["payment_adjustment"])
        participation = p["participation"].strip().lower()
        return {
            "Neurosurgeon, %": 100.0 * (pre_info(n, event)[1] == "neurosurgery"),
            "Baseline MIPS final score": num(p["final_score"]),
            "Baseline quality score": num(p["quality_score"]),
            "Positive payment adjustment, %": None if adj is None else 100.0*(adj > 0),
            "Group reporting, %": None if not participation else 100.0*(participation == "group"),
            "APM reporting, %": None if not participation else 100.0*("apm" in participation),
            "Individual reporting, %": None if not participation else 100.0*(participation == "individual"),
            "Practice size (clinicians)": num(p["practice_size"]),
            "Small practice, %": None if p["small_practice"].lower() not in ("true", "false") else 100.0*(p["small_practice"].lower()=="true"),
            "Rural, %": None if p["rural"].lower() not in ("true", "false") else 100.0*(p["rural"].lower()=="true"),
            "Years in Medicare": num(p["years_in_medicare"]),
            "Medicare patients": num(p["medicare_patients"]),
        }
    baseline_rows = []
    for (cell, n), u in units.items():
        baseline_rows.append({"cell": cell, "npi": n, **u, **baseline(n, u)})
    csvout(outdir / "baseline_unit_data.csv", baseline_rows)
    balance = []
    for characteristic in baseline_rows[0].keys():
        if characteristic in {"cell", "npi", "year", "treated", "weight"}:
            continue
        group = {}
        for d in (1, 0):
            all_rows = [p for p in baseline_rows if p["treated"] == d]
            valid = [p for p in all_rows if p[characteristic] is not None]
            v = np.array([p[characteristic] for p in valid], float)
            w = np.array([p["weight"] for p in valid], float)
            mean = float(np.average(v, weights=w)) if len(v) else None
            variance = float(np.average((v-mean)**2, weights=w)) if len(v) else None
            group[d] = {"mean": mean, "variance": variance, "observed": len(v),
                        "missing": len(all_rows)-len(v)}
        denom = np.sqrt((group[1]["variance"]+group[0]["variance"])/2) if all(
            group[d]["variance"] is not None for d in (1, 0)) else 0
        smd = (group[1]["mean"]-group[0]["mean"])/denom if denom > 0 else None
        balance.append({"characteristic": characteristic,
                        "exposed_mean_or_percent": group[1]["mean"],
                        "comparison_weighted_mean_or_percent": group[0]["mean"],
                        "standardized_difference": smd,
                        "exposed_observed_units": group[1]["observed"],
                        "exposed_missing_units": group[1]["missing"],
                        "comparison_observed_units": group[0]["observed"],
                        "comparison_missing_units": group[0]["missing"]})
    csvout(outdir / "balance_table.csv", balance)
    dump(outdir / "balance_table.json", balance)
    return result

if __name__=="__main__":
    run(Path(__file__).resolve().parent)
