"""Single spine-only weighted stacked final-score model and event study."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import codes, fit_fe, summarize, wald

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scope import COHORT, PANEL, RESULTS  # noqa: E402
K_MIN, K_MAX = -4, 4
OUTCOMES = {"final_score": "MIPS final score"}


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_panel():
    records = defaultdict(dict)
    for r in csv.DictReader(PANEL.open()):
        records[r["npi"]][int(r["year"])] = {
            "final_score": num(r["final_score"]),
            "individual": r["participation"].strip().lower() == "individual",
            "hardship": r["extreme_hardship"].strip().lower() == "true"}
    return records


def load_people():
    exposed = [dict(r, event=int(r["deal_date"][:4]), cluster="practice:" + r["parent"])
               for r in csv.DictReader((COHORT / "exposed_surgeons.csv").open())]
    pool = list(csv.DictReader((COHORT / "comparison_pool.csv").open()))
    return exposed, pool


def usable(years, outcome, event, opts):
    keep = {}
    for t, v in years.items():
        k = t - event
        if not (K_MIN <= k <= K_MAX) or v[outcome] is None:
            continue
        if opts.get("before") is not None and t >= opts["before"]:
            continue
        if t in opts.get("drop_years", ()) or (opts.get("drop_hardship") and v["hardship"]):
            continue
        if opts.get("individual_only") and not v["individual"]:
            continue
        keep[t] = v[outcome]
    pre = [t for t in keep if t < event]
    post = [t for t in keep if t > event]
    return keep if pre and post else None


def build_stack(exposed, pool, records, outcome, opts):
    cells = defaultdict(list)
    for p in exposed:
        cells[(p["state"], p["event"])].append(p)
    controls_by_state = defaultdict(list)
    for c in pool:
        controls_by_state[c["state"]].append(c)
    rows, units_t, units_c = [], {}, {}
    for (state, event), people in sorted(cells.items()):
        cell = f"{state}:{event}"
        treated_rows = []
        for p in people:
            obs = usable(records.get(p["npi"], {}), outcome, event, {**opts, "before": p.get("before")})
            if obs is None:
                continue
            units_t[(cell, p["npi"])] = p
            treated_rows += [(cell, p["npi"], t, y, 1, t - event, p["cluster"]) for t, y in obs.items()]
        if not treated_rows:
            continue
        control_rows = []
        for c in opts.get("controls_fn", lambda s, e: controls_by_state[s])(state, event):
            obs = usable(records.get(c["npi"], {}), outcome, event, {**opts, "before": c.get("before")})
            if obs is None:
                continue
            units_c[(cell, c["npi"])] = c
            control_rows += [(cell, c["npi"], t, y, 0, t - event, c["cluster"]) for t, y in obs.items()]
        if not control_rows:
            for key in [k for k in units_t if k[0] == cell]:
                del units_t[key]
            continue
        rows += treated_rows + control_rows
    nt, nc = len(units_t), len(units_c)
    if nt == 0:
        return None
    nt_cell, nc_cell = defaultdict(int), defaultdict(int)
    for cell, _ in units_t:
        nt_cell[cell] += 1
    for cell, _ in units_c:
        nc_cell[cell] += 1
    cols = list(zip(*rows))
    stack = {k: np.array(v) for k, v in zip(("cell", "npi", "year", "y", "treated", "k", "cluster"), cols)}
    stack["weight"] = np.array([1.0 if d else (nt_cell[c] / nt) / (nc_cell[c] / nc)
                                for c, d in zip(stack["cell"], stack["treated"])])
    stack["counts"] = {"exposed_surgeons": len({n for _, n in units_t}),
                       "exposed_practices": len({p["cluster"] for p in units_t.values()}),
                       "exposed_neurosurgeons": len({n for (_, n), p in units_t.items() if p.get("specialty") == "neurosurgery"}),
                       "comparison_surgeons": len({n for _, n in units_c}), "comparison_cell_units": nc,
                       "cells": len(nt_cell), "observations": int(len(stack["y"]))}
    return stack


def model(stack, columns):
    groups = [codes(np.char.add(np.char.add(stack["cell"].astype(str), "|"), stack["npi"].astype(str))),
              codes(np.char.add(np.char.add(stack["cell"].astype(str), "|"), stack["year"].astype(str)))]
    return fit_fe(stack["y"], columns, groups, stack["weight"], stack["cluster"], stack["treated"] == 1)


def subset(stack, mask):
    out = {k: v[mask] for k, v in stack.items() if isinstance(v, np.ndarray)}
    out["counts"] = stack["counts"]
    return out


def group_means(stack):
    out = {}
    for label, d in (("exposed", 1), ("comparison", 0)):
        units = defaultdict(lambda: {"pre": [], "post": [], "w": 0.0})
        sel = stack["treated"] == d
        for cell, npi, k, y, w in zip(stack["cell"][sel], stack["npi"][sel], stack["k"][sel], stack["y"][sel],
                                      stack["weight"][sel]):
            u = units[(cell, npi)]
            u["pre" if k < 0 else "post"].append(y)
            u["w"] = w
        w = np.array([u["w"] for u in units.values()])
        pre = np.array([np.mean(u["pre"]) for u in units.values()])
        post = np.array([np.mean(u["post"]) for u in units.values()])
        out[label] = {"surgeons": len(units), "pre_mean": float(np.average(pre, weights=w)),
                      "post_mean": float(np.average(post, weights=w)), "change": float(np.average(post - pre, weights=w))}
    return out


def pooled(stack):
    s = subset(stack, stack["k"] != 0)
    post = ((s["treated"] == 1) & (s["k"] > 0)).astype(float)
    m = model(s, [post])
    out = {"counts": stack["counts"], "estimate": summarize(m), "group_means": group_means(s)}
    return out


def event_study(stack):
    d = stack["treated"] == 1
    ks = [k for k in range(K_MIN, K_MAX + 1) if k != -1 and np.any(d & (stack["k"] == k))]
    m = model(stack, [(d & (stack["k"] == k)).astype(float) for k in ks])
    coefs = {k: summarize(m, i) for i, k in enumerate(ks)}
    pre_idx = [i for i, k in enumerate(ks) if k <= -2]
    return {"coefficients": {str(k): v for k, v in coefs.items()}, "reference": -1,
            "pretrend_joint_test": wald(m, pre_idx) if pre_idx else None}

