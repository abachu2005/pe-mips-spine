"""Exposed-surgeon cohort and comparison pool.

Inputs: hand-adjudicated deal/organization decisions (cohort/deal_decisions.csv), the professional affiliation index, the Doctors & Clinicians affiliation panel and the spine frame.
Outputs: cohort/exposed_surgeons.csv, cohort/comparison_pool.csv, cohort/flow.json.
"""
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import csv
import json

from scope import AFFILIATIONS, COHORT, EXPANSION, FRAME, PANEL, V2

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT.parent / "affiliation_index"
QUEUE = EXPANSION / "match_queue.csv"
WINDOW = ("2016-01-01", "2023-12-31")
EARLY_POST_DAYS = 365
ARCHIVE_START = date(2019, 4, 17)
TIN_TOLERANCE = 0.10
CONNECTIONS = {"practice_or_linked_mso", "operator_professional_group"} if V2 else {"practice_or_linked_mso"}


def day(text):
    return date.fromisoformat(text[:10])


def frame():
    spec, states = {}, defaultdict(Counter)
    for r in csv.DictReader(FRAME.open()):
        spec[r["Rndrng_NPI"]] = "neurosurgery" if r["Rndrng_Prvdr_Type"] == "Neurosurgery" else spec.get(
            r["Rndrng_NPI"], "orthopedic_spine")
        states[r["Rndrng_NPI"]][r["Rndrng_Prvdr_State_Abrvtn"]] += 1
    return spec, {n: min(c, key=lambda state: (-c[state], state)) for n, c in states.items() if c}


def practice_sizes():
    sizes = {}
    for r in csv.DictReader(PANEL.open()):
        if r["practice_size"]:
            sizes[(r["npi"], int(r["year"]))] = float(r["practice_size"])
    return sizes


def tin_match(npi, members, deal_day, sizes):
    """Before the CMS archive: same pre-deal TIN size as the modal size of the org's other members."""
    year = deal_day.year - 1
    own = sizes.get((npi, year))
    others = [sizes[(m, year)] for m in members if m != npi and (m, year) in sizes]
    if own is None or not others:
        return False
    counts = Counter(others)
    modal = min(counts, key=lambda size: (-counts[size], size))
    return abs(own - modal) <= TIN_TOLERANCE * modal


def affiliation_index():
    by_org = defaultdict(lambda: defaultdict(set))
    by_npi = defaultdict(lambda: defaultdict(set))
    for r in csv.DictReader(AFFILIATIONS.open()):
        if r["org_pac_id"]:
            by_org[r["org_pac_id"]][r["snapshot"]].add(r["npi"])
            by_npi[r["npi"]][r["snapshot"]].add(r["org_pac_id"])
    return by_org, by_npi


def basis(snapshots_present, deal_day):
    """Pre-deal affiliation basis from the snapshot dates where the NPI was in the org."""
    dates = sorted(day(s) for s in snapshots_present)
    if any(d <= deal_day for d in dates):
        return "pre_deal_cms"
    if any(deal_day < d <= deal_day + timedelta(days=EARLY_POST_DAYS) for d in dates):
        return "early_post_deal_cms"
    return None


def main():
    spec, home_state = frame()
    by_org, by_npi = affiliation_index()
    sizes = practice_sizes()
    decisions = list(csv.DictReader((COHORT / "deal_decisions.csv").open()))
    index = list(csv.DictReader((INDEX_DIR / "professional_index.csv").open()))
    flow_frame = len(spec)
    seen = set()
    for r in index:
        key = (r["organization_id"], r["ownership_start"][:10])
        if r["connection_type"] not in CONNECTIONS or not r["organization_id"].isdigit() or key in seen:
            continue
        seen.add(key)
        decisions.append({"deal_ref": "affiliation_index:" + r["evidence_reference"], "org_pac_id": r["organization_id"],
                          "org_name": r["organization_name"], "state": r["state"],
                          "deal_date": r["ownership_start"][:10], "sponsor": r["sponsor"],
                          "parent": r["operator_parent"], "decision": "include", "reason_code": "",
                          "connection": r["connection_type"]})
    flow = Counter()
    flow["operative_spine_frame_surgeons"] = flow_frame
    candidates = []

    for d in decisions:
        flow["deal_org_pairs_reviewed"] += 1
        if d["decision"] != "include":
            flow[f"pair_excluded:{d['reason_code']}"] += 1
            continue
        if not (WINDOW[0] <= d["deal_date"] <= WINDOW[1]):
            flow["pair_excluded:deal_outside_window"] += 1
            continue
        flow["deal_org_pairs_included"] += 1
        deal_day = day(d["deal_date"])
        members = defaultdict(set)
        for snapshot, npis in by_org.get(d["org_pac_id"], {}).items():
            for npi in npis:
                members[npi].add(snapshot)
        for npi, snaps in members.items():
            if npi not in spec:
                continue
            b = basis(snaps, deal_day)
            flow["pitchbook_surgeon_candidates"] += 1
            if b is None and deal_day < ARCHIVE_START and min(day(s) for s in snaps) <= date(2019, 12, 31) \
                    and tin_match(npi, members, deal_day, sizes):
                b = "pre_archive_tin_match"
            if b is None:
                flow["pitchbook_surgeon_excluded:joined_after_deal"] += 1
                continue
            candidates.append({"npi": npi, "specialty": spec[npi], "practice": d["org_pac_id"],
                               "practice_name": d["org_name"], "parent": d["parent"], "state": d["state"],
                               "deal_date": d["deal_date"], "sponsor": d["sponsor"], "source": "pitchbook",
                               "affiliation_basis": b, "deal_ref": d["deal_ref"],
                               "connection": d.get("connection") or ("operator_professional_group"
                                                                     if d["deal_ref"] == "manual|lifepoint-apollo"
                                                                     else "practice_or_linked_mso")})

    for r in index:
        if r["connection_type"] not in CONNECTIONS:
            continue
        flow["affiliation_index_practice_surgeons"] += 1
        if r["npi"] not in spec:
            flow["affiliation_index_excluded:not_in_spine_frame"] += 1
            continue
        start = r["ownership_start"][:10]
        if not (WINDOW[0] <= start <= WINDOW[1]):
            flow["affiliation_index_excluded:pe_entry_outside_window"] += 1
            continue
        deal_day = day(start)
        observed = day(r["observation_lower"])
        if observed <= deal_day:
            b = "pre_deal_roster"
        elif observed <= deal_day + timedelta(days=EARLY_POST_DAYS):
            b = "early_post_deal_roster"
        else:
            b = basis([s for s, orgs in by_npi.get(r["npi"], {}).items() if r["organization_id"] in orgs], deal_day)
        if b is None:
            flow["affiliation_index_excluded:no_pre_deal_affiliation"] += 1
            continue
        candidates.append({"npi": r["npi"], "specialty": spec[r["npi"]], "practice": r["organization_id"],
                           "practice_name": r["organization_name"], "parent": r["operator_parent"],
                           "state": r["state"], "deal_date": start, "sponsor": r["sponsor"], "source": "affiliation_index",
                           "affiliation_basis": b, "deal_ref": r["evidence_reference"],
                           "connection": r["connection_type"]})

    exposed = {}
    for c in sorted(candidates, key=lambda c: (
            c["deal_date"], c["source"] != "affiliation_index", c["npi"], c["parent"], c["practice"], c["deal_ref"])):
        if c["npi"] not in exposed:
            exposed[c["npi"]] = c
    # First ownership is established before billing eligibility is evaluated.
    # A post-ownership spine procedure cannot reset treatment to a later acquisition.
    billing_years = defaultdict(set)
    for r in csv.DictReader(FRAME.open()):
        billing_years[r["Rndrng_NPI"]].add(int(r["year"]))
    spine_excluded = []
    for npi, c in list(exposed.items()):
        if not any(y < int(c["deal_date"][:4]) for y in billing_years[npi]):
            flow["exposed_excluded:no_preacquisition_spine_billing"] += 1
            spine_excluded.append(dict(c, exclusion="no_preacquisition_spine_billing"))
            del exposed[npi]
    if spine_excluded:
        with (COHORT / "spine_eligibility_exclusions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(spine_excluded[0]))
            w.writeheader()
            w.writerows(spine_excluded)
    flow["exposed_surgeons"] = len(exposed)
    flow["exposed_practices"] = len({c["parent"] for c in exposed.values()})

    linked = set()
    flagged_orgs = set()
    for q in csv.DictReader(QUEUE.open()):
        linked |= set(q["spine_surgeon_npis"].split(";"))
        flagged_orgs.add(q["org_pac_id"])
    linked |= {r["npi"] for r in index}
    flagged_orgs |= {r["organization_id"] for r in index}
    flagged_orgs |= {d["org_pac_id"] for d in decisions}
    for org in flagged_orgs:
        for npis in by_org.get(org, {}).values():
            linked |= npis
    linked |= {r["npi"] for r in csv.DictReader((INDEX_DIR / "comparator_exclusions.csv").open())}
    pool = []
    for npi in spec:
        if npi in linked or npi in exposed:
            continue
        orgs = Counter(o for s in by_npi.get(npi, {}).values() for o in s)
        pool.append({"npi": npi, "specialty": spec[npi], "state": home_state[npi],
                     "cluster": "org:" + min(orgs, key=lambda o: (-orgs[o], o)) if orgs else "npi:" + npi})
    flow["comparison_pool"] = len(pool)
    flow["frame_excluded_as_linked"] = len([n for n in spec if n in linked and n not in exposed])

    fields = ["npi", "specialty", "practice", "practice_name", "parent", "state", "deal_date", "sponsor",
              "source", "affiliation_basis", "deal_ref", "connection"]
    with (COHORT / "exposed_surgeons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(exposed.values(), key=lambda c: (c["parent"], c["npi"])))
    with (COHORT / "comparison_pool.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npi", "specialty", "state", "cluster"])
        w.writeheader()
        w.writerows(sorted(pool, key=lambda c: c["npi"]))
    (COHORT / "flow.json").write_text(json.dumps(dict(sorted(flow.items())), indent=2) + "\n")
    print(json.dumps(dict(sorted(flow.items())), indent=2))
    print(Counter(c["specialty"] for c in exposed.values()), Counter(c["affiliation_basis"] for c in exposed.values()))


if __name__ == "__main__":
    main()
