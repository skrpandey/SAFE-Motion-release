"""R18. Verified persons and clusters.

1. Interval of the confidence-only MWI on the rows of the 18 test identifiers that the identity audit
   matched to 18 distinct persons (18 clusters; the scenario-C construction of
   experiments/cluster_scenarios.py), for the historical recipe (3 seeds), the corrected recipe
   (assignment 0, 3 seeds; 9 seeds), ST-GCN, regenerated assignments 1-3 and their pool; logit and raw.
2. Cluster description of the host test set: identifiers with SMM and non-SMM segments, segments per
   identifier (min, median, max), for all 40 and for the 18.
3. Leave-one-identifier-out range of the point MWI (corrected three-seed; historical three-seed).
4. Extra: BCa interval for the 40-identifier MWI (jackknife over identifiers for the
   acceleration) and the alternative clusterings A and B of the post hoc cluster protocol for the
   corrected three-seed cell (the historical cells are in paper/tables/cluster_scenarios.tex).
5. Distinct persons among the matched development identifiers (results/identity_audit).
Writes results/reanalyses/r18_*.csv and r18_summary.json.
    python3 experiments/reanalyses/r18_verified_persons.py
"""
import csv
import json

import numpy as np
import yaml
from scipy.stats import norm

import common as C

CELLS = [("hist3", "historical recipe, 3 seeds"), ("corr3", "corrected recipe, assignment 0, 3 seeds"),
         ("corr9", "corrected recipe, assignment 0, 9 seeds"), ("stgcn", "ST-GCN, corrected recipe, 3 seeds"),
         ("w1", "corrected recipe, assignment 1, 3 seeds"), ("w2", "corrected recipe, assignment 2, 3 seeds"),
         ("w3", "corrected recipe, assignment 3, 3 seeds"), ("w123", "assignments 1-3 pooled, 9 seeds")]


def audit_identifiers():
    """Matched test identifiers and matched development identifiers with their persons."""
    man = list(csv.DictReader(open("results/identity_audit/manifest_ids.csv")))
    summ = json.load(open("results/identity_audit/summary.json"))
    test = {r["annotated_id"]: r["canonical_person"] for r in man if r["partition"] == "test" and r["status"] == "accepted"}
    test.update({k: str(v) for k, v in summ["pass2"]["resolved_test_ids"].items()})
    dev_parts = ("train", "validation", "train(short only)")
    dev_all = [r for r in man if r["partition"] in dev_parts]
    dev = {r["annotated_id"]: r["canonical_person"] for r in dev_all if r["status"] == "accepted"}
    return test, dev, len(dev_all), man


def point_mwi(sc, g, sel, scale="logit"):
    f = C.logit if scale == "logit" else (lambda a: a)
    y = sc.y[sel]

    def R(m):
        return np.mean([f(C.M.auc(sc.col(m["name"], f"armruns_S_mid_m{ms}")[sel], y))
                        - f(C.M.auc(sc.col(m["name"], f"armruns_I_mid_m{ms}")[sel], y)) for ms in C.MASK_SEEDS])
    return np.mean([R(m) for m in g["S"]]) - np.mean([R(m) for m in g["I"]])


def bca(boot, theta, jack):
    """BCa percentile interval (Efron 1987) from bootstrap draws and leave-one-cluster-out values."""
    b = np.asarray(boot)
    b = b[np.isfinite(b)]
    z0 = norm.ppf(np.mean(b < theta) + 0.5 * np.mean(b == theta))
    d = jack.mean() - jack
    a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)
    out = []
    for q in (0.025, 0.975):
        z = norm.ppf(q)
        out.append(float(np.percentile(b, 100 * norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))
    return out, float(z0), float(a)


def main():
    cfg = yaml.safe_load(open(C.CLUSTER_CFG))
    test_ids, dev_ids, n_dev_all, man = audit_identifiers()
    assert sorted(int(i) for i in test_ids) == sorted(cfg["matched_identifiers"]), (sorted(test_ids), cfg["matched_identifiers"])
    assert len(set(test_ids.values())) == 18
    cells = C.confonly_cells()
    sc = C.Scores([m for g in cells.values() for w in ("S", "I") for m in g[w]])
    assert set(sc.child) == {str(i) for i in cfg["test_identifiers"]}
    matched = sorted(test_ids, key=int)
    sel18 = np.flatnonzero(np.isin(sc.child, matched))
    summary = {}

    # 1. intervals on the 18 matched identifiers (and the 40-identifier values beside them)
    full, e18 = C.Engine(sc), C.Engine(sc, sel18)
    single = sum(1 for idx in e18.rs.boots[1:] if len(np.unique(e18.rs.y[idx])) < 2)
    rows = []
    for key, label in CELLS:
        rec = {"cell": key, "description": label, "seeds_per_world": len(cells[key]["S"])}
        for tag, eng in (("40", full), ("18", e18)):
            lg, rS, rI, _, _ = eng.cell(cells[key])
            rw = eng.cell(cells[key], scale="raw")[0]
            p, lo, hi = C.ci(lg)
            rp, rlo, rhi = C.ci(rw)
            rec.update({f"mwi_logit_{tag}": round(p, 6), f"lo_logit_{tag}": round(lo, 6), f"hi_logit_{tag}": round(hi, 6),
                        f"mwi_raw_{tag}": round(rp, 6), f"lo_raw_{tag}": round(rlo, 6), f"hi_raw_{tag}": round(rhi, 6),
                        f"R_fS_{tag}": C.fmt(rS), f"R_fI_{tag}": C.fmt(rI)})
        rows.append(rec)
        print(f"{key:6s} 40: {rec['mwi_logit_40']:+.3f} [{rec['lo_logit_40']:+.3f}, {rec['hi_logit_40']:+.3f}]  "
              f"18: {rec['mwi_logit_18']:+.3f} [{rec['lo_logit_18']:+.3f}, {rec['hi_logit_18']:+.3f}] "
              f"raw {rec['mwi_raw_18']:+.3f} [{rec['lo_raw_18']:+.3f}, {rec['hi_raw_18']:+.3f}]", flush=True)
    C.write_csv("r18_matched18_intervals.csv", rows,
                "R18: confidence-only MWI on all 40 host test identifiers and on the rows of the 18 identifiers matched to 18\n"
                f"distinct persons (18 clusters; {len(sel18)} rows); identifier-clustered percentile bootstrap, 2,000 resamples,\n"
                f"RowSet seed 0, mask mean; resamples of the 18 with a single class: {single}")
    summary["matched18"] = {"identifiers": [int(i) for i in matched], "rows": int(len(sel18)),
                            "smm_rows": int(sc.y[sel18].sum()), "resamples_single_class": single}

    # 2. cluster description
    def describe(ids_subset):
        n, smm, non = [], [], []
        for c in ids_subset:
            k = sc.child == c
            n.append(int(k.sum())); smm.append(int(sc.y[k].sum())); non.append(int((sc.y[k] == 0).sum()))
        n, smm, non = map(np.array, (n, smm, non))
        return {"identifiers": len(n), "rows": int(n.sum()), "smm_rows": int(smm.sum()),
                "with_smm": int((smm > 0).sum()), "with_non_smm": int((non > 0).sum()),
                "with_both": int(((smm > 0) & (non > 0)).sum()),
                "segments_min_median_max": [int(n.min()), float(np.median(n)), int(n.max())],
                "smm_segments_min_median_max": [int(smm.min()), float(np.median(smm)), int(smm.max())],
                "non_smm_segments_min_median_max": [int(non.min()), float(np.median(non)), int(non.max())],
                "largest_identifier_share_of_rows": float(n.max() / n.sum()),
                "five_largest_share_of_rows": float(np.sort(n)[-5:].sum() / n.sum()),
                "smm_rows_largest_identifier_share": float(smm.max() / smm.sum())}
    all40 = sorted(set(sc.child), key=int)
    summary["clusters_40"] = describe(all40)
    summary["clusters_18"] = describe(matched)
    per_id = [{"identifier": c, "matched_person": test_ids.get(c, ""), "segments": int((sc.child == c).sum()),
               "smm": int(sc.y[sc.child == c].sum()), "non_smm": int((sc.y[sc.child == c] == 0).sum())} for c in all40]

    # 3. leave-one-identifier-out (point estimates) and 4. BCa
    loo = {}
    for key in ("corr3", "hist3"):
        vals, vals_raw = [], []
        for c in all40:
            keep = np.flatnonzero(sc.child != c)
            vals.append(point_mwi(sc, cells[key], keep)); vals_raw.append(point_mwi(sc, cells[key], keep, "raw"))
        vals, vals_raw = np.array(vals), np.array(vals_raw)
        lg = full.cell(cells[key])[0]
        (blo, bhi), z0, a = bca(lg[1:], lg[0], vals)
        loo[key] = {"logit_min": float(vals.min()), "logit_max": float(vals.max()),
                    "argmin_identifier": all40[int(vals.argmin())], "argmax_identifier": all40[int(vals.argmax())],
                    "raw_min": float(vals_raw.min()), "raw_max": float(vals_raw.max()),
                    "bca_logit": [blo, bhi], "bca_z0": z0, "bca_acceleration": a,
                    "percentile_logit": list(C.M.ci(lg[1:])), "point": float(lg[0])}
        for r, v, vr in zip(per_id, vals, vals_raw):
            r[f"loo_mwi_logit_{key}"] = round(float(v), 6)
            r[f"loo_mwi_raw_{key}"] = round(float(vr), 6)
        print(f"LOO {key}: logit {vals.min():+.3f} (drop {loo[key]['argmin_identifier']}) to {vals.max():+.3f} "
              f"(drop {loo[key]['argmax_identifier']}); BCa [{blo:+.3f}, {bhi:+.3f}] z0 {z0:+.3f} a {a:+.4f}", flush=True)
    summary["leave_one_identifier_out"] = loo
    C.write_csv("r18_per_identifier.csv", per_id,
                "R18: host test identifiers (1,966 rows), matched person of the identity audit (pass 1 + pass 2),\n"
                "segment counts by class, and the point MWI with that identifier left out (logit and raw, mask mean)")

    # alternative clusterings A (unmatched merged in adjacent pairs) and B (unmatched as one cluster), corrected 3 seeds
    pairs = {}
    for a_, b_ in cfg["scenarios"]["A_pairwise"]["pairs"]:
        pairs[str(a_)] = pairs[str(b_)] = f"pair_{a_}_{b_}"
    unmatched = {str(i) for i in cfg["unmatched_identifiers"]}
    alt = {}
    for scen, cl in (("A_pairwise", np.array([pairs.get(c, c) for c in sc.child])),
                     ("B_one_cluster", np.array(["unmatched" if c in unmatched else c for c in sc.child]))):
        e = C.Engine(sc, clusters=cl)
        alt[scen] = {"clusters": e.n_clusters}
        for key in ("corr3", "hist3"):
            alt[scen][key] = {"logit": C.fmt(e.cell(cells[key])[0]), "raw": C.fmt(e.cell(cells[key], scale="raw")[0])}
        print(scen, alt[scen], flush=True)
    summary["alternative_clusterings"] = alt

    # 5. persons among matched development identifiers
    persons = set(dev_ids.values())
    multi = {p: sorted([i for i, q in dev_ids.items() if q == p], key=int) for p in persons
             if sum(1 for q in dev_ids.values() if q == p) > 1}
    summary["development"] = {"development_identifiers": n_dev_all, "matched_identifiers": len(dev_ids),
                              "distinct_persons": len(persons), "persons_with_several_identifiers": multi,
                              "persons_shared_with_matched_test": sorted(persons & set(test_ids.values())),
                              "matched_by_partition": {p: sum(1 for r in man if r["partition"] == p and r["status"] == "accepted")
                                                       for p in ("train", "validation", "train(short only)")}}
    print("development:", summary["development"], flush=True)
    C.write_json("r18_summary.json", summary)


if __name__ == "__main__":
    main()
