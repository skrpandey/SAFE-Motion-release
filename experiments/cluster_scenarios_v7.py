"""Post hoc, illustrative test-cluster scenarios; definitions in
configs/protocols/cluster_scenarios_post_hoc.yaml (fixed before computation).

The primary estimator is kept (paired identifier-clustered bootstrap, 2,000 resamples, seeds and
masks averaged inside each resample); only the clustering of the 40 test identifiers changes.
A: unmatched identifiers merged in adjacent pairs (29 clusters). B: all 22 unmatched identifiers as
one cluster (19 clusters, not a bound). C: the 18 matched identifiers only. Quantities: the five
Table I interactions and the two v12 Group DRO contrasts, logit and raw.
Writes results/v7/analysis/cluster_scenarios.csv and paper/tables/cluster_scenarios.tex.
"""
import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                  # noqa: E402
from safe_motion.paths import guard_output                            # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit          # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                    # noqa: E402

NB = 2000
CFG = "configs/protocols/cluster_scenarios_post_hoc.yaml"
TABLE_I = [("v7", "runs", "mid"), ("v7", "runs", "low"), ("v7", "whole", "mid"), ("v8", "frozen", "mid"), ("v13", "armruns", "mid")]
MECH = {"runs": "missing runs", "whole": "whole-joint absence", "frozen": "frozen frames", "armruns": "arm confidence"}


def models():
    out = []
    for d in ("results/v7", "results/v13", "results/v12"):
        for m in json.load(open(f"{d}/registry.json")):
            p = f"{d}/scores/{m['name']}.csv"
            if not os.path.exists(p) or m["world"] not in ("S", "I") or m.get("world_seed", 0) != 0:
                continue
            key = (m["proto"], m["mechanism"], m["strength"], m["cell"])
            if key[:3] in TABLE_I and m["cell"] == "A_none" or (m["proto"] == "v12" and m["cell"] in ("A_none", "C_groupdro")):
                out.append(dict(m, path=p))
    return out


def main():
    cfg = yaml.safe_load(open(CFG))
    unmatched = [str(i) for i in cfg["unmatched_identifiers"]]
    matched = [str(i) for i in cfg["matched_identifiers"]]
    pairs = {str(a): f"pair_{a}_{b}" for a, b in cfg["scenarios"]["A_pairwise"]["pairs"]}
    pairs.update({str(b): f"pair_{a}_{b}" for a, b in cfg["scenarios"]["A_pairwise"]["pairs"]})
    reg = models()
    loaded = {m["name"]: load_scores(m["path"]) for m in reg}
    common = set.intersection(*[set(meta["identifier"]) for meta, _ in loaded.values()])
    meta0 = next(iter(loaded.values()))[0]
    keep = [k for k, i in enumerate(meta0["identifier"]) if i in common]
    ids = [meta0["identifier"][k] for k in keep]
    y = np.array([int(meta0["label"][k]) for k in keep])
    child = np.array([str(meta0["child_id"][k]) for k in keep])
    assert set(child) == set(str(i) for i in cfg["test_identifiers"]), sorted(set(child))
    S = {}
    for n, (meta, sc) in loaded.items():
        pos = {i: k for k, i in enumerate(meta["identifier"])}
        order = [pos[i] for i in ids]
        assert np.array_equal(np.array([int(meta["label"][k]) for k in order]), y), n
        S[n] = {c: v[order] for c, v in sc.items()}

    def cluster_map(name):
        if name == "primary":
            return child, np.ones(len(ids), bool)
        if name == "A_pairwise":
            return np.array([pairs.get(c, c) for c in child]), np.ones(len(ids), bool)
        if name == "B_one_cluster":
            return np.array(["unmatched" if c in unmatched else c for c in child]), np.ones(len(ids), bool)
        if name == "C_matched_only":
            sel = np.isin(child, matched)
            return child[sel], sel
        raise KeyError(name)

    rows = []
    for scen in ("primary", "A_pairwise", "B_one_cluster", "C_matched_only"):
        cl, sel = cluster_map(scen)
        rs = RowSet([i for i, s in zip(ids, sel) if s], y[sel], cl, NB)
        sizes = np.unique(cl, return_counts=True)[1]
        single_class = sum(1 for idx in rs.boots[1:] if len(np.unique(rs.y[idx])) < 2)
        cache = {}

        def R(name, mech, st, raw):
            key = (name, mech, st, raw)
            if key not in cache:
                f = (lambda a: a) if raw else logit
                cache[key] = np.mean([f(rs.aucs(S[name][f"{mech}_S_{st}_m{ms}"][sel])) - f(rs.aucs(S[name][f"{mech}_I_{st}_m{ms}"][sel]))
                                      for ms in MASK_SEEDS], axis=0)
            return cache[key]

        def SE(proto, mech, st, cell, raw):
            g = {w: sorted(m["name"] for m in reg if (m["proto"], m["mechanism"], m["strength"], m["cell"], m["world"]) == (proto, mech, st, cell, w))
                 for w in ("S", "I")}
            assert len(g["S"]) == 3 and len(g["I"]) == 3, (proto, mech, st, cell, g)
            return np.mean([R(n, mech, st, raw) for n in g["S"]], 0) - np.mean([R(n, mech, st, raw) for n in g["I"]], 0)

        def rec(quantity, arr, raw_arr):
            lo, hi = M.ci(arr[1:]); rlo, rhi = M.ci(raw_arr[1:])
            rows.append({"scenario": scen, "n_clusters": len(sizes), "cluster_sizes_min_median_max": f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}",
                         "n_rows": int(sel.sum()), "resamples_single_class": single_class, "quantity": quantity,
                         "point": round(float(arr[0]), 4), "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                         "raw_point": round(float(raw_arr[0]), 4), "raw_ci_lo": round(float(rlo), 4), "raw_ci_hi": round(float(rhi), 4)})
            print(rows[-1], flush=True)

        for proto, mech, st in TABLE_I:
            rec(f"MWI {MECH[mech]} {st}", SE(proto, mech, st, "A_none", False), SE(proto, mech, st, "A_none", True))
        for mech, base_proto in (("runs", "v7"), ("frozen", "v8")):
            d = SE("v12", mech, "mid", "C_groupdro", False) - SE(base_proto, mech, "mid", "A_none", False)
            dr = SE("v12", mech, "mid", "C_groupdro", True) - SE(base_proto, mech, "mid", "A_none", True)
            rec(f"Group DRO - none, {MECH[mech]}", d, dr)
    out = "results/v7/analysis/cluster_scenarios.csv"
    guard_output(out)
    with open(out, "w", newline="") as f:
        f.write("# post hoc: illustrative hypothetical test-cluster scenarios, definitions in "
                "configs/protocols/cluster_scenarios_post_hoc.yaml; primary estimator unchanged; identity remains open\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)

    # supplement table: one row per quantity, one column per scenario (logit; raw in the CSV)
    scen_lab = {"primary": "40 identifiers (published)", "A_pairwise": "A: 29 clusters", "B_one_cluster": "B: 19 clusters", "C_matched_only": "C: 18 matched"}
    qs = []
    for r in rows:
        if r["quantity"] not in qs:
            qs.append(r["quantity"])
    L = ["% generated by experiments/cluster_scenarios_v7.py (post hoc); do not edit",
         r"\begin{tabular}{l" + "c" * len(scen_lab) + "}", r"\toprule",
         "Quantity (logit) & " + " & ".join(scen_lab.values()) + r" \\", r"\midrule"]
    for q in qs:
        cells = []
        for s in scen_lab:
            r = next(x for x in rows if x["scenario"] == s and x["quantity"] == q)
            cells.append(f"{r['point']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]".replace("-", "$-$"))
        L.append(f"{q} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    tp = "paper/tables/cluster_scenarios.tex"
    guard_output(tp); open(tp, "w").write("\n".join(L) + "\n"); print("wrote", tp)


if __name__ == "__main__":
    main()
