"""Protocol v19 analysis (configs/protocols/mapped_persons.yaml): Experiment A, verified-person split.

Six CompactCTRGCN no-remedy fits (Conditions S and I, seeds 0-2) trained on matched development
persons only, scored on all host test rows by score_confonly_corrected.py --proto v19; this analysis uses the rows
of the 18 matched test persons. Gate G19_clean, primary P19_1 (TTI interval above zero on the logit
and raw scales), the reported quantities and the descriptive comparison D19_2 with the published
corrected three-seed cell (v15conf A_none) on the same rows. Estimator as analyze_confonly_primary.py: AUROC
per fitted model and test mask inside each identifier-clustered resample (RowSet of analyze_operators,
seed 0, 2,000 resamples), transformed, then averaged over masks and seeds.
Writes <out>/{effects.csv, criteria.json}.
    python experiments/analyze_mapped_persons.py [--v19 results/mapped_persons] [--out results/mapped_persons/analysis]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                                  # noqa: E402
from experiments.analyze_operators import RowSet, load_scores, logit, p_inversion, R_MARGIN  # noqa: E402
from experiments.analyze_operators_masks import MASK_SEEDS                                   # noqa: E402

PROTO = "configs/protocols/mapped_persons.yaml"
V15 = "results/confonly_primary"
MECH = "armruns"
SEEDS = (0, 1, 2)
CROSSED_SEED = 20261118
PUBLISHED_V15_18 = (0.595, 0.500, 0.719)        # results/reanalyses/r18_matched18_intervals.csv (corrected, 3 seeds)


def ci(a):
    lo, hi = M.ci(a[1:])
    return float(a[0]), lo, hi


def fmt(a, d=3):
    p, lo, hi = ci(a)
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v19", default="results/mapped_persons")
    ap.add_argument("--out", default="results/mapped_persons/analysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    test18 = {int(k) for k in yaml.safe_load(open(PROTO))["person_split"]["test_identifiers"]}

    def names(src, proto):
        return {w: [f"{proto}_{w}_w0_{MECH}_mid_A_none_s{k}" for k in SEEDS] for w in ("S", "I")}
    groups = {"v19": (args.v19, names(args.v19, "v19")), "v15conf": (V15, names(V15, "v15conf"))}
    for key, (src, g) in groups.items():
        miss = [n for w in g for n in g[w] if not os.path.exists(f"{src}/scores/{n}.csv")]
        if miss:
            raise SystemExit(f"{key}: score files missing: {miss}")
    # row order: the first v15conf file (the order of every published resample), restricted to the 18
    meta0, _ = load_scores(f"{V15}/scores/{groups['v15conf'][1]['S'][0]}.csv")
    keep = [i for i, c in enumerate(meta0["child_id"]) if int(c) in test18]
    ids = [meta0["identifier"][i] for i in keep]
    y = np.array([int(meta0["label"][i]) for i in keep])
    child = np.array([str(meta0["child_id"][i]) for i in keep])
    rs = RowSet(ids, y, child, args.n_boot)
    pos = {i: k for k, i in enumerate(ids)}
    S = {}
    for key, (src, g) in groups.items():
        for w in g:
            for n in g[w]:
                meta, cols = load_scores(f"{src}/scores/{n}.csv")
                idx = {i: k for k, i in enumerate(meta["identifier"])}
                order = np.array([idx[i] for i in ids])
                if not np.array_equal(np.array([int(meta["label"][k]) for k in order]), y):
                    raise SystemExit(f"{n}: labels differ")
                S[n] = {c: v[order] for c, v in cols.items()}
    cache = {}

    def aucs(n, c):
        if (n, c) not in cache:
            cache[(n, c)] = rs.aucs(S[n][c])
        return cache[(n, c)]

    def R(n, scale):
        f = logit if scale == "logit" else (lambda a: a)
        return np.stack([f(aucs(n, f"{MECH}_S_mid_m{ms}")) - f(aucs(n, f"{MECH}_I_mid_m{ms}")) for ms in MASK_SEEDS])

    def cell(g, scale):
        aS = np.stack([R(n, scale) for n in g["S"]])
        aI = np.stack([R(n, scale) for n in g["I"]])
        return aS.mean((0, 1)) - aI.mean((0, 1)), aS.mean((0, 1)), aI.mean((0, 1)), aS, aI

    def cell_mean(g, w, p, scale):
        f = logit if scale == "logit" else (lambda a: a)
        return np.mean([[f(aucs(n, f"{MECH}_{p}_mid_m{ms}")) for ms in MASK_SEEDS] for n in g[w]], axis=(0, 1))

    def clean(g, w):
        return np.mean([aucs(n, "clean") for n in g[w]], axis=0)

    rows, crit = [], {"protocol": PROTO, "n_boot": args.n_boot, "n_rows": len(ids), "n_smm": int(y.sum()),
                      "n_clusters": len(set(child))}
    res = {}
    for key, (_, g) in groups.items():
        tl, rS, rI, aS, aI = cell(g, "logit")
        tr, _, _, aSr, aIr = cell(g, "raw")
        res[key] = (tl, tr, aS, aI, aSr, aIr)
        row = {"model_set": key, "TTI_logit": fmt(tl), "TTI_raw": fmt(tr), "R_S": fmt(rS), "R_I": fmt(rI),
               "clean_S": fmt(clean(g, "S")), "clean_I": fmt(clean(g, "I")),
               "clean_S_minus_I": fmt(clean(g, "S") - clean(g, "I"))}
        for w in ("S", "I"):
            for p in ("S", "I"):
                row[f"auroc_{w}{p}"] = round(float(cell_mean(g, w, p, "raw")[0]), 4)
                row[f"logit_{w}{p}"] = round(float(cell_mean(g, w, p, "logit")[0]), 4)
        lo, hi = M.ci(rI[1:])
        row["R_I_within_margin"] = bool(-R_MARGIN < lo and hi < R_MARGIN)
        row["per_seed_TTI_logit"] = [round(float(aS[k].mean(0)[0] - aI[k].mean(0)[0]), 3) for k in range(len(SEEDS))]
        row["per_mask_TTI_logit"] = [round(float(aS[:, m, 0].mean() - aI[:, m, 0].mean()), 3) for m in range(len(MASK_SEEDS))]
        rows.append(row)
    v19 = rows[0]
    gate_val = ci(clean(groups["v19"][1], "I"))[0]
    crit["G19_clean"] = {"mean_clean_auroc_condition_I": gate_val, "pass": bool(gate_val >= 0.75)}
    tl, tr = res["v19"][0], res["v19"][1]
    crit["P19_1"] = {"tti_logit": list(ci(tl)), "tti_raw": list(ci(tr)),
                     "pass": bool(M.ci(tl[1:])[0] > 0 and M.ci(tr[1:])[0] > 0),
                     "p_inv_logit": p_inversion(tl[1:]), "p_inv_raw": p_inversion(tr[1:])}
    # seed-and-mask sensitivity interval (crossed resampling of seeds and masks, shared identifiers)
    rng = np.random.RandomState(CROSSED_SEED)
    nb = args.n_boot
    mask_draw = rng.randint(0, len(MASK_SEEDS), size=(nb, len(MASK_SEEDS)))
    sens = {}
    for scale, (A_S, A_I), point in (("logit", (res["v19"][2], res["v19"][3]), tl),
                                     ("raw", (res["v19"][4], res["v19"][5]), tr)):
        sd_S, sd_I = rng.randint(0, A_S.shape[0], size=(nb, A_S.shape[0])), rng.randint(0, A_I.shape[0], size=(nb, A_I.shape[0]))
        out = np.array([A_S[np.ix_(sd_S[b], mask_draw[b])][..., b + 1].mean() - A_I[np.ix_(sd_I[b], mask_draw[b])][..., b + 1].mean()
                        for b in range(nb)])
        sens[scale] = [float(point[0]), *M.ci(out)]
    crit["seed_mask_sensitivity"] = sens
    # D19_2: published full-cohort cell on the same rows, and the paired difference
    v15p = ci(res["v15conf"][0])
    crit["D19_2"] = {"v15conf_tti_logit_18": list(v15p),
                     "reproduces_published": bool(tuple(round(x, 3) for x in v15p) == PUBLISHED_V15_18),
                     "v19_minus_v15conf_logit": list(ci(res["v19"][0] - res["v15conf"][0])),
                     "v19_minus_v15conf_raw": list(ci(res["v19"][1] - res["v15conf"][1])),
                     "descriptive": True}
    if round(v15p[0], 3) != PUBLISHED_V15_18[0]:
        raise SystemExit(f"v15conf point on the 18 persons {v15p[0]:.4f} does not reproduce {PUBLISHED_V15_18[0]}")
    os.makedirs(args.out, exist_ok=True)
    cols = list(rows[0].keys())
    with open(f"{args.out}/effects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    json.dump({"rows": rows, **crit}, open(f"{args.out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(r)
    print(json.dumps(crit, indent=1, default=float))


if __name__ == "__main__":
    main()
