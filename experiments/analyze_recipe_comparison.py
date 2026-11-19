"""Protocol v20 analysis (configs/protocols/recipe_comparison.yaml): the velocity plus whole-joint-
dropout recipe (A_both) in the primary confidence-only cell.

Gate G20_clean; P20_1 (dTTI = TTI(A_both) - TTI(A_none, v15conf), logit and raw, verdict by the rule of
the article); P20_2 (clean-AUROC change per condition against +-0.02); the reported quantities; the
seed-and-mask sensitivity interval; and, once the phase-2 fits exist, D20_3 (the same contrast against
the local refit of the no-remedy cell). Estimator as analyze_confonly_primary.py: AUROC per fitted model and
test mask inside each identifier-clustered resample (RowSet of analyze_operators, seed 0, 2,000 resamples),
transformed, then averaged over masks and seeds.
Writes <out>/{effects.csv, criteria.json}.
    python experiments/analyze_recipe_comparison.py [--v20 results/recipe_comparison] [--out results/recipe_comparison/analysis]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                                  # noqa: E402
from experiments.analyze_operators import RowSet, load_scores, logit, p_inversion, align, R_MARGIN  # noqa: E402
from experiments.analyze_operators_masks import MASK_SEEDS                                   # noqa: E402

PROTO = "configs/protocols/recipe_comparison.yaml"
V15 = "results/confonly_primary"
MECH = "armruns"
SEEDS = (0, 1, 2)
CROSSED_SEED = 20261120
CLEAN_MARGIN = 0.02
GATE = 0.75


def ci(a):
    lo, hi = M.ci(a[1:])
    return float(a[0]), lo, hi


def fmt(a, d=5):
    p, lo, hi = ci(a)
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def verdict(dl, dr):
    """Rule of the article (Section III-F, step 6) on the conditional paired intervals."""
    (_, llo, lhi), (_, rlo, rhi) = ci(dl), ci(dr)
    if lhi < 0 and rhi < 0:
        return "lowers TTI"
    if llo > 0 and rlo > 0:
        return "raises TTI"
    return "no detected change"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v20", default="results/recipe_comparison")
    ap.add_argument("--out", default="results/recipe_comparison/analysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    def names(proto, cell):
        return {w: [f"{proto}_{w}_w0_{MECH}_mid_{cell}_s{k}" for k in SEEDS] for w in ("S", "I")}
    groups = {"A_both": (args.v20, names("v20", "A_both")), "A_none_v15conf": (V15, names("v15conf", "A_none"))}
    local = (args.v20, names("v20", "A_none"))
    have_local = all(os.path.exists(f"{local[0]}/scores/{n}.csv") for w in local[1] for n in local[1][w])
    if have_local:
        groups["A_none_local"] = local
    for key, (src, g) in groups.items():
        miss = [n for w in g for n in g[w] if not os.path.exists(f"{src}/scores/{n}.csv")]
        if miss:
            raise SystemExit(f"{key}: score files missing: {miss}")
    # row order of the first v15conf file, as in every published resample of the primary cell
    meta0, _ = load_scores(f"{V15}/scores/{groups['A_none_v15conf'][1]['S'][0]}.csv")
    ids = list(meta0["identifier"])
    rs = RowSet(ids, [int(x) for x in meta0["label"]], list(meta0["child_id"]), args.n_boot)
    S = {}
    for key, (src, g) in groups.items():
        for w in g:
            for n in g[w]:
                meta, sc = load_scores(f"{src}/scores/{n}.csv")
                S[n] = align(rs, meta["identifier"], meta["label"], sc)
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

    rows, res = [], {}
    crit = {"protocol": PROTO, "n_boot": args.n_boot, "n_rows": len(ids), "n_clusters": len(set(rs.child))}
    for key, (_, g) in groups.items():
        tl, rS, rI, aS, aI = cell(g, "logit")
        tr, _, _, aSr, aIr = cell(g, "raw")
        res[key] = {"tl": tl, "tr": tr, "aS": aS, "aI": aI, "aSr": aSr, "aIr": aIr, "g": g}
        row = {"model_set": key, "TTI_logit": fmt(tl), "TTI_raw": fmt(tr), "R_S": fmt(rS), "R_I": fmt(rI),
               "clean_S": fmt(clean(g, "S")), "clean_I": fmt(clean(g, "I")),
               "clean_S_minus_I": fmt(clean(g, "S") - clean(g, "I"))}
        for w in ("S", "I"):
            for p in ("S", "I"):
                row[f"auroc_{w}{p}"] = round(float(cell_mean(g, w, p, "raw")[0]), 6)
                row[f"logit_{w}{p}"] = round(float(cell_mean(g, w, p, "logit")[0]), 6)
        lo, hi = M.ci(rI[1:])
        row["R_I_within_margin"] = bool(-R_MARGIN < lo and hi < R_MARGIN)
        row["per_seed_TTI_logit"] = [round(float(aS[k].mean(0)[0] - aI[k].mean(0)[0]), 4) for k in range(len(SEEDS))]
        rows.append(row)

    gate_val = ci(clean(groups["A_both"][1], "I"))[0]
    crit["G20_clean"] = {"mean_clean_auroc_condition_I": gate_val, "pass": bool(gate_val >= GATE)}
    nb = args.n_boot
    rng = np.random.RandomState(CROSSED_SEED)
    mask_draw = rng.randint(0, len(MASK_SEEDS), size=(nb, len(MASK_SEEDS)))

    def crossed(key, scale):
        A_S, A_I = (res[key]["aS"], res[key]["aI"]) if scale == "logit" else (res[key]["aSr"], res[key]["aIr"])
        sd_S = rng.randint(0, A_S.shape[0], size=(nb, A_S.shape[0]))
        sd_I = rng.randint(0, A_I.shape[0], size=(nb, A_I.shape[0]))
        return np.array([A_S[np.ix_(sd_S[b], mask_draw[b])][..., b + 1].mean()
                         - A_I[np.ix_(sd_I[b], mask_draw[b])][..., b + 1].mean() for b in range(nb)])

    def contrast(base):
        dl, dr = res["A_both"]["tl"] - res[base]["tl"], res["A_both"]["tr"] - res[base]["tr"]
        sens = {}
        for scale, d in (("logit", dl), ("raw", dr)):
            c = crossed("A_both", scale) - crossed(base, scale)
            sens[scale] = [float(d[0]), *M.ci(c)]
        v = verdict(dl, dr)
        nominal = v != "no detected change" and (sens["logit"][1] <= 0 <= sens["logit"][2])
        g, gb = groups["A_both"][1], groups[base][1]
        cc = {w: clean(g, w) - clean(gb, w) for w in ("S", "I")}
        pres = all(-CLEAN_MARGIN < ci(cc[w])[1] and ci(cc[w])[2] < CLEAN_MARGIN for w in ("S", "I"))
        return {"baseline": base, "dTTI_logit": list(ci(dl)), "dTTI_raw": list(ci(dr)), "verdict": v,
                "nominal_only": bool(nominal), "largest_reduction_not_excluded_logit": ci(dl)[1],
                "p_inv_logit": p_inversion(dl[1:]), "p_inv_raw": p_inversion(dr[1:]),
                "seed_mask_sensitivity": sens,
                "clean_change": {w: list(ci(cc[w])) for w in ("S", "I")}, "clean_preserved": bool(pres)}

    crit["P20_1_P20_2"] = contrast("A_none_v15conf")
    if have_local:
        d3 = contrast("A_none_local")
        dd = res["A_none_local"]["tl"] - res["A_none_v15conf"]["tl"]
        ddr = res["A_none_local"]["tr"] - res["A_none_v15conf"]["tr"]
        d3["local_minus_published_baseline_TTI_logit"] = list(ci(dd))
        d3["local_minus_published_baseline_TTI_raw"] = list(ci(ddr))
        d3["verdict_agrees_with_primary"] = d3["verdict"] == crit["P20_1_P20_2"]["verdict"]
        crit["D20_3"] = d3
    else:
        crit["D20_3"] = {"status": "phase-2 local no-remedy fits not scored yet"}

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/effects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    json.dump({"rows": rows, **crit}, open(f"{args.out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(r)
    print(json.dumps(crit, indent=1, default=float))


if __name__ == "__main__":
    main()
