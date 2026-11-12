"""Protocol v15conf analysis (configs/protocols/remedies_confonly_v15.yaml): Group DRO against no
remedy under confidence-only planting, both arms trained under the v15 recipe.

P15_1 (primary): MWI(C_groupdro) - MWI(A_none) on the mean over test masks 2000/3000/4000, seeds
averaged inside identifier-clustered resamples; passes iff the logit and raw intervals both lie below
zero. P15_2: clean-AUROC change per world. P15_3: R(f_I) against the +-0.05 margin. P15_4: seed-and-mask
crossed interval. D15_5 (descriptive): MWI(A_none) of v15conf minus that of v13 (legacy recipe).
Writes <v15>/analysis/{effects.csv, criteria.json}; experiments/make_tables_v15conf.py builds the table.
    conda run -n torch python experiments/analyze_v15conf.py [--n-boot 2000] [--v15 DIR] [--v13 DIR]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                                  # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, p_inversion, align, verdict, R_MARGIN  # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                                   # noqa: E402

MECH = "armruns"
CELLS = ("A_none", "C_groupdro")
CROSSED_SEED = 20261109
SEEDS = (0, 1, 2)


def ci_str(a, d=5):
    lo, hi = M.ci(a[1:])
    return float(a[0]), lo, hi, f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--v15", default="results/v15conf/clean")
    ap.add_argument("--v13", default="build/release_v15_clean/results/v13",
                    help="cleaned-selection v13 score set (legacy recipe) for the descriptive D15_5 contrast")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"{args.v15}/analysis"
    # every registered fit needs a score file, each world-cell group exactly the protocol seeds,
    # and all files the same rows
    reg15 = [dict(m, src=args.v15) for m in json.load(open(f"{args.v15}/registry.json"))]
    absent = [m["name"] for m in reg15 if not os.path.exists(f"{args.v15}/scores/{m['name']}.csv")]
    if absent:
        raise SystemExit(f"v15conf score files missing: {absent}")
    for w in ("S", "I"):
        for c in CELLS:
            seeds = sorted(m["seed"] for m in reg15 if m["world"] == w and m["cell"] == c)
            if seeds != list(SEEDS):
                raise SystemExit(f"v15conf {w}/{c}: seeds {seeds}, expected {list(SEEDS)}")
    if len(reg15) != 2 * len(CELLS) * len(SEEDS):
        raise SystemExit(f"v15conf: {len(reg15)} registered fits, expected {2 * len(CELLS) * len(SEEDS)}")
    reg13 = []
    if os.path.exists(f"{args.v13}/registry.json"):
        reg13 = [dict(m, src=args.v13) for m in json.load(open(f"{args.v13}/registry.json"))
                 if m["cell"] == "A_none" and m["mechanism"] == MECH and m["strength"] == "mid"
                 and m.get("world_seed", 0) == 0]
        absent = [m["name"] for m in reg13 if not os.path.exists(f"{args.v13}/scores/{m['name']}.csv")]
        if absent:
            raise SystemExit(f"v13 comparator score files missing: {absent}")
    loaded = {m["name"]: load_scores(f"{m['src']}/scores/{m['name']}.csv") for m in reg15 + reg13}
    id_sets = {frozenset(meta["identifier"]) for meta, _ in loaded.values()}
    if len(id_sets) != 1:
        raise SystemExit("v15conf: score files do not cover the same identifiers")
    meta0 = next(iter(loaded.values()))[0]
    ids = list(meta0["identifier"])
    rs = RowSet(ids, [int(x) for x in meta0["label"]], list(meta0["child_id"]), args.n_boot)
    S = {}
    for n, (meta, sc) in loaded.items():       # align rows by identifier
        S[n] = align(rs, meta["identifier"], meta["label"], sc)
    cache = {}

    def aucs(n, c):
        if (n, c) not in cache:
            cache[(n, c)] = rs.aucs(S[n][c])
        return cache[(n, c)]

    def group(models, world, cell):
        return sorted([m for m in models if m["world"] == world and m["cell"] == cell], key=lambda m: m["name"])

    def R_arrays(ms_models):
        """(n_seeds, 3 masks, NB+1) logit and raw R arrays of a world-cell group."""
        lg = np.stack([[logit(aucs(m["name"], f"{MECH}_S_mid_m{ms}")) - logit(aucs(m["name"], f"{MECH}_I_mid_m{ms}"))
                        for ms in MASK_SEEDS] for m in ms_models])
        rw = np.stack([[aucs(m["name"], f"{MECH}_S_mid_m{ms}") - aucs(m["name"], f"{MECH}_I_mid_m{ms}")
                        for ms in MASK_SEEDS] for m in ms_models])
        return lg, rw

    def mwi(gS, gI):
        """Mask-mean MWI (logit, raw) with seeds averaged inside resamples, plus the R arrays."""
        (lS, rS), (lI, rI) = R_arrays(gS), R_arrays(gI)
        return (lS.mean((0, 1)) - lI.mean((0, 1)), rS.mean((0, 1)) - rI.mean((0, 1)),
                (lS, rS, lI, rI))

    def clean_auc(ms_models):
        return np.mean([aucs(m["name"], "clean") for m in ms_models], axis=0)

    rows, crit = [], {"n_boot": args.n_boot, "n_rows": len(ids), "n_clusters": len(set(rs.child)),
                      "protocol": "configs/protocols/remedies_confonly_v15.yaml"}
    G = {(w, c): group(reg15, w, c) for w in ("S", "I") for c in CELLS}
    missing = [k for k, v in G.items() if not v]
    if missing:
        raise SystemExit(f"missing v15conf groups: {missing}")
    res = {}
    for cell in CELLS:
        se, se_raw, arrs = mwi(G[("S", cell)], G[("I", cell)])
        res[cell] = (se, se_raw, arrs)
        rI = arrs[2].mean((0, 1))
        rows.append({"cell": cell, "contrast": f"MWI({cell})", "logit": ci_str(se)[3], "raw": ci_str(se_raw)[3],
                     "R_S": ci_str(arrs[0].mean((0, 1)))[3], "R_I": ci_str(rI)[3],
                     "R_I_verdict": verdict(rI[0], M.ci(rI[1:]), R_MARGIN),
                     "n_seeds": f"{len(G[('S', cell)])}/{len(G[('I', cell)])}"})
        crit.setdefault("P15_3", {})[cell] = {"R_I": list(ci_str(rI)[:3]),
                                              "verdict": verdict(rI[0], M.ci(rI[1:]), R_MARGIN)}
    d = res["C_groupdro"][0] - res["A_none"][0]
    dr = res["C_groupdro"][1] - res["A_none"][1]
    dp, dlo, dhi, ds = ci_str(d)
    rp, rlo, rhi, rstr = ci_str(dr)
    lowers = bool(dhi < 0 and rhi < 0)
    rows.append({"cell": "C_groupdro", "contrast": "MWI(C_groupdro) - MWI(A_none)", "logit": ds, "raw": rstr,
                 "lowers_P15_1": str(lowers), "p_inv": f"{p_inversion(d[1:]):.4f}",
                 "raw_p_inv": f"{p_inversion(dr[1:]):.4f}"})
    crit["P15_1"] = {"diff_logit": [dp, dlo, dhi], "diff_raw": [rp, rlo, rhi], "pass": lowers,
                     "p_inv_logit": p_inversion(d[1:]), "p_inv_raw": p_inversion(dr[1:])}
    for world in ("S", "I"):
        cd = clean_auc(G[(world, "C_groupdro")]) - clean_auc(G[(world, "A_none")])
        rows.append({"cell": "C_groupdro", "contrast": f"clean AUROC(C_groupdro) - clean AUROC(A_none), World {world}",
                     "raw": ci_str(cd)[3]})
        crit.setdefault("P15_2", {})[world] = {"clean_auc_change": list(ci_str(cd)[:3]),
                                               "clean_auc_A_none": list(ci_str(clean_auc(G[(world, "A_none")]))[:3]),
                                               "clean_auc_C_groupdro": list(ci_str(clean_auc(G[(world, "C_groupdro")]))[:3])}

    # P15_4: seeds and masks resampled as crossed factors, identifier resamples shared
    nb = args.n_boot
    rng = np.random.RandomState(CROSSED_SEED)
    mask_draw = rng.randint(0, len(MASK_SEEDS), size=(nb, len(MASK_SEEDS)))

    def crossed(cell, scale):
        lS, rS, lI, rI = res[cell][2]
        A_S, A_I = (lS, lI) if scale == "logit" else (rS, rI)
        kS, kI = A_S.shape[0], A_I.shape[0]
        sd_S, sd_I = rng.randint(0, kS, size=(nb, kS)), rng.randint(0, kI, size=(nb, kI))
        out = np.empty(nb)
        for b in range(nb):
            ms = mask_draw[b]
            out[b] = A_S[np.ix_(sd_S[b], ms)][..., b + 1].mean() - A_I[np.ix_(sd_I[b], ms)][..., b + 1].mean()
        return out

    sens = {}
    for scale in ("logit", "raw"):
        cross = crossed("C_groupdro", scale) - crossed("A_none", scale)
        point = (d if scale == "logit" else dr)[0]
        lo, hi = M.ci(cross)
        sens[scale] = [float(point), lo, hi]
        rows.append({"cell": "C_groupdro", "contrast": f"MWI(C_groupdro) - MWI(A_none), seeds and masks resampled ({scale})",
                     scale: f"{point:+.5f} [{lo:+.5f}, {hi:+.5f}]"})
    crit["P15_4"] = {"crossed_interval": sens, "seed": CROSSED_SEED,
                     "note": "sensitivity display over three fitted seeds and three masks, not a calibrated interval"}

    # D15_5 (descriptive): v15conf A_none against v13 A_none
    g13 = {w: group(reg13, w, "A_none") for w in ("S", "I")}
    if g13["S"] and g13["I"]:
        se13, se13_raw, a13 = mwi(g13["S"], g13["I"])
        rows.append({"cell": "A_none (v13, legacy recipe)", "contrast": "MWI(A_none, v13 cleaned)",
                     "logit": ci_str(se13)[3], "raw": ci_str(se13_raw)[3], "R_S": ci_str(a13[0].mean((0, 1)))[3],
                     "R_I": ci_str(a13[2].mean((0, 1)))[3], "n_seeds": f"{len(g13['S'])}/{len(g13['I'])}"})
        dd, ddr = res["A_none"][0] - se13, res["A_none"][1] - se13_raw
        rows.append({"cell": "A_none", "contrast": "MWI(A_none, v15conf) - MWI(A_none, v13 cleaned) [descriptive]",
                     "logit": ci_str(dd)[3], "raw": ci_str(ddr)[3]})
        crit["D15_5"] = {"diff_logit": list(ci_str(dd)[:3]), "diff_raw": list(ci_str(ddr)[:3]),
                         "v13_source": args.v13, "descriptive": True}
    else:
        crit["D15_5"] = {"status": "v13 comparator not found", "v13_source": args.v13}

    os.makedirs(out, exist_ok=True)
    cols = ["cell", "contrast", "logit", "raw", "R_S", "R_I", "R_I_verdict", "n_seeds", "lowers_P15_1",
            "p_inv", "raw_p_inv"]
    with open(f"{out}/effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(f"{r['contrast']:75s} {r.get('logit', ''):34s} {r.get('raw', '')}")
    print(f"P15_1 lowers: {lowers}; rows {len(ids)}, clusters {crit['n_clusters']}")


if __name__ == "__main__":
    main()
