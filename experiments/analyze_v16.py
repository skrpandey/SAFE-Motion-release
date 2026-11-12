"""Protocol v16 analysis (configs/protocols/confonly_extension_v16.yaml), cleaned selection.

Part A: ST-GCN no remedy. Gate G16_A_convergence (mean clean AUROC of the World I models >= 0.75),
P16_A1 (MWI interval above zero on both scales), R(f_S), R(f_I), clean AUROC S - I, seed-and-mask
sensitivity interval, D16_A2 (MWI ST-GCN minus MWI CompactCTRGCN, v15conf seeds 0-2).
Part B: CompactCTRGCN no remedy with nine seeds per world (v15conf seeds 0-2 pooled with v16 seeds
3-8): conditional interval, Welch t interval on per-seed mask-mean responses, seed SDs, sensitivity.
Part C: Group DRO with observed groups against v15conf A_none seeds 0-2 (P16_C1), against all nine
A_none seeds (sensitivity), clean-AUROC change per world, and against v15conf Group DRO (descriptive).
Writes <out>/{effects.csv, criteria.json}.
    python experiments/analyze_v16.py --v16 results/v16 --v15 results/v15conf --out results/v16/analysis
--v16 takes one or more result sets (scores/ and registry.json each). The group agreement of part C is
read from the training logs (--epochs), which are not redistributed; criteria.json records it.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                                  # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, p_inversion, align, verdict, R_MARGIN  # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                                   # noqa: E402

MECH = "armruns"
CROSSED_SEED = 20261110
CLEAN_MARGIN = 0.02
GATE_CLEAN_AUROC = 0.75


def ci_str(a, d=5):
    lo, hi = M.ci(a[1:])
    return float(a[0]), lo, hi, f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def load_set(path, proto):
    reg = [dict(m, src=path, proto_tag=proto) for m in json.load(open(f"{path}/registry.json"))]
    absent = [m["name"] for m in reg if not os.path.exists(f"{path}/scores/{m['name']}.csv")]
    if absent:
        raise SystemExit(f"{path}: score files missing: {absent}")
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--v16", nargs="+", default=["results/v16"])
    ap.add_argument("--v15", default="results/v15conf")
    ap.add_argument("--epochs", default="results/fits/v16bc/results/epochs", help="train logs of part C")
    ap.add_argument("--out", default="results/v16/analysis")
    args = ap.parse_args()
    reg = [m for p in args.v16 for m in load_set(p, "v16")] + load_set(args.v15, "v15conf")

    def pick(proto, cell, seeds):
        g = sorted([m for m in reg if m["proto_tag"] == proto and m["cell"] == cell and m["seed"] in seeds],
                   key=lambda m: (m["world"], m["seed"]))
        out = {w: [m for m in g if m["world"] == w] for w in ("S", "I")}
        for w in ("S", "I"):
            got = sorted(m["seed"] for m in out[w])
            if got != sorted(seeds):
                raise SystemExit(f"{proto} {cell} world {w}: seeds {got}, expected {sorted(seeds)}")
        return out

    G = {"A": pick("v16", "A_none_stgcn", (0, 1, 2)),
         "B_new": pick("v16", "A_none", tuple(range(3, 9))),
         "C": pick("v16", "C_groupdro_obs", (0, 1, 2)),
         "base3": pick("v15conf", "A_none", (0, 1, 2)),
         "dro_oracle": pick("v15conf", "C_groupdro", (0, 1, 2))}
    G["base9"] = {w: G["base3"][w] + G["B_new"][w] for w in ("S", "I")}
    used = {m["name"]: m for g in G.values() for w in ("S", "I") for m in g[w]}
    loaded = {n: load_scores(f"{m['src']}/scores/{n}.csv") for n, m in used.items()}
    if len({frozenset(meta["identifier"]) for meta, _ in loaded.values()}) != 1:
        raise SystemExit("score files do not cover the same identifiers")
    meta0 = next(iter(loaded.values()))[0]
    ids = list(meta0["identifier"])
    rs = RowSet(ids, [int(x) for x in meta0["label"]], list(meta0["child_id"]), args.n_boot)
    S = {n: align(rs, meta["identifier"], meta["label"], sc) for n, (meta, sc) in loaded.items()}
    cache = {}

    def aucs(n, c):
        if (n, c) not in cache:
            cache[(n, c)] = rs.aucs(S[n][c])
        return cache[(n, c)]

    def R_arrays(models):
        """(n_seeds, 3 masks, NB+1) logit and raw responses."""
        lg = np.stack([[logit(aucs(m["name"], f"{MECH}_S_mid_m{ms}")) - logit(aucs(m["name"], f"{MECH}_I_mid_m{ms}"))
                        for ms in MASK_SEEDS] for m in models])
        rw = np.stack([[aucs(m["name"], f"{MECH}_S_mid_m{ms}") - aucs(m["name"], f"{MECH}_I_mid_m{ms}")
                        for ms in MASK_SEEDS] for m in models])
        return lg, rw

    def mwi(g):
        (lS, rS), (lI, rI) = R_arrays(g["S"]), R_arrays(g["I"])
        return lS.mean((0, 1)) - lI.mean((0, 1)), rS.mean((0, 1)) - rI.mean((0, 1)), (lS, rS, lI, rI)

    def clean(models):
        return np.mean([aucs(m["name"], "clean") for m in models], axis=0)

    rng = np.random.RandomState(CROSSED_SEED)
    nb = args.n_boot
    mask_draw = rng.randint(0, len(MASK_SEEDS), size=(nb, len(MASK_SEEDS)))

    def crossed(arrs, scale):
        """Seed-and-mask sensitivity draws of one cell's MWI (shared identifier and mask draws)."""
        lS, rS, lI, rI = arrs
        A_S, A_I = (lS, lI) if scale == "logit" else (rS, rI)
        kS, kI = A_S.shape[0], A_I.shape[0]
        sS, sI = rng.randint(0, kS, size=(nb, kS)), rng.randint(0, kI, size=(nb, kI))
        out = np.empty(nb)
        for b in range(nb):
            ms = mask_draw[b]
            out[b] = A_S[np.ix_(sS[b], ms)][..., b + 1].mean() - A_I[np.ix_(sI[b], ms)][..., b + 1].mean()
        return out

    rows, crit = [], {"n_boot": nb, "n_rows": len(ids), "n_clusters": len(set(rs.child)),
                      "protocol": "configs/protocols/confonly_extension_v16.yaml", "selection": "cleaned"}
    res = {k: mwi(G[k]) for k in ("A", "C", "base3", "base9", "dro_oracle")}
    draws = {}                      # (key, scale) -> the seed-and-mask draw vector of that cell (shared mask draw)

    def add_cell(label, key):
        se, sr, arrs = res[key]
        rS, rI = arrs[0].mean((0, 1)), arrs[2].mean((0, 1))
        cd = clean(G[key]["S"]) - clean(G[key]["I"])
        rows.append({"part": label, "quantity": f"MWI ({key})", "logit": ci_str(se)[3], "raw": ci_str(sr)[3],
                     "R_S": ci_str(rS)[3], "R_I": ci_str(rI)[3], "R_I_verdict": verdict(rI[0], M.ci(rI[1:]), R_MARGIN),
                     "clean_S_minus_I": ci_str(cd)[3], "clean_verdict": verdict(cd[0], M.ci(cd[1:]), CLEAN_MARGIN),
                     "n_seeds": f"{len(G[key]['S'])}/{len(G[key]['I'])}"})
        sens = {}
        for scale in ("logit", "raw"):
            x = crossed(arrs, scale)
            draws[(key, scale)] = x
            lo, hi = M.ci(x)
            sens[scale] = [float((se if scale == "logit" else sr)[0]), lo, hi]
            rows.append({"part": label, "quantity": f"MWI ({key}), seeds and masks resampled ({scale})",
                         scale: f"{sens[scale][0]:+.5f} [{lo:+.5f}, {hi:+.5f}]"})
        return se, sr, sens

    def contrast(label, name, a, b):
        d, dr = res[a][0] - res[b][0], res[a][1] - res[b][1]
        rows.append({"part": label, "quantity": name, "logit": ci_str(d)[3], "raw": ci_str(dr)[3],
                     "p_inv": f"{p_inversion(d[1:]):.4f}", "raw_p_inv": f"{p_inversion(dr[1:]):.4f}"})
        return d, dr

    # part A
    gate_clean = float(np.mean([aucs(m["name"], "clean")[0] for m in G["A"]["I"]]))
    se, sr, sens = add_cell("A", "A")
    passA = bool(M.ci(se[1:])[0] > 0 and M.ci(sr[1:])[0] > 0)
    dA, dAr = contrast("A", "MWI(ST-GCN) - MWI(CompactCTRGCN, v15conf seeds 0-2) [descriptive]", "A", "base3")
    crit["G16_A_convergence"] = {"mean_clean_auroc_world_I": gate_clean, "threshold": GATE_CLEAN_AUROC,
                                 "pass": gate_clean >= GATE_CLEAN_AUROC}
    crit["P16_A1_reproduction"] = {"mwi_logit": list(ci_str(se)[:3]), "mwi_raw": list(ci_str(sr)[:3]),
                                   "pass": passA if gate_clean >= GATE_CLEAN_AUROC else None,
                                   "sensitivity": sens}
    crit["D16_A2_architecture"] = {"diff_logit": list(ci_str(dA)[:3]), "diff_raw": list(ci_str(dAr)[:3])}

    # part B: nine seeds per world
    seB, srB, sensB = add_cell("B", "base9")
    lS, _, lI, _ = res["base9"][2]
    per_S, per_I = lS[:, :, 0].mean(1), lI[:, :, 0].mean(1)          # per-seed mask-mean responses (point)
    vS, vI, nS, nI = per_S.var(ddof=1), per_I.var(ddof=1), len(per_S), len(per_I)
    se_w = np.sqrt(vS / nS + vI / nI)
    df = (vS / nS + vI / nI) ** 2 / ((vS / nS) ** 2 / (nS - 1) + (vI / nI) ** 2 / (nI - 1))
    tq = stats.t.ppf(0.975, df)
    diff = per_S.mean() - per_I.mean()
    rows.append({"part": "B", "quantity": "MWI (base9), Welch t interval over per-seed responses (logit)",
                 "logit": f"{diff:+.5f} [{diff - tq * se_w:+.5f}, {diff + tq * se_w:+.5f}]",
                 "note": f"df {df:.1f}; seed SD World S {np.sqrt(vS):.4f}, World I {np.sqrt(vI):.4f}"})
    crit["P16_B1_precision"] = {"mwi_logit_conditional": list(ci_str(seB)[:3]), "mwi_raw_conditional": list(ci_str(srB)[:3]),
                                "welch_t": [float(diff), float(diff - tq * se_w), float(diff + tq * se_w)],
                                "welch_df": float(df), "seed_sd_S": float(np.sqrt(vS)), "seed_sd_I": float(np.sqrt(vI)),
                                "per_seed_R_S": per_S.round(5).tolist(), "per_seed_R_I": per_I.round(5).tolist(),
                                "sensitivity": sensB}

    # part C
    add_cell("C", "C")
    dC, dCr = contrast("C", "MWI(C_groupdro_obs) - MWI(A_none, v15conf seeds 0-2)", "C", "base3")
    passC = bool(M.ci(dC[1:])[1] < 0 and M.ci(dCr[1:])[1] < 0)
    d9, d9r = contrast("C", "MWI(C_groupdro_obs) - MWI(A_none, nine seeds) [sensitivity]", "C", "base9")
    dO, dOr = contrast("C", "MWI(C_groupdro_obs) - MWI(C_groupdro, v15conf, severity groups) [descriptive]", "C", "dro_oracle")
    clean_change = {}
    for w in ("S", "I"):
        cc = clean(G["C"][w]) - clean(G["base3"][w])
        clean_change[w] = list(ci_str(cc)[:3])
        rows.append({"part": "C", "quantity": f"clean AUROC(C_groupdro_obs) - clean AUROC(A_none), World {w}",
                     "raw": ci_str(cc)[3]})
    agree = {}
    for p in sorted(glob.glob(f"{args.epochs}/*C_groupdro_obs*/train_log.json")):
        log = json.load(open(p))
        agree[os.path.basename(os.path.dirname(p))] = log.get("cell", {}).get("group_agreement_with_severity_groups")
    crit["P16_C1_reduction"] = {"diff_logit": list(ci_str(dC)[:3]), "diff_raw": list(ci_str(dCr)[:3]),
                                "p_inv_logit": p_inversion(dC[1:]), "p_inv_raw": p_inversion(dCr[1:]),
                                "pass": passC, "sensitivity_nine_seeds": {"logit": list(ci_str(d9)[:3]), "raw": list(ci_str(d9r)[:3])},
                                "vs_severity_groups": {"logit": list(ci_str(dO)[:3]), "raw": list(ci_str(dOr)[:3])},
                                "clean_auroc_change": clean_change, "group_agreement": agree}

    # ---- added 2026-11-11 (revision after reviews A and B; every draw below comes AFTER the draws
    # of parts A, B and C, so the numbers above are unchanged).
    # (1) The corrected-recipe three-seed no-remedy cell (v15conf seeds 0-2) as a reported cell: its
    #     seed-and-mask sensitivity interval and clean AUROC S - I, which the main-text table needs.
    seB0, srB0, sensB0 = add_cell("B0", "base3")
    cdB0 = clean(G["base3"]["S"]) - clean(G["base3"]["I"])
    crit["B0_v15conf_no_remedy"] = {"mwi_logit": list(ci_str(seB0)[:3]), "mwi_raw": list(ci_str(srB0)[:3]),
                                    "clean_S_minus_I": list(ci_str(cdB0)[:3]), "sensitivity": sensB0}
    # (2) Paired seed-and-mask sensitivity interval of the observed-group Group DRO contrast (review B,
    #     M4A): the difference of the two cells' draw vectors, which share the identifier and mask
    #     draws (the v15conf P15_4 construction). No new random draws.
    paired = {}
    for comp, tag in (("base3", "v15conf seeds 0-2"), ("base9", "nine seeds")):
        paired[comp] = {}
        for scale, pt in (("logit", dC if comp == "base3" else d9), ("raw", dCr if comp == "base3" else d9r)):
            x = draws[("C", scale)] - draws[(comp, scale)]
            lo, hi = M.ci(x)
            paired[comp][scale] = [float(pt[0]), lo, hi]
            rows.append({"part": "C", "quantity": f"MWI(C_groupdro_obs) - MWI(A_none, {tag}), seeds and masks resampled ({scale})",
                         scale: f"{pt[0]:+.5f} [{lo:+.5f}, {hi:+.5f}]"})
    crit["P16_C1_reduction"]["sensitivity_seeds_masks"] = paired["base3"]
    crit["P16_C1_reduction"]["sensitivity_nine_seeds"]["seeds_masks"] = paired["base9"]
    # (3) Crossed responses of the current coordinate-rule confidence-only models to the operators
    #     they were not trained against (first test mask, seeds averaged inside resamples; the
    #     convention of analyze_v13.py), for the "current models" rows of the operator-specificity
    #     table. No random draws.
    crit["crossed"] = {}
    for key, label in (("base9", "CompactCTRGCN no remedy, nine seeds"), ("A", "ST-GCN no remedy")):
        for w in ("S", "I"):
            for other in ("runs", "whole", "frozen", "armruns"):
                r = np.mean([logit(aucs(m["name"], f"{other}_S_mid_m2000")) - logit(aucs(m["name"], f"{other}_I_mid_m2000"))
                             for m in G[key][w]], axis=0)
                rows.append({"part": "X", "quantity": f"crossed R under {other} (mask 2000), {key}, World {w}",
                             "logit": ci_str(r)[3], "note": label})
                crit["crossed"][f"{key}|{w}|{other}"] = list(ci_str(r)[:3])

    os.makedirs(args.out, exist_ok=True)
    keys = ["part", "quantity", "logit", "raw", "R_S", "R_I", "R_I_verdict", "clean_S_minus_I", "clean_verdict",
            "n_seeds", "p_inv", "raw_p_inv", "note"]
    with open(f"{args.out}/effects.csv", "w", newline="") as f:
        f.write("# protocol v16, cleaned selection; mask mean over test masks 2000/3000/4000, seeds averaged inside "
                f"{nb} identifier-clustered resamples\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    json.dump(crit, open(f"{args.out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(f"{r['part']} | {r['quantity'][:70]:70s} | {r.get('logit', '')} | {r.get('raw', '')}")
    print("G16_A_convergence:", crit["G16_A_convergence"])
    print("P16_A1 pass:", crit["P16_A1_reproduction"]["pass"], "| P16_C1 pass:", passC)


if __name__ == "__main__":
    main()
