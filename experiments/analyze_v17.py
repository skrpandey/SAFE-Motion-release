"""Protocol v17 analysis (configs/protocols/confonly_assignment_v17.yaml), cleaned selection.

Replication of the corrected confidence-only no-remedy contrast under independently regenerated
training assignments (world seeds 1, 2, 3; three optimizer seeds per world each). For every
assignment, and for world seed 0 (the v15conf A_none fits) beside them: gate G17_convergence (mean
clean AUROC of the World I models >= 0.75), MWI with the conditional paired interval on both scales,
R(f_S), R(f_I), clean AUROC S - I and the seed-and-mask sensitivity interval (P17_1). Then the
assignment spread (D17_2: mean, sample SD and range of the point MWI across world seeds 1-3 and
0-3; each new assignment minus world seed 0, paired) and the pooled nine-fit-per-world estimate
(D17_3). Writes <out>/{effects.csv, criteria.json}.
    python experiments/analyze_v17.py --v17 results/fits/v17/results/clean \
        --v15 results/fits/v15conf/results/clean --out results/v17/analysis
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                                  # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, p_inversion, align, verdict, R_MARGIN  # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                                   # noqa: E402
from experiments.analyze_v16 import ci_str, load_set, CLEAN_MARGIN, GATE_CLEAN_AUROC, MECH   # noqa: E402

CROSSED_SEED = 20261111
WORLD_SEEDS = (1, 2, 3)
PROTOCOL = "configs/protocols/confonly_assignment_v17.yaml"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--v17", nargs="+", default=["results/fits/v17/results/clean"])
    ap.add_argument("--v15", default="results/fits/v15conf/results/clean")
    ap.add_argument("--epochs", default="results/fits/v17/results/epochs", help="train logs (selected epochs)")
    ap.add_argument("--out", default="results/v17/analysis")
    args = ap.parse_args(argv)
    reg = [m for p in args.v17 for m in load_set(p, "v17")] + load_set(args.v15, "v15conf")

    def pick(proto, cell, seeds, world_seed):
        g = sorted([m for m in reg if m["proto_tag"] == proto and m["cell"] == cell and m["seed"] in seeds
                    and int(m.get("world_seed", 0)) == world_seed], key=lambda m: (m["world"], m["seed"]))
        out = {w: [m for m in g if m["world"] == w] for w in ("S", "I")}
        for w in ("S", "I"):
            got = sorted(m["seed"] for m in out[w])
            if got != sorted(seeds):
                raise SystemExit(f"{proto} {cell} world seed {world_seed} world {w}: seeds {got}, expected {sorted(seeds)}")
        return out

    G = {"w0": pick("v15conf", "A_none", (0, 1, 2), 0)}
    for ws in WORLD_SEEDS:
        G[f"w{ws}"] = pick("v17", "A_none", (0, 1, 2), ws)
    G["pooled"] = {w: sum((G[f"w{ws}"][w] for ws in WORLD_SEEDS), []) for w in ("S", "I")}
    used = {m["name"]: m for k in G for w in ("S", "I") for m in G[k][w]}
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
                      "protocol": PROTOCOL, "selection": "cleaned"}
    keys = ["w0"] + [f"w{ws}" for ws in WORLD_SEEDS] + ["pooled"]
    res = {k: mwi(G[k]) for k in keys}
    crit["G17_convergence"], crit["P17_1_per_assignment"] = {}, {}
    for key in keys:
        se, sr, arrs = res[key]
        rS, rI = arrs[0].mean((0, 1)), arrs[2].mean((0, 1))
        cd = clean(G[key]["S"]) - clean(G[key]["I"])
        gate = float(np.mean([aucs(m["name"], "clean")[0] for m in G[key]["I"]]))
        rows.append({"part": key, "quantity": f"MWI ({key})", "logit": ci_str(se)[3], "raw": ci_str(sr)[3],
                     "R_S": ci_str(rS)[3], "R_I": ci_str(rI)[3], "R_I_verdict": verdict(rI[0], M.ci(rI[1:]), R_MARGIN),
                     "clean_S_minus_I": ci_str(cd)[3], "clean_verdict": verdict(cd[0], M.ci(cd[1:]), CLEAN_MARGIN),
                     "n_seeds": f"{len(G[key]['S'])}/{len(G[key]['I'])}", "note": f"World I clean AUROC {gate:.4f}"})
        sens = {}
        for scale in ("logit", "raw"):
            x = crossed(arrs, scale)
            lo, hi = M.ci(x)
            sens[scale] = [float((se if scale == "logit" else sr)[0]), lo, hi]
            rows.append({"part": key, "quantity": f"MWI ({key}), seeds and masks resampled ({scale})",
                         scale: f"{sens[scale][0]:+.5f} [{lo:+.5f}, {hi:+.5f}]"})
        crit["G17_convergence"][key] = {"mean_clean_auroc_world_I": gate, "threshold": GATE_CLEAN_AUROC,
                                        "pass": gate >= GATE_CLEAN_AUROC}
        crit["P17_1_per_assignment"][key] = {
            "mwi_logit": list(ci_str(se)[:3]), "mwi_raw": list(ci_str(sr)[:3]),
            "R_S_logit": list(ci_str(rS)[:3]), "R_I_logit": list(ci_str(rI)[:3]),
            "clean_S_minus_I": list(ci_str(cd)[:3]), "sensitivity": sens,
            "models": sorted(m["name"] for w in ("S", "I") for m in G[key][w])}

    # D17_2: assignment spread (point estimates) and paired differences against world seed 0
    spread = {}
    for label, ks in (("world seeds 1-3", [f"w{ws}" for ws in WORLD_SEEDS]),
                      ("world seeds 0-3", ["w0"] + [f"w{ws}" for ws in WORLD_SEEDS])):
        for scale, j in (("logit", 0), ("raw", 1)):
            pts = np.array([res[k][j][0] for k in ks])
            spread[f"{label} ({scale})"] = {"points": {k: float(res[k][j][0]) for k in ks},
                                            "mean": float(pts.mean()), "sd": float(pts.std(ddof=1)),
                                            "min": float(pts.min()), "max": float(pts.max()), "n": len(ks)}
            rows.append({"part": "D17_2", "quantity": f"MWI across {label}, mean (SD) [min, max] ({scale})",
                         scale: f"{pts.mean():+.5f} ({pts.std(ddof=1):.5f}) [{pts.min():+.5f}, {pts.max():+.5f}]",
                         "n_seeds": str(len(ks))})
    diffs = {}
    for ws in WORLD_SEEDS:
        d, dr = res[f"w{ws}"][0] - res["w0"][0], res[f"w{ws}"][1] - res["w0"][1]
        diffs[f"w{ws}"] = {"logit": list(ci_str(d)[:3]), "raw": list(ci_str(dr)[:3]),
                           "p_inv_logit": p_inversion(d[1:]), "p_inv_raw": p_inversion(dr[1:])}
        rows.append({"part": "D17_2", "quantity": f"MWI (w{ws}) - MWI (w0) [paired, descriptive]",
                     "logit": ci_str(d)[3], "raw": ci_str(dr)[3],
                     "p_inv": f"{p_inversion(d[1:]):.4f}", "raw_p_inv": f"{p_inversion(dr[1:]):.4f}"})
    crit["D17_2_assignment_spread"] = {"spread": spread, "minus_w0": diffs}
    crit["D17_3_pooled"] = crit["P17_1_per_assignment"]["pooled"]
    sel = {}
    for p in sorted(glob.glob(f"{args.epochs}/*/train_log.json")):
        log = json.load(open(p))
        sel[os.path.basename(os.path.dirname(p))] = {k: log.get(k) for k in ("selected_epoch_clean", "selected_epoch_orig", "train_recipe")}
    crit["selected_epochs"] = sel

    os.makedirs(args.out, exist_ok=True)
    cols = ["part", "quantity", "logit", "raw", "R_S", "R_I", "R_I_verdict", "clean_S_minus_I", "clean_verdict",
            "n_seeds", "p_inv", "raw_p_inv", "note"]
    with open(f"{args.out}/effects.csv", "w", newline="") as f:
        f.write("# protocol v17, cleaned selection; mask mean over test masks 2000/3000/4000, seeds averaged inside "
                f"{nb} identifier-clustered resamples; w0 = v15conf A_none seeds 0-2\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    json.dump(crit, open(f"{args.out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(f"{r['part']:6s} | {r['quantity'][:66]:66s} | {r.get('logit', '')} | {r.get('raw', '')}")
    print("G17_convergence:", {k: round(v["mean_clean_auroc_world_I"], 4) for k, v in crit["G17_convergence"].items()})
    return crit


if __name__ == "__main__":
    main()
