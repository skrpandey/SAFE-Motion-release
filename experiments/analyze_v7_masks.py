"""Mask-averaged analysis of the ASDPose planted cells (v7 and v8). The mean over test-mask seeds
2000/3000/4000 is taken inside each child resample.

Per planted group: R per mask and as the mask mean with CI, mask range, seed and bootstrap SD, on
logit and raw scales, plus the +-0.05 margin verdict for World I. Effects: SE(A_none), remedy
contrasts SE(cell) - SE(A_none) and SE(mid) - SE(low), all on shared paired child resamples.
Reads results/v7/{registry.json,scores}; writes results/v7/analysis/{groups_masks,effects_masks}.csv.
    conda run -n torch python experiments/analyze_v7_masks.py [--n-boot 2000]
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                      # noqa: E402
from experiments.analyze_v7 import RowSet, align, load_scores, logit, verdict, p_inversion, R_MARGIN  # noqa: E402

MASK_SEEDS = (2000, 3000, 4000)
CELLS = ["A_none", "A_velocity", "A_dropout", "A_both", "B_coorddrop"]


def fmt_ci(arr, d=6):
    lo, hi = M.ci(arr[1:])
    return f"{arr[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    base, out = "results/v7", "results/v7/analysis"
    reg = [m for m in json.load(open(f"{base}/registry.json"))
           if m["world"] in ("S", "I") and m.get("world_seed", 0) == 0
           and os.path.exists(f"{base}/scores/{m['name']}.csv")]
    rowset, S, A, Araw = None, {}, {}, {}
    for m in reg:
        meta, sc = load_scores(f"{base}/scores/{m['name']}.csv")
        if rowset is None:
            rowset = RowSet(meta["identifier"], [int(v) for v in meta["label"]], meta["child_id"], args.n_boot)
        S[m["name"]] = align(rowset, meta["identifier"], meta["label"], sc)
    groups = defaultdict(list)
    for m in reg:
        groups[(m["proto"], m["world"], m["mechanism"], m["strength"], m["cell"])].append(m)

    def aucs(name, col):
        key = (name, col)
        if key not in A:
            A[key] = rowset.aucs(S[name][col])
        return A[key]

    def R_arrays(name, mech, st, ms):
        """(logit R, raw AUC difference) bootstrap arrays for one model and mask seed, or None."""
        cs, ci_ = f"{mech}_S_{st}_m{ms}", f"{mech}_I_{st}_m{ms}"
        if cs not in S[name]:
            return None
        aS, aI = aucs(name, cs), aucs(name, ci_)
        return logit(aS) - logit(aI), aS - aI

    def group_masks(g):
        """Dict mask -> (seed-mean logit array, seed-mean raw array, per-seed logit points,
        per-seed raw points), for the masks present."""
        proto, world, mech, st, cell = g
        res = {}
        for ms in MASK_SEEDS:
            per = [R_arrays(m["name"], mech, st, ms) for m in groups[g]]
            per = [p for p in per if p is not None]
            if not per:
                continue
            res[ms] = (np.mean([p[0] for p in per], axis=0), np.mean([p[1] for p in per], axis=0),
                       [p[0][0] for p in per], [p[1][0] for p in per])
        return res

    def mask_mean(res):
        """Mask mean inside each resample (logit, raw), per-seed mask-mean points for seed SD."""
        masks = sorted(res)
        lg = np.mean([res[ms][0] for ms in masks], axis=0)
        rw = np.mean([res[ms][1] for ms in masks], axis=0)
        n_seeds = len(res[masks[0]][2])
        seed_pts_lg = [np.mean([res[ms][2][k] for ms in masks]) for k in range(n_seeds)]
        seed_pts_rw = [np.mean([res[ms][3][k] for ms in masks]) for k in range(n_seeds)]
        return lg, rw, seed_pts_lg, seed_pts_rw, masks

    # groups_masks.csv
    G = {}
    grows = []
    for g in sorted(groups, key=str):
        proto, world, mech, st, cell = g
        res = group_masks(g)
        if not res:
            continue
        lg, rw, sp_lg, sp_rw, masks = mask_mean(res)
        G[g] = (lg, rw, sp_lg, sp_rw, masks, res)
        per_mask_lg = {ms: res[ms][0][0] for ms in masks}
        row = {"proto": proto, "world": world, "mechanism": mech, "strength": st, "cell": cell,
               "n_seeds": len(sp_lg), "n_masks": len(masks),
               "R_maskmean": fmt_ci(lg), "R_maskmean_raw": fmt_ci(rw),
               "R_per_mask": "; ".join(f"m{ms} {per_mask_lg[ms]:+.6f}" for ms in masks),
               "mask_range": f"{max(per_mask_lg.values()) - min(per_mask_lg.values()):.6f}",
               "sd_seeds": f"{np.std(sp_lg, ddof=1):.6f}", "sd_boot": f"{np.std(lg[1:], ddof=0):.6f}",
               "sd_seeds_raw": f"{np.std(sp_rw, ddof=1):.6f}"}
        if world == "I":
            vs = []
            for ms in masks:
                arr = res[ms][0]
                vs.append(f"m{ms} {verdict(arr[0], M.ci(arr[1:]), R_MARGIN)}")
            row["margin_verdict_per_mask"] = "; ".join(vs)
            row["margin_verdict_maskmean"] = verdict(lg[0], M.ci(lg[1:]), R_MARGIN)
            row["per_seed_per_mask_points"] = "; ".join(
                f"m{ms} " + "/".join(f"{v:+.3f}" for v in res[ms][2]) for ms in masks)
        else:
            row["margin_verdict_per_mask"] = row["margin_verdict_maskmean"] = row["per_seed_per_mask_points"] = ""
        grows.append(row)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/groups_masks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0])); w.writeheader(); w.writerows(grows)

    # effects_masks.csv
    def SE(proto, mech, st, cell):
        gS, gI = (proto, "S", mech, st, cell), (proto, "I", mech, st, cell)
        if gS not in G or gI not in G:
            return None
        lS, rS, spS, _, mS, resS = G[gS]
        lI, rI, spI, _, mI, resI = G[gI]
        masks = [ms for ms in mS if ms in mI]
        per_mask = {ms: resS[ms][0][0] - resI[ms][0][0] for ms in masks}
        seed_pts = [a - b for a, b in zip(spS, spI)]
        return {"lg": lS - lI, "raw": rS - rI, "per_mask": per_mask, "seed_pts": seed_pts,
                "R_S": lS[0], "R_I": lI[0], "R_I_arr": lI, "masks": masks}

    erows = []

    def add(proto, mech, st, contrast, lg, raw, extra=None):
        lo, hi = M.ci(lg[1:]); rlo, rhi = M.ci(raw[1:])
        row = {"proto": proto, "mechanism": mech, "strength": st, "contrast": contrast,
               "point": f"{lg[0]:+.6f}", "ci_lo": f"{lo:+.6f}", "ci_hi": f"{hi:+.6f}",
               "verdict": verdict(lg[0], (lo, hi), R_MARGIN),
               "raw_point": f"{raw[0]:+.6f}", "raw_ci_lo": f"{rlo:+.6f}", "raw_ci_hi": f"{rhi:+.6f}",
               "raw_excludes_zero": str(rlo > 0 or rhi < 0),
               "R_S": "", "R_I": "", "R_I_ci": "", "per_mask": "", "mask_range": "", "sd_seeds": "",
               # child-bootstrap SD of the mask-mean estimate
               "sd_boot": f"{np.std(lg[1:], ddof=0):.6f}",
               # post hoc, for the Holm sensitivity: smallest alpha at which the central (1 - alpha)
               # bootstrap interval excludes zero (alpha = 0.05 gives the interval rule)
               "p_inv": f"{p_inversion(lg[1:]):.4f}", "raw_p_inv": f"{p_inversion(raw[1:]):.4f}"}
        row.update(extra or {})
        erows.append(row)

    se_cache = {}
    for g in sorted(G, key=str):
        proto, world, mech, st, cell = g
        if world != "S":
            continue
        e = SE(proto, mech, st, cell)
        if e is None:
            continue
        se_cache[(proto, mech, st, cell)] = e
        pm = e["per_mask"]
        add(proto, mech, st, f"SE({cell})", e["lg"], e["raw"],
            {"R_S": f"{e['R_S']:+.6f}", "R_I": f"{e['R_I']:+.6f}", "R_I_ci": fmt_ci(e["R_I_arr"]),
             "per_mask": "; ".join(f"m{ms} {v:+.6f}" for ms, v in pm.items()),
             "mask_range": f"{max(pm.values()) - min(pm.values()):.6f}",
             "sd_seeds": f"{np.std(e['seed_pts'], ddof=1):.6f}"})
    for (proto, mech, st, cell), e in sorted(se_cache.items(), key=str):
        if cell == "A_none":
            continue
        base_e = se_cache.get((proto, mech, st, "A_none"))
        if base_e is None:
            continue
        add(proto, mech, st, f"SE({cell}) - SE(A_none)", e["lg"] - base_e["lg"], e["raw"] - base_e["raw"],
            {"R_S": f"{e['R_S']:+.6f}", "R_I": f"{e['R_I']:+.6f}"})
    mid, low = se_cache.get(("v7", "runs", "mid", "A_none")), se_cache.get(("v7", "runs", "low", "A_none"))
    if mid and low:
        add("v7", "runs", "mid-low", "SE(mid) - SE(low), A_none", mid["lg"] - low["lg"], mid["raw"] - low["raw"])
    with open(f"{out}/effects_masks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0])); w.writeheader(); w.writerows(erows)
    for r in erows:
        print(f"{r['proto']} {r['mechanism']:8s} {r['strength']:7s} {r['contrast']:34s} {r['point']} [{r['ci_lo']}, {r['ci_hi']}] "
              f"raw {r['raw_point']} [{r['raw_ci_lo']}, {r['raw_ci_hi']}] masks {r['per_mask']} sd_seeds {r['sd_seeds']}")


if __name__ == "__main__":
    main()
