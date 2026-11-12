"""Realized severities and planted cue strength of the confidence-only training worlds, per
training assignment (world seed), without any model.

The confidence-only (armruns) worlds do not reuse the missing-run severity draws summarized in
results/v7/analysis/severities_realized.csv: safe_motion/planted.py seeds the severity generator
with the tag sev|{world}|{strength}|armruns{seed}. This script rebuilds the optimization rows
exactly as experiments/train_asdpose.py does for the planted host (length protocol, subject-level
validation split, QC in train and validation), draws the severities and masks of World S and
World I for world seeds 0 (v15conf/v16) and 1-3 (v17), and reports per class the mean and SD of
the severity, the cue-only AUROC of the severity, and the cue-only AUROC of the measured arm
low-confidence fraction of the planted segment (the fourth SAFE-SMM quality statistic, computed
as safe_motion/remedies.py observed_groups does) with a child-clustered bootstrap interval.
Writes results/v17/analysis/assignment_severities.csv.
    python experiments/assignment_severities_v17.py [--world-seeds 0 1 2 3] [--n-boot 1000]
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                   # noqa: E402
from safe_motion import planted as P                                   # noqa: E402
from safe_motion import remedies as RM                                 # noqa: E402
from safe_motion.data import asdpose as D                              # noqa: E402

PROTO = "configs/protocols/pilot_asdpose_v1.yaml"
MECH, MODE, STRENGTH = "armruns", "confscale", "mid"


def host_rows(cfg):
    """The optimization rows (train + validation) of the planted host, as train_asdpose.py builds them."""
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    tr = [r for r in tr if D.passes_qc(r, cfg["qc_filter"])]
    va = [r for r in va if D.passes_qc(r, cfg["qc_filter"])]
    return tr + va


def measured_stat(rows, masks, data):
    stat = np.empty(len(rows))
    for i, r in enumerate(rows):
        start = r["byte_offset"] // 4
        arr = data[start:start + r["n_frames"] * 51].reshape(r["n_frames"], 17, 3).copy()
        stat[i] = RM.quality_stats(D.apply_planted(arr, masks[r["identifier"]], MODE))[3]
    return stat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="results/v17/analysis/assignment_severities.csv")
    ap.add_argument("--cache-dir", default=D.CACHE_DIR)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(PROTO))
    rows = host_rows(cfg)
    y = np.array([int(r["binary_label"]) for r in rows])
    child = np.array([r.get("child_id", str(r["identifier"]).split("_")[0]) for r in rows])
    boots = list(M.cluster_bootstrap_indices(child, args.n_boot, seed=0))
    data = D.load_cache(args.cache_dir)
    out = []
    for ws in args.world_seeds:
        for world in ("S", "I"):
            masks, eps = P.build_masks(rows, world, STRENGTH, ws, mechanism=MECH)
            stat = measured_stat(rows, masks, data)
            # AUROC in the direction of the planted association (non-SMM degraded more), as the
            # paper reports cue-only separation: the score is minus the degradation statistic.
            a_stat = M.auc(-stat, y)
            bs = np.array([M.auc(-stat[b], y[b]) for b in boots])
            lo, hi = M.ci(bs)
            out.append({"world_seed": ws, "world": world, "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum()),
                        "eps_mean_neg": eps[y == 0].mean(), "eps_mean_pos": eps[y == 1].mean(),
                        "eps_sd_neg": eps[y == 0].std(ddof=1), "eps_sd_pos": eps[y == 1].std(ddof=1),
                        "auc_eps": M.auc(-eps, y), "auc_stat": a_stat, "auc_stat_lo": lo, "auc_stat_hi": hi,
                        "stat_median": float(np.median(stat))})
            print(f"world seed {ws} world {world}: eps non-SMM {out[-1]['eps_mean_neg']:.3f} SMM {out[-1]['eps_mean_pos']:.3f}; "
                  f"AUROC(eps) {out[-1]['auc_eps']:.3f}; AUROC(arm low-conf fraction) {a_stat:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {args.out} ({len(out)} rows, {len(rows)} optimization rows)")
    return out


if __name__ == "__main__":
    main()
