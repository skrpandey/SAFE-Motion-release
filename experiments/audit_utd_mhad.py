"""UTD-MHAD dataset statistics and natural-cue audit (CPU only).

Computes the four tracking statistics of safe_motion.audit.natural_cue on the UTD-MHAD clips.
The Kinect v1 skeletons carry no tracking state, so these are constant; the 3D bone-length
coefficient of variation is added as a proxy (inferred joints break rigid bone length), with
clip length as a separate predictor. Leave-one-subject-out CV (8 subjects).

Usage: python experiments/audit_utd_mhad.py [--n-perm-lr 200] [--n-perm-gb 50]
Outputs: results/audit_utd_mhad/{audit.json, clip_stats.csv, summary.txt}
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.audit import natural_cue as NC
from safe_motion.data import utd_mhad as D

OUT = "results/audit_utd_mhad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm-lr", type=int, default=200)
    ap.add_argument("--n-perm-gb", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    rows = D.load_clips()
    out = {"config": vars(args)}
    L = np.array([r["n_frames"] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["subject"] for r in rows])
    present = {(r["action"], r["subject"], r["trial"]) for r in rows}
    absent = [f"a{a}_s{s}_t{t}" for a in range(1, 28) for s in range(1, 9) for t in range(1, 5)
              if (a, s, t) not in present]
    allxyz = np.concatenate([r["xyz"] for r in rows])
    frames_rep = sum(int((r["xyz"][1:] == r["xyz"][:-1]).all(axis=(1, 2)).sum()) for r in rows)
    joint_rep = sum(int((r["xyz"][1:] == r["xyz"][:-1]).all(axis=2).sum()) for r in rows)
    out["dataset"] = {"clips": len(rows), "frames": int(L.sum()), "subjects": len(set(groups)),
                      "classes": 27, "absent_clips": absent,
                      "len_pct_0_5_50_95_100": np.percentile(L, [0, 5, 50, 95, 100]).tolist(),
                      "zero_joint_frames": int((np.abs(allxyz).sum(-1) == 0).sum()),
                      "nan_values": int(np.isnan(allxyz).sum()),
                      "repeated_frames": frames_rep, "repeated_joint_frames": joint_rep,
                      "z_range_m": [float(allxyz[..., 2].min()), float(allxyz[..., 2].max())],
                      "clips_per_subject": dict(Counter(groups.tolist()))}
    log(f"UTD-MHAD: {len(rows)} clips, {L.sum()} frames, {len(set(groups))} subjects, 27 "
        f"actions; absent {absent}")
    log(f"length p0/p5/p50/p95/p100 {out['dataset']['len_pct_0_5_50_95_100']}; zero joint-frames "
        f"{out['dataset']['zero_joint_frames']}, NaN {out['dataset']['nan_values']}, repeated "
        f"frames {frames_rep}, repeated joint-frames {joint_rep}, z {out['dataset']['z_range_m']}")
    per_class = {}
    log("action                 clips  len p5  med  p95  mean")
    for k in range(27):
        m = y == k
        q = np.percentile(L[m], [5, 50, 95])
        per_class[D.ACTIONS[k]] = {"clips": int(m.sum()), "len_pct_5_50_95": q.tolist(),
                                   "len_mean": float(L[m].mean())}
        log(f"{D.ACTIONS[k]:22s} {m.sum():4d}  " + " ".join(f"{v:4.0f}" for v in q)
            + f" {L[m].mean():5.1f}")
    out["dataset"]["per_class"] = per_class

    Q = np.array([[NC.tracking_stats(r["kp"], D.ARM_JOINTS)[s] for s in NC.STAT_NAMES]
                  for r in rows])
    bcv = np.array([D.bone_length_cv(r["xyz"]) for r in rows])
    logL = np.log(L)
    with open(f"{OUT}/clip_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "action", "subject", "trial", "n_frames"] + NC.STAT_NAMES
                   + ["bone_len_cv"])
        for r, q, b in zip(rows, Q, bcv):
            w.writerow([r["identifier"], r["action"], r["subject"], r["trial"], r["n_frames"]]
                       + [f"{v:.6f}" for v in q] + [f"{b:.6f}"])
    out["stat_ranges"] = {s: [float(Q[:, i].min()), float(Q[:, i].max())]
                          for i, s in enumerate(NC.STAT_NAMES)}
    out["bone_len_cv_pct_5_50_95"] = np.percentile(bcv, [5, 50, 95]).tolist()
    out["spearman_length_vs_bonecv"] = float(spearmanr(L, bcv)[0])
    log(f"tracking statistic ranges: {out['stat_ranges']}")
    log(f"bone-length CV p5/p50/p95 {np.round(out['bone_len_cv_pct_5_50_95'], 4).tolist()}")

    folds = NC.make_folds(y, groups)          # leave one subject out
    out["audit"] = {}
    log("\naudit, 27 actions, leave-one-subject-out:")
    for fs, X in (("tracking", Q), ("length", logL), ("bone_len_cv", bcv),
                  ("bone_len_cv+length", np.c_[bcv, logL])):
        for kind, n_perm in (("lr", args.n_perm_lr), ("gb", args.n_perm_gb)):
            if fs == "tracking" and kind == "gb":
                n_perm = 0
            r = NC.evaluate(X, y, groups, kind, folds, n_perm=n_perm)
            out["audit"][f"{fs}|{kind}"] = r
            ci = r.get("macro_auc_ci", (np.nan, np.nan))
            log(f"  {fs:19s} {kind}: macro AUC {r['macro_auc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
                f"bacc {r['bacc']:.3f}"
                + (f"  null AUC {r['null_auc_mean']:.3f} (95th {r['null_auc_95']:.3f}) "
                   f"p={r['p_auc']:.3f}" if "p_auc" in r else "")
                + (f"  [{r['note']}]" if r.get("degenerate") else ""))
    for key in ("length|gb", "bone_len_cv|gb"):
        pc = out["audit"][key]["per_class_auc"]
        top = sorted(pc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:6]
        log(f"per-class OvR AUC {key}, six furthest from 0.5: "
            + ", ".join(f"{D.ACTIONS[k]} {v:.3f}" for k, v in top))

    with open(f"{OUT}/audit.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
