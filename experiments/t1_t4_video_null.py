"""Post hoc video-restricted permutation null for the T1 and T4 quality-only AUCs.

Labels are permuted only within each source video (child_id), unlike the v11 nulls that permute
across all windows. Same video-grouped 5-fold out-of-fold fits as analyze_v11.quality_only;
1,000 permutations for logistic regression, 200 for gradient boosting.
Writes results/v11/analysis/t1_t4_video_null.csv.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_v11 as V  # noqa: E402
from safe_motion import metrics as M  # noqa: E402
from safe_motion.data import asdpose as D  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def oof(X, yy, rows, folds, kind):
    o = np.full(len(yy), np.nan)
    for fv in folds:
        te = np.array([r["child_id"] in set(fv) for r in rows]); tr = ~te
        if kind == "lr":
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), yy[tr])
            o[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        else:
            clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05,
                                                 class_weight="balanced").fit(X[tr], yy[tr])
            o[te] = clf.predict_proba(X[te])[:, 1]
    return o


def within_video(y, vid, rng):
    yp = y.copy()
    for v in np.unique(vid):
        k = np.flatnonzero(vid == v)
        yp[k] = y[rng.permutation(k)]
    return yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lr", type=int, default=1000)
    ap.add_argument("--n-gb", type=int, default=200)
    a = ap.parse_args()
    base = "results/v11"
    folds = json.load(open(f"{base}/folds.json"))["folds"]
    prior = {}
    for t, f in (("T1 RTMPose", "t1_quality.csv"), ("T4 MediaPipe", "t4_quality.csv")):
        for r in csv.DictReader(open(f"{base}/analysis/{f}")):
            prior[(t, r["classifier"])] = r.get("perm_p", "")
    out = []
    for test, idx in (("T1 RTMPose", f"{V.EXTERNAL_CACHE}/index.csv"), ("T4 MediaPipe", f"{base}/mediapipe_cache/index.csv")):
        rows = D.load_index(idx)
        X = np.array([[r[q] for q in V.Q] for r in rows], float)
        y = np.array([r["binary_label"] for r in rows]); vid = np.array([r["child_id"] for r in rows])
        n_mixed = sum(len(set(y[vid == v])) > 1 for v in np.unique(vid))
        for kind, n in (("lr", a.n_lr), ("gb", a.n_gb)):
            obs = M.auc(oof(X, y, rows, folds, kind), y)
            rng = np.random.RandomState(0)
            null = np.array([M.auc(oof(X, within_video(y, vid, rng), rows, folds, kind), y) for _ in range(n)])
            p = float((np.sum(null >= obs) + 1) / (n + 1))
            out.append({"test": test, "classifier": f"within_set_{kind}", "auc": round(obs, 4), "n_perm": n,
                        "null_mean": round(float(null.mean()), 4), "null_p95": round(float(np.percentile(null, 95)), 4),
                        "p_within_video": round(p, 4), "p_floor": round(1 / (n + 1), 4),
                        "p_across_windows_v11": prior.get((test, f"within_set_{kind}"), ""),
                        "videos_with_both_classes": int(n_mixed), "post_hoc": "yes"})
            print(out[-1], flush=True)
    with open(f"{base}/analysis/t1_t4_video_null.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)


if __name__ == "__main__":
    main()
