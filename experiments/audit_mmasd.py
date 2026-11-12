"""MMASD dataset statistics and natural-cue audit (CPU only).

Tests whether the four tracking statistics (safe_motion.audit.natural_cue) predict the activity
label under child-grouped CV, with length as a separate predictor. Also covers the clip tag
(y / n / i, meaning undocumented), track vs maxconf person selection, and child-level Spearman
correlations with ADOS scores (descriptive only).
    python experiments/audit_mmasd.py [--n-perm-lr 200] [--n-perm-gb 100]
Writes results/audit_mmasd/{audit.json, clip_stats.csv, summary.txt}.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.audit import natural_cue as NC
from safe_motion.data import mmasd as D

OUT = "results/audit_mmasd"


def stats_table(rows, joints):
    arm = D.ARM_JOINTS_COCO17 if joints == "coco17" else D.ARM_JOINTS_BODY25
    sel = D.COCO17_FROM_BODY25 if joints == "coco17" else slice(None)
    return np.array([[NC.tracking_stats(r["kp25"][:, sel], arm)[k] for k in NC.STAT_NAMES]
                     for r in rows])


def run_audit(Q, logL, y, groups, folds, n_lr, n_gb, log):
    res = {}
    for fs, X in (("tracking", Q), ("length", logL[:, None]),
                  ("tracking+length", np.c_[Q, logL])):
        for kind, n_perm in (("lr", n_lr), ("gb", n_gb)):
            r = NC.evaluate(X, y, groups, kind, folds, n_perm=n_perm)
            res[f"{fs}|{kind}"] = r
            log(f"  {fs:16s} {kind}: macro AUC {r['macro_auc']:.3f} "
                f"[{r.get('macro_auc_ci', (np.nan, np.nan))[0]:.3f}, "
                f"{r.get('macro_auc_ci', (np.nan, np.nan))[1]:.3f}]  bacc {r['bacc']:.3f}"
                + (f"  null AUC {r['null_auc_mean']:.3f} (95th {r['null_auc_95']:.3f}) "
                   f"p={r['p_auc']:.3f}" if "p_auc" in r else "")
                + f"  ({r.get('seconds', 0)} s)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm-lr", type=int, default=200)
    ap.add_argument("--n-perm-gb", type=int, default=100)
    ap.add_argument("--n-folds", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    out = {"config": vars(args)}
    rows, classes = D.load_clips("track", joints="body25")
    for r in rows:
        r["kp25"] = r["kp"]
    out["classes"] = classes

    # dataset statistics
    raw = np.load(f"{D.CACHE}/mmasd_raw.npz")
    folders = [str(n) for n in raw["names"]]
    empty = Counter(D.parse_clip_name(n)["activity"] for n, k in zip(folders, raw["n_frames"]) if k == 0)
    L = np.array([r["n_frames"] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["child_id"] for r in rows])
    log(f"MMASD: {len(folders)} clip folders, {len(rows)} non-empty clips, {L.sum()} frames, "
        f"{len(set(groups))} children, {len(classes)} classes with frames")
    log(f"empty folders by activity: {dict(empty)}")
    per_class = {}
    log("class                     clips children  len p5   p25   med   p75   p95   mean")
    for k, c in enumerate(classes):
        m = y == k
        q = np.percentile(L[m], [5, 25, 50, 75, 95])
        per_class[c] = {"clips": int(m.sum()), "children": int(len(set(groups[m]))),
                        "len_pct_5_25_50_75_95": q.tolist(), "len_mean": float(L[m].mean())}
        log(f"{c:25s} {m.sum():5d} {len(set(groups[m])):6d}  " + " ".join(f"{v:5.0f}" for v in q)
            + f" {L[m].mean():6.1f}")
    clips_per_child = Counter(groups.tolist())
    out["dataset"] = {"folders": len(folders), "clips": len(rows), "frames": int(L.sum()),
                      "children": len(clips_per_child), "empty_by_activity": dict(empty),
                      "per_class": per_class,
                      "clips_per_child_pct_0_50_100": np.percentile(list(clips_per_child.values()),
                                                                     [0, 50, 100]).tolist(),
                      "days": dict(Counter(r["day"] for r in rows)),
                      "tags": dict(Counter(r["tag"] for r in rows)),
                      "len_pct_5_50_95": np.percentile(L, [5, 50, 95]).tolist()}
    log(f"clips per child min/median/max {out['dataset']['clips_per_child_pct_0_50_100']}; "
        f"days {out['dataset']['days']}; tags {out['dataset']['tags']}")
    tag_by_class = {c: dict(Counter(r["tag"] for r in rows if r["label"] == k))
                    for k, c in enumerate(classes)}
    out["dataset"]["tag_by_class"] = tag_by_class

    # selection diagnostics
    npers = np.array([r["n_persistent"] for r in rows])
    sh = np.array([r["sel_shortest"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    out["selection"] = {"people_per_frame_median": float(np.median([r["n_people_mean"] for r in rows])),
                        "n_persistent_tracks": dict(Counter(npers.tolist())),
                        "selected_coverage_pct_5_50": np.percentile(cov, [5, 50]).tolist(),
                        "selected_is_shortest_frac": float((sh == 1).mean() / max((sh >= 0).mean(), 1e-9)),
                        "n_clips_multi_persistent": int((sh >= 0).sum())}
    log(f"selection: {out['selection']}")

    # tracking statistics
    Q17 = stats_table(rows, "coco17")
    Q25 = stats_table(rows, "body25")
    logL = np.log(L)
    with open(f"{OUT}/clip_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "activity", "child_id", "day", "tag", "n_frames"]
                   + [f"{s}_coco17" for s in NC.STAT_NAMES] + [f"{s}_body25" for s in NC.STAT_NAMES])
        for r, a, b in zip(rows, Q17, Q25):
            w.writerow([r["identifier"], r["activity"], r["child_id"], r["day"], r["tag"],
                        r["n_frames"]] + [f"{v:.6f}" for v in a] + [f"{v:.6f}" for v in b])
    desc = {}
    log("tracking statistics (COCO-17 subset), per class mean:")
    log("class                     " + " ".join(f"{s:>13s}" for s in NC.STAT_NAMES))
    for k, c in enumerate(classes):
        desc[c] = Q17[y == k].mean(0).tolist()
        log(f"{c:25s} " + " ".join(f"{v:13.3f}" for v in desc[c]))
    out["stats_by_class_coco17"] = desc
    out["stats_overall_pct_5_50_95_coco17"] = {s: np.percentile(Q17[:, i], [5, 50, 95]).tolist()
                                               for i, s in enumerate(NC.STAT_NAMES)}
    out["frozen_frac_clips_over_0.10"] = float((Q17[:, 2] >= 0.10).mean())
    out["armlow_frac_clips_over_0.30"] = float((Q17[:, 3] >= 0.30).mean())
    out["spearman_length_vs_stats"] = {s: float(spearmanr(L, Q17[:, i])[0])
                                       for i, s in enumerate(NC.STAT_NAMES)}

    # natural-cue audit: activity (multi-class)
    folds = NC.make_folds(y, groups, n_splits=args.n_folds)
    log(f"\nactivity, {len(classes)} classes, child-grouped StratifiedGroupKFold({args.n_folds}):")
    out["activity_coco17"] = run_audit(Q17, logL, y, groups, folds, args.n_perm_lr,
                                       args.n_perm_gb, log)
    log("activity, BODY_25 statistics (no permutations):")
    out["activity_body25"] = run_audit(Q25, logL, y, groups, folds, 0, 0, log)
    for key in ("tracking|gb", "length|gb", "tracking+length|gb"):
        log(f"per-class OvR AUC {key}: " + ", ".join(
            f"{classes[k]} {v:.3f}" for k, v in out["activity_coco17"][key]["per_class_auc"].items()))

    # selection-rule sensitivity
    rows_mc, _ = D.load_clips("maxconf", joints="body25")
    assert [r["identifier"] for r in rows_mc] == [r["identifier"] for r in rows]
    for r in rows_mc:
        r["kp25"] = r["kp"]
    Qmc = stats_table(rows_mc, "coco17")
    log("activity, selection rule maxconf (no permutations):")
    out["activity_coco17_maxconf"] = run_audit(Qmc, logL, y, groups, folds, 0, 0, log)
    out["maxconf_vs_track_stat_spearman"] = {s: float(spearmanr(Q17[:, i], Qmc[:, i])[0])
                                             for i, s in enumerate(NC.STAT_NAMES)}

    # theme (3-class)
    themes = sorted({r["theme"] for r in rows})
    yt = np.array([themes.index(r["theme"]) for r in rows])
    ft = NC.make_folds(yt, groups, n_splits=args.n_folds)
    log(f"\ntheme {themes}:")
    out["theme_coco17"] = run_audit(Q17, logL, yt, groups, ft, 0, 0, log)

    # within theme (children nest in themes)
    child_themes = defaultdict(set)
    for r in rows:
        child_themes[r["child_id"]].add(r["theme"])
    out["child_theme_sets"] = dict(Counter("+".join(sorted(v)) for v in child_themes.values()))
    log(f"\nchildren by theme set: {out['child_theme_sets']}")
    out["within_theme"], out["pairs"] = {}, {}
    for th in themes:
        idx = np.flatnonzero(yt == themes.index(th))
        cls = sorted(set(y[idx]))
        yl = np.searchsorted(cls, y[idx])
        g = groups[idx]
        nf = min(args.n_folds, len(set(g)))
        f_th = NC.make_folds(yl, g, n_splits=nf)
        log(f"within theme {th}: {[classes[c] for c in cls]}, {len(idx)} clips, "
            f"{len(set(g))} children, {nf} folds")
        out["within_theme"][th] = run_audit(Q17[idx], logL[idx], yl, g, f_th, 0, 0, log)
        out["within_theme"][th]["n_children"] = len(set(g))
        for a in range(len(cls)):
            for b in range(a + 1, len(cls)):
                jdx = idx[(y[idx] == cls[a]) | (y[idx] == cls[b])]
                yb = (y[jdx] == cls[b]).astype(int)
                gb_ = groups[jdx]
                both = [c for c in set(gb_) if len(set(yb[gb_ == c])) == 2]
                fb = NC.make_folds(yb, gb_, n_splits=min(args.n_folds, len(set(gb_))))
                key = f"{classes[cls[a]]} vs {classes[cls[b]]}"
                res = {fs: NC.evaluate(X, yb, gb_, "gb", fb, n_boot=1000)
                       for fs, X in (("tracking", Q17[jdx]), ("length", logL[jdx]))}
                res["tracking_lr"] = NC.evaluate(Q17[jdx], yb, gb_, "lr", fb, n_boot=1000)
                res.update(clips=int(len(jdx)), children=len(set(gb_)),
                           children_with_both=len(both), pos_frac=float(yb.mean()))
                out["pairs"][key] = res
                log(f"  pair {key:45s} clips {len(jdx):4d} children {len(set(gb_)):2d} "
                    f"(both {len(both):2d})  tracking AUC gb {res['tracking']['macro_auc']:.3f} "
                    f"lr {res['tracking_lr']['macro_auc']:.3f}  length AUC "
                    f"{res['length']['macro_auc']:.3f}")

    # tag y vs n (binary, meaning undocumented)
    m = np.array([r["tag"] in ("y", "n") for r in rows])
    yy = np.array([int(r["tag"] == "y") for r in rows])[m]
    fy = NC.make_folds(yy, groups[m], n_splits=args.n_folds)
    log(f"\ntag y vs n ({m.sum()} clips; tag meaning undocumented):")
    out["tag_yn_coco17"] = run_audit(Q17[m], logL[m], yy, groups[m], fy, 0, 0, log)

    # ADOS, child level (descriptive)
    ados = D.load_ados()
    by_child = defaultdict(list)
    for r, q in zip(rows, Q17):
        by_child[r["child_id"]].append(np.r_[q, np.log(r["n_frames"])])
    matched = sorted(c for c in by_child if c in ados)
    out["ados"] = {"children_with_skeletons": len(by_child), "rows_in_table": len(ados),
                   "matched_children": len(matched),
                   "unmatched_skeleton_children": sorted(set(by_child) - set(ados)), "rho": {}}
    log(f"\nADOS: {len(matched)} of {len(by_child)} children matched by id")
    feats = NC.STAT_NAMES + ["log_length"]
    for score_name in ("comparison", "sa_total", "rrb_total", "total"):
        s = np.array([ados[c][score_name] for c in matched])
        ok = np.isfinite(s)
        for i, fn in enumerate(feats):
            v = np.array([np.mean(by_child[c], axis=0)[i] for c in matched])
            rho, p = spearmanr(v[ok], s[ok])
            out["ados"]["rho"][f"{fn}|{score_name}"] = {"rho": float(rho), "p": float(p),
                                                        "n": int(ok.sum())}
        log(f"  {score_name:10s} " + " ".join(
            f"{fn} {out['ados']['rho'][f'{fn}|{score_name}']['rho']:+.2f}" for fn in feats)
            + f" (n={ok.sum()})")

    with open(f"{OUT}/audit.json", "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
