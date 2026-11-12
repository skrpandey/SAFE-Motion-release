"""Protocol v11 analysis (configs/protocols/transfer_v11.yaml): tests T1 to T9 on the external set.

Reads the score files of score_v11.py and within_set_v11.py under results/v11, the external
and ASDPose indexes and results/v11/folds.json. Video-clustered paired bootstrap over the 127
source videos (seeds averaged inside resamples), logit AUC beside raw AUC. Writes
results/v11/analysis/*.csv, criteria.json, label_noise_sample.csv and the T9 contact sheets.

    conda run -n torch python experiments/analyze_v11.py [--n-boot 2000] [--n-perm 200]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D
from analyze_v7 import logit, verdict, RowSet, stat

STIMMING = os.environ.get("SM_STIMMING_ROOT", "data/stimming")
EXTERNAL_CACHE = os.environ.get("SM_EXTERNAL_CACHE", f"{D.SMM_ROOT}/cache/external")
OUT = "results/v11"
Q = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]
MECHS = ("runs", "whole", "frozen", "armruns")
OP_POINT = 0.85
CELLS = ["A_none", "A_velocity", "A_dropout", "A_both", "B_coorddrop"]
PUBLISHED_GROUPS = {"v1": ["stage1_detector"],
                    "v2_family": ["stage1_detector_v2", "stage1_v2_seed1", "stage1_v2_seed2"],
                    "v3": ["stage1_detector_v3"],
                    "v3_noconf_family": ["stage1_v3_noconf_seed0", "stage1_v3_noconf_seed1", "stage1_v3_noconf_seed2"],
                    "base_no_remedies": ["wacv_ours_base"],
                    "abl_velocity": ["stage1_abl_velocity"], "abl_dropout": ["stage1_abl_dropout"],
                    "abl_velocity_v3norm": ["stage1_abl_velocity_v3norm"],
                    "abl_dropout_v3norm": ["stage1_abl_dropout_v3norm"],
                    "permuted_v1": ["stage1_permuted"], "permuted_v2": ["stage1_v2_permuted"],
                    "stgcn_base": ["wacv_stgcn_base"], "stgcn_remedies": ["wacv_stgcn_remedies"],
                    "agcn_base": ["wacv_agcn_base"], "agcn_remedies": ["wacv_agcn_remedies"]}
NEG_RE = re.compile(r"^(v_[A-Za-z]+_\d+(?:_\d+)?)_neg_(\d+)_(\d+)$")


def fmt(x, d=4):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (fmt(v) if isinstance(v, float) else v) for k, v in r.items()})


def load_scores(path):
    rd = list(csv.DictReader(open(path)))
    cols = [c for c in rd[0] if c not in ("identifier", "child_id", "label", "n_frames", "qc_pass")]
    return [r["identifier"] for r in rd], {c: np.array([float(r[c]) for r in rd]) for c in cols}


def groups_of(models):
    """name -> group id and the list of members (seed families averaged inside resamples)."""
    g = {}
    for name in models:
        m = re.match(r"^(ours_v7|ours_v10)_([AB]_[a-z]+)_s(\d)$", name)
        if m:
            g.setdefault(f"{m.group(1)}_{m.group(2)}", []).append(name)
            continue
        for gid, members in PUBLISHED_GROUPS.items():
            if name in members:
                g.setdefault(gid, []).append(name)
    return g


class Ext:
    """Score store with shared video resamples. aucs(group, col) -> array over resamples."""

    def __init__(self, score_dir, rows, n_boot):
        self.rows = rows
        self.ids = [r["identifier"] for r in rows]
        self.y = np.array([r["binary_label"] for r in rows])
        self.child = np.array([r["child_id"] for r in rows])
        self.rs = RowSet(self.ids, self.y, self.child, n_boot)
        self.S = {}
        for p in sorted(glob.glob(f"{score_dir}/*.csv")):
            ids, cols = load_scores(p)
            assert ids == self.ids, p
            self.S[os.path.basename(p)[:-4]] = cols
        self.groups = groups_of(self.S)
        self._cache = {}

    def aucs(self, group, col, subset=None):
        key = (group, col, None if subset is None else subset.tobytes())
        if key not in self._cache:
            per = []
            for name in self.groups[group]:
                s = self.S[name][col]
                if subset is None:
                    per.append(self.rs.aucs(s))
                else:
                    per.append(np.array([M.auc(s[idx][subset[idx]], self.y[idx][subset[idx]])
                                         for idx in self.rs.boots]))
            self._cache[key] = np.mean(per, axis=0)
        return self._cache[key]

    def point_ci(self, arr):
        st = stat(arr)
        return st["point"], st["ci"][0], st["ci"][1]


def rec(name, arr, margin=0.05, **extra):
    st = stat(arr)
    return {"contrast": name, "point": st["point"], "ci_lo": st["ci"][0], "ci_hi": st["ci"][1],
            "verdict": verdict(st["point"], st["ci"], margin), **extra}


# T1 quality-only
def quality_only(rows, folds, rs, n_perm, asd_rows=None, seed=0):
    X = np.array([[r[q] for q in Q] for r in rows])
    y = np.array([r["binary_label"] for r in rows])
    res = {}
    if asd_rows is not None:
        Xa = np.array([[r[q] for q in Q] for r in asd_rows])
        ya = np.array([r["binary_label"] for r in asd_rows])
        sc = StandardScaler().fit(Xa)
        lr = LogisticRegression(max_iter=5000).fit(sc.transform(Xa), ya)
        s = lr.predict_proba(sc.transform(X))[:, 1]
        res["asdpose_fit_lr"] = {"scores": s, "aucs": rs.aucs(s)}
    for kind in ("lr", "gb"):
        def oof(yy):
            o = np.full(len(yy), np.nan)
            for fv in folds:
                te = np.array([r["child_id"] in set(fv) for r in rows])
                tr = ~te
                if kind == "lr":
                    sc = StandardScaler().fit(X[tr])
                    clf = LogisticRegression(max_iter=5000, class_weight="balanced").fit(sc.transform(X[tr]), yy[tr])
                    o[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
                else:
                    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05,
                                                         class_weight="balanced").fit(X[tr], yy[tr])
                    o[te] = clf.predict_proba(X[te])[:, 1]
            return o
        s = oof(y)
        rng = np.random.RandomState(seed)
        null = np.array([M.auc(oof(rng.permutation(y)), y) for _ in range(n_perm)])
        obs = M.auc(s, y)
        res[f"within_set_{kind}"] = {"scores": s, "aucs": rs.aucs(s), "null": null,
                                     "p": float((np.sum(null >= obs) + 1) / (n_perm + 1))}
    return res


# T7 calibration
def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        k = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if k.any():
            e += k.mean() * abs(p[k].mean() - y[k].mean())
    return float(e)


def fit_temperature(z, y):
    def nll(logT):
        q = 1 / (1 + np.exp(-z / np.exp(logT)))
        q = np.clip(q, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    r = minimize_scalar(nll, bounds=(-4, 4), method="bounded")
    return float(np.exp(r.x))


def best_f1_threshold(p, y):
    ths = np.unique(np.round(p, 4))
    best, bt = -1, 0.5
    for t in ths:
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        if tp == 0:
            continue
        prec, recall = tp / pred.sum(), tp / (y == 1).sum()
        f1 = 2 * prec * recall / (prec + recall)
        if f1 > best:
            best, bt = f1, float(t)
    return bt


def op_metrics(p, y, t):
    pred = p >= t
    tp = (pred & (y == 1)).sum()
    return {"precision": float(tp / pred.sum()) if pred.sum() else float("nan"),
            "recall": float(tp / (y == 1).sum()), "firing": float(pred.mean()),
            "false_alarm": float(pred[y == 0].mean())}


def calibration(ext, folds):
    rows_out = []
    for name, cols in ext.S.items():
        p = np.clip(cols["clean"], 1e-6, 1 - 1e-6)
        y = ext.y
        z = logit(p)
        after = np.full(len(p), np.nan)
        thr_before = np.full(len(p), np.nan)
        thr_after = np.full(len(p), np.nan)
        auc_ok = True
        temps = []
        for fv in folds:
            te = np.array([r["child_id"] in set(fv) for r in ext.rows])
            tr = ~te
            T = fit_temperature(z[tr], y[tr])
            temps.append(T)
            q_tr = 1 / (1 + np.exp(-z[tr] / T))
            after[te] = 1 / (1 + np.exp(-z[te] / T))
            thr_before[te] = best_f1_threshold(p[tr], y[tr])
            thr_after[te] = best_f1_threshold(q_tr, y[tr])
            if abs(M.auc(after[te], y[te]) - M.auc(p[te], y[te])) > 1e-9:
                auc_ok = False
        row = {"model": name, "temperature_mean": float(np.mean(temps)),
               "auc_unchanged_per_fold": auc_ok,
               "ece_before": ece(p, y), "ece_after": ece(after, y),
               "brier_before": float(np.mean((p - y) ** 2)), "brier_after": float(np.mean((after - y) ** 2))}
        for tag, pp, thr in (("before", p, thr_before), ("after", after, thr_after)):
            m85 = op_metrics(pp, y, OP_POINT)
            row.update({f"{tag}_op085_{k}": v for k, v in m85.items()})
            pred = pp >= thr
            tp = (pred & (y == 1)).sum()
            row.update({f"{tag}_f1thr_precision": float(tp / pred.sum()) if pred.sum() else float("nan"),
                        f"{tag}_f1thr_recall": float(tp / (y == 1).sum()),
                        f"{tag}_f1thr_firing": float(pred.mean()),
                        f"{tag}_f1thr_mean_threshold": float(np.nanmean(thr))})
        rows_out.append(row)
    return rows_out


# T8 / T9 helpers
def video_frames(rows):
    import cv2
    counts = {}
    for r in rows:
        vid, cat = video_of(r)
        if vid in counts:
            continue
        path = f"{STIMMING}/unified/{cat}/{vid}.mp4"
        if not os.path.exists(path):
            hits = glob.glob(f"{STIMMING}/unified/*/{vid}.mp4")
            path = hits[0] if hits else None
        if path is None:
            counts[vid] = (None, None)
            continue
        c = cv2.VideoCapture(path)
        counts[vid] = (int(c.get(cv2.CAP_PROP_FRAME_COUNT)), path)
        c.release()
    return counts


def video_of(r):
    m = NEG_RE.match(r["identifier"])
    if m:
        z = np.load(f"{STIMMING}/external-negatives/{r['identifier']}.npz")
        return m.group(1), str(z["category"])
    return r["identifier"], r["action_name"]


def span_of(r):
    m = NEG_RE.match(r["identifier"])
    if m:
        return int(m.group(2)), int(m.group(3))
    z = np.load(f"{STIMMING}/ntu-format-rtm/{r['action_name']}/{r['identifier']}.npz")
    return int(z["seg_start_frame"]), int(z["seg_end_frame"])


def contact_sheet(path, start, end, out, n=8, width=1200):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    for t in np.linspace(start, max(start, end - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        tw = width // n
        frames.append(cv2.resize(fr, (tw, int(h * tw / w))))
    cap.release()
    if not frames:
        return False
    hmin = min(f.shape[0] for f in frames)
    cv2.imwrite(out, np.concatenate([f[:hmin] for f in frames], axis=1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--base", default=OUT)
    ap.add_argument("--skip-sheets", action="store_true")
    args = ap.parse_args()
    base = args.base
    A = f"{base}/analysis"
    os.makedirs(A, exist_ok=True)
    rows = D.load_index(f"{EXTERNAL_CACHE}/index.csv")
    folds = json.load(open(f"{base}/folds.json"))["folds"]
    ext = Ext(f"{base}/scores", rows, args.n_boot)
    rs = ext.rs
    crit = {"n_boot": args.n_boot, "n_perm": args.n_perm, "n_rows": len(rows),
            "n_videos": int(len(set(ext.child)))}
    print(f"models {len(ext.S)}, groups {len(ext.groups)}", flush=True)

    # T1
    asd = [r for r in D.load_index() if r["split"] == "train"]
    t1 = quality_only(rows, folds, rs, args.n_perm, asd_rows=asd)
    t1_rows = []
    for k, v in t1.items():
        p, lo, hi = ext.point_ci(v["aucs"])
        t1_rows.append({"classifier": k, "auc": p, "ci_lo": lo, "ci_hi": hi,
                        "null_mean": float(np.mean(v["null"])) if "null" in v else float("nan"),
                        "null_p95": float(np.percentile(v["null"], 95)) if "null" in v else float("nan"),
                        "perm_p": v.get("p", float("nan")), "asdpose_standard_reference": 0.714})
    write_csv(f"{A}/t1_quality.csv", t1_rows)
    gb = next(r for r in t1_rows if r["classifier"] == "within_set_gb")
    lr_ = next(r for r in t1_rows if r["classifier"] == "within_set_lr")
    worst_hi = max(gb["ci_hi"], lr_["ci_hi"])
    crit["T1"] = {"within_set_gb": gb, "within_set_lr": lr_,
                  "release_specific": bool(worst_hi < 0.60 and min(gb["perm_p"], lr_["perm_p"]) > 0.05),
                  "reading": "release-specific" if (worst_hi < 0.60 and min(gb["perm_p"], lr_["perm_p"]) > 0.05)
                  else "weaker on this set"}

    # T2
    asd_auc = asdpose_aucs(ext)
    cats = sorted({r["action_name"] for r in rows if r["binary_label"] == 1})
    t2 = []
    for gid in ext.groups:
        a = ext.aucs(gid, "clean")
        p, lo, hi = ext.point_ci(a)
        la = stat(logit(a))
        row = {"group": gid, "n_members": len(ext.groups[gid]), "auc": p, "ci_lo": lo, "ci_hi": hi,
               "logit_auc": la["point"], "logit_lo": la["ci"][0], "logit_hi": la["ci"][1],
               "asdpose_auc": asd_auc.get(gid, float("nan"))}
        sc = np.mean([ext.S[n]["clean"] for n in ext.groups[gid]], axis=0)
        m85 = op_metrics(sc, ext.y, OP_POINT)
        row.update({"firing_085": m85["firing"], "false_alarm_085": m85["false_alarm"], "recall_085": m85["recall"]})
        for c in cats:
            sub = (ext.y == 0) | np.array([r["action_name"] == c for r in rows])
            pc, lc, hc = ext.point_ci(ext.aucs(gid, "clean", subset=sub))
            row[f"auc_{c}"] = pc
            row[f"auc_{c}_lo"], row[f"auc_{c}_hi"] = lc, hc
        t2.append(row)
    write_csv(f"{A}/t2_transfer.csv", t2)
    contrasts = []

    def contrast(name, g1, g2, col="clean"):
        if g1 in ext.groups and g2 in ext.groups:
            contrasts.append(rec(name, ext.aucs(g1, col) - ext.aucs(g2, col), margin=0.02,
                                 group_a=g1, group_b=g2, scale="raw AUC"))
            contrasts.append(rec(name, logit(ext.aucs(g1, col)) - logit(ext.aucs(g2, col)),
                                 group_a=g1, group_b=g2, scale="logit AUC"))
    contrast("v3 - v1", "v3", "v1")
    contrast("v3 - base_no_remedies", "v3", "base_no_remedies")
    contrast("v2_family - v1", "v2_family", "v1")
    for c in CELLS[1:]:
        contrast(f"ours_v7 {c} - A_none", f"ours_v7_{c}", "ours_v7_A_none")
    for c in CELLS:
        contrast(f"ours_v7 {c} - ours_v10 {c}", f"ours_v7_{c}", f"ours_v10_{c}")
    write_csv(f"{A}/t2_contrasts.csv", contrasts)
    crit["T2"] = {"released_checkpoint": "not available in this repository (base project's "
                  "score_baseline.py and the authors' model.pth loader were not retained); the "
                  "contrast is reported against v1 and against the no-remedy base cell instead",
                  "n_groups": len(t2)}

    # T3
    t3 = []
    for gid in ext.groups:
        d_raw = ext.aucs(gid, "conf0") - ext.aucs(gid, "clean")
        d_log = logit(ext.aucs(gid, "conf0")) - logit(ext.aucs(gid, "clean"))
        t3.append({**rec("conf0 - clean (raw AUC)", d_raw, margin=0.02, group=gid,
                         auc_conf0=ext.point_ci(ext.aucs(gid, "conf0"))[0]),
                   "logit_point": stat(d_log)["point"], "logit_lo": stat(d_log)["ci"][0],
                   "logit_hi": stat(d_log)["ci"][1]})
    write_csv(f"{A}/t3_confidence.csv", t3)

    # T4
    mp_dir = f"{base}/scores_mediapipe"
    gate = json.load(open(f"{base}/mediapipe_gate.json")) if os.path.exists(f"{base}/mediapipe_gate.json") else None
    t4, t4q = [], []
    if os.path.isdir(mp_dir) and glob.glob(f"{mp_dir}/*.csv"):
        mp_rows = D.load_index(f"{base}/mediapipe_cache/index.csv")
        mp = Ext(mp_dir, mp_rows, args.n_boot)
        assert mp.ids == ext.ids
        fallback = bool(gate and gate.get("fallback_applies"))
        pos = ext.y == 1
        for gid in ext.groups:
            if gid not in mp.groups:
                continue
            a_r, a_m = ext.aucs(gid, "clean"), mp.aucs(gid, "clean")
            row = rec("mediapipe - rtmpose (raw AUC)", a_m - a_r, margin=0.02, group=gid,
                      auc_rtmpose=stat(a_r)["point"], auc_mediapipe=stat(a_m)["point"])
            dl = logit(a_m) - logit(a_r)
            row.update({"logit_point": stat(dl)["point"], "logit_lo": stat(dl)["ci"][0], "logit_hi": stat(dl)["ci"][1]})
            # positive-class score sensitivity (the fallback quantity, always reported)
            sr = np.mean([ext.S[n]["clean"] for n in ext.groups[gid]], axis=0)
            sm = np.mean([mp.S[n]["clean"] for n in mp.groups[gid]], axis=0)
            row["pos_mean_logit_shift"] = float(np.mean(logit(sm[pos]) - logit(sr[pos])))
            row["pos_frac_lower_under_mediapipe"] = float(np.mean(sm[pos] < sr[pos]))
            row["fallback_positive_only"] = fallback
            t4.append(row)
        # the ASDPose-fitted quality score is applied with its direction fixed
        t1m = quality_only(mp_rows, folds, mp.rs, args.n_perm, asd_rows=asd)
        for k, v in t1m.items():
            p, lo, hi = mp.point_ci(v["aucs"])
            t4q.append({"extractor": "mediapipe", "classifier": k, "auc": p, "ci_lo": lo, "ci_hi": hi,
                        "perm_p": v.get("p", ""), "null_p95": float(np.percentile(v["null"], 95)) if "null" in v else ""})
        for q in Q:
            for ex, rr in (("rtmpose", rows), ("mediapipe", mp_rows)):
                for lab in (0, 1):
                    v = np.array([r[q] for r in rr if r["binary_label"] == lab])
                    t4q.append({"extractor": ex, "classifier": f"stat {q} class {lab}", "auc": float("nan"),
                                "ci_lo": float(np.median(v)), "ci_hi": float(np.mean(v)), "perm_p": float("nan"),
                                "null_p95": float("nan")})
        crit["T4"] = {"gate": {k: gate[k] for k in ("layout", "parity", "fallback_applies", "mediapipe_version")} if gate else None,
                      "fallback_positive_only": fallback}
    write_csv(f"{A}/t4_estimator.csv", t4)
    write_csv(f"{A}/t4_quality.csv", t4q)

    # T5
    t5 = []
    for c in CELLS:
        for mech in MECHS + ("gapfill",):
            seeds = (2000, 3000, 4000) if mech == "runs" else (2000,)
            for ms in seeds:
                cs, ci_ = f"{mech}_S_mid_m{ms}", f"{mech}_I_mid_m{ms}"
                g7, g10 = f"ours_v7_{c}", f"ours_v10_{c}"
                if g7 not in ext.groups:
                    continue
                r7 = logit(ext.aucs(g7, cs)) - logit(ext.aucs(g7, ci_))
                row = rec(f"R v7 {c} {mech} m{ms}", r7, cell=c, mechanism=mech, mask_seed=ms,
                          auc_S=stat(ext.aucs(g7, cs))["point"], auc_I=stat(ext.aucs(g7, ci_))["point"])
                if g10 in ext.groups:
                    r10 = logit(ext.aucs(g10, cs)) - logit(ext.aucs(g10, ci_))
                    d = stat(r7 - r10)
                    row.update({"R_ref": stat(r10)["point"], "R_ref_lo": stat(r10)["ci"][0], "R_ref_hi": stat(r10)["ci"][1],
                                "diff_point": d["point"], "diff_lo": d["ci"][0], "diff_hi": d["ci"][1],
                                "diff_excludes_zero": bool(d["ci"][0] > 0 or d["ci"][1] < 0)})
                t5.append(row)
    write_csv(f"{A}/t5_operators.csv", t5)

    # T6
    t6, t6c = [], []
    dfr_dir = f"{base}/dfr"
    if os.path.isdir(dfr_dir):
        dfr = {os.path.basename(p)[:-4]: np.array([float(r["dfr"]) for r in csv.DictReader(open(p))])
               for p in sorted(glob.glob(f"{dfr_dir}/*.csv"))}
        for gid, members in ext.groups.items():
            if not all(n in dfr for n in members):
                continue
            a_full = ext.aucs(gid, "clean")
            a_dfr = np.mean([rs.aucs(dfr[n]) for n in members], axis=0)
            row = rec("dfr - full (raw AUC)", a_dfr - a_full, margin=0.02, group=gid,
                      auc_full=stat(a_full)["point"], auc_dfr=stat(a_dfr)["point"],
                      auc_dfr_lo=stat(a_dfr)["ci"][0], auc_dfr_hi=stat(a_dfr)["ci"][1])
            t6.append(row)
    ws_dir = f"{base}/within_set"
    if os.path.isdir(ws_dir):
        by_arch = defaultdict(list)
        for p in sorted(glob.glob(f"{ws_dir}/*.csv")):
            arch = os.path.basename(p).split("_s")[0]
            by_arch[arch].append(np.array([float(r["oof"]) for r in csv.DictReader(open(p))]))
        for arch, ss in by_arch.items():
            a = np.mean([rs.aucs(s) for s in ss], axis=0)
            p, lo, hi = ext.point_ci(a)
            t6c.append({"arch": arch, "n_seeds": len(ss), "oof_auc": p, "ci_lo": lo, "ci_hi": hi,
                        "per_seed": " ".join(f"{M.auc(s, ext.y):.3f}" for s in ss)})
    write_csv(f"{A}/t6_dfr.csv", t6)
    write_csv(f"{A}/t6_ceiling.csv", t6c)

    # T7
    t7 = calibration(ext, folds)
    write_csv(f"{A}/t7_calibration.csv", t7)
    crit["T7"] = {"auc_unchanged_every_model": bool(all(r["auc_unchanged_per_fold"] for r in t7))}

    # T8
    counts = video_frames(rows)
    pos_frac = np.array([span_of(r)[0] / counts[video_of(r)[0]][0] if counts[video_of(r)[0]][0] else np.nan
                         for r in rows])
    ok = np.isfinite(pos_frac)
    a_pos = rs.aucs(np.where(ok, pos_frac, 0.5))
    rng = np.random.RandomState(0)
    null8 = np.array([M.auc(pos_frac[ok], rng.permutation(ext.y[ok])) for _ in range(args.n_perm)])
    obs8 = M.auc(pos_frac[ok], ext.y[ok])
    oof8 = np.full(len(rows), np.nan)
    for fv in folds:
        te = np.array([r["child_id"] in set(fv) for r in rows]) & ok
        tr = ~te & ok
        lr8 = LogisticRegression().fit(pos_frac[tr, None], ext.y[tr])
        oof8[te] = lr8.predict_proba(pos_frac[te, None])[:, 1]
    p8, lo8, hi8 = ext.point_ci(a_pos)
    t8 = [{"statistic": "start position fraction, raw AUC", "auc": obs8, "ci_lo": lo8, "ci_hi": hi8,
           "null_p95": float(np.percentile(null8, 95)), "null_p05": float(np.percentile(null8, 5)),
           "perm_p_two_sided": float((np.sum(np.abs(null8 - 0.5) >= abs(obs8 - 0.5)) + 1) / (args.n_perm + 1)),
           "n_windows": int(ok.sum()), "no_position_cue": bool(lo8 <= 0.5 <= hi8)},
          {"statistic": "video-grouped 5-fold logistic on position, OOF AUC",
           "auc": M.auc(oof8[ok], ext.y[ok]), "ci_lo": float("nan"), "ci_hi": float("nan"),
           "null_p95": float("nan"), "null_p05": float("nan"), "perm_p_two_sided": float("nan"),
           "n_windows": int(ok.sum()), "no_position_cue": None}]
    write_csv(f"{A}/t8_position.csv", t8)
    crit["T8"] = t8[0]

    # T9 sample and sheets
    neg = [r for r in rows if r["binary_label"] == 0]
    rng = np.random.RandomState(0)
    sample = [neg[i] for i in sorted(rng.choice(len(neg), 100, replace=False))]
    sheet_dir = f"{base}/label_noise_sheets"
    os.makedirs(sheet_dir, exist_ok=True)
    srows = []
    for r in sample:
        vid, cat = video_of(r)
        s, e = span_of(r)
        path = counts[vid][1]
        made = False
        if path and not args.skip_sheets:
            made = contact_sheet(path, s, e, f"{sheet_dir}/{r['identifier']}.png")
        srows.append({"identifier": r["identifier"], "video_id": vid, "category_of_video": cat,
                      "start_frame": s, "end_frame": e, "sheet": f"{r['identifier']}.png" if made else "",
                      "answer": ""})
    write_csv(f"{base}/label_noise_sample.csv", srows)
    crit["T9"] = {"status": "pending user viewing", "n_sampled": len(srows),
                  "n_sheets": sum(1 for r in srows if r["sheet"]),
                  "answer_file": f"{base}/label_noise_answers.csv (columns identifier, answer in yes/no/unsure)"}
    json.dump(crit, open(f"{A}/criteria.json", "w"), indent=1, default=float)
    print("done", flush=True)


def asdpose_aucs(ext):
    """ASDPose AUC of the same models from their own score files (full test split, clean)."""
    out = {}
    for gid, members in ext.groups.items():
        vals = []
        for n in members:
            m = re.match(r"^ours_(v7|v10)_([AB]_[a-z]+)_s(\d)$", n)
            if m:
                v, cell, s = m.groups()
                p = f"results/v7/scores/v7_real_{cell}_s{s}.csv" if v == "v7" else \
                    f"results/v10/scores/v10_real_I_{cell}_s{s}.csv"
            else:
                p = f"results/pilots/pilot_v5/scores/{n}.csv"
            if os.path.exists(p):
                rd = list(csv.DictReader(open(p)))
                lab = [int(r.get("label", r.get("binary_label"))) for r in rd]
                vals.append(M.auc([float(r["clean"]) for r in rd], lab))
        if vals:
            out[gid] = float(np.mean(vals))
    return out


if __name__ == "__main__":
    main()
