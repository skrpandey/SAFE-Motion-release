"""Analysis of the v9 replication round (configs/protocols/replication_v9.yaml, criteria R1 to R7).

Pools the out-of-fold scores of experiments/train_v9.py so every subject is scored once by a model
that never saw it, then computes R, SE, remedy contrasts, SE(mid) - SE(low), the permutation null of
SE, the crossed table and per-host gates on a subject-clustered paired bootstrap (statistics from
analyze_v7.py). The verdict column uses the v8 margins for reference only. Writes results/v9/analysis/*.csv, criteria.json and results/v9/<host>/gates.json.
    conda run -n torch python experiments/analyze_v9.py [--base results/v9_smoke] [--n-boot 2000]
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
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M
from analyze_v7 import logit, verdict, RowSet, align, R_MARGIN, AUC_MARGIN   # noqa: E402

MASK_SEEDS = (2000, 3000, 4000)
Q_COLS = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]
META = ("identifier", "subject_id", "label", "n_frames", "qc_pass") + tuple(Q_COLS)
GATES = {"mmasd_maracas": {"clean_auc_min": 0.65, "quality_only_band": 0.08},
         "mmasd_chest_squat": {"clean_auc_min": 0.65, "quality_only_band": 0.08},
         "utd_mhad_body_vs_arm": {"clean_auc_max": 0.97},
         "utd_mhad_arm_only": {"clean_auc_max": 0.97}}
DIR_RE = re.compile(r"^(?P<host>[^/]+)/(?:(?P<mech>runs|whole|frozen|armruns)_(?P<st>mid|low)/(?P<world>S|I)|real/none)"
                    r"/(?P<cell>[AB]_[a-z]+)/(?:seed(?P<seed>\d)|perm(?P<perm>\d+))/fold(?P<fold>\d+)$")


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else v


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def discover(base):
    """Models: key (host, mech, st, world, cell, seed, perm) -> sorted fold dirs."""
    models = defaultdict(list)
    for p in sorted(glob.glob(f"{base}/*/*/*/*/*/fold*/test_scores.csv") +
                    glob.glob(f"{base}/*/*/*/*/*/*/fold*/test_scores.csv")):
        rel = os.path.relpath(os.path.dirname(p), base)
        m = DIR_RE.match(rel)
        if not m:
            continue
        g = m.groupdict()
        key = (g["host"], g["mech"], g["st"], g["world"], g["cell"],
               int(g["seed"]) if g["seed"] is not None else 0,
               int(g["perm"]) if g["perm"] is not None else None)
        models[key].append(os.path.dirname(p))
    return models


def pool(dirs):
    """Concatenate the fold files of one model: meta dict of lists, score columns."""
    meta, cols = defaultdict(list), defaultdict(list)
    for d in dirs:
        rd = list(csv.DictReader(open(f"{d}/test_scores.csv")))
        for r in rd:
            for k in META:
                meta[k].append(r[k])
            for k, v in r.items():
                if k not in META:
                    cols[k].append(float(v))
    return dict(meta), {k: np.array(v) for k, v in cols.items()}


def pool_features(dirs):
    zs = [np.load(f"{d}/features.npz") for d in dirs]
    return {k: np.concatenate([z[k] for z in zs]) for k in ("feats", "n_frames", "label", "subject")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/v9")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-gates", action="store_true", help="do not load the host data")
    args = ap.parse_args()
    base, out = args.base, f"{args.base}/analysis"
    os.makedirs(out, exist_ok=True)
    models = discover(base)
    assert models, f"no fold outputs under {base}"

    # pool folds, shared resamples per host
    rowsets, S, meta, A = {}, {}, {}, {}
    for key, dirs in models.items():
        meta[key], S[key] = pool(dirs)
        host = key[0]
        if host not in rowsets:
            mt = meta[key]
            rowsets[host] = RowSet(mt["identifier"], [int(v) for v in mt["label"]],
                                   mt["subject_id"], args.n_boot)
    for key in models:
        rs = rowsets[key[0]]
        ids = meta[key]["identifier"]
        if set(ids) != set(rs.ids):              # a model with missing folds: score only its rows
            sub = RowSet(ids, [int(v) for v in meta[key]["label"]], meta[key]["subject_id"], args.n_boot)
            order = list(range(len(ids)))
            for c, s in S[key].items():
                A[(key, c)] = sub.aucs(s[order])
            continue
        S[key] = align(rs, ids, meta[key]["label"], S[key])
        for c, s in S[key].items():
            A[(key, c)] = rs.aucs(s)
    groups = defaultdict(list)                   # (host, mech, st, world, cell, perm) -> model keys
    for key in models:
        groups[(key[0], key[1], key[2], key[3], key[4], key[6])].append(key)

    def R_model(key, mech, st, ms):
        cS, cI = (key, f"{mech}_S_{st}_m{ms}"), (key, f"{mech}_I_{st}_m{ms}")
        if cS not in A or cI not in A:
            return None
        return logit(A[cS]) - logit(A[cI])

    def R_group(g, mech, st="mid", ms=2000):
        per = [r for r in (R_model(k, mech, st, ms) for k in groups.get(g, [])) if r is not None]
        if not per:
            return None, None
        return np.mean(per, axis=0), float(np.std([p[0] for p in per], ddof=1))

    def clean_group(g):
        per = [A[(k, "clean")] for k in groups.get(g, []) if (k, "clean") in A]
        return np.mean(per, axis=0) if per else None

    hosts = sorted({k[0] for k in models})
    mechs_of = {h: sorted({k[1] for k in models if k[0] == h and k[1]}) for h in hosts}

    # groups.csv
    grows = []
    for g in sorted(groups, key=str):
        host, mech, st, world, cell, perm = g
        ks = groups[g]
        cl = clean_group(g)
        row = {"host": host, "mechanism": mech, "strength": st, "world": world or "real", "cell": cell,
               "perm": perm, "n_seeds": len(ks), "n_rows": len(meta[ks[0]]["identifier"]),
               "auc_clean": f"{cl[0]:.4f} [{M.ci(cl[1:])[0]:.4f}, {M.ci(cl[1:])[1]:.4f}]"}
        conds = sorted({c.replace("_S_", "_X_") for c in S[ks[0]] if "_S_" in c})
        for cnd in conds:
            mch, _, stx, msd = cnd.split("_")
            r, sd = R_group(g, mch, stx, int(msd[1:]))
            if r is None:
                continue
            ci = M.ci(r[1:])
            row[f"R_{mch}_{stx}_{msd}"] = f"{r[0]:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] sd_seeds {sd:.4f}"
            raw = [float(np.mean([A[(k, f"{mch}_{w}_{stx}_{msd}")][0] for k in ks])) for w in ("S", "I")]
            row[f"AUC_{mch}_{stx}_{msd}"] = f"S {raw[0]:.4f} I {raw[1]:.4f}"
        grows.append(row)
    keys = sorted({k for r in grows for k in r}, key=lambda k: (k.startswith(("R_", "AUC_")), k))
    with open(f"{out}/groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(grows)

    # effects.csv and null.csv
    effects, null_rows = [], []

    def SE(host, mech, st, cell, ms=2000, perm=None):
        rS, sdS = R_group((host, mech, st, "S", cell, perm), mech, st, ms)
        rI, sdI = R_group((host, mech, st, "I", cell, perm), mech, st, ms)
        if rS is None or rI is None:
            return None
        return {"arr": rS - rI, "R_S": rS, "R_I": rI, "sd_S": sdS, "sd_I": sdI}

    for host in hosts:
        for mech in mechs_of[host]:
            for st in ("mid", "low"):
                b = SE(host, mech, st, "A_none")
                if b is None:
                    continue
                masks = [SE(host, mech, st, "A_none", ms)["arr"][0] for ms in MASK_SEEDS
                         if SE(host, mech, st, "A_none", ms) is not None]
                e, ci = b["arr"], M.ci(b["arr"][1:])
                perms = sorted({g[5] for g in groups if g[0] == host and g[1] == mech and g[2] == st
                                and g[5] is not None})
                nulls = [SE(host, mech, st, "A_none", perm=k) for k in perms]
                nulls = [n["arr"][0] for n in nulls if n is not None]
                p95 = float(np.percentile(nulls, 95)) if nulls else None
                emp_p = float((1 + sum(n >= e[0] for n in nulls)) / (1 + len(nulls))) if nulls else None
                effects.append({"host": host, "mechanism": mech, "strength": st, "contrast": "SE(A_none)",
                                "point": e[0], "ci_lo": ci[0], "ci_hi": ci[1], "verdict": verdict(e[0], ci, R_MARGIN),
                                "R_S": b["R_S"][0], "R_I": b["R_I"][0], "R_I_ci": M.ci(b["R_I"][1:]),
                                "sd_seeds_S": b["sd_S"], "sd_seeds_I": b["sd_I"],
                                "mask_range": (max(masks) - min(masks)) if masks else None,
                                "null_n": len(nulls), "null_p95": p95, "null_p": emp_p,
                                "above_null": (p95 is not None and e[0] > p95)})
                for k, n in zip(perms, nulls):
                    null_rows.append({"host": host, "mechanism": mech, "strength": st, "perm": k, "SE_perm": n})
                for cell in ("A_dropout", "B_coorddrop", "A_velocity", "A_both"):
                    r = SE(host, mech, st, cell)
                    if r is None:
                        continue
                    d, ci2 = r["arr"] - e, M.ci((r["arr"] - e)[1:])
                    effects.append({"host": host, "mechanism": mech, "strength": st,
                                    "contrast": f"SE({cell}) - SE(A_none)", "point": d[0], "ci_lo": ci2[0],
                                    "ci_hi": ci2[1], "verdict": verdict(d[0], ci2, R_MARGIN),
                                    "R_S": r["R_S"][0], "R_I": r["R_I"][0]})
                    cS, cI = clean_group((host, mech, st, "S", cell, None)), clean_group((host, mech, st, "I", cell, None))
                    bS, bI = clean_group((host, mech, st, "S", "A_none", None)), clean_group((host, mech, st, "I", "A_none", None))
                    if all(x is not None for x in (cS, cI, bS, bI)):
                        it = (cS - bS) - (cI - bI)
                        ci3 = M.ci(it[1:])
                        effects.append({"host": host, "mechanism": mech, "strength": st,
                                        "contrast": f"interaction({cell}) gain_S - gain_I (clean AUC)",
                                        "point": it[0], "ci_lo": ci3[0], "ci_hi": ci3[1],
                                        "verdict": verdict(it[0], ci3, AUC_MARGIN)})
                cS, cI = clean_group((host, mech, st, "S", "A_none", None)), clean_group((host, mech, st, "I", "A_none", None))
                dq, ci4 = cS - cI, M.ci((cS - cI)[1:])
                effects.append({"host": host, "mechanism": mech, "strength": st,
                                "contrast": "AUC_clean(f_S) - AUC_clean(f_I), A_none", "point": dq[0],
                                "ci_lo": ci4[0], "ci_hi": ci4[1], "verdict": verdict(dq[0], ci4, AUC_MARGIN)})
            # scaling: SE(mid) - SE(low), paired on the same resamples
            bm, bl = SE(host, mech, "mid", "A_none"), SE(host, mech, "low", "A_none")
            if bm is not None and bl is not None:
                d, ci5 = bm["arr"] - bl["arr"], M.ci((bm["arr"] - bl["arr"])[1:])
                effects.append({"host": host, "mechanism": mech, "strength": "mid-low",
                                "contrast": "SE(mid) - SE(low), A_none", "point": d[0], "ci_lo": ci5[0],
                                "ci_hi": ci5[1], "verdict": verdict(d[0], ci5, R_MARGIN)})
    keys = ["host", "mechanism", "strength", "contrast", "point", "ci_lo", "ci_hi", "verdict", "R_S", "R_I",
            "R_I_ci", "sd_seeds_S", "sd_seeds_I", "mask_range", "null_n", "null_p95", "null_p", "above_null"]
    with open(f"{out}/effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in effects:
            w.writerow({k: fmt(v) for k, v in r.items()})
    write_csv(f"{out}/null.csv", null_rows)

    # crossed table
    crossed = []
    for g in groups:
        host, mech, st, world, cell, perm = g
        if cell != "A_none" or world is None or st != "mid" or perm is not None:
            continue
        for other in mechs_of[host]:
            r, sd = R_group(g, other, "mid", 2000)
            if r is None:
                continue
            ci = M.ci(r[1:])
            crossed.append({"host": host, "trained_world": world, "trained_mechanism": mech,
                            "test_mechanism": other, "R": r[0], "ci_lo": ci[0], "ci_hi": ci[1],
                            "sd_seeds": sd, "verdict": verdict(r[0], ci, R_MARGIN)})
    write_csv(f"{out}/crossed.csv", crossed)

    # real cells (R7) beside the references
    real = []
    for g in groups:
        host, mech, st, world, cell, perm = g
        if world is not None:
            continue
        ks, rs = groups[g], rowsets[host]
        for mch in list(mechs_of[host]) + ["gapfill"]:
            r, sd = R_group(g, mch, "mid", 2000)
            if r is None:
                continue
            ci = M.ci(r[1:])
            ref, _ = R_group((host, mch, "mid", "I", "A_none", None), mch, "mid", 2000)
            cS, cI = f"{mch}_S_mid_m2000", f"{mch}_I_mid_m2000"
            sign = 2 * rs.y - 1
            d = sign * np.mean([logit(S[k][cS]) - logit(S[k][cI]) for k in ks], axis=0)
            dc = np.array([d[rs.child == c].mean() for c in np.unique(rs.child)])
            real.append({"host": host, "cell": cell, "mechanism_tested": mch, "R": r[0], "ci_lo": ci[0],
                         "ci_hi": ci[1], "sd_seeds": sd, "reference_R_I": ref[0] if ref is not None else None,
                         "reference_ci": M.ci(ref[1:]) if ref is not None else None,
                         "subjects_opposite_sign": int((np.sign(dc) == -np.sign(r[0])).sum()),
                         "n_subjects": int(len(dc))})
    write_csv(f"{out}/real.csv", real)

    # length recoverability (R6): real and planted A_none cells
    length = []
    for key, dirs in models.items():
        host, mech, st, world, cell, seed, perm = key
        if cell != "A_none" or perm is not None or not all(os.path.exists(f"{d}/features.npz") for d in dirs):
            continue
        z = pool_features(dirs)
        y, n, ch = z["label"], z["n_frames"].astype(float), z["subject"]
        s = np.array(S[key]["clean"]) if set(meta[key]["identifier"]) != set(rowsets[host].ids) else None
        if s is None:                            # pooled order was re-sorted; realign through identifiers
            ids = [i for d in dirs for i in [r["identifier"] for r in csv.DictReader(open(f"{d}/test_scores.csv"))]]
            s = S[key]["clean"][[rowsets[host].pos[i] for i in ids]]
        rho = {c: float(spearmanr(s[y == c], n[y == c]).correlation) if (y == c).sum() > 2 else float("nan")
               for c in (0, 1)}
        k = min(5, len(np.unique(ch)))
        pred = np.zeros(len(n))
        if k >= 2 and len(np.unique(n)) > 1:
            for tr, te in GroupKFold(n_splits=k).split(z["feats"], n, ch):
                pred[te] = Ridge(alpha=10.0).fit(z["feats"][tr], np.log(n[tr])).predict(z["feats"][te])
            r2 = float(1 - ((np.log(n) - pred) ** 2).sum() / ((np.log(n) - np.log(n).mean()) ** 2).sum())
        else:
            r2 = float("nan")
        length.append({"host": host, "mechanism": mech, "strength": st, "world": world or "real", "cell": cell,
                       "seed": seed, "spearman_neg": rho[0], "spearman_pos": rho[1], "ridge_r2_log_length": r2,
                       "length_not_exploited": bool(abs(rho[0]) < 0.10 and abs(rho[1]) < 0.10)})
    write_csv(f"{out}/length.csv", length)

    # comparators with subject-bootstrap Spearman intervals
    comps, crit = [], {}
    real_keys = [k for k in models if k[3] is None]
    boot_measures = defaultdict(list)            # measure -> per-model list of bootstrap arrays
    for key in real_keys:
        host, rs, s = key[0], rowsets[key[0]], S[key]
        y, cl = rs.y, s["clean"]
        qc = np.array([int(v) for v in meta[key]["qc_pass"]])
        if set(meta[key]["identifier"]) == set(rs.ids):
            qc = qc[[list(meta[key]["identifier"]).index(i) for i in rs.ids]]
        rec = {"model": "/".join(str(x) for x in key if x is not None), "host": host, "cell": key[4],
               "seed": key[5], "auc_clean": M.auc(cl, y)}

        def gap(idx):
            a, b = M.auc(cl[idx][qc[idx] == 1], y[idx][qc[idx] == 1]), M.auc(cl[idx][qc[idx] == 0], y[idx][qc[idx] == 0])
            return a - b
        rec["filtered_gap"] = gap(np.arange(len(y)))
        rec["qc_fail_n"] = int((qc == 0).sum())
        boot_measures["filtered_gap"].append(np.array([gap(i) for i in rs.boots]))
        for mch in list(mechs_of[host]) + ["gapfill"]:
            r = R_model(key, mch, "mid", 2000)
            if r is None:
                continue
            rec[f"R_{mch}"] = float(r[0])
            boot_measures[f"R_{mch}"].append(r)
            if mch != "gapfill":
                dd = np.abs(logit(s[f"{mch}_S_mid_m2000"]) - logit(s[f"{mch}_I_mid_m2000"]))
                rec[f"sens_{mch}"] = float(dd.mean())
                boot_measures[f"sens_{mch}"].append(np.array([dd[i].mean() for i in rs.boots]))
        # conditional permutation importance of the tracking statistics (point only, as v7)
        qm = np.array([[float(v) for v in meta[key][c]] for c in Q_COLS]).T
        if set(meta[key]["identifier"]) == set(rs.ids):
            qm = qm[[list(meta[key]["identifier"]).index(i) for i in rs.ids]]
        X = np.column_stack([logit(cl), qm])
        k = min(5, len(np.unique(rs.child)))
        auc_full, auc_perm, rng = [], [], np.random.RandomState(0)
        for tr, te in GroupKFold(n_splits=k).split(X, y, rs.child):
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(X[tr], y[tr])
            auc_full.append(M.auc(gb.predict_proba(X[te])[:, 1], y[te]))
            Xp = X[te].copy()
            strata = y[te] * 10 + np.digitize(cl[te], np.quantile(cl[te], np.linspace(0, 1, 11)[1:-1]))
            for st_ in np.unique(strata):
                kk = np.flatnonzero(strata == st_)
                Xp[kk, 1:] = Xp[rng.permutation(kk), 1:]
            auc_perm.append(M.auc(gb.predict_proba(Xp)[:, 1], y[te]))
        rec["cpi_tracking_stats"] = float(np.mean(auc_full) - np.mean(auc_perm))
        comps.append(rec)
    write_csv(f"{out}/comparators.csv", comps)
    if len(comps) >= 3:
        names = [k for k in boot_measures if len(boot_measures[k]) == len(comps)]
        sp = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ma, mb = np.array(boot_measures[a]), np.array(boot_measures[b])   # (models, 1 + n_boot)
                rhos = np.array([spearmanr(ma[:, j], mb[:, j]).correlation for j in range(ma.shape[1])], float)
                fin = rhos[1:][np.isfinite(rhos[1:])]
                sp[f"{a}~{b}"] = {"rho": float(rhos[0]), "ci": M.ci(fin) if len(fin) else None,
                                  "n_models": len(comps), "n_boot_finite": int(len(fin))}
        crit["spearman_between_measures"] = sp

    # gates (clean AUC of A_none World I runs mid seed 0; quality-only)
    gates = {}
    for host in hosts:
        gk = (host, "runs", "mid", "I", "A_none", 0, None)
        g = dict(GATES.get(host, {}))
        if gk in models:
            cl, y = S[gk]["clean"], rowsets[host].y if set(meta[gk]["identifier"]) == set(rowsets[host].ids) \
                else np.array([int(v) for v in meta[gk]["label"]])
            g["clean_auc_oof"] = M.auc(cl, y)
            g["clean_auc_n_folds"] = len(models[gk])
            if "clean_auc_min" in g:
                g["clean_auc_pass"] = bool(g["clean_auc_oof"] >= g["clean_auc_min"])
            if "clean_auc_max" in g:
                g["clean_auc_pass"] = bool(g["clean_auc_oof"] <= g["clean_auc_max"])
        if not args.skip_gates:
            try:
                from safe_motion.data import hosts as HH
                rows = HH.load_host(host)
                folds = HH.make_folds(host, rows, seed=0, path=f"{base}/{host}/folds.json")
                Xq = np.array([[r[c] for c in Q_COLS] for r in rows])
                yq = np.array([r["binary_label"] for r in rows])
                sub = np.array([r["subject"] for r in rows])
                nq = np.array([float(r["n_frames"]) for r in rows])
                oof = np.zeros(len(rows))
                oof_len = np.zeros(len(rows))
                for fs in folds:
                    te = np.isin(sub, fs)
                    gb = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05,
                                                        class_weight="balanced").fit(Xq[~te], yq[~te])
                    oof[te] = gb.predict_proba(Xq[te])[:, 1]
                    gl = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05,
                                                        class_weight="balanced").fit(nq[~te, None], yq[~te])
                    oof_len[te] = gl.predict_proba(nq[te, None])[:, 1]
                g["quality_only_auc_oof"] = M.auc(oof, yq)
                g["length_auc_oof"] = M.auc(oof_len, yq)
                g["n_clips"], g["n_subjects"], g["n_folds"] = len(rows), int(len(np.unique(sub))), len(folds)
                if "quality_only_band" in g:
                    g["quality_only_pass"] = bool(abs(g["quality_only_auc_oof"] - 0.5) <= g["quality_only_band"])
            except Exception as ex:               # host data not on this machine
                g["quality_only_error"] = str(ex)
        gates[host] = g
        os.makedirs(f"{base}/{host}", exist_ok=True)
        json.dump(g, open(f"{base}/{host}/gates.json", "w"), indent=1)

    # criteria R1 to R7
    def find(host, mech, st, contrast):
        for r in effects:
            if (r["host"], r["mechanism"], r["strength"], r["contrast"]) == (host, mech, st, contrast):
                return r
        return None

    crit["R1_se_positive"] = {}
    crit["R2_scaling"] = {}
    crit["R3_dropout_does_not_remove"] = {}
    crit["R4_dropout_raises_somewhere"] = {}
    crit["R5_reference"] = {}
    for host in hosts:
        r1, r3, r4, r5 = {}, {}, 0, {}
        for mech in mechs_of[host]:
            e = find(host, mech, "mid", "SE(A_none)")
            if e:
                r1[mech] = {"SE": [e["point"], e["ci_lo"], e["ci_hi"]], "null_p95": e["null_p95"],
                            "null_p": e["null_p"], "pass": bool(e["ci_lo"] > 0 and (e["above_null"] or e["null_p95"] is None)),
                            "null_available": e["null_p95"] is not None}
                r5[mech] = {"R_I": e["R_I"], "R_I_ci": e["R_I_ci"]}
            d = find(host, mech, "mid", "SE(A_dropout) - SE(A_none)")
            if d:
                r3[mech] = {"diff": [d["point"], d["ci_lo"], d["ci_hi"]], "pass": bool(not d["ci_hi"] < 0)}
                r4 += int(d["ci_lo"] > 0)
        crit["R1_se_positive"][host] = {"per_mechanism": r1, "pass": bool(r1) and all(v["pass"] for v in r1.values())}
        sc = find(host, "runs", "mid-low", "SE(mid) - SE(low), A_none")
        crit["R2_scaling"][host] = ({"diff": [sc["point"], sc["ci_lo"], sc["ci_hi"]], "pass": bool(sc["ci_lo"] > 0)}
                                    if sc else {"pass": None})
        crit["R3_dropout_does_not_remove"][host] = {"per_mechanism": r3, "pass": bool(r3) and all(v["pass"] for v in r3.values())}
        crit["R4_dropout_raises_somewhere"][host] = {"count_raises": r4, "n_mechanisms": len(r3), "asdpose_count": "3 of 4"}
        crit["R5_reference"][host] = r5
    crit["R6_length"] = {"all_not_exploited": bool(length) and all(r["length_not_exploited"] for r in length),
                         "flagged": [f"{r['host']}/{r['world']}/{r['mechanism']}/{r['cell']}/s{r['seed']}"
                                     for r in length if not r["length_not_exploited"]]}
    crit["R7_real_mmasd"] = [r for r in real if r["host"].startswith("mmasd")]
    crit["gates"] = gates
    crit["replicates"] = {h: bool(crit["R1_se_positive"][h]["pass"] and crit["R2_scaling"][h].get("pass")
                                  and crit["R3_dropout_does_not_remove"][h]["pass"]) for h in hosts}
    crit["n_models"], crit["n_boot"] = len(models), args.n_boot
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1, default=float)
    print(json.dumps({k: crit[k] for k in ("R1_se_positive", "R2_scaling", "R3_dropout_does_not_remove",
                                            "R4_dropout_raises_somewhere", "replicates", "gates")}, indent=1, default=float))


if __name__ == "__main__":
    main()
