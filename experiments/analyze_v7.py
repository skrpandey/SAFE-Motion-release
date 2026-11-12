"""Analysis of the v7 + v8 round (criteria P1-P5 of pilot_asdpose_v7.yaml, M1-M5 of
pilot_asdpose_v8.yaml) from the scores and features written by score_v7.py.

R = logit AUC_S - logit AUC_I; shortcut effect SE = R(f_S) - R(f_I); remedy effect
SE(remedy) - SE(A_none). Child-clustered paired bootstrap on shared resamples, seed-averaged.
Writes results/v7/analysis/{groups,effects,crossed,companions,length,comparators}.csv and criteria.json.
    conda run -n torch python experiments/analyze_v7.py [--smoke] [--n-boot 2000]
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M

MECHS = ("runs", "whole", "frozen", "armruns")
MASK_SEEDS = (2000, 3000, 4000)
R_MARGIN, AUC_MARGIN = 0.05, 0.02
Q_COLS = ["q_lowconf", "q_missing", "q_frozen", "q_arm_lowconf"]


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def verdict(point, ci, margin):
    lo, hi = ci
    if lo > 0 or hi < 0:
        return "differs_from_zero"
    if lo >= -margin and hi <= margin:
        return "within_margin"
    return "undetermined"


class RowSet:
    """Shared child-clustered resamples for one test row set (host or full)."""

    def __init__(self, ids, y, child, n_boot):
        self.ids, self.y, self.child = ids, np.asarray(y), np.asarray(child)
        self.pos = {i: k for k, i in enumerate(ids)}
        self.boots = [np.arange(len(ids))] + list(M.cluster_bootstrap_indices(child, n_boot, seed=0))

    def aucs(self, s):
        return np.array([M.auc(s[idx], self.y[idx]) for idx in self.boots])


def p_inversion(boot):
    """Two-sided bootstrap p by percentile-interval inversion, add-one form:
    2 * min((#{b <= 0} + 1) / (B + 1), (#{b >= 0} + 1) / (B + 1)), capped at 1."""
    b = np.asarray(boot, float)
    n = len(b)
    lo = (np.sum(b <= 0) + 1) / (n + 1)
    hi = (np.sum(b >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def align(rs, ids, labels, cols):
    """Reorder each score column into the RowSet's row order by identifier.

    Identifiers must be unique and cover the RowSet exactly; labels must agree row by row.
    """
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate identifiers in score file")
    if set(ids) != set(rs.ids):
        raise ValueError(f"score file rows ({len(ids)}) do not match the RowSet ({len(rs.ids)})")
    pos = np.array([rs.pos[i] for i in ids])
    y = np.asarray([int(v) for v in labels])
    if not np.array_equal(rs.y[pos], y):
        raise ValueError("labels differ from the RowSet for the same identifiers")
    out = {}
    for c, s in cols.items():
        a = np.empty_like(np.asarray(s, dtype=float))
        a[pos] = s
        out[c] = a
    return out


def load_scores(path):
    rd = list(csv.DictReader(open(path)))
    meta = {k: [r[k] for r in rd] for k in ("identifier", "child_id", "label", "n_frames", "qc_pass")}
    cols = [c for c in rd[0] if c not in meta]
    return meta, {c: np.array([float(r[c]) for r in rd]) for c in cols}


def stat(samples, point_override=None):
    s = np.asarray(samples, float)
    point = float(s[0]) if point_override is None else point_override
    return {"point": point, "ci": M.ci(s[1:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    base = "results/v7" + ("_smoke" if args.smoke else "")
    out = f"{base}/analysis"
    os.makedirs(out, exist_ok=True)
    reg = json.load(open(f"{base}/registry.json"))
    reg = [m for m in reg if os.path.exists(f"{base}/scores/{m['name']}.csv")]

    # load, group, bootstrap
    rowsets, S, meta = {}, {}, {}
    for m in reg:
        meta[m["name"]], S[m["name"]] = load_scores(f"{base}/scores/{m['name']}.csv")
        tag = "full" if m["world"] is None else "host"
        if tag not in rowsets:
            mt = meta[m["name"]]
            rowsets[tag] = RowSet(mt["identifier"], [int(v) for v in mt["label"]],
                                  mt["child_id"], args.n_boot)
        m["tag"] = tag
    groups = defaultdict(list)
    for m in reg:
        groups[(m["proto"], m["world"], m["world_seed"], m["mechanism"], m["strength"], m["cell"])].append(m)
    A = {}                                    # (model, col) -> bootstrap AUC array
    for m in reg:
        rs = rowsets[m["tag"]]
        S[m["name"]] = align(rs, meta[m["name"]]["identifier"], meta[m["name"]]["label"], S[m["name"]])
        for c, s in S[m["name"]].items():
            A[(m["name"], c)] = rs.aucs(s)

    def R_model(name, mech, st, ms):
        return logit(A[(name, f"{mech}_S_{st}_m{ms}")]) - logit(A[(name, f"{mech}_I_{st}_m{ms}")])

    def R_group(g, mech, st="mid", ms=2000):
        """Seed-averaged bootstrap array, between-seed SD of the point estimate."""
        per = [R_model(m["name"], mech, st, ms) for m in groups[g]
               if (m["name"], f"{mech}_S_{st}_m{ms}") in A]
        if not per:
            return None, None
        return np.mean(per, axis=0), float(np.std([p[0] for p in per], ddof=1))

    # groups.csv (R per group and condition)
    rows_out = []
    for g, ms_ in sorted(groups.items(), key=lambda kv: str(kv[0])):
        proto, world, ws, mech, st, cell = g
        clean = np.mean([A[(m["name"], "clean")] for m in ms_], axis=0)
        row = {"proto": proto, "world": world or "real", "world_seed": ws, "mechanism": mech,
               "strength": st, "cell": cell, "n_seeds": len(ms_),
               "auc_clean": f"{clean[0]:.4f} [{M.ci(clean[1:])[0]:.4f}, {M.ci(clean[1:])[1]:.4f}]"}
        conds = sorted({c[:-len("_S_mid_m2000")] + c[-len("_mid_m2000"):] for c in S[ms_[0]["name"]]
                        if "_S_" in c})
        for cnd in conds:
            mch, stx, msd = cnd.rsplit("_", 2)
            msd = int(msd[1:])
            r, sd = R_group(g, mch, stx, msd)
            if r is None:
                continue
            ci = M.ci(r[1:])
            row[f"R_{cnd}"] = f"{r[0]:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] sd_seeds {sd:.4f} {verdict(r[0], ci, R_MARGIN)}"
            raw = [float(np.mean([A[(m['name'], f'{mch}_{w}_{stx}_m{msd}')][0] for m in ms_])) for w in ("S", "I")]
            row[f"AUC_{cnd}"] = f"S {raw[0]:.4f} I {raw[1]:.4f}"
        rows_out.append(row)
    keys = sorted({k for r in rows_out for k in r}, key=lambda k: (k.startswith(("R_", "AUC_")), k))
    with open(f"{out}/groups.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_out)

    # effects.csv (SE, remedy-by-mechanism, equivalence)
    effects, crit = [], {}

    def SE(proto, mech, st, cell, ms=2000, ws=0):
        gS, gI = (proto, "S", ws, mech, st, cell), (proto, "I", ws, mech, st, cell)
        if gS not in groups or gI not in groups:
            return None
        rS, sdS = R_group(gS, mech, st, ms)
        rI, sdI = R_group(gI, mech, st, ms)
        if rS is None or rI is None:
            return None
        return {"arr": rS - rI, "R_S": rS, "R_I": rI, "sd_S": sdS, "sd_I": sdI}

    for proto in ("v7", "v8", "v3"):
        for mech in MECHS:
            for st in ("mid", "low"):
                base_se = SE(proto, mech, st, "A_none")
                if base_se is None:
                    continue
                masks = [SE(proto, mech, st, "A_none", ms)["arr"][0] for ms in MASK_SEEDS
                         if SE(proto, mech, st, "A_none", ms) is not None]
                e = base_se["arr"]
                ci = M.ci(e[1:])
                rec = {"proto": proto, "mechanism": mech, "strength": st, "contrast": "SE(A_none)",
                       "point": e[0], "ci_lo": ci[0], "ci_hi": ci[1],
                       "R_S": base_se["R_S"][0], "R_I": base_se["R_I"][0],
                       "R_I_ci": M.ci(base_se["R_I"][1:]), "sd_seeds_S": base_se["sd_S"],
                       "sd_seeds_I": base_se["sd_I"], "mask_range": (max(masks) - min(masks)) if masks else None,
                       "verdict": verdict(e[0], ci, R_MARGIN),
                       "R_I_verdict": verdict(base_se["R_I"][0], M.ci(base_se["R_I"][1:]), R_MARGIN)}
                effects.append(rec)
                for cell in ("A_velocity", "A_dropout", "A_both", "B_coorddrop"):
                    r_se = SE(proto, mech, st, cell)
                    if r_se is None:
                        continue
                    d = r_se["arr"] - e
                    ci2 = M.ci(d[1:])
                    effects.append({"proto": proto, "mechanism": mech, "strength": st,
                                    "contrast": f"SE({cell}) - SE(A_none)", "point": d[0],
                                    "ci_lo": ci2[0], "ci_hi": ci2[1], "R_S": r_se["R_S"][0],
                                    "R_I": r_se["R_I"][0], "verdict": verdict(d[0], ci2, R_MARGIN)})
                    # remedy interaction on clean AUC: gain_S - gain_I
                    gS = np.mean([A[(m["name"], "clean")] for m in groups[(proto, "S", 0, mech, st, cell)]], axis=0) \
                        - np.mean([A[(m["name"], "clean")] for m in groups[(proto, "S", 0, mech, st, "A_none")]], axis=0)
                    gI = np.mean([A[(m["name"], "clean")] for m in groups[(proto, "I", 0, mech, st, cell)]], axis=0) \
                        - np.mean([A[(m["name"], "clean")] for m in groups[(proto, "I", 0, mech, st, "A_none")]], axis=0)
                    it = gS - gI
                    ci3 = M.ci(it[1:])
                    effects.append({"proto": proto, "mechanism": mech, "strength": st,
                                    "contrast": f"interaction({cell}) gain_S - gain_I (clean AUC)",
                                    "point": it[0], "ci_lo": ci3[0], "ci_hi": ci3[1],
                                    "verdict": verdict(it[0], ci3, AUC_MARGIN)})
                # equivalence of clean accuracy between worlds (P5)
                cS = np.mean([A[(m["name"], "clean")] for m in groups[(proto, "S", 0, mech, st, "A_none")]], axis=0)
                cI = np.mean([A[(m["name"], "clean")] for m in groups[(proto, "I", 0, mech, st, "A_none")]], axis=0)
                dq = cS - cI
                ci4 = M.ci(dq[1:])
                effects.append({"proto": proto, "mechanism": mech, "strength": st,
                                "contrast": "AUC_clean(f_S) - AUC_clean(f_I), A_none", "point": dq[0],
                                "ci_lo": ci4[0], "ci_hi": ci4[1], "verdict": verdict(dq[0], ci4, AUC_MARGIN)})
    with open(f"{out}/effects.csv", "w", newline="") as f:
        keys = ["proto", "mechanism", "strength", "contrast", "point", "ci_lo", "ci_hi", "verdict",
                "R_S", "R_I", "R_I_ci", "R_I_verdict", "sd_seeds_S", "sd_seeds_I", "mask_range"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in effects:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # crossed table (M2): A_none models tested under the other operators
    crossed = []
    for g, ms_ in groups.items():
        proto, world, ws, mech, st, cell = g
        if cell != "A_none" or world is None or st != "mid":
            continue
        for other in MECHS:
            r, sd = R_group(g, other, "mid", 2000)
            if r is None:
                continue
            ci = M.ci(r[1:])
            crossed.append({"proto": proto, "trained_world": world, "trained_mechanism": mech,
                            "test_mechanism": other, "R": r[0], "ci_lo": ci[0], "ci_hi": ci[1],
                            "sd_seeds": sd, "verdict": verdict(r[0], ci, R_MARGIN)})
    with open(f"{out}/crossed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(crossed[0]) if crossed else ["empty"])
        w.writeheader()
        for r in crossed:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # per-child companion
    comp = []
    for g, ms_ in groups.items():
        proto, world, ws, mech, st, cell = g
        rs = rowsets[ms_[0]["tag"]]
        for mch in (MECHS if world is None else [mech]):
            cS, cI = f"{mch}_S_{st or 'mid'}_m2000", f"{mch}_I_{st or 'mid'}_m2000"
            if cS not in S[ms_[0]["name"]]:
                continue
            sign = 2 * rs.y - 1                      # class-aligned: positive = toward own label
            d = sign * np.mean([logit(S[m["name"]][cS]) - logit(S[m["name"]][cI]) for m in ms_], axis=0)
            pooled = np.mean([R_model(m["name"], mch, st or "mid", 2000)[0] for m in ms_])
            dc = np.array([d[rs.child == c].mean() for c in np.unique(rs.child)])
            comp.append({"proto": proto, "world": world or "real", "mechanism_trained": mech,
                         "mechanism_tested": mch, "cell": cell, "pooled_R": pooled,
                         "mean_abs_child": float(np.abs(dc).mean()),
                         "mean_abs_segment": float(np.abs(d).mean()),
                         "children_opposite_sign": int((np.sign(dc) == -np.sign(pooled)).sum()),
                         "n_children": int(len(dc)),
                         "flip_rate_0.5": float(np.mean([((S[m["name"]][cS] > 0.5) != (S[m["name"]][cI] > 0.5)).mean() for m in ms_]))})
    with open(f"{out}/companions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0]) if comp else ["empty"])
        w.writeheader()
        for r in comp:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # length recoverability (P4), real cells
    length = []
    for m in reg:
        if m["world"] is not None:
            continue
        z = np.load(f"{base}/features/{m['name']}.npz")
        y, n, ch, s = z["label"], z["n_frames"].astype(float), z["child"], S[m["name"]]["clean"]
        rho = {c: float(spearmanr(s[y == c], n[y == c]).correlation) for c in (0, 1)}
        gkf = GroupKFold(n_splits=min(5, len(np.unique(ch))))
        pred = np.zeros(len(n))
        for tr, te in gkf.split(z["feats"], n, ch):
            pred[te] = Ridge(alpha=10.0).fit(z["feats"][tr], np.log(n[tr])).predict(z["feats"][te])
        r2 = float(1 - ((np.log(n) - pred) ** 2).sum() / ((np.log(n) - np.log(n).mean()) ** 2).sum())
        length.append({"model": m["name"], "cell": m["cell"], "seed": m["seed"],
                       "spearman_neg": rho[0], "spearman_pos": rho[1], "ridge_r2_log_length": r2,
                       "length_not_exploited": bool(abs(rho[0]) < 0.10 and abs(rho[1]) < 0.10)})
    with open(f"{out}/length.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(length[0]) if length else ["empty"])
        w.writeheader()
        for r in length:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # comparator table (M5), real cells
    comps = []
    qidx = {r["identifier"]: [float(r[c]) for c in Q_COLS]
            for r in csv.DictReader(open(f"{D.CACHE_DIR}/index.csv"))}
    for m in reg:
        if m["world"] is not None:
            continue
        rs, s = rowsets["full"], S[m["name"]]
        qc = align(rs, meta[m["name"]]["identifier"], meta[m["name"]]["label"],
                   {"qc": np.array([int(v) for v in meta[m["name"]]["qc_pass"]], dtype=float)})["qc"].astype(int)
        y, cl = rs.y, s["clean"]
        gap = M.auc(cl[qc == 1], y[qc == 1]) - M.auc(cl[qc == 0], y[qc == 0])
        q = np.array([qidx[i] for i in rs.ids])
        X = np.column_stack([logit(cl), q])
        gkf = GroupKFold(n_splits=5)
        auc_full, auc_perm = [], []
        rng = np.random.RandomState(0)
        for tr, te in gkf.split(X, y, rs.child):
            g = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05).fit(X[tr], y[tr])
            auc_full.append(M.auc(g.predict_proba(X[te])[:, 1], y[te]))
            Xp = X[te].copy()
            strata = y[te] * 10 + np.digitize(cl[te], np.quantile(cl[te], np.linspace(0, 1, 11)[1:-1]))
            for st_ in np.unique(strata):
                k = np.flatnonzero(strata == st_)
                Xp[k, 1:] = Xp[rng.permutation(k), 1:]
            auc_perm.append(M.auc(g.predict_proba(Xp)[:, 1], y[te]))
        rec = {"model": m["name"], "cell": m["cell"], "seed": m["seed"], "auc_clean": M.auc(cl, y),
               "filtered_gap": gap, "cpi_tracking_stats": float(np.mean(auc_full) - np.mean(auc_perm)),
               "stacked_auc": float(np.mean(auc_full))}
        for mch in MECHS:
            rec[f"R_{mch}"] = float(R_model(m["name"], mch, "mid", 2000)[0])
            rec[f"sens_{mch}"] = float(np.abs(logit(s[f"{mch}_S_mid_m2000"]) - logit(s[f"{mch}_I_mid_m2000"])).mean())
        rec["R_gapfill"] = float(logit(A[(m["name"], "gapfill_S_mid_m2000")][0]) - logit(A[(m["name"], "gapfill_I_mid_m2000")][0]))
        comps.append(rec)
    with open(f"{out}/comparators.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comps[0]) if comps else ["empty"])
        w.writeheader()
        for r in comps:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    if len(comps) >= 4:
        keys_c = ["R_runs", "R_frozen", "R_armruns", "R_whole", "filtered_gap", "cpi_tracking_stats", "sens_runs"]
        mat = np.array([[c[k] for k in keys_c] for c in comps])
        crit["M5_spearman_between_measures"] = {f"{a}~{b}": float(spearmanr(mat[:, i], mat[:, j]).correlation)
                                                for i, a in enumerate(keys_c) for j, b in enumerate(keys_c) if i < j}

    # criteria
    def find(proto, mech, st, contrast):
        for r in effects:
            if (r["proto"], r["mechanism"], r["strength"], r["contrast"]) == (proto, mech, st, contrast):
                return r
        return None

    p1 = find("v7", "runs", "mid", "SE(A_none)")
    p1l = find("v7", "runs", "low", "SE(A_none)")
    if p1:
        crit["P1"] = {"SE_mid": [p1["point"], p1["ci_lo"], p1["ci_hi"]],
                      "SE_low": [p1l["point"], p1l["ci_lo"], p1l["ci_hi"]] if p1l else None,
                      "R_I_mid_verdict": p1["R_I_verdict"], "mask_range": p1["mask_range"],
                      "pass": bool(p1["ci_lo"] > 0 and p1["R_I_verdict"] == "within_margin"
                                   and (p1["mask_range"] is None or p1["mask_range"] < (p1["ci_hi"] - p1["ci_lo"]) / 2))}
    for name, mech in (("M1_frozen", "frozen"), ("M1_armruns", "armruns")):
        e = find("v8", mech, "mid", "SE(A_none)")
        if e:
            crit[name] = {"SE": [e["point"], e["ci_lo"], e["ci_hi"]], "R_I": e["R_I"],
                          "R_I_verdict": e["R_I_verdict"],
                          "pass": bool(e["ci_lo"] > 0 and e["R_I_verdict"] == "within_margin")}
    p5 = find("v7", "runs", "mid", "AUC_clean(f_S) - AUC_clean(f_I), A_none")
    if p5:
        crit["P5"] = {"diff": [p5["point"], p5["ci_lo"], p5["ci_hi"]], "pass": p5["verdict"] == "within_margin"}
    if length:
        crit["P4"] = {"all_real_cells_length_not_exploited": all(r["length_not_exploited"] for r in length),
                      "flagged": [r["model"] for r in length if not r["length_not_exploited"]]}
    crit["M3_remedy_by_mechanism"] = [{k: r[k] for k in ("proto", "mechanism", "contrast", "point", "ci_lo", "ci_hi", "verdict")}
                                      for r in effects if r["contrast"].startswith("SE(A_dropout)")]
    crit["n_models"] = len(reg)
    crit["n_boot"] = args.n_boot
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1)
    print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    main()
