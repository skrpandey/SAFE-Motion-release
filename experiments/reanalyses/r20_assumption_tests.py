"""R20. Partial tests of the identifying assumption (equal non-cue responses) and the placebo floor.

Inputs: per-segment scores (common.py) and the regenerated per-segment planted severities
(test_planted_severities.csv, written and verified by regen_test_planting.py).

(a) Severity-matched AUROC. For each model, test policy (S, I) and test mask, the AUROC over
    SMM/non-SMM pairs whose planted severities fall in the same stratum (K = 10 quantile strata of the
    policy-and-mask severities, pooled over classes; pair-weighted, i.e. the AUROC over all
    within-stratum pairs). Matched response per model: mean over masks of logit matched AUROC under
    policy S minus policy I; matched MWI: World S minus World I (seeds averaged). Identifier-clustered
    intervals on the shared RowSet resamples (seed 0). Within a stratum the planted cue carries
    (almost) no label information, so the matched response measures how each model's non-cue
    discrimination changes between the policies; equal non-cue responses predict a matched MWI near 0.
    Sensitivity: K = 5, and strata of the realized masked arm fraction in the scored window.
    Diagnostic: the stratified AUROC of the severity itself (residual cue within strata).
(b) Within-class slopes. Under policy I, per segment and mask, the shift of the seed-averaged
    score (softmax probability of SMM) from its clean score, regressed on the planted severity by
    ordinary least squares within each class, pooling the three masks (each a fresh severity draw);
    for World S and World I models and their difference, with identifier-clustered intervals.
    Secondary: the logit-score scale and policy S.
Placebo floor: R(f_I) of the four corrected-recipe assignments, their spread and the
    pairwise reference-against-reference differences with paired intervals.
Writes results/reanalyses/r20_*.csv and r20_summary.json.
    python3 experiments/reanalyses/r20_assumption_tests.py
"""
import csv
import itertools

import numpy as np

import common as C

K_MAIN = 10


def load_planting():
    d = {}
    for r in csv.DictReader(l for l in open(C.OUT / "test_planted_severities.csv") if not l.startswith("#")):
        d[(r["identifier"], r["policy"], int(r["mask_seed"]))] = r
    return d


def counts_matrix(eng):
    n = len(eng.sel)
    return np.stack([np.bincount(idx, minlength=n) for idx in eng.rs.boots]).astype(np.float64)


def pair_matrix(sp, sn):
    """1 if positive score > negative score, 0.5 if equal, else 0 (the average-rank AUROC kernel)."""
    return (sp[:, None] > sn[None, :]) + 0.5 * (sp[:, None] == sn[None, :])


def strat_auc(cnt, s, y, strata):
    """Pair-weighted within-stratum AUROC for every resample (rows of cnt); one stratum = ordinary AUROC."""
    num = np.zeros(cnt.shape[0])
    den = np.zeros(cnt.shape[0])
    for k in np.unique(strata):
        p = np.flatnonzero((strata == k) & (y == 1))
        q = np.flatnonzero((strata == k) & (y == 0))
        if len(p) == 0 or len(q) == 0:
            continue
        P = pair_matrix(s[p], s[q])
        cp, cq = cnt[:, p], cnt[:, q]
        num += np.einsum("bi,ij,bj->b", cp, P, cq)
        den += cp.sum(1) * cq.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def quantile_strata(x, k):
    edges = np.quantile(x, np.linspace(0, 1, k + 1))[1:-1]
    return np.digitize(x, edges)


def main():
    cells = C.confonly_cells()
    cells["w0123"] = {w: cells["corr3"][w] + cells["w123"][w] for w in ("S", "I")}
    use = ("hist3", "corr3", "corr9", "w0123")
    sc = C.Scores([m for k in use for w in ("S", "I") for m in cells[k][w]] +
                  [m for k in ("w1", "w2", "w3") for w in ("S", "I") for m in cells[k][w]])
    eng = C.Engine(sc)
    cnt = counts_matrix(eng)
    y = sc.y
    pl = load_planting()
    eps = {(p, ms): np.array([float(pl[(i, p, ms)]["eps"]) for i in sc.ids]) for p in ("S", "I") for ms in C.MASK_SEEDS}
    realw = {(p, ms): np.array([float(pl[(i, p, ms)]["masked_frac_window"]) for i in sc.ids]) for p in ("S", "I") for ms in C.MASK_SEEDS}
    summary = {}

    # the count-weighted kernel reproduces the published resampled AUROC exactly (one stratum)
    m0 = cells["corr3"]["S"][0]
    chk = strat_auc(cnt, sc.col(m0["name"], "armruns_S_mid_m2000"), y, np.zeros(len(y), int))
    ref = eng.aucs(m0["name"], "armruns_S_mid_m2000")
    summary["kernel_check_max_abs_diff"] = float(np.nanmax(np.abs(chk - ref)))
    print("kernel check", summary["kernel_check_max_abs_diff"], flush=True)

    # (a) severity-matched AUROC
    def matched(strata_of, label):
        cache = {}

        def A(m, p, ms):
            key = (m["name"], p, ms)
            if key not in cache:
                cache[key] = strat_auc(cnt, sc.col(m["name"], f"armruns_{p}_mid_m{ms}"), y, strata_of[(p, ms)])
            return cache[key]
        out = []
        for key in use:
            g = cells[key]
            Rw, lev = {}, {}
            for w in ("S", "I"):
                Rw[w] = np.mean([np.mean([C.logit(A(m, "S", ms)) - C.logit(A(m, "I", ms)) for ms in C.MASK_SEEDS], 0) for m in g[w]], 0)
                for p in ("S", "I"):
                    lev[(w, p)] = np.mean([np.mean([A(m, p, ms) for ms in C.MASK_SEEDS], 0) for m in g[w]], 0)
            mwi = Rw["S"] - Rw["I"]
            ordinary = eng.cell(g)
            out.append({"strata": label, "cell": key, "seeds_per_world": len(g["S"]),
                        "matched_mwi_logit": C.fmt(mwi), "matched_R_fS": C.fmt(Rw["S"]), "matched_R_fI": C.fmt(Rw["I"]),
                        "ordinary_mwi_logit": C.fmt(ordinary[0]), "ordinary_R_fS": C.fmt(ordinary[1]), "ordinary_R_fI": C.fmt(ordinary[2]),
                        "matched_auroc_SS": f"{lev[('S', 'S')][0]:.3f}", "matched_auroc_SI": f"{lev[('S', 'I')][0]:.3f}",
                        "matched_auroc_IS": f"{lev[('I', 'S')][0]:.3f}", "matched_auroc_II": f"{lev[('I', 'I')][0]:.3f}",
                        "share_resamples_mwi_le_0": float(np.mean(mwi[1:] <= 0)),
                        "mwi_point": float(mwi[0]), "mwi_lo": C.ci(mwi)[1], "mwi_hi": C.ci(mwi)[2]})
            print(label, key, out[-1]["matched_mwi_logit"], "R_fS", out[-1]["matched_R_fS"], "R_fI", out[-1]["matched_R_fI"],
                  "| ordinary", out[-1]["ordinary_mwi_logit"], flush=True)
        return out

    st10 = {k: quantile_strata(v, K_MAIN) for k, v in eps.items()}
    st5 = {k: quantile_strata(v, 5) for k, v in eps.items()}
    st20 = {k: quantile_strata(v, 20) for k, v in eps.items()}
    stw = {k: quantile_strata(v, K_MAIN) for k, v in realw.items()}
    rows = matched(st10, "planted severity, 10 strata (primary)")
    rows += [r for r in matched(st5, "planted severity, 5 strata") if r["cell"] in ("hist3", "corr3", "w0123")]
    rows += matched(st20, "planted severity, 20 strata")
    rows += [r for r in matched(stw, "realized masked arm fraction in the scored window, 10 strata") if r["cell"] in ("hist3", "corr3", "w0123")]
    C.write_csv("r20a_severity_matched.csv", rows,
                "R20(a): AUROC over SMM/non-SMM pairs within strata of the regenerated planted severity of each test\n"
                "policy and mask (pair-weighted), averaged over masks 2000/3000/4000 and seeds on the logit scale; matched MWI =\n"
                "matched response of World S minus World I; 95% identifier-clustered percentile intervals (RowSet seed 0, 2,000).\n"
                "Cells: hist3 historical 3 seeds; corr3 corrected assignment 0, 3 seeds; corr9 nine seeds; w0123 the four\n"
                "corrected assignments pooled (12 seeds per world). matched_auroc_XY: model of World X under policy Y (mean raw)")
    # residual cue within strata: stratified AUROC of -severity (non-SMM degraded more), and of the planted window statistic
    diag = []
    for p in ("S", "I"):
        for label, st in (("10 strata", st10), ("5 strata", st5), ("20 strata", st20)):
            a = np.mean([strat_auc(cnt[:1], -eps[(p, ms)], y, st[(p, ms)])[0] for ms in C.MASK_SEEDS])
            u = np.mean([strat_auc(cnt[:1], -eps[(p, ms)], y, np.zeros(len(y), int))[0] for ms in C.MASK_SEEDS])
            stat = {ms: np.array([float(pl[(i, p, ms)]["arm_lowconf_window"]) for i in sc.ids]) for ms in C.MASK_SEEDS}
            b = np.mean([strat_auc(cnt[:1], -stat[ms], y, st[(p, ms)])[0] for ms in C.MASK_SEEDS])
            bu = np.mean([strat_auc(cnt[:1], -stat[ms], y, np.zeros(len(y), int))[0] for ms in C.MASK_SEEDS])
            diag.append({"policy": p, "strata": label, "auroc_minus_eps_unstratified": u, "auroc_minus_eps_within_strata": a,
                         "auroc_planted_window_stat_unstratified": bu, "auroc_planted_window_stat_within_strata": b})
    C.write_csv("r20a_matching_diagnostic.csv", diag,
                "R20(a) diagnostic: cue-only AUROC (non-SMM degraded more) of the planted severity and of the planted arm\n"
                "low-confidence fraction of the scored window, unstratified and within severity strata (mean over 3 masks)")
    summary["matching_diagnostic"] = diag

    # (b) within-class slopes of the score shift on severity
    def slopes(g, pol, scale):
        f = (lambda a: C.logit(a)) if scale == "logit-score" else (lambda a: a)
        out = {}
        for w in ("S", "I"):
            sh = np.stack([np.mean([f(sc.col(m["name"], f"armruns_{pol}_mid_m{ms}")) - f(sc.col(m["name"], "clean"))
                                    for m in g[w]], 0) for ms in C.MASK_SEEDS], 1)      # (n, 3)
            X = np.stack([eps[(pol, ms)] for ms in C.MASK_SEEDS], 1)
            for c in (0, 1):
                k = y == c
                cw = cnt[:, k]
                Sw = cw.sum(1) * 3
                Sx, Sy = cw @ X[k].sum(1), cw @ sh[k].sum(1)
                Sxy, Sxx = cw @ (X[k] * sh[k]).sum(1), cw @ (X[k] ** 2).sum(1)
                out[(w, c)] = ((Sxy - Sx * Sy / Sw) / (Sxx - Sx ** 2 / Sw), Sy / Sw)
        return out
    srows = []
    for key in use:
        for pol, scale in (("I", "probability"), ("I", "logit-score"), ("S", "probability")):
            o = slopes(cells[key], pol, scale)
            for c, cname in ((0, "non-SMM"), (1, "SMM")):
                sS, mS = o[("S", c)]
                sI, mI = o[("I", c)]
                srows.append({"cell": key, "policy": pol, "score_scale": scale, "class": cname,
                              "slope_world_S": C.fmt(sS), "slope_world_I": C.fmt(sI), "slope_S_minus_I": C.fmt(sS - sI),
                              "mean_shift_world_S": C.fmt(mS), "mean_shift_world_I": C.fmt(mI), "mean_shift_S_minus_I": C.fmt(mS - mI)})
            # class difference of slopes (SMM minus non-SMM), per world and World S minus World I
            dS = o[("S", 1)][0] - o[("S", 0)][0]
            dI = o[("I", 1)][0] - o[("I", 0)][0]
            mS = o[("S", 1)][1] - o[("S", 0)][1]
            mI = o[("I", 1)][1] - o[("I", 0)][1]
            srows.append({"cell": key, "policy": pol, "score_scale": scale, "class": "SMM minus non-SMM",
                          "slope_world_S": C.fmt(dS), "slope_world_I": C.fmt(dI), "slope_S_minus_I": C.fmt(dS - dI),
                          "mean_shift_world_S": C.fmt(mS), "mean_shift_world_I": C.fmt(mI), "mean_shift_S_minus_I": C.fmt(mS - mI)})
            if key in ("hist3", "corr3"):
                for r in srows[-3:]:
                    print(r, flush=True)
    eps_I = np.concatenate([eps[("I", ms)] for ms in C.MASK_SEEDS])
    summary["severity_policy_I"] = {"mean": float(eps_I.mean()), "sd": float(eps_I.std()), "p05": float(np.percentile(eps_I, 5)),
                                    "p95": float(np.percentile(eps_I, 95))}
    C.write_csv("r20b_within_class_slopes.csv", srows,
                "R20(b): OLS slope of the per-segment score shift (score under the test policy minus clean score; seed-averaged;\n"
                "softmax probability of SMM, or its logit) on the regenerated planted severity eps (0 to 0.95), within class, pooling\n"
                "the three test masks; World S, World I and their difference; mean shift by class; 95% identifier-clustered\n"
                "percentile intervals (RowSet seed 0, 2,000 resamples; all mask observations of a resampled segment move together)")

    # placebo floor: reference responses over the four corrected assignments
    keys = ("corr3", "w1", "w2", "w3")
    RI = {k: eng.cell(cells[k])[2] for k in keys}
    RS = {k: eng.cell(cells[k])[1] for k in keys}
    prow = [{"assignment": k, "R_fI": C.fmt(RI[k]), "R_fS": C.fmt(RS[k])} for k in keys]
    pts = np.array([RI[k][0] for k in keys])
    for a, b in itertools.combinations(keys, 2):
        prow.append({"assignment": f"R_fI({a}) - R_fI({b})", "R_fI": C.fmt(RI[a] - RI[b]), "R_fS": C.fmt(RS[a] - RS[b])})
    prow.append({"assignment": "spread max - min (point)", "R_fI": f"{pts.max() - pts.min():.6f}",
                 "R_fS": f"{max(RS[k][0] for k in keys) - min(RS[k][0] for k in keys):.6f}"})
    prow.append({"assignment": "sample SD over the four assignments (point)", "R_fI": f"{pts.std(ddof=1):.6f}",
                 "R_fS": f"{np.std([RS[k][0] for k in keys], ddof=1):.6f}"})
    C.write_csv("r20_placebo_floor.csv", prow,
                "R20 placebo floor: own-operator responses of the corrected-recipe references R(f_I) and of the\n"
                "World S models R(f_S) for the four training assignments (logit, mask mean), and pairwise reference-against-reference\n"
                "differences with paired identifier-clustered intervals (shared resamples)")
    for r in prow:
        print(r, flush=True)
    C.write_json("r20_summary.json", summary)


if __name__ == "__main__":
    main()
