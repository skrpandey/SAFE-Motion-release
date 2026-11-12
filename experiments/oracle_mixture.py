"""Known-coefficient score mixtures as a check of the shortcut effect (post hoc).

In each training world W, a cue-only score (logistic regression on the four tracking statistics,
fit on host validation segments planted in W) and the interpolated control of protocol v10 (not
blind to planted severity) are mixed at the logit level, f_alpha = alpha * z_cue + (1 - alpha) *
z_control, each z standardised over the pooled test scores of its world. SE(alpha) uses test
mask 2000, R averaged over the three control seeds on the logit scale, and 2,000 child-clustered
resamples. Missing runs and whole-joint absence at mid only (the mechanisms with control models).
Writes results/v10/analysis/oracle_mixture.csv and paper/figures/fig_oracle_mixture.{pdf,svg}.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_v7 as S7  # noqa: E402
from safe_motion import metrics as M  # noqa: E402
from safe_motion import planted as P  # noqa: E402
from safe_motion.data import asdpose as D  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ALPHAS = np.round(np.linspace(0, 1, 11), 2)
N_BOOT = 2000


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def blind_scores(world, mech, ids):
    """Per-seed logit scores of the interpolated control, one dict {"S", "I"} per seed."""
    out = []
    for s in range(3):
        p = f"results/v10/scores/v10_{world}_w0_{mech}_mid_A_blind_s{s}.csv"
        rd = {r["identifier"]: r for r in csv.DictReader(open(p))}
        out.append({t: logit(np.array([float(rd[i][f"{mech}_{t}_mid_m2000"]) for i in ids])) for t in ("S", "I")})
    return out


def main():
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
    _, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    va = [r for r in va if D.passes_qc(r, S7.V1["qc_filter"])]
    ids = [r["identifier"] for r in te]
    y = np.array([r["binary_label"] for r in te]); yv = [r["binary_label"] for r in va]
    boots = [np.arange(len(te))] + list(M.cluster_bootstrap_indices([r["child_id"] for r in te], N_BOOT, seed=0))
    out = []
    curves = {}
    for mech in ("runs", "whole"):
        mode = P.MECHANISM_MODE[mech]
        xt = {t: S7.tracking_stats(te, P.build_masks(te, t, "mid", 2000, mechanism=mech)[0], mode) for t in ("S", "I")}
        R = {}
        for w in ("S", "I"):
            xv = S7.tracking_stats(va, P.build_masks(va, w, "mid", 2001, mechanism=mech)[0], mode)
            sc = StandardScaler().fit(xv)
            lr = LogisticRegression(max_iter=2000).fit(sc.transform(xv), yv)
            zc = {t: lr.decision_function(sc.transform(xt[t])) for t in ("S", "I")}
            zms = blind_scores(w, mech, ids)
            sd_c = np.std(np.concatenate([zc["S"], zc["I"]])) or 1.0
            for a in ALPHAS:
                per = []
                for zm in zms:                       # one mixture per seed; R averaged over seeds on the logit scale
                    sd_m = np.std(np.concatenate([zm["S"], zm["I"]])) or 1.0
                    f = {t: a * zc[t] / sd_c + (1 - a) * zm[t] / sd_m for t in ("S", "I")}
                    aS = np.array([M.auc(f["S"][b], y[b]) for b in boots]); aI = np.array([M.auc(f["I"][b], y[b]) for b in boots])
                    per.append(logit(aS) - logit(aI))
                R[(w, a)] = np.mean(per, axis=0)
        for a in ALPHAS:
            se = R[("S", a)] - R[("I", a)]
            lo, hi = M.ci(se[1:])
            rS, rI = R[("S", a)], R[("I", a)]
            out.append({"mechanism": mech, "alpha": a, "SE": round(float(se[0]), 4), "SE_lo": round(lo, 4), "SE_hi": round(hi, 4),
                        "R_fS": round(float(rS[0]), 4), "R_fI": round(float(rI[0]), 4), "post_hoc": "yes"})
        curves[mech] = [r for r in out if r["mechanism"] == mech]
        se_pts = [r["SE"] for r in curves[mech]]
        mono = all(b >= a - 1e-9 for a, b in zip(se_pts, se_pts[1:]))
        print(mech, "SE by alpha:", se_pts, "monotone non-decreasing:", mono, flush=True)
    os.makedirs("results/v10/analysis", exist_ok=True)
    with open("results/v10/analysis/oracle_mixture.csv", "w", newline="") as fh:
        fh.write("# post hoc oracle-mixture validation; alpha = weight of the cue-only score\n")
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for mech, col, lab in (("runs", "#0072B2", "missing runs"), ("whole", "#D55E00", "whole-joint absence")):
        c = curves[mech]
        a = [r["alpha"] for r in c]
        ax.plot(a, [r["SE"] for r in c], "o-", color=col, ms=3, lw=1.3, label=lab)
        ax.fill_between(a, [r["SE_lo"] for r in c], [r["SE_hi"] for r in c], color=col, alpha=0.18, lw=0)
    ax.axhline(0, color="#888888", lw=0.7)
    ax.set_xlabel("weight of the cue-only score in the mixture")
    ax.set_ylabel("matched-world interaction MWI (logit AUC)")
    ax.set_title("Known-coefficient score mixtures (post hoc)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for ext in ("pdf", "svg"):
        fig.savefig(f"paper/figures/fig_oracle_mixture.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    main()
