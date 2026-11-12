"""Interval width of SE against the number of test children (post hoc).

For each no-remedy cell (ASDPose host test split, v7 and v8), SE on the mask mean (test masks
2000/3000/4000) from scores averaged over the three training seeds. For k in 10, 20, 30, 40
children, 100 subsets are drawn without replacement, and a child-clustered bootstrap (200
resamples) gives the 95% interval half-width in each; median and quartiles are reported. Seed 0.
Writes results/v7/analysis/se_sample_size.csv and paper/figures/fig_sample_size.{pdf,svg};
--plot-only redraws the figure from the csv.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M  # noqa: E402

CELLS = [("runs", "mid", "v7"), ("runs", "low", "v7"), ("whole", "mid", "v7"), ("frozen", "mid", "v8"), ("armruns", "mid", "v13")]   # corrected validity rule
NAME = {("runs", "mid"): "missing runs", ("runs", "low"): "missing runs, low", ("whole", "mid"): "whole-joint absence",
        ("frozen", "mid"): "frozen frames", ("armruns", "mid"): "arm confidence"}
KS, N_SUB, N_BOOT = (10, 20, 30, 40), 100, 200


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def fast_auc(s, y):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan


def load(proto, world, mech, st):
    cols = [f"{mech}_{t}_{st}_m{m}" for t in ("S", "I") for m in (2000, 3000, 4000)]
    acc, ids = None, None
    for s in range(3):
        p = f"results/{'v13' if proto == 'v13' else 'v7'}/scores/{proto}_{world}_w0_{mech}_{st}_A_none_s{s}.csv"
        rd = list(csv.DictReader(open(p)))
        a = np.array([[float(r[c]) for c in cols] for r in rd])
        acc = a if acc is None else acc + a
        ids = ids or rd
    y = np.array([int(r["label"]) for r in ids]); child = np.array([r["child_id"] for r in ids])
    return acc / 3, y, child, cols


def se_from(idx, sc, y):
    """SE on the mask mean for rows idx; sc maps world -> (n, 6) scores [S m2000..m4000, I m2000..m4000]."""
    R = {}
    for w, a in sc.items():
        lS = np.mean([logit(fast_auc(a[idx, j], y[idx])) for j in range(3)])
        lI = np.mean([logit(fast_auc(a[idx, 3 + j], y[idx])) for j in range(3)])
        R[w] = lS - lI
    return R["S"] - R["I"]


def main():
    rng = np.random.RandomState(0)
    out = []
    for mech, st, proto in CELLS:
        aS, y, child, _ = load(proto, "S", mech, st)
        aI, y2, _, _ = load(proto, "I", mech, st)
        assert (y == y2).all()
        sc = {"S": aS, "I": aI}
        kids = np.unique(child); members = {c: np.flatnonzero(child == c) for c in kids}
        full = se_from(np.arange(len(y)), sc, y)
        for k in KS:
            hw = []
            for _ in range(N_SUB if k < len(kids) else 1):
                sub = rng.choice(kids, k, replace=False)
                vals = []
                for _ in range(N_BOOT):
                    pick = rng.choice(sub, k, replace=True)
                    idx = np.concatenate([members[c] for c in pick])
                    vals.append(se_from(idx, sc, y))
                lo, hi = np.nanpercentile(vals, [2.5, 97.5])
                hw.append((hi - lo) / 2)
            out.append({"mechanism": NAME[(mech, st)], "SE_full": round(float(full), 4), "k_children": k,
                        "median_half_width": round(float(np.median(hw)), 4),
                        "q25_half_width": round(float(np.percentile(hw, 25)), 4),
                        "q75_half_width": round(float(np.percentile(hw, 75)), 4), "post_hoc": "yes"})
            print(out[-1], flush=True)
    with open("results/v7/analysis/se_sample_size.csv", "w", newline="") as f:
        f.write("# post hoc; SE on the mask mean from seed-ensembled scores; half-width of the 95% child bootstrap interval\n")
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    cols = ["#0072B2", "#56B4E9", "#009E73", "#7F7F7F", "#D55E00"]
    for (mech, st, _), c in zip(CELLS, cols):
        rr = [r for r in out if r["mechanism"] == NAME[(mech, st)]]
        ax.plot([r["k_children"] for r in rr], [r["median_half_width"] for r in rr], "o-", color=c, ms=3, lw=1.2,
                label=NAME[(mech, st)])
    ax.set_xlabel("test subject identifiers"); ax.set_ylabel("95% interval half-width of MWI\n(logit AUROC)")
    ax.set_xticks(KS); ax.set_title("Precision of MWI by number of\ntest subject identifiers (post hoc)", fontsize=8)
    # opaque legend frame over the curves
    ax.legend(frameon=True, framealpha=1, facecolor="white", edgecolor="none", fontsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for ext in ("pdf", "svg"):
        fig.savefig(f"paper/figures/fig_sample_size.{ext}", bbox_inches="tight")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:          # redraw from the saved csv, no recomputation
        rows_ = [r for r in csv.DictReader(l for l in open("results/v7/analysis/se_sample_size.csv")
                                            if not l.startswith("#"))]
        for r in rows_:
            r["k_children"] = int(r["k_children"]); r["median_half_width"] = float(r["median_half_width"])
        plot(rows_)
    else:
        main()
