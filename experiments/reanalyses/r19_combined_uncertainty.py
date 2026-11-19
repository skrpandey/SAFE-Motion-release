"""R19. Combined uncertainty.

1. Hierarchical bootstrap of the corrected-recipe, no-remedy confidence-only MWI over four training
   assignments (assignment 0 = v15conf seeds 0-2; assignments 1-3 = v17 seeds 0-2). In each of 2,000
   replicates: draw 4 assignments with replacement; within each drawn assignment draw 3 optimizer seeds
   with replacement, independently for World S and World I; draw the 3 test masks with replacement
   (one draw shared by all models of the replicate); use identifier resample b of the shared
   40-identifier RowSet (seed 0) for every model. The replicate value is the mean over the 4 drawn
   assignments of (mean logit response of the drawn World S seeds and masks) minus (the same for World I).
   Point estimate: mean over the four assignments of their MWI. RNG seed 20261116 (one set of draws,
   reused for the raw scale). Also the intervals that drop levels, for comparison.
2. Per-seed MWI (seed k of World S paired with seed k of World I, mask mean, point AUROCs) for every
   three-seed no-remedy cell of Table 2 (and of Table 4), their sample SD, and the alternative
   definition behind the printed range 0.637 to 0.821 (each World S seed minus the World I seed mean,
   experiments/procedure_checks.py section 5, worldseed_realizations.csv).
3. First-mask (mask 2000) estimates of the confidence-only cell, historical and corrected three seeds.
Writes results/reanalyses/r19_*.csv and r19_summary.json.
    python3 experiments/reanalyses/r19_combined_uncertainty.py
"""
import numpy as np
from scipy import stats

import common as C

RNG_SEED = 20261116
ASSIGN = ("corr3", "w1", "w2", "w3")


def hierarchical(A, nb, rng_seed=RNG_SEED, levels=("assignments", "seeds", "masks", "identifiers")):
    """A[scale][a][w]: (3 seeds, 3 masks, NB+1). Returns {scale: draws}."""
    rng = np.random.RandomState(rng_seed)
    nA = len(ASSIGN)
    out = {s: np.empty(nb) for s in A}
    for b in range(nb):
        asg = rng.randint(0, nA, nA)
        md = rng.randint(0, 3, 3)
        sd = [(rng.randint(0, 3, 3), rng.randint(0, 3, 3)) for _ in range(nA)]
        if "assignments" not in levels:
            asg = np.arange(nA)
        if "masks" not in levels:
            md = np.arange(3)
        if "seeds" not in levels:
            sd = [(np.arange(3), np.arange(3))] * nA
        col = b + 1 if "identifiers" in levels else 0
        for s in A:
            v = [A[s][ASSIGN[a]]["S"][np.ix_(sS, md)][..., col].mean() - A[s][ASSIGN[a]]["I"][np.ix_(sI, md)][..., col].mean()
                 for a, (sS, sI) in zip(asg, sd)]
            out[s][b] = np.mean(v)
    return out


def per_seed(sc, g, mech, st):
    """Index-paired per-seed MWI and World S seed minus World I seed mean (logit, mask mean, point)."""
    y = sc.y

    def R(m):
        return np.mean([C.logit(C.M.auc(sc.col(m["name"], f"{mech}_S_{st}_m{ms}"), y))
                        - C.logit(C.M.auc(sc.col(m["name"], f"{mech}_I_{st}_m{ms}"), y)) for ms in C.MASK_SEEDS])
    rS = np.array([R(m) for m in g["S"]])
    rI = np.array([R(m) for m in g["I"]])
    paired = rS - rI
    vs_mean = rS - rI.mean()
    return rS, rI, paired, vs_mean


def main():
    cells = C.confonly_cells()
    sc = C.Scores([m for k in ("hist3",) + ASSIGN + ("stgcn",) for w in ("S", "I") for m in cells[k][w]])
    eng = C.Engine(sc)
    summary = {}

    # 1. hierarchical bootstrap
    A = {s: {a: {w: np.stack([eng.R(m, scale=s) for m in cells[a][w]]) for w in ("S", "I")} for a in ASSIGN}
         for s in ("logit", "raw")}
    pts = {s: np.array([A[s][a]["S"][..., 0].mean() - A[s][a]["I"][..., 0].mean() for a in ASSIGN]) for s in A}
    rows = []
    variants = [("identifiers only (fits, masks and assignments fixed)", ("identifiers",)),
                ("identifiers, masks and seeds (assignments fixed)", ("identifiers", "masks", "seeds")),
                ("assignments, seeds and masks (identifiers fixed)", ("assignments", "seeds", "masks")),
                ("all four levels: assignments, seeds, masks, identifiers", ("assignments", "seeds", "masks", "identifiers"))]
    for label, lev in variants:
        d = hierarchical(A, C.NB, levels=lev)
        rec = {"procedure": label}
        for s in ("logit", "raw"):
            lo, hi = np.percentile(d[s], [2.5, 97.5])
            rec.update({f"point_{s}": float(pts[s].mean()), f"lo_{s}": float(lo), f"hi_{s}": float(hi),
                        f"draw_sd_{s}": float(d[s].std(ddof=1)), f"share_draws_le_0_{s}": float(np.mean(d[s] <= 0))})
        rows.append(rec)
        print(f"{label}: logit {rec['point_logit']:+.3f} [{rec['lo_logit']:+.3f}, {rec['hi_logit']:+.3f}] "
              f"raw {rec['point_raw']:+.3f} [{rec['lo_raw']:+.3f}, {rec['hi_raw']:+.3f}]", flush=True)
    for s in ("logit", "raw"):        # t interval over the four assignment points (descriptive comparator)
        p = pts[s]
        h = stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p))
        rows.append({"procedure": f"t interval over the four assignment MWIs, df 3 ({s})", f"point_{s}": float(p.mean()),
                     f"lo_{s}": float(p.mean() - h), f"hi_{s}": float(p.mean() + h)})
    summary["assignment_points"] = {s: dict(zip(ASSIGN, pts[s].round(6).tolist())) for s in pts}
    C.write_csv("r19_hierarchical.csv", rows,
                "R19: corrected-recipe no-remedy confidence-only MWI over four training assignments (assignment 0 = v15conf A_none\n"
                "seeds 0-2, assignments 1-3 = v17 A_none seeds 0-2); point = mean of the four assignment MWIs; 95% percentile\n"
                f"intervals from 2,000 replicates, RNG seed {RNG_SEED}; identifier resamples = RowSet seed 0 (shared)")

    # 2. per-seed values
    t2 = C.table2_cells()
    sc2 = C.Scores([m for g, _, _ in t2.values() for w in ("S", "I") for m in g[w]])
    assert sc2.ids and set(sc2.ids) == set(sc.ids)
    prow = []
    for label, (g, mech, st) in list(t2.items()) + [(f"Table 4: {k}", (cells[k], "armruns", "mid")) for k in ASSIGN + ("stgcn",)]:
        s_ = sc2 if label in t2 else sc
        rS, rI, paired, vsm = per_seed(s_, g, mech, st)
        prow.append({"cell": label, "seed_models": " | ".join(f"{a['name']} - {b['name']}" for a, b in zip(g["S"], g["I"])),
                     "R_fS_per_seed": "; ".join(f"{v:+.3f}" for v in rS), "R_fI_per_seed": "; ".join(f"{v:+.3f}" for v in rI),
                     "mwi_paired_per_seed": "; ".join(f"{v:+.3f}" for v in paired),
                     "mwi_paired_mean": float(paired.mean()), "mwi_paired_sd": float(paired.std(ddof=1)),
                     "mwi_paired_min": float(paired.min()), "mwi_paired_max": float(paired.max()),
                     "mwi_S_seed_minus_I_mean_per_seed": "; ".join(f"{v:+.3f}" for v in vsm),
                     "mwi_S_seed_minus_I_mean_sd": float(vsm.std(ddof=1)),
                     "mwi_S_seed_minus_I_mean_min": float(vsm.min()), "mwi_S_seed_minus_I_mean_max": float(vsm.max())})
        print(prow[-1]["cell"], prow[-1]["mwi_paired_per_seed"], f"SD {paired.std(ddof=1):.3f}",
              "| vs I mean", prow[-1]["mwi_S_seed_minus_I_mean_per_seed"], f"SD {vsm.std(ddof=1):.3f}", flush=True)
    # extra missing-run assignments (one World S seed each) against the World I mean, as printed (0.707, 0.751)
    gI = t2["missing runs, mid"][0]["I"]
    extra = [m for m in C._registry("v7") if m["proto"] == "v7" and m["world"] == "S" and m["mechanism"] == "runs"
             and m["strength"] == "mid" and m["cell"] == "A_none" and int(m.get("world_seed") or 0) in (1, 2)]
    sc3 = C.Scores(extra + gI)
    _, rI, _, _ = per_seed(sc3, {"S": gI, "I": gI}, "runs", "mid")
    for m in sorted(extra, key=lambda m: m["world_seed"]):
        rS = per_seed(sc3, {"S": [m], "I": [m]}, "runs", "mid")[0][0]
        prow.append({"cell": f"missing runs, mid, World S assignment {m['world_seed']} ({m['name']}) minus World I seed mean",
                     "R_fS_per_seed": f"{rS:+.3f}", "mwi_S_seed_minus_I_mean_per_seed": f"{rS - rI.mean():+.3f}"})
        print(prow[-1]["cell"], prow[-1]["mwi_S_seed_minus_I_mean_per_seed"], flush=True)
    C.write_csv("r19_per_seed.csv", prow,
                "R19: per-seed MWI of the three-seed no-remedy cells (logit, mean over test masks 2000/3000/4000, point AUROCs on the\n"
                "1,966 host test rows). 'paired': World S seed k minus World I seed k (experiments/procedure_checks.py variance\n"
                "components, the source of the printed SD 0.107). 'S_seed_minus_I_mean': World S seed k minus the World I seed\n"
                "mean (procedure_checks.py worldseed_realizations, the source of the printed range 0.637 to 0.821).\n"
                "Table 2 destructive cells: results/operators/scores (cleaned selection)")

    # 3. first-mask estimates
    fm = []
    for key in ("hist3", "corr3"):
        for masks, tag in (((2000,), "mask 2000 only (first test mask)"), (C.MASK_SEEDS, "mean over masks 2000/3000/4000")):
            lg, rS, rI, _, _ = eng.cell(cells[key], masks=masks)
            rw = eng.cell(cells[key], masks=masks, scale="raw")[0]
            fm.append({"cell": key, "masks": tag, "mwi_logit": C.fmt(lg), "mwi_raw": C.fmt(rw), "R_fS": C.fmt(rS), "R_fI": C.fmt(rI)})
            print(fm[-1], flush=True)
        for ms in (3000, 4000):
            lg = eng.cell(cells[key], masks=(ms,))[0]
            fm.append({"cell": key, "masks": f"mask {ms} only", "mwi_logit": C.fmt(lg),
                       "mwi_raw": C.fmt(eng.cell(cells[key], masks=(ms,), scale="raw")[0])})
    C.write_csv("r19_first_mask.csv", fm,
                "R19: confidence-only MWI with the first test mask only (2000, the pre-specified single-mask estimate of the\n"
                "main-experiment cells) next to the post hoc mean over three masks; 95% identifier-clustered intervals")
    C.write_json("r19_summary.json", summary)


if __name__ == "__main__":
    main()
