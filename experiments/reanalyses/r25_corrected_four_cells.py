"""Four condition-by-policy AUROCs of the corrected-loader confidence-only comparisons (2026-11-18).

Cells: no remedy (v15conf A_none), Group DRO with planted-severity groups (v15conf C_groupdro) and
Group DRO with input-derived groups (v16 C_groupdro_obs); CompactCTRGCN, assignment 0, seeds 0-2,
coordinate validity rule, test masks 2000/3000/4000. Every statistic is computed from the stored
per-segment scores with the estimator of experiments/analyze_confonly_primary.py and analyze_confonly_extension.py: AUROC per
fitted model and test mask inside each of the 2,000 shared identifier-clustered resamples, then
transformed (logit scale) or not (raw scale), then averaged over seeds and masks. Mean logit values
are never obtained from the logit of a mean raw AUROC.

The published interactions, responses and clean changes (results/confonly_primary/analysis
and results/confonly_extension/analysis) are checked to the printed five decimals before anything is written.
Writes results/reanalyses/r25_corrected_four_cells.csv and r25_summary.json.
    python3 experiments/reanalyses/r25_corrected_four_cells.py
"""
import numpy as np

import common as C

POL = ("S", "I")
# published values (effects.csv), five decimals: cell -> (TTI logit, lo, hi, TTI raw, lo, hi, R_S, R_I)
PUBLISHED = {
    "none": (0.58686, 0.52017, 0.66727, 0.09074, 0.07177, 0.11292, 0.61919, 0.03232),
    "dro_severity": (0.23040, 0.18420, 0.27810, 0.03690, 0.02740, 0.04860, 0.21510, -0.01520),
    "dro_input": (0.25143, 0.21007, 0.29876, 0.03953, 0.02975, 0.05119, 0.26155, 0.01012),
}
# clean change from no remedy, world S and I (criteria.json / effects.csv), five decimals
PUBLISHED_CLEAN = {"dro_input": {"S": (0.00244, -0.01085, 0.01616), "I": (-0.00748, -0.01651, 0.00125)}}


def main():
    cells = {
        "none": C.models("v15conf", "A_none", (0, 1, 2)),
        "dro_severity": C.models("v15conf", "C_groupdro", (0, 1, 2)),
        "dro_input": C.models("v16bc", "C_groupdro_obs", (0, 1, 2)),
    }
    recs = [m for g in cells.values() for w in POL for m in g[w]]
    sc = C.Scores(recs)
    eng = C.Engine(sc)
    print(f"rows {len(sc.ids)}, clusters {eng.n_clusters}")

    def cell_mean(g, w, p, scale):
        f = C.logit if scale == "logit" else (lambda a: a)
        return np.mean([[f(eng.aucs(m["name"], f"{C.MECH}_{p}_mid_m{ms}")) for ms in C.MASK_SEEDS]
                        for m in g[w]], axis=(0, 1))

    def clean(g, w):
        return np.mean([eng.aucs(m["name"], "clean") for m in g[w]], axis=0)

    rows, summary, problems = [], {}, []
    for key, g in cells.items():
        tti_l, rS, rI, _, _ = eng.cell(g)
        tti_r = eng.cell(g, scale="raw")[0]
        raw = {(w, p): cell_mean(g, w, p, "raw") for w in POL for p in POL}
        lgt = {(w, p): cell_mean(g, w, p, "logit") for w in POL for p in POL}
        # internal identities: responses from the logit cell means, raw TTI from the raw cell means
        for w, r in (("S", rS), ("I", rI)):
            if not np.allclose(lgt[(w, "S")] - lgt[(w, "I")], r):
                problems.append(f"{key}: logit cell means do not give R({w})")
        tti_raw_cells = (raw[("S", "S")] - raw[("S", "I")]) - (raw[("I", "S")] - raw[("I", "I")])
        if not np.allclose(tti_raw_cells, tti_r):
            problems.append(f"{key}: raw cell means do not give raw TTI")
        got = C.ci(tti_l) + C.ci(tti_r) + (C.ci(rS)[0], C.ci(rI)[0])
        want = PUBLISHED[key]
        tol = 5e-6 if key != "dro_severity" else 5e-5   # dro_severity published to four decimals
        if any(abs(a - b) > tol for a, b in zip(got, want)):
            problems.append(f"{key}: {tuple(round(x, 5) for x in got)} differs from published {want}")
        row = {"cell": key}
        for (w, p), a in raw.items():
            row[f"auroc_{w}{p}"] = round(float(a[0]), 6)
        for (w, p), a in lgt.items():
            row[f"logit_{w}{p}"] = round(float(a[0]), 6)
        row.update({"R_S": C.fmt(rS, 6), "R_I": C.fmt(rI, 6), "TTI_logit": C.fmt(tti_l, 6),
                    "TTI_raw": C.fmt(tti_r, 6)})
        for w in POL:
            row[f"clean_{w}"] = C.fmt(clean(g, w), 6)
            if key != "none":
                d = clean(g, w) - clean(cells["none"], w)
                row[f"clean_change_{w}"] = C.fmt(d, 6)
                if key in PUBLISHED_CLEAN:
                    if any(abs(a - b) > 5e-6 for a, b in zip(C.ci(d), PUBLISHED_CLEAN[key][w])):
                        problems.append(f"{key}: clean change {w} {C.fmt(d, 6)} differs from published")
        if key != "none":
            base_l = eng.cell(cells["none"])[0]
            base_r = eng.cell(cells["none"], scale="raw")[0]
            row["dTTI_logit"] = C.fmt(tti_l - base_l, 6)
            row["dTTI_raw"] = C.fmt(tti_r - base_r, 6)
        rows.append(row)
        summary[key] = row
    if problems:
        raise SystemExit("check failed:\n  " + "\n  ".join(problems))
    C.write_csv("r25_corrected_four_cells.csv", rows,
                header_comment="corrected loader, coordinate rule, assignment 0, CompactCTRGCN seeds 0-2, "
                               "mask mean 2000/3000/4000; AUROC per model and mask, transformed, then averaged")
    C.write_json("r25_summary.json", summary)
    for r in rows:
        print(r)
    print("all published values reproduced")


if __name__ == "__main__":
    main()
