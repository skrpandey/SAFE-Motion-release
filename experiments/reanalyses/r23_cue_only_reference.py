"""R23. Reference point for the magnitude.

Cue-only scorer: the planted arm low-confidence fraction of each test segment (fourth SAFE-SMM quality
statistic, remedies.quality_stats()[3], of the planted input), direction fixed so that a higher
fraction scores as non-SMM (score = minus the fraction). Its AUROC on the 1,966 host test rows under
policy S and policy I for the three test masks, the mean logit AUROC per policy, and its response
logit AUROC(S) - logit AUROC(I) (mask mean inside each resample), with the identifier-clustered
interval of the main analysis (RowSet seed 0, 2,000 resamples). Primary: whole segment (the
statistic as defined); secondary: the centre 200-frame window the models score. Cross-check: the
unplanted arm low-confidence fraction gives the article's 0.527 [0.495, 0.562].
Writes results/reanalyses/r23_cue_only_reference.csv and r23_summary.json.
    python3 experiments/reanalyses/r23_cue_only_reference.py
"""
import csv

import numpy as np

import common as C
from safe_motion.data import asdpose as D


def main():
    cells = C.confonly_cells()
    sc = C.Scores([m for k in ("hist3", "corr3") for w in ("S", "I") for m in cells[k][w]])
    eng = C.Engine(sc)
    rs = eng.rs
    pl = {}
    for r in csv.DictReader(l for l in open(C.OUT / "test_planted_severities.csv") if not l.startswith("#")):
        pl[(r["identifier"], r["policy"], int(r["mask_seed"]))] = r
    out, summary = [], {}
    for stat in ("arm_lowconf_full", "arm_lowconf_window"):
        A = {p: np.stack([rs.aucs(-np.array([float(pl[(i, p, ms)][stat]) for i in sc.ids])) for ms in C.MASK_SEEDS])
             for p in ("S", "I")}
        resp = C.logit(A["S"]).mean(0) - C.logit(A["I"]).mean(0)
        raw = A["S"].mean(0) - A["I"].mean(0)
        rec = {"statistic": stat,
               "auroc_policy_S_per_mask": "; ".join(f"m{ms} {a:.3f}" for ms, a in zip(C.MASK_SEEDS, A["S"][:, 0])),
               "auroc_policy_I_per_mask": "; ".join(f"m{ms} {a:.3f}" for ms, a in zip(C.MASK_SEEDS, A["I"][:, 0])),
               "auroc_policy_S_mean": C.fmt(A["S"].mean(0)), "auroc_policy_I_mean": C.fmt(A["I"].mean(0)),
               "logit_auroc_S_mean": C.fmt(C.logit(A["S"]).mean(0)), "logit_auroc_I_mean": C.fmt(C.logit(A["I"]).mean(0)),
               "response_logit": C.fmt(resp), "response_raw": C.fmt(raw)}
        # the model quantities on the same resamples, for the ratio
        for key in ("hist3", "corr3"):
            mwi, rS, rI, _, _ = eng.cell(cells[key])
            rec[f"MWI_{key}"] = C.fmt(mwi)
            rec[f"MWI_{key}_over_cue_response"] = C.fmt(mwi / resp)
            rec[f"R_fS_{key}_over_cue_response"] = C.fmt(rS / resp)
        out.append(rec)
        print(rec, flush=True)
    # cross-check: unplanted arm low-confidence fraction of the host test rows (article: 0.527 [0.495, 0.562])
    q = {r["identifier"]: r["q_arm_lowconf"] for r in D.load_index()}
    nat = rs.aucs(-np.array([q[i] for i in sc.ids]))
    summary["natural_arm_lowconf_auroc_host_test"] = C.fmt(nat)
    print("natural:", summary["natural_arm_lowconf_auroc_host_test"])
    C.write_csv("r23_cue_only_reference.csv", out,
                "R23: cue-only scorer = minus the planted arm low-confidence fraction (non-SMM scored high when degraded), 1,966 host\n"
                "test rows, test masks 2000/3000/4000; response = mean logit AUROC under policy S minus under policy I; 95%\n"
                "identifier-clustered percentile intervals (RowSet seed 0, 2,000 resamples; shared with the model quantities)")
    C.write_json("r23_summary.json", summary)


if __name__ == "__main__":
    main()
