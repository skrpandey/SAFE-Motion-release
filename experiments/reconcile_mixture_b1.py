"""Post hoc: the score-mixture endpoint at alpha = 0 against the interpolated control of Table S4.

Both from the same v10 A_blind score files (host test, length protocol, QC pass, mask 2000): the
logit AUROC of the seed-averaged score, as in oracle_mixture.py, against the mean over seeds of
logit AUROC, as in analyze_v10.py. Writes results/v10/analysis/b1_reconcile.csv."""
import csv, sys, os
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
from safe_motion import metrics as M
from safe_motion.data import asdpose as D
import score_v7 as S7

def logit(a): a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6); return np.log(a / (1 - a))
rows = [r for r in D.load_index() if r["n_frames"] >= 200]
te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, S7.V1["qc_filter"])]
ids = [r["identifier"] for r in te]; y = np.array([r["binary_label"] for r in te])
out = []
for mech in ("runs", "whole"):
    sc = {}
    for w in ("S", "I"):
        for s in range(3):
            p = f"results/v10/scores/v10_{w}_w0_{mech}_mid_A_blind_s{s}.csv"
            rd = {r["identifier"]: r for r in csv.DictReader(open(p))}
            assert set(rd) == set(ids), (p, len(set(rd) ^ set(ids)))
            for t in ("S", "I"):
                sc[w, s, t] = np.array([float(rd[i][f"{mech}_{t}_mid_m2000"]) for i in ids])
    per = {}
    for w in ("S", "I"):
        for t in ("S", "I"):
            per[w, t] = [M.auc(sc[w, s, t], y) for s in range(3)]
            out.append({"mechanism": mech, "world": w, "policy": t,
                        **{f"auc_s{s}": round(per[w, t][s], 5) for s in range(3)},
                        "auc_ensemble": round(M.auc(np.mean([sc[w, s, t] for s in range(3)], 0), y), 5)})
    se_seed = np.mean([logit(per["S", "S"][s]) - logit(per["S", "I"][s]) for s in range(3)]) - \
              np.mean([logit(per["I", "S"][s]) - logit(per["I", "I"][s]) for s in range(3)])
    ens = {k: M.auc(np.mean([sc[k[0], s, k[1]] for s in range(3)], 0), y) for k in per}
    se_ens = (logit(ens["S", "S"]) - logit(ens["S", "I"])) - (logit(ens["I", "S"]) - logit(ens["I", "I"]))
    # also the ensemble of logits, in case the oracle averages on the logit scale
    ens2 = {k: M.auc(np.mean([logit(sc[k[0], s, k[1]]) for s in range(3)], 0), y) for k in per}
    se_ens2 = (logit(ens2["S", "S"]) - logit(ens2["S", "I"])) - (logit(ens2["I", "S"]) - logit(ens2["I", "I"]))
    print(f"{mech}: SE per-seed estimator {se_seed:.4f} | ensemble of probabilities {se_ens:.4f} | ensemble of logits {se_ens2:.4f}")
    out.append({"mechanism": mech, "world": "SE", "policy": "", "auc_s0": round(se_seed, 5), "auc_s1": round(se_ens, 5), "auc_s2": round(se_ens2, 5),
                "auc_ensemble": "per-seed / prob-ensemble / logit-ensemble"})
with open("results/v10/analysis/b1_reconcile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
