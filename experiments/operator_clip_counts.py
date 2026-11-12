"""Planted severity draws affected by the 0.95 cap and by entry-probability clipping
(eps > L/(L+1)) on the ASDPose host as used in v7/v8/v12/v13 (post hoc; length protocol, QC-pass
pool; train/validation masks world seed 0, test masks 2000/3000/4000). The clipped chain's
stationary probability is a/(a + 1/L) with a = min(eps/(L(1-eps)), 1), i.e. eps when
eps <= L/(L+1) and L/(L+1) otherwise. Writes results/v7/analysis/operator_clip_counts.csv."""
import csv, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "experiments")
from safe_motion import planted as P
from safe_motion.data import asdpose as D
import score_v7 as S7
rows = [r for r in D.load_index() if r["n_frames"] >= 200]
tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
te = [r for r in rows if r["split"] == "test"]
q = lambda rr: [r for r in rr if D.passes_qc(r, S7.V1["qc_filter"])]
sets = {"train": (q(tr), [0]), "validation": (q(va), [0]), "test": (q(te), [2000, 3000, 4000])}
out = []
for strength in ("mid", "low"):
    for w in ("S", "I"):
        for split, (rr, seeds) in sets.items():
            for sd in seeds:
                e = P.severities(rr, w, strength, sd)
                out.append({"strength": strength, "world": w, "split": split, "mask_seed": sd, "n_draws": len(e),
                            "max_eps": round(float(e.max()), 4), "n_above_0.95_cap": int((e > 0.95).sum()),
                            "n_above_15/16 (L=15 clipped)": int((e > 15 / 16).sum()),
                            "n_above_20/21 (L=20, unreachable after cap)": int((e > 20 / 21).sum())})
with open("results/v7/analysis/operator_clip_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
tot = {k: sum(r[k] for r in out) for k in ("n_draws", "n_above_0.95_cap", "n_above_15/16 (L=15 clipped)")}
print(tot, "max", max(r["max_eps"] for r in out))
