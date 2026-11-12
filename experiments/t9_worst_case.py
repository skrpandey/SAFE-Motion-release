"""T9 worst-case sensitivity: external clean AUC per model group with the "yes" negatives removed
and with "yes" and "unsure" negatives removed, under both rating rules (lower: 5 yes + 70 unsure;
upper: 16 yes + 59 unsure). Video-clustered paired bootstrap over the 127 source videos.
Reads results/v11/scores and the answers from t9_finish.py; writes
results/v11/analysis/t9_worst_case{,_counts}.csv.
"""
import csv
import glob
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score

R, A_, N_BOOT = "results/v11", "results/v11/analysis", 2000


def read(p):
    return {r["identifier"]: r["answer"] for r in csv.DictReader(open(p))}


lower, upper = read(f"{R}/label_noise_answers_lower.csv"), read(f"{R}/label_noise_answers_upper.csv")
files = sorted(glob.glob(f"{R}/scores/*.csv"))
groups = {}
for p in files:
    groups.setdefault(re.sub(r"_s\d$", "", os.path.basename(p)[:-4]), []).append(p)
first = list(csv.DictReader(open(files[0])))
ids = [r["identifier"] for r in first]
y = np.array([int(r["label"]) for r in first])
vid = np.array([r["child_id"] for r in first])
scores = {g: np.mean([[float(r["clean"]) for r in csv.DictReader(open(p))] for p in ps], axis=0)
          for g, ps in groups.items()}
rules = {}
for name, d in (("lower", lower), ("upper", upper)):
    yes = np.array([d.get(i) == "yes" for i in ids])
    uns = np.array([d.get(i) == "unsure" for i in ids])
    rules[f"{name}_yes_removed"] = ~yes
    rules[f"{name}_yes_and_unsure_removed"] = ~(yes | uns)
counts = {k: int((~v).sum()) for k, v in rules.items()}
rng = np.random.RandomState(0)
uv = np.unique(vid)
idx_by_v = {v: np.flatnonzero(vid == v) for v in uv}
boots = [np.concatenate([idx_by_v[v] for v in rng.choice(uv, len(uv), replace=True)]) for _ in range(N_BOOT)]


def auc_sub(s, keep, idx):
    k = idx[keep[idx]]
    return roc_auc_score(y[k], s[k]) if len(np.unique(y[k])) == 2 else np.nan


out = []
for g, s in sorted(scores.items()):
    full = roc_auc_score(y, s)
    rec = {"group": g, "n_seeds": len(groups[g]), "auc_all": round(full, 4)}
    for name, keep in rules.items():
        a = roc_auc_score(y[keep], s[keep])
        diffs = np.array([auc_sub(s, keep, b) - auc_sub(s, np.ones_like(keep), b) for b in boots])
        rec[f"auc_{name}"] = round(a, 4)
        rec[f"diff_{name}"] = round(a - full, 4)
        rec[f"diff_{name}_lo"] = round(float(np.nanpercentile(diffs, 2.5)), 4)
        rec[f"diff_{name}_hi"] = round(float(np.nanpercentile(diffs, 97.5)), 4)
    out.append(rec)
os.makedirs(A_, exist_ok=True)
with open(f"{A_}/t9_worst_case.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
with open(f"{A_}/t9_worst_case_counts.csv", "w") as f:
    f.write("rule,windows_removed,negatives_remaining\n")
    for k, n in counts.items():
        f.write(f"{k},{n},{int(((y == 0) & rules[k]).sum())}\n")
print({k: n for k, n in counts.items()})
for r in out:
    if any(k in r["group"] for k in ("stage1_detector", "ours_v7_A_none", "ours_v7_A_both", "ours_v10_A_none")) and "noconf" not in r["group"]:
        print(r["group"][:26].ljust(26), "all", r["auc_all"],
              "| upper yes+unsure", r["auc_upper_yes_and_unsure_removed"], r["diff_upper_yes_and_unsure_removed"],
              f"[{r['diff_upper_yes_and_unsure_removed_lo']}, {r['diff_upper_yes_and_unsure_removed_hi']}]",
              "| lower yes+unsure", r["auc_lower_yes_and_unsure_removed"], r["diff_lower_yes_and_unsure_removed"])
