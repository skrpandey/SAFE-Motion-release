"""T9 label-noise check on 100 sampled external windows, with lower and upper bounds (no human
adjudication). Raters A and B rate all 100 windows; both rerun on the 27 windows either flagged.
Lower bound: yes where both reruns say yes; upper: where either does; unflagged windows are no if
both first-pass ratings are no, else unsure. The rule was fixed before any AUC effect was computed.
Writes results/v11/label_noise_answers{,_lower,_upper}.csv, analysis/t9_label_noise.csv (Wilson
intervals, kappa), analysis/t9_sensitivity.csv (AUC without flagged negatives, video-clustered
bootstrap) and updates criteria.json["T9"].
"""
import csv, glob, json, os, re
import numpy as np
from sklearn.metrics import roc_auc_score

R = "results/v11"; A_ = f"{R}/analysis"; N_BOOT = 2000
def read(p): return {r["identifier"]: r for r in csv.DictReader(open(p))}
A1 = read(f"{R}/label_noise_rater_A_pass1.csv"); B1 = read(f"{R}/label_noise_rater_B_pass1.csv")
A2 = read(f"{R}/label_noise_rater_A_pass2_27.csv"); B2 = read(f"{R}/label_noise_rater_B_pass2_27.csv")
order = [r["identifier"] for r in csv.DictReader(open(f"{R}/label_noise_sample.csv"))]
flag = [i for i in order if "yes" in (A1[i]["answer"], B1[i]["answer"])]
assert set(flag) == set(A2) == set(B2) and len(order) == 100
def unflagged(i): return "no" if A1[i]["answer"] == B1[i]["answer"] == "no" else "unsure"
lower, upper = {}, {}
for i in order:
    if i in A2:
        a, b = A2[i]["answer"], B2[i]["answer"]
        lower[i] = "yes" if a == b == "yes" else ("no" if a == b == "no" else "unsure")
        upper[i] = "yes" if "yes" in (a, b) else ("no" if a == b == "no" else "unsure")
    else:
        lower[i] = upper[i] = unflagged(i)
for name, d in (("lower", lower), ("upper", upper)):
    with open(f"{R}/label_noise_answers_{name}.csv", "w") as f:
        f.write("identifier,answer\n"); [f.write(f"{i},{d[i]}\n") for i in order]
with open(f"{R}/label_noise_answers.csv", "w") as f:
    f.write("identifier,answer\n"); [f.write(f"{i},{lower[i]}\n") for i in order]

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h
cats = ("yes", "no", "unsure")
def counts(d): return {c: sum(v == c for v in d.values()) for c in cats}
a1 = {i: A1[i]["answer"] for i in order}; b1 = {i: B1[i]["answer"] for i in order}
ag = sum(a1[i] == b1[i] for i in order); pe = sum(counts(a1)[c] * counts(b1)[c] for c in cats) / 1e4
kappa1 = (ag / 100 - pe) / (1 - pe)
a2 = {i: A2[i]["answer"] for i in flag}; b2 = {i: B2[i]["answer"] for i in flag}; n27 = len(flag)
ag2 = sum(a2[i] == b2[i] for i in flag); pe2 = sum(counts(a2)[c] * counts(b2)[c] for c in cats) / n27**2
kappa2 = (ag2 / n27 - pe2) / (1 - pe2)
os.makedirs(A_, exist_ok=True)
rows = []
for name, d in (("model_A_first_pass", a1), ("model_B_first_pass", b1), ("lower_bound_both_reruns_yes", lower), ("upper_bound_any_rerun_yes", upper)):
    c = counts(d); p, lo, hi = wilson(c["yes"], 100)
    rows.append({"rating": name, "yes": c["yes"], "no": c["no"], "unsure": c["unsure"], "yes_fraction": round(p, 3), "wilson_lo": round(lo, 3), "wilson_hi": round(hi, 3)})
rows.append({"rating": "rerun_A_on_27", "yes": counts(a2)["yes"], "no": counts(a2)["no"], "unsure": counts(a2)["unsure"], "yes_fraction": "", "wilson_lo": "", "wilson_hi": ""})
rows.append({"rating": "rerun_B_on_27", "yes": counts(b2)["yes"], "no": counts(b2)["no"], "unsure": counts(b2)["unsure"], "yes_fraction": "", "wilson_lo": "", "wilson_hi": ""})
rows.append({"rating": "agreement_first_pass_100", "yes": ag, "no": "", "unsure": "", "yes_fraction": f"kappa {kappa1:.3f}", "wilson_lo": "", "wilson_hi": ""})
rows.append({"rating": "agreement_reruns_27", "yes": ag2, "no": "", "unsure": "", "yes_fraction": f"kappa {kappa2:.3f}", "wilson_lo": "", "wilson_hi": ""})
with open(f"{A_}/t9_label_noise.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# sensitivity: clean external AUC per model group, all windows vs without the flagged negatives
files = sorted(glob.glob(f"{R}/scores/*.csv"))
groups = {}
for p in files:
    n = os.path.basename(p)[:-4]; groups.setdefault(re.sub(r"_s\d$", "", n), []).append(p)
first = list(csv.DictReader(open(files[0])))
ids = [r["identifier"] for r in first]; y = np.array([int(r["label"]) for r in first]); vid = np.array([r["child_id"] for r in first])
scores = {g: np.mean([[float(r["clean"]) for r in csv.DictReader(open(p))] for p in ps], axis=0) for g, ps in groups.items()}
drop = {"lower": np.array([lower.get(i) == "yes" for i in ids]), "upper": np.array([upper.get(i) == "yes" for i in ids])}
rng = np.random.RandomState(0); uv = np.unique(vid); idx_by_v = {v: np.flatnonzero(vid == v) for v in uv}
boots = [np.concatenate([idx_by_v[v] for v in rng.choice(uv, len(uv), replace=True)]) for _ in range(N_BOOT)]
def auc_keep(s, keep, idx):
    k = idx[keep[idx]]; 
    return roc_auc_score(y[k], s[k]) if len(np.unique(y[k])) == 2 else np.nan
out = []
for g, s in sorted(scores.items()):
    full = roc_auc_score(y, s); rec = {"group": g, "n_seeds": len(groups[g]), "auc_all": round(full, 4)}
    for name, dr in drop.items():
        keep = ~dr; a = roc_auc_score(y[keep], s[keep])
        diffs = np.array([auc_keep(s, keep, b) - auc_keep(s, np.ones_like(keep), b) for b in boots])
        rec[f"auc_without_{name}"] = round(a, 4); rec[f"diff_{name}"] = round(a - full, 4)
        rec[f"diff_{name}_lo"] = round(np.nanpercentile(diffs, 2.5), 4); rec[f"diff_{name}_hi"] = round(np.nanpercentile(diffs, 97.5), 4)
    out.append(rec)
with open(f"{A_}/t9_sensitivity.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
crit = json.load(open(f"{A_}/criteria.json"))
crit["T9"] = {"status": "independently rated, no human adjudication; lower and upper bounds", "n_sampled": 100, "flagged_by_first_pass": len(flag),
              "yes_lower": counts(lower)["yes"], "yes_upper": counts(upper)["yes"], "unsure_lower": counts(lower)["unsure"], "unsure_upper": counts(upper)["unsure"],
              "kappa_first_pass": round(kappa1, 3), "kappa_reruns_27": round(kappa2, 3), "n_dropped_lower": int(drop["lower"].sum()), "n_dropped_upper": int(drop["upper"].sum())}
json.dump(crit, open(f"{A_}/criteria.json", "w"), indent=1, default=float)
print(json.dumps(crit["T9"]))
