"""Full-minus-QC and pass-minus-fail AUC gaps (post hoc) for the five real-split variants, as
trained (v7) and retrained with label-independent missing runs (v10 real_I). Logit and raw
scales, seeds averaged inside each resample, 2,000 child-clustered resamples of the 40 test
children shared by all models. QC set = all SMM segments plus QC-passing non-SMM segments.
Writes results/v7/analysis/gap_table_b4.csv."""
import csv, sys
import numpy as np
sys.path.insert(0, ".")
from safe_motion import metrics as M
VARS = ["A_none", "A_velocity", "A_dropout", "A_both", "B_coorddrop"]
NB = 2000


def load(p):
    rd = list(csv.DictReader(open(p)))
    return ([r["identifier"] for r in rd], np.array([int(r["label"]) for r in rd]), np.array([int(r["qc_pass"]) for r in rd]),
            [r["child_id"] for r in rd], np.array([float(r["clean"]) for r in rd]))


ids0, y, qc, child, _ = load("results/v7/scores/v7_real_A_none_s0.csv")
pos = {i: k for k, i in enumerate(ids0)}
boots = [np.arange(len(y))] + list(M.cluster_bootstrap_indices(child, NB, seed=0))
lg = lambda a: np.log(np.clip(a, 1e-6, 1 - 1e-6) / (1 - np.clip(a, 1e-6, 1 - 1e-6)))


def gaps(s):
    full, qcp, pas, fai = [], [], [], []
    for b in boots:
        yb, qb, sb = y[b], qc[b], s[b]
        full.append(M.auc(sb, yb)); k = (yb == 1) | (qb == 1); qcp.append(M.auc(sb[k], yb[k]))
        pas.append(M.auc(sb[qb == 1], yb[qb == 1])); fai.append(M.auc(sb[qb == 0], yb[qb == 0]))
    full, qcp, pas, fai = map(np.array, (full, qcp, pas, fai))
    return {"full_qc_logit": lg(full) - lg(qcp), "full_qc_raw": full - qcp, "pass_fail_raw": pas - fai, "pass_fail_logit": lg(pas) - lg(fai)}


G = {}
for v in VARS:
    for st, pat in (("original", "results/v7/scores/v7_real_{v}_s{s}.csv"), ("retrained", "results/v10/scores/v10_real_I_{v}_s{s}.csv")):
        per = []
        for s in range(3):
            ids, yy, _, _, sc = load(pat.format(v=v, s=s))
            o = np.array([pos[i] for i in ids]); x = np.empty(len(y)); x[o] = sc
            assert (yy == y[o]).all()
            per.append(gaps(x))
        G[v, st] = {k: np.mean([p[k] for p in per], 0) for k in per[0]}
        G[v, st]["seed_points"] = {k: [round(float(p[k][0]), 4) for p in per] for k in per[0]}
out = []
fmt = lambda a: f"{a[0]:+.3f} [{M.ci(a[1:])[0]:+.3f}, {M.ci(a[1:])[1]:+.3f}]"
for v in VARS:
    row = {"variant": v}
    for k in ("full_qc_logit", "full_qc_raw", "pass_fail_raw"):
        row[f"{k} original"] = fmt(G[v, "original"][k]); row[f"{k} retrained"] = fmt(G[v, "retrained"][k])
        row[f"{k} retrained-original"] = fmt(G[v, "retrained"][k] - G[v, "original"][k])
    row["seed points full_qc_logit original/retrained"] = f"{G[v,'original']['seed_points']['full_qc_logit']} / {G[v,'retrained']['seed_points']['full_qc_logit']}"
    out.append(row); print(row, flush=True)
with open("results/v7/analysis/gap_table_b4.csv", "w", newline="") as f:
    f.write("# post hoc; paired child-clustered 95% intervals, 2,000 resamples, seeds averaged inside resamples\n")
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
