"""The two filtered-gap statistics, computed side by side (post hoc, descriptive).

Delta_full_to_QC: AUC on all test segments minus AUC under the QC protocol (all SMM segments plus
QC-passing non-SMM segments), the statistic SAFE-SMM published. Delta_pass_minus_fail: AUC on
QC-passing minus AUC on QC-failing segments, as in the v7 comparators. Length protocol, 15 v7 real
and 15 v10 real-data World I models, raw and logit scale, child-clustered bootstrap (2,000) with
seeds averaged inside resamples. Writes results/v7/analysis/qc_gap_definitions.csv.
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M  # noqa: E402

N_BOOT = 2000


def logit(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


def load(p):
    rd = list(csv.DictReader(open(p)))
    return ([r["identifier"] for r in rd], np.array([int(r["label"]) for r in rd]),
            np.array([r["child_id"] for r in rd]), np.array([int(r["qc_pass"]) for r in rd]),
            np.array([float(r["clean"]) for r in rd]))


def main():
    files = sorted(glob.glob("results/v7/scores/v7_real_*.csv")) + sorted(glob.glob("results/v10/scores/v10_real_I_*.csv"))
    ids0, y, child, qc, _ = load(files[0])
    pos = {i: k for k, i in enumerate(ids0)}
    boots = [np.arange(len(ids0))] + list(M.cluster_bootstrap_indices(child, N_BOOT, seed=0))
    mask_qcprot = (y == 1) | (qc == 1)
    stats = {}
    for p in files:
        ids, yy, _, qq, s = load(p)
        order = np.array([pos[i] for i in ids])
        sc = np.empty(len(ids0)); sc[order] = s
        vals = {"full": [], "qcprot": [], "pass": [], "fail": []}
        for b in boots:
            vals["full"].append(M.auc(sc[b], y[b]))
            m = mask_qcprot[b]; vals["qcprot"].append(M.auc(sc[b][m], y[b][m]))
            m = qc[b] == 1; vals["pass"].append(M.auc(sc[b][m], y[b][m]))
            m = qc[b] == 0; vals["fail"].append(M.auc(sc[b][m], y[b][m]))
        stats[os.path.basename(p)[:-4]] = {k: np.array(v) for k, v in vals.items()}
    rows = []
    groups = defaultdict(list)
    for name in stats:
        groups[re.sub(r"_s\d$", "", name)].append(name)
    def add(label, members):
        v = {k: np.mean([stats[n][k] for n in members], axis=0) for k in ("full", "qcprot", "pass", "fail")}
        rec = {"model": label, "n_seeds": len(members)}
        for nm, d_raw, d_logit in (
                ("full_to_QC", v["full"] - v["qcprot"], logit(v["full"]) - logit(v["qcprot"])),
                ("pass_minus_fail", v["pass"] - v["fail"], logit(v["pass"]) - logit(v["fail"]))):
            lo, hi = M.ci(d_raw[1:]); llo, lhi = M.ci(d_logit[1:])
            rec.update({f"Delta_{nm}_raw": round(float(d_raw[0]), 4), f"Delta_{nm}_raw_lo": round(lo, 4),
                        f"Delta_{nm}_raw_hi": round(hi, 4), f"Delta_{nm}_logit": round(float(d_logit[0]), 4),
                        f"Delta_{nm}_logit_lo": round(llo, 4), f"Delta_{nm}_logit_hi": round(lhi, 4)})
        rec["auc_full"] = round(float(v["full"][0]), 4); rec["auc_qc_protocol"] = round(float(v["qcprot"][0]), 4)
        rows.append(rec)
    for g, mem in sorted(groups.items()):
        add(g, mem)
    for n in sorted(stats):
        add(n, [n])
    os.makedirs("results/v7/analysis", exist_ok=True)
    with open("results/v7/analysis/qc_gap_definitions.csv", "w", newline="") as f:
        f.write("# post hoc descriptive recomputation; Delta_full_to_QC is the SAFE-SMM statistic; length protocol\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    single = [r for r in rows if r["n_seeds"] == 1]
    for key in ("Delta_full_to_QC_raw", "Delta_full_to_QC_logit", "Delta_pass_minus_fail_raw"):
        vs = [r[key] for r in single]
        print(key, "range over", len(vs), "models:", min(vs), "to", max(vs))
    for r in rows:
        if r["n_seeds"] > 1:
            print(r["model"], r["Delta_full_to_QC_raw"], [r["Delta_full_to_QC_raw_lo"], r["Delta_full_to_QC_raw_hi"]],
                  "logit", r["Delta_full_to_QC_logit"], "| pass-fail", r["Delta_pass_minus_fail_raw"])


if __name__ == "__main__":
    main()
