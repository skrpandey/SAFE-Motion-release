"""Protocol v16: permutation reliance on the confidence channel (definitions as analyze_conf_reliance.py).

Reliance = logit AUROC on the input minus the mean logit AUROC over five permutations of the confidence
channel across segments, on clean input and on planted policy S and I input (first test mask). Seeds are
averaged inside 2,000 identifier-clustered resamples shared across models. Groups: ST-GCN (v16, seeds 0-2);
CompactCTRGCN no remedy with nine seeds per world (v15conf seeds 0-2 scored by score_conf_reliance.py, pooled
with v16 seeds 3-8, the pooling planned in protocol v16) and the v16 seeds 3-8 alone; Group DRO with observed
groups (v16, seeds 0-2).
Writes results/v16/analysis/conf_reliance.csv.
    python experiments/analyze_conf_reliance_v16.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                  # noqa: E402
from experiments.analyze_v7 import RowSet, logit, align                # noqa: E402

SRC16 = "results/v16/conf_reliance"
SRC15 = "results/posthoc/conf_reliance"
OUT = "results/v16/analysis/conf_reliance.csv"
N_PERM = 5
GROUPS = (("ST-GCN, no remedy", [("v16", "A_none_stgcn", (0, 1, 2))]),
          ("CompactCTRGCN, no remedy, nine seeds", [("v15conf", "A_none", (0, 1, 2)), ("v16", "A_none", tuple(range(3, 9)))]),
          ("CompactCTRGCN, no remedy, seeds 3-8", [("v16", "A_none", tuple(range(3, 9)))]),
          ("CompactCTRGCN, no remedy, seeds 0-2", [("v15conf", "A_none", (0, 1, 2))]),
          ("CompactCTRGCN, Group DRO with observed groups", [("v16", "C_groupdro_obs", (0, 1, 2))]))


def fmt(a, d=3):
    lo, hi = M.ci(a[1:])
    return f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    models = {}
    for src in (SRC16, SRC15):
        for m in json.load(open(f"{src}/registry.json"))["models"]:
            models[m["name"]] = dict(m, src=src)
    need = {(p, c, s) for _, parts in GROUPS for p, c, seeds in parts for s in seeds}
    use = {n: m for n, m in models.items() if (m["proto"], m["cell"], m["seed"]) in need}
    for p, c, s in need:
        for w in ("S", "I"):
            assert any(m["proto"] == p and m["cell"] == c and m["seed"] == s and m["world"] == w for m in use.values()), (p, c, s, w)
    data = {n: list(csv.DictReader(open(f"{m['src']}/scores/{n}.csv"))) for n, m in use.items()}
    first = next(iter(data.values()))
    ids = [r["identifier"] for r in first]
    rs = RowSet(ids, [int(r["label"]) for r in first], [r["child_id"] for r in first], 2000)
    cols = ["clean"] + [f"perm{k}" for k in range(N_PERM)] + \
           [f"pol{w}{x}" for w in ("S", "I") for x in [""] + [f"_perm{k}" for k in range(N_PERM)]]
    A = {}
    for n, rd in data.items():
        S = align(rs, [r["identifier"] for r in rd], [r["label"] for r in rd],
                  {c: np.array([float(r[c]) for r in rd]) for c in cols})
        A[n] = {c: logit(rs.aucs(S[c])) for c in cols}

    def reliance(names, base):
        perm = (lambda k: f"perm{k}") if base == "clean" else (lambda k: f"{base}_perm{k}")
        return np.mean([np.mean([A[n][base] - A[n][perm(k)] for k in range(N_PERM)], axis=0) for n in names], axis=0)

    rows = []
    for label, parts in GROUPS:
        g = {w: [n for n, m in use.items() if m["world"] == w
                 and any(m["proto"] == p and m["cell"] == c and m["seed"] in seeds for p, c, seeds in parts)] for w in ("S", "I")}
        rec = {"group": label, "n_seeds": f"{len(g['S'])}/{len(g['I'])}"}
        for base, tag in (("clean", "clean"), ("polS", "policyS"), ("polI", "policyI")):
            rS, rI = reliance(g["S"], base), reliance(g["I"], base)
            rec[f"{tag}_S"], rec[f"{tag}_I"], rec[f"{tag}_S_minus_I"] = fmt(rS), fmt(rI), fmt(rS - rI)
        rows.append(rec)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write("# protocol v16: permutation reliance on the confidence channel, logit AUROC drop, seeds averaged inside "
                "2,000 identifier-clustered resamples\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['group']:46s} {r['n_seeds']:5s} clean S-I {r['clean_S_minus_I']} | policy S S-I {r['policyS_S_minus_I']} "
              f"| policy I S-I {r['policyI_S_minus_I']}")


if __name__ == "__main__":
    main()
