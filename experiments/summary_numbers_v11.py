"""Post hoc summary numbers: external AUROC ranges (all groups and without the velocity-only
ablation and permuted controls), positives per category, head-banging AUROC per model group, the
arm-confidence R(f_I) at four decimals, and the one-seed training-mask realization cells.
Writes results/v11/analysis/summary_ranges.csv and prints the numbers.
"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "experiments")
from analyze_v11 import EXTERNAL_CACHE  # noqa: E402
from safe_motion.data import asdpose as D  # noqa: E402

EXCL = {"abl_velocity_v3norm", "permuted_v1", "permuted_v2"}


def main():
    t2 = list(csv.DictReader(open("results/v11/analysis/t2_transfer.csv")))
    out = []
    allv = [(r["group"], float(r["auc"])) for r in t2]
    sub = [(g, a) for g, a in allv if g not in EXCL]
    out.append({"item": "external AUROC range, all 25 groups", "value": f"{min(a for _, a in allv):.3f} to {max(a for _, a in allv):.3f}",
                "note": f"min {min(allv, key=lambda x: x[1])[0]}, max {max(allv, key=lambda x: x[1])[0]}"})
    out.append({"item": "external AUROC range, excluding velocity-only v3-norm ablation and the two permuted controls",
                "value": f"{min(a for _, a in sub):.3f} to {max(a for _, a in sub):.3f}",
                "note": f"min {min(sub, key=lambda x: x[1])[0]}, max {max(sub, key=lambda x: x[1])[0]}; 22 groups"})
    rows = D.load_index(f"{EXTERNAL_CACHE}/index.csv")
    cats = {}
    for r in rows:
        if r["binary_label"] == 1:
            cats[r["action_name"]] = cats.get(r["action_name"], 0) + 1
    out.append({"item": "positives per category", "value": "; ".join(f"{k} {v}" for k, v in sorted(cats.items())), "note": f"negatives {sum(r['binary_label']==0 for r in rows)}"})
    for r in t2:
        out.append({"item": f"head-banging AUROC, {r['group']}", "value": f"{float(r['auc_headbanging']):.3f} [{float(r['auc_headbanging_lo']):.3f}, {float(r['auc_headbanging_hi']):.3f}]",
                    "note": "33 head-banging positives against all 423 negatives"})
    g = list(csv.DictReader(open("results/v7/analysis/groups_masks.csv")))
    for r in g:
        if r["world"] == "I" and r["mechanism"] == "armruns" and r["cell"] == "A_none":
            out.append({"item": "arm-confidence reference R(f_I), mask mean, four decimals", "value": r["R_maskmean"], "note": "groups_masks.csv"})
    e = list(csv.DictReader(open("results/v7/analysis/effects_masks.csv")))
    for r in e:
        if r["mechanism"] == "armruns" and r["contrast"] == "SE(A_none)":
            out.append({"item": "arm-confidence R_I_ci from effects_masks", "value": r["R_I_ci"], "note": ""})
    reg = json.load(open("results/v7/registry.json"))
    one_seed = [m["name"] for m in reg if m.get("world_seed", 0) not in (0, None)]
    out.append({"item": "one-seed training-mask realization cells (exceptions to three seeds)", "value": "; ".join(one_seed), "note": "World S missing runs mid, training-mask seeds 1 and 2, seed 0 only"})
    with open("results/v11/analysis/summary_ranges.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value", "note"]); w.writeheader(); w.writerows(out)
    for r in out:
        print(r["item"], "|", r["value"], "|", r["note"])


if __name__ == "__main__":
    main()
