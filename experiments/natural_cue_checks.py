"""Post hoc checks of the natural tracking cue on ASDPose (cache only, no model).

Per population (length-protocol segments of the train and test splits, and the QC-pass host
test): AUROC of each QC statistic with non-SMM as the high class (fixed direction, identifier-
clustered interval), the share of missing joint-frames that fall in person-absent frames (all 17
joints missing), and the QC statistics recomputed on visible frames only, since absent frames
count as both frozen and low confidence. Writes results/v7/analysis/natural_cue_checks.csv and
paper/tables/natural_cue_checks.tex.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                 # noqa: E402
from safe_motion.data import asdpose as D            # noqa: E402
from safe_motion.planted import ARM_JOINTS           # noqa: E402

MIN_FRAMES, LOW, NB = 200, 0.3, 2000
QC = {"q_frozen": 0.10, "q_arm_lowconf": 0.30}
OUT = "results/v7/analysis/natural_cue_checks.csv"
TEX = "paper/tables/natural_cue_checks.tex"


def seg_stats(a):
    kp, sc = a[..., :2], a[..., 2]
    absent = (sc == 0).all(1)                                    # person absent: all 17 joints missing
    miss = sc == 0
    frozen = np.concatenate([[False], (kp[1:] == kp[:-1]).all(axis=(1, 2))])
    vis = ~absent
    arm = sc[:, ARM_JOINTS]
    return {"absent_frac": float(absent.mean()),
            "missing_jf": float(miss.sum()), "missing_jf_in_absent": float(miss[absent].sum()),
            "q_frozen": float(frozen[1:].mean()), "q_arm_lowconf": float((arm < LOW).mean()),
            "q_frozen_visible": float(frozen[vis][1:].mean()) if vis.sum() > 1 else 0.0,
            "q_arm_lowconf_visible": float((arm[vis] < LOW).mean()) if vis.any() else 0.0}


def auc_ci(score, y, groups):
    score, y = np.asarray(score), np.asarray(y)
    pt = M.auc(score, y)
    bs = [M.auc(score[i], y[i]) for i in M.cluster_bootstrap_indices(np.asarray(groups), NB, seed=0)]
    lo, hi = M.ci(bs)
    return pt, lo, hi


def main():
    data = D.load_cache()
    rows = [r for r in D.load_index() if r["n_frames"] >= MIN_FRAMES]
    stats = {}
    for r in rows:
        start = r["byte_offset"] // 4
        stats[r["identifier"]] = seg_stats(data[start:start + r["n_frames"] * 51].reshape(r["n_frames"], 17, 3))
    out = []
    pops = {"train, length protocol": [r for r in rows if r["split"] == "train"],
            "test, length protocol": [r for r in rows if r["split"] == "test"],
            "host test (QC pass)": [r for r in rows if r["split"] == "test" and D.passes_qc(r, {"q_frozen_max": 0.10, "q_arm_lowconf_max": 0.30})]}
    for pop, sel in pops.items():
        y = [1 - r["binary_label"] for r in sel]                 # non-SMM = 1: degraded more scores high
        g = [r["child_id"] for r in sel]
        rec = {"population": pop, "segments": len(sel), "identifiers": len(set(g))}
        for k in ("q_arm_lowconf", "q_frozen", "q_arm_lowconf_visible", "q_frozen_visible", "absent_frac"):
            pt, lo, hi = auc_ci([stats[r["identifier"]][k] for r in sel], y, g)
            rec[f"auc_{k}"] = f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"
        for cls, name in ((0, "nonSMM"), (1, "SMM")):
            s = [stats[r["identifier"]] for r in sel if r["binary_label"] == cls]
            rec[f"absent_share_of_missing_{name}"] = f"{sum(x['missing_jf_in_absent'] for x in s) / max(1, sum(x['missing_jf'] for x in s)):.3f}"
            rec[f"qc_fail_{name}"] = f"{np.mean([x['q_frozen'] >= QC['q_frozen'] or x['q_arm_lowconf'] >= QC['q_arm_lowconf'] for x in s]):.3f}"
            rec[f"qc_fail_visible_{name}"] = f"{np.mean([x['q_frozen_visible'] >= QC['q_frozen'] or x['q_arm_lowconf_visible'] >= QC['q_arm_lowconf'] for x in s]):.3f}"
        out.append(rec)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write("# post hoc: AUROC of each statistic with non-SMM as the high class "
                "(fixed direction), identifier-clustered 95% interval (2,000); 'visible' statistics exclude frames with all 17 joints missing\n")
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Population & Arm low-conf. & Frozen & Arm low-conf., visible & Frozen, visible & QC fail non-SMM / SMM & QC fail, visible \\",
         r"\midrule"]
    for r in out:
        L.append(f"{r['population']} & {r['auc_q_arm_lowconf']} & {r['auc_q_frozen']} & {r['auc_q_arm_lowconf_visible']} & "
                 f"{r['auc_q_frozen_visible']} & {float(r['qc_fail_nonSMM']):.1%} / {float(r['qc_fail_SMM']):.1%} & "
                 f"{float(r['qc_fail_visible_nonSMM']):.1%} / {float(r['qc_fail_visible_SMM']):.1%} \\\\".replace("%", "\\%"))
    L += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(TEX), exist_ok=True)
    with open(TEX, "w") as f:
        f.write("% generated by experiments/natural_cue_checks.py from the SAFE-SMM cache; do not edit\n" + "\n".join(L) + "\n")
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
