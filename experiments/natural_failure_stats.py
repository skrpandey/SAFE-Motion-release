"""Natural tracker-failure statistics on ASDPose for comparison with the planted operators (post hoc).

Uses whole length-protocol segments (>= 200 frames) of the train and test splits, by class, from
the SAFE-SMM cache: missing joint-frames (per joint, co-missingness, run lengths, whole-segment
absence), frozen frames (fraction, run lengths) and arm-joint confidence. Matching marginals do
not validate the joint failure model.
Writes results/v7/analysis/natural_failure_stats.csv and paper/tables/natural_failure_stats.tex.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.data import asdpose as D    # noqa: E402
from safe_motion.paths import guard_output    # noqa: E402
from safe_motion.planted import ARM_JOINTS, RUN_LEN, FROZEN_RUN_LEN   # noqa: E402

MIN_FRAMES = 200


def runs(mask1d):
    """Lengths of the runs of True in a 1-D boolean array."""
    if not mask1d.any():
        return []
    d = np.diff(np.concatenate([[0], mask1d.astype(int), [0]]))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist()


def main():
    data = D.load_cache()
    rows = [r for r in D.load_index() if r["n_frames"] >= MIN_FRAMES]
    out = []
    for split in ("train", "test"):
        for cls in (0, 1):
            sel = [r for r in rows if r["split"] == split and r["binary_label"] == cls]
            per_joint_missing = np.zeros(17); n_jf = 0
            co_counts, all17 = [], 0; n_missing_jf = 0
            miss_runs, frozen_runs, arm_conf = [], [], []
            whole_missing_joints, joint_segments = 0, 0
            frozen_frames, frames = 0, 0
            for r in sel:
                start = r["byte_offset"] // 4
                a = data[start:start + r["n_frames"] * 51].reshape(r["n_frames"], 17, 3)
                sc = a[..., 2]
                miss = sc == 0                                   # (T, 17)
                per_joint_missing += miss.sum(0); n_jf += miss.size
                any_miss = miss.any(1)
                if any_miss.any():
                    k = miss[any_miss].sum(1); co_counts.extend(k.tolist())
                    all17 += int((k == 17).sum() * 17); n_missing_jf += int(miss.sum())
                for j in range(17):
                    miss_runs.extend(runs(miss[:, j]))
                whole_missing_joints += int(miss.all(0).sum()); joint_segments += 17
                fz = np.concatenate([[False], (a[1:] == a[:-1]).all(axis=(1, 2))])
                frozen_frames += int(fz.sum()); frames += len(fz); frozen_runs.extend(runs(fz))
                arm_conf.append(sc[:, ARM_JOINTS].ravel())
            arm = np.concatenate(arm_conf) if arm_conf else np.array([])
            mr = np.array(miss_runs) if miss_runs else np.array([np.nan]); fr = np.array(frozen_runs) if frozen_runs else np.array([np.nan])
            out.append({"split": split, "class": "SMM" if cls else "non-SMM", "segments": len(sel),
                        "missing_jointframe_fraction": round(per_joint_missing.sum() / n_jf, 4),
                        "per_joint_missing_min": round(per_joint_missing.min() / (n_jf / 17), 4),
                        "per_joint_missing_max": round(per_joint_missing.max() / (n_jf / 17), 4),
                        "per_joint_missing_argmax": int(per_joint_missing.argmax()),
                        "comissing_mean_joints_given_any": round(float(np.mean(co_counts)), 2) if co_counts else "",
                        "share_missing_jf_in_all17_frames": round(all17 / n_missing_jf, 4) if n_missing_jf else "",
                        "missing_run_mean": round(float(np.nanmean(mr)), 1), "missing_run_median": round(float(np.nanmedian(mr)), 1),
                        "missing_run_p90": round(float(np.nanpercentile(mr, 90)), 1), "n_missing_runs": len(miss_runs),
                        "whole_segment_missing_joint_fraction": round(whole_missing_joints / joint_segments, 4),
                        "frozen_frame_fraction": round(frozen_frames / frames, 4),
                        "frozen_run_mean": round(float(np.nanmean(fr)), 1), "frozen_run_median": round(float(np.nanmedian(fr)), 1),
                        "frozen_run_p90": round(float(np.nanpercentile(fr, 90)), 1), "n_frozen_runs": len(frozen_runs),
                        "arm_conf_below_0.3": round(float((arm < 0.3).mean()), 4) if arm.size else "",
                        "arm_conf_zero": round(float((arm == 0).mean()), 4) if arm.size else "",
                        "arm_conf_median_nonzero": round(float(np.median(arm[arm > 0])), 3) if (arm > 0).any() else "",
                        "arm_conf_hist_0.0_0.3_0.5_0.7_1.0": "/".join(f"{x:.3f}" for x in np.histogram(arm, bins=[0, 0.3, 0.5, 0.7, 1.0001])[0] / max(1, arm.size))})
            print(out[-1], flush=True)
    path = "results/v7/analysis/natural_failure_stats.csv"; guard_output(path)
    with open(path, "w", newline="") as f:
        f.write(f"# post hoc: natural tracker-failure statistics on whole length-protocol segments; operators: missing runs mean {RUN_LEN:.0f} frames per joint independent, frozen runs mean {FROZEN_RUN_LEN:.0f} frames, arm confidence x0.3 floor 0.05\n")
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("wrote", path)

    L = ["% generated by experiments/natural_failure_stats.py (post hoc) from the SAFE-SMM cache; do not edit",
         r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r"Split & Class & Missing jf & Co-missing & All-17 share & Run mean / p90 & Whole-seg. joints & Frozen fr. & Frozen run mean / p90 & Arm conf $<$0.3 \\", r"\midrule"]
    for r in out:
        L.append(f"{r['split']} & {r['class']} & {100 * r['missing_jointframe_fraction']:.1f}\\% & {r['comissing_mean_joints_given_any']} & "
                 f"{100 * float(r['share_missing_jf_in_all17_frames']):.0f}\\% & {r['missing_run_mean']} / {r['missing_run_p90']} & "
                 f"{100 * r['whole_segment_missing_joint_fraction']:.2f}\\% & {100 * r['frozen_frame_fraction']:.1f}\\% & "
                 f"{r['frozen_run_mean']} / {r['frozen_run_p90']} & {100 * float(r['arm_conf_below_0.3']):.1f}\\% \\\\")
    L += [r"\midrule", f"planted operators & & chain, mean run {RUN_LEN:.0f} & independent joints & n/a & {RUN_LEN:.0f} (mean) & whole-joint operator & chain & {FROZEN_RUN_LEN:.0f} (mean) & $\\times$0.3, floor 0.05 \\\\",
          r"\bottomrule", r"\end{tabular}"]
    tp = "paper/tables/natural_failure_stats.tex"; guard_output(tp); open(tp, "w").write("\n".join(L) + "\n"); print("wrote", tp)


if __name__ == "__main__":
    main()
