"""R22. Natural rate of the planted confidence pattern.

On the host segments (QC-passing, at least 200 frames; development = optimization training plus
validation, 12,572 rows, as the planted worlds are built; test = 1,966 rows), by class, over whole
segments and the six arm joints (COCO-17 shoulders, elbows, wrists):
  share of arm joint-frames with 0 < confidence < 0.3 and nonzero coordinates (x or y nonzero), the
  form the operator plants; its mean run length (consecutive frames of one joint);
  share with confidence 0 and zero coordinates (a missing joint);
  the remaining combinations, for completeness.
The same on all length-controlled segments (QC-failing included), the population of the article's
natural-failure paragraph, as a cross-check against results/operators/analysis/
natural_failure_stats.csv (arm confidence below 0.3 and equal to 0).
Operator counts: arm joint-frames that the planted operator turns from (zero coordinates, confidence 0)
into (zero coordinates, confidence 0.05), as a share of all arm joint-frames, for the training worlds
S and I of assignment 0 (development rows, whole segments) and for test policies S and I (three masks;
from test_planted_severities.csv).
Verification of the regenerated development severities (assignment 0) against
results/confonly_assignments/analysis/assignment_severities.csv (class means and SDs of eps, cue-only AUROCs).
Writes results/reanalyses/r22_*.csv and r22_summary.json.
    python3 experiments/reanalyses/r22_natural_rate.py
"""
import csv

import numpy as np

import common as C
from safe_motion import planted as P
from safe_motion import remedies as RM
from safe_motion.data import asdpose as D

ARM = P.ARM_JOINTS


def run_lengths(m):
    """Run lengths of True along axis 0 of a (T, J) boolean array, all joints."""
    out = []
    for j in range(m.shape[1]):
        d = np.diff(np.concatenate([[0], m[:, j].astype(np.int8), [0]]))
        out.extend((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist())
    return out


def natural(rows, data):
    acc = {}
    for c in (0, 1):
        acc[c] = {"segments": 0, "arm_jf": 0, "low_nonzero_xy": 0, "zero_conf_zero_xy": 0, "zero_conf_nonzero_xy": 0,
                  "low_zero_xy": 0, "high_zero_xy": 0, "conf_below_0.3": 0, "conf_zero": 0, "runs": [],
                  "segments_with_pattern": 0}
    for r in rows:
        a = C.segment(data, r)[:, ARM, :]
        x, y, s = a[..., 0], a[..., 1], a[..., 2]
        zxy = (x == 0) & (y == 0)
        p1 = (s > 0) & (s < 0.3) & ~zxy
        A = acc[r["binary_label"]]
        A["segments"] += 1
        A["arm_jf"] += s.size
        A["low_nonzero_xy"] += int(p1.sum())
        A["zero_conf_zero_xy"] += int(((s == 0) & zxy).sum())
        A["zero_conf_nonzero_xy"] += int(((s == 0) & ~zxy).sum())
        A["low_zero_xy"] += int(((s > 0) & (s < 0.3) & zxy).sum())
        A["high_zero_xy"] += int(((s >= 0.3) & zxy).sum())
        A["conf_below_0.3"] += int((s < 0.3).sum())
        A["conf_zero"] += int((s == 0).sum())
        A["segments_with_pattern"] += int(p1.any())
        if p1.any():
            A["runs"].extend(run_lengths(p1))
    out = []
    for c in (0, 1):
        A = acc[c]
        n = A["arm_jf"]
        runs = np.array(A["runs"]) if A["runs"] else np.array([np.nan])
        out.append({"class": "SMM" if c else "non-SMM", "segments": A["segments"], "arm_joint_frames": n,
                    "share_conf_between_0_and_0.3_nonzero_xy": A["low_nonzero_xy"] / n,
                    "share_conf_0_zero_xy": A["zero_conf_zero_xy"] / n,
                    "share_conf_0_nonzero_xy": A["zero_conf_nonzero_xy"] / n,
                    "share_conf_between_0_and_0.3_zero_xy": A["low_zero_xy"] / n,
                    "share_conf_at_least_0.3_zero_xy": A["high_zero_xy"] / n,
                    "share_conf_below_0.3": A["conf_below_0.3"] / n, "share_conf_0": A["conf_zero"] / n,
                    "pattern_runs": int(np.isfinite(runs).sum()), "pattern_run_mean": float(np.nanmean(runs)),
                    "pattern_run_median": float(np.nanmedian(runs)), "pattern_run_p90": float(np.nanpercentile(runs, 90)),
                    "share_segments_with_pattern": A["segments_with_pattern"] / A["segments"],
                    "pattern_share_of_conf_below_0.3": A["low_nonzero_xy"] / max(1, A["conf_below_0.3"])})
    return out


def main():
    data = C.keypoints()
    tr, va = C.dev_host_rows()
    dev = tr + va
    test = C.test_host_rows()
    import yaml
    v1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
    lc = [r for r in D.load_index() if r["n_frames"] >= 200]
    summary = {"n_dev_host": len(dev), "n_train_host": len(tr), "n_val_host": len(va), "n_test_host": len(test)}
    nat = []
    for label, rows in (("development host (QC-passing, >=200 frames)", dev), ("test host (QC-passing, >=200 frames)", test),
                        ("development, all length-controlled (QC-failing included)", [r for r in lc if r["split"] == "train"]),
                        ("test, all length-controlled (QC-failing included)", [r for r in lc if r["split"] == "test"])):
        for rec in natural(rows, data):
            nat.append({"population": label, **rec})
            print(label, rec["class"], f"low&nonzero {rec['share_conf_between_0_and_0.3_nonzero_xy']:.4f} "
                  f"zero&zero {rec['share_conf_0_zero_xy']:.4f} run mean {rec['pattern_run_mean']:.2f} "
                  f"below0.3 {rec['share_conf_below_0.3']:.4f} zero {rec['share_conf_0']:.4f}", flush=True)
    # cross-check against the stored natural-failure statistics (all length-controlled segments)
    stored = {(r["split"], r["class"]): r for r in csv.DictReader(l for l in open("results/operators/analysis/natural_failure_stats.csv") if not l.startswith("#"))}
    xc = []
    for rec in nat:
        if "all length-controlled" not in rec["population"]:
            continue
        sp = "train" if rec["population"].startswith("development") else "test"
        s = stored[(sp, rec["class"])]
        xc.append({"split": sp, "class": rec["class"], "below_0.3_stored": float(s["arm_conf_below_0.3"]),
                   "below_0.3_recomputed": round(rec["share_conf_below_0.3"], 4),
                   "zero_stored": float(s["arm_conf_zero"]), "zero_recomputed": round(rec["share_conf_0"], 4),
                   "segments_stored": int(s["segments"]), "segments_recomputed": rec["segments"]})
    summary["crosscheck_natural_failure_stats"] = xc
    summary["crosscheck_equal"] = all(r["below_0.3_stored"] == r["below_0.3_recomputed"] and r["zero_stored"] == r["zero_recomputed"]
                                      and r["segments_stored"] == r["segments_recomputed"] for r in xc)
    print("cross-check with natural_failure_stats.csv:", summary["crosscheck_equal"], xc, flush=True)
    C.write_csv("r22_natural_rate.csv", nat,
                "R22: natural arm joint-frame patterns (six COCO-17 arm joints, whole segments) by class. Pattern planted by the operator:\n"
                "0 < confidence < 0.3 with nonzero coordinates (x or y nonzero). Missing joint: confidence 0, x = y = 0.\n"
                "pattern_run_*: lengths of runs of consecutive frames of one joint in that pattern. Script: r22_natural_rate.py")

    # operator counts, training worlds of assignment 0 (development host rows, whole segments)
    y = np.array([r["binary_label"] for r in dev])
    ops, ver = [], []
    ref = {(r["world_seed"], r["world"]): r for r in csv.DictReader(open("results/confonly_assignments/analysis/assignment_severities.csv"))}
    for w in ("S", "I"):
        masks, eps = P.build_masks(dev, w, "mid", 0, mechanism=C.MECH)
        n_floor = np.zeros(2); n_jf = np.zeros(2); n_masked = np.zeros(2); stat = np.empty(len(dev))
        for i, r in enumerate(dev):
            raw = C.segment(data, r)
            m = masks[r["identifier"]]
            zero = (raw[..., 0] == 0) & (raw[..., 1] == 0) & (raw[..., 2] == 0)
            c = r["binary_label"]
            n_floor[c] += (m & zero)[:, ARM].sum()
            n_masked[c] += m[:, ARM].sum()
            n_jf[c] += 6 * r["n_frames"]
            stat[i] = RM.quality_stats(D.apply_planted(raw, m, "confscale"))[3]
        for c, cname in ((0, "non-SMM"), (1, "SMM"), (None, "both")):
            nf = n_floor.sum() if c is None else n_floor[c]
            nj = n_jf.sum() if c is None else n_jf[c]
            nm = n_masked.sum() if c is None else n_masked[c]
            ops.append({"data": "training world, assignment 0 (development host)", "world_or_policy": w, "mask_seed": "",
                        "class": cname, "arm_joint_frames": int(nj), "masked_arm_joint_frames": int(nm),
                        "zero_to_floor_joint_frames": int(nf), "zero_to_floor_share_of_all_arm_jf": nf / nj,
                        "zero_to_floor_share_of_masked_arm_jf": nf / nm})
        rr = ref[("0", w)]
        got = {"eps_mean_neg": eps[y == 0].mean(), "eps_mean_pos": eps[y == 1].mean(), "eps_sd_neg": eps[y == 0].std(ddof=1),
               "eps_sd_pos": eps[y == 1].std(ddof=1), "auc_eps": C.M.auc(-eps, y), "auc_stat": C.M.auc(-stat, y),
               "n_neg": int((y == 0).sum()), "n_pos": int((y == 1).sum())}
        ver.append({"world": w, **{f"{k}_stored": rr[k] for k in got}, **{f"{k}_regenerated": (f"{v:.5f}" if isinstance(v, float) else v) for k, v in got.items()}})
        ver[-1]["all_equal_5dp"] = all(str(rr[k]) == (f"{v:.5f}" if isinstance(v, float) else str(v)) for k, v in got.items())
        print("verification", ver[-1], flush=True)
    summary["dev_severity_verification"] = ver
    # test policies, from the regenerated test planting
    tp = list(csv.DictReader(l for l in open(C.OUT / "test_planted_severities.csv") if not l.startswith("#")))
    for pol in ("S", "I"):
        for ms in C.MASK_SEEDS + ("mean",):
            for cname, lab in (("non-SMM", "0"), ("SMM", "1"), ("both", None)):
                sel = [r for r in tp if r["policy"] == pol and (ms == "mean" or int(r["mask_seed"]) == ms) and (lab is None or r["label"] == lab)]
                nf = sum(int(r["arm_zero_to_floor"]) for r in sel)
                nj = sum(int(r["n_arm_jf"]) for r in sel)
                nm = sum(float(r["masked_frac_full"]) * int(r["n_arm_jf"]) for r in sel)
                ops.append({"data": "test policy (test host)", "world_or_policy": pol, "mask_seed": ms, "class": cname,
                            "arm_joint_frames": nj, "masked_arm_joint_frames": int(round(nm)), "zero_to_floor_joint_frames": nf,
                            "zero_to_floor_share_of_all_arm_jf": nf / nj, "zero_to_floor_share_of_masked_arm_jf": nf / nm})
    C.write_csv("r22_operator_floor_counts.csv", ops,
                "R22: arm joint-frames that the confidence-only operator turns from (x = y = 0, confidence 0) into (x = y = 0,\n"
                "confidence 0.05) (masked and naturally missing), whole segments; training worlds of assignment 0 on the 12,572\n"
                "development host rows; test policies S and I for masks 2000/3000/4000 on the 1,966 test host rows ('mean' pools the masks)")
    for r in ops:
        if r["class"] == "both" or r["mask_seed"] in ("", "mean"):
            print(r, flush=True)
    C.write_json("r22_summary.json", summary)


if __name__ == "__main__":
    main()
