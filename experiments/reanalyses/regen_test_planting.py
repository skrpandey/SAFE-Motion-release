"""Regenerate the per-segment planted severities and planted statistics of the confidence-only test
policies (arm-confidence operator "armruns", mode confscale, mid strength) for the three test masks,
exactly as the scorers build them (experiments/score_confonly_historical.py, score_confonly_corrected.py: P.build_masks(te_host,
policy, "mid", mask_seed, mechanism="armruns") on the host test rows, then
safe_motion.data.asdpose.apply_planted, centre 200-frame window at test).

Per segment, policy (S, I) and test mask (2000, 3000, 4000):
  eps                  planted severity (the Beta draw of safe_motion/planted.py severities())
  masked_frac_full     share of the six arm joints x frames selected by the mask, whole segment
  masked_frac_window   the same in the centre 200-frame window that the models score
  arm_lowconf_full     planted arm low-confidence fraction (remedies.quality_stats()[3]), whole segment
  arm_lowconf_window   the same in the centre window
  arm_zero_to_floor    masked arm joint-frames with zero coordinates and confidence 0 (set to 0.05)
  n_arm_jf             arm joint-frames of the segment (6 x n_frames)
Verification (written to regen_test_planting_checks.json):
  (1) class means of the four quality statistics of the planted inputs at mask 2000 against the
      stored values in results/operators/oracles.json (score_operators.py tracking_stats, same rows and masks);
  (2) a sample of 25 segments per policy: planted array of this script equals the array that
      SegmentDataset.load_segment (the scorer's loader) returns, and the centre crop equals _crop;
  (3) whether severities are redrawn across test masks (correlation of eps between mask seeds).
Writes results/reanalyses/test_planted_severities.csv and regen_test_planting_checks.json.
    python3 experiments/reanalyses/regen_test_planting.py
"""
import json

import numpy as np

import common as C
from safe_motion import planted as P
from safe_motion import remedies as RM
from safe_motion.data import asdpose as D

MODE = "confscale"
WINDOW = 200


def centre(arr, w=WINDOW):
    t = arr.shape[0]
    if t < w:
        return np.tile(arr, (int(np.ceil(w / t)), 1, 1))[:w]
    s = (t - w) // 2
    return arr[s:s + w]


def main():
    rows = C.test_host_rows()
    data = C.keypoints()
    arm = P.ARM_JOINTS
    # the score files cover exactly these rows
    meta, _ = C.load_scores("results/confonly_primary/scores/v15conf_S_w0_armruns_mid_A_none_s0.csv")
    assert set(meta["identifier"]) == {r["identifier"] for r in rows} and len(rows) == len(meta["identifier"])
    y = np.array([r["binary_label"] for r in rows])
    print(f"host test rows {len(rows)} (SMM {int(y.sum())})", flush=True)
    out, checks = [], {"n_rows": len(rows), "n_smm": int(y.sum())}
    stats2000, eps_all = {}, {}
    for pol in ("S", "I"):
        for ms in C.MASK_SEEDS:
            masks, eps = P.build_masks(rows, pol, "mid", ms, mechanism=C.MECH)
            eps_all[(pol, ms)] = eps
            q = np.empty((len(rows), 4))
            for i, r in enumerate(rows):
                raw = C.segment(data, r)
                m = masks[r["identifier"]]
                assert m.shape == (r["n_frames"], 17) and not m[:, [j for j in range(17) if j not in arm]].any()
                zero = (raw[..., 0] == 0) & (raw[..., 1] == 0) & (raw[..., 2] == 0)
                to_floor = int((m & zero)[:, arm].sum())
                pl = D.apply_planted(raw.copy(), m, MODE)
                q[i] = RM.quality_stats(pl)
                win = centre(pl)
                mw = centre(m[..., None].astype(np.float32))[..., 0] > 0
                out.append({"identifier": r["identifier"], "child_id": r["child_id"], "label": r["binary_label"],
                            "n_frames": r["n_frames"], "policy": pol, "mask_seed": ms, "eps": float(eps[i]),
                            "masked_frac_full": float(m[:, arm].mean()), "masked_frac_window": float(mw[:, arm].mean()),
                            "arm_lowconf_full": float(q[i, 3]), "arm_lowconf_window": float(RM.quality_stats(win)[3]),
                            "arm_zero_to_floor": to_floor, "n_arm_jf": 6 * r["n_frames"]})
            if ms == 2000:
                stats2000[pol] = {"smm": q[y == 1].mean(0).round(4).tolist(), "non_smm": q[y == 0].mean(0).round(4).tolist()}
            # (2) the scorer's loader returns the same planted array and centre window for a sample
            ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=MODE, valid_rule="coords")
            rng = np.random.RandomState(ms + (0 if pol == "S" else 1))
            samp = rng.choice(len(rows), 25, replace=False)
            same = all(np.array_equal(ds.load_segment(rows[k]), D.apply_planted(C.segment(data, rows[k]), masks[rows[k]["identifier"]], MODE))
                       and np.array_equal(ds._crop(ds.load_segment(rows[k])), centre(ds.load_segment(rows[k]))) for k in samp)
            checks[f"loader_equal_{pol}_m{ms}"] = bool(same)
            print(f"policy {pol} mask {ms}: eps non-SMM {eps[y == 0].mean():.4f} SMM {eps[y == 1].mean():.4f}; "
                  f"loader sample equal {same}", flush=True)
    # (1) stored planted statistics of score_operators.py (results/operators/oracles.json)
    oracle = json.load(open("results/operators/oracles.json"))
    for pol in ("S", "I"):
        stored = oracle[f"armruns_{pol}"]["mean_stats_by_class"]
        checks[f"oracle_stats_{pol}"] = {"stored": stored, "regenerated": stats2000[pol],
                                         "equal_4dp": stored == stats2000[pol]}
        print(f"oracle stats {pol}: stored {stored} regenerated {stats2000[pol]}", flush=True)
    # (3) severities redrawn across test masks?
    red = {}
    for pol in ("S", "I"):
        e = [eps_all[(pol, ms)] for ms in C.MASK_SEEDS]
        red[pol] = {"corr_2000_3000": float(np.corrcoef(e[0], e[1])[0, 1]), "corr_2000_4000": float(np.corrcoef(e[0], e[2])[0, 1]),
                    "share_identical_2000_3000": float(np.mean(e[0] == e[1])),
                    "corr_within_class_2000_3000": {c: float(np.corrcoef(e[0][y == c], e[1][y == c])[0, 1]) for c in (0, 1)}}
    red["S_vs_I_same_mask_2000_corr"] = float(np.corrcoef(eps_all[("S", 2000)], eps_all[("I", 2000)])[0, 1])
    checks["severity_redraw"] = red
    C.write_csv("test_planted_severities.csv", out,
                "Per-segment planted severities and statistics of the confidence-only test policies, regenerated with\n"
                "safe_motion/planted.py build_masks(host test rows, policy, 'mid', mask seed, 'armruns') and apply_planted(confscale),\n"
                "as experiments/score_confonly_historical.py and score_confonly_corrected.py build them; window = centre 200 frames (test-time crop).\n"
                "Script: experiments/reanalyses/regen_test_planting.py")
    C.write_json("regen_test_planting_checks.json", checks)
    ok = all(checks[f"oracle_stats_{p}"]["equal_4dp"] for p in ("S", "I")) and all(v for k, v in checks.items() if k.startswith("loader_equal"))
    print("VERIFIED" if ok else "VERIFICATION FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
