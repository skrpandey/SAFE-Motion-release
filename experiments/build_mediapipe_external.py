"""Protocol v11 T4: the external set re-extracted with MediaPipe Pose for both classes.

Negatives are re-extracted on CPU from the source videos over the native frame spans of the
RTMPose negatives; positives are the existing MediaPipe clips sliced to the annotated span
(end exclusive). Both are mapped to COCO-17 with visibility as confidence, then resampled to
30 fps and windowed to 200 frames by the rule that reproduces the RTMPose cache (max abs
difference 6e-5 on five windows). Writes results/v11/mediapipe_cache/{index.csv, keypoints.f32}
in the same order as the RTMPose index, results/v11/mediapipe_gate.json (layout and parity
checks, failures listed, nothing dropped) and raw negatives under $SM_STIMMING_ROOT/mediapipe-negatives/.

    conda run -n torch python experiments/build_mediapipe_external.py --workers 6
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion.data import asdpose as D

EXTERNAL_CACHE = os.environ.get("SM_EXTERNAL_CACHE", f"{D.SMM_ROOT}/cache/external")
STIMMING = os.environ.get("SM_STIMMING_ROOT", "data/stimming")   # source videos and pose files
UNIFIED = f"{STIMMING}/unified"
RTM_POS = f"{STIMMING}/ntu-format-rtm"
RTM_NEG = f"{STIMMING}/external-negatives"
MP_POS = f"{STIMMING}/mediapipe/pose-data"
MP_NEG_OUT = f"{STIMMING}/mediapipe-negatives"
OUT = "results/v11/mediapipe_cache"
GATE = "results/v11/mediapipe_gate.json"
COCO17_FROM_MP33 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LOW_CONF, ARM_JOINTS = 0.3, [5, 6, 7, 8, 9, 10]
NEG_RE = re.compile(r"^(v_[A-Za-z]+_\d+(?:_\d+)?)_neg_(\d+)_(\d+)$")
WINDOW = 200


def resample_30(arr, fps):
    """(T, J, C) at fps -> (T', J, C) at 30 fps, linear in time; the base project's rule."""
    T = arr.shape[0]
    Tp = int(round(T * 30.0 / fps))
    if Tp < 2 or T < 2:
        return np.repeat(arr[:1], max(Tp, 1), axis=0)
    p = np.linspace(0, T - 1, Tp)
    x = np.arange(T)
    return np.stack([np.stack([np.interp(p, x, arr[:, j, k]) for k in range(arr.shape[2])], 1)
                     for j in range(arr.shape[1])], 1)


def window(arr, n=WINDOW):
    t = arr.shape[0]
    if t >= n:
        s = (t - n) // 2
        return arr[s:s + n]
    return np.tile(arr, (int(np.ceil(n / t)), 1, 1))[:n]


def quality_features(kp, sc):
    """SAFE-SMM stage1/build_cache.py definitions, cutoff 0.3."""
    lowconf = float((sc < LOW_CONF).mean())
    missing = float((sc == 0).mean())
    frozen = float((kp[1:] == kp[:-1]).all(axis=(1, 2)).mean()) if kp.shape[0] > 1 else 0.0
    energy = float(np.linalg.norm(np.diff(kp, axis=0), axis=2).mean()) if kp.shape[0] > 1 else 0.0
    arm_lowconf = float((sc[:, ARM_JOINTS] < LOW_CONF).mean())
    return lowconf, missing, frozen, energy, arm_lowconf


def mp33_to_coco17(lm, img_w, img_h):
    """(T, 33, 4) normalised x, y, z, visibility -> (T, 17, 3) pixel x, y, visibility."""
    sel = lm[:, COCO17_FROM_MP33, :]
    out = np.zeros((lm.shape[0], 17, 3), np.float32)
    out[..., 0] = sel[..., 0] * img_w
    out[..., 1] = sel[..., 1] * img_h
    out[..., 2] = sel[..., 3]
    detected = (lm[..., 3] > 0).any(axis=1)          # frames with no detection stay zero
    out[~detected] = 0.0
    return out


def extract_negative(job):
    """Run MediaPipe Pose on one negative window's native frames. Returns (identifier, status)."""
    ident, video, start, end, img_w, img_h, fps, out_path = job
    if os.path.exists(out_path):
        return ident, "cached"
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return ident, "video_unreadable"
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    lm = np.zeros((end - start, 33, 4), np.float32)
    n_read = n_det = 0
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for t in range(end - start):
            ok, frame = cap.read()
            if not ok:
                break
            n_read += 1
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks is not None:
                n_det += 1
                for j, p in enumerate(res.pose_landmarks.landmark):
                    lm[t, j] = (p.x, p.y, p.z, p.visibility)
    cap.release()
    if n_read == 0:
        return ident, "no_frames"
    np.savez_compressed(out_path, landmarks=lm[:n_read], fps=fps, img_w=img_w, img_h=img_h,
                        video_id=os.path.basename(video)[:-4], seg_start_frame=start,
                        seg_end_frame=end, n_read=n_read, n_detected=n_det,
                        mediapipe="solutions.pose model_complexity=1")
    return ident, f"ok read {n_read} of {end - start} detected {n_det}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="first N rows (smoke)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--gate", default=GATE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.gate), exist_ok=True)
    os.makedirs(MP_NEG_OUT, exist_ok=True)
    rows = list(csv.DictReader(open(f"{EXTERNAL_CACHE}/index.csv")))
    if args.limit:
        rows = rows[:args.limit]
    import mediapipe
    gate = {"mediapipe_version": mediapipe.__version__, "n_rows": len(rows), "failures": [],
            "layout": {}, "parity": {}, "statuses": {}}

    # negatives: extraction jobs
    jobs, neg_meta = [], {}
    for r in rows:
        if int(r["binary_label"]) == 1:
            continue
        m = NEG_RE.match(r["identifier"])
        if not m:
            gate["failures"].append({"identifier": r["identifier"], "reason": "unparsed identifier"})
            continue
        vid, start, end = m.group(1), int(m.group(2)), int(m.group(3))
        cat = r["action_name"]
        raw = np.load(f"{RTM_NEG}/{r['identifier']}.npz")
        cat = str(raw["category"]) if cat in ("none", "") else cat
        video = f"{UNIFIED}/{cat}/{vid}.mp4"
        if not os.path.exists(video):
            hits = glob.glob(f"{UNIFIED}/*/{vid}.mp4")
            video = hits[0] if hits else video
        neg_meta[r["identifier"]] = (vid, start, end, float(raw["fps"]), int(raw["img_w"]),
                                     int(raw["img_h"]), int(raw["n_frames"]))
        jobs.append((r["identifier"], video, start, end, int(raw["img_w"]), int(raw["img_h"]),
                     float(raw["fps"]), f"{MP_NEG_OUT}/{r['identifier']}.npz"))
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, (ident, status) in enumerate(pool.imap_unordered(extract_negative, jobs)):
            gate["statuses"][ident] = status
            if k % 50 == 0:
                print(f"negatives {k}/{len(jobs)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"negatives extracted in {(time.time()-t0)/60:.1f} min", flush=True)

    # build the cache in index order
    kp_all, index_rows = [], []
    offset = 0
    ranges = {"x_min": [], "x_max": [], "y_min": [], "y_max": []}
    for r in rows:
        ident, img_w, img_h = r["identifier"], int(r["img_w"]), int(r["img_h"])
        status = "ok"
        arr17 = None
        try:
            if int(r["binary_label"]) == 1:
                rtm = np.load(f"{RTM_POS}/{r['action_name']}/{ident}.npz")
                s, e, fps = int(rtm["seg_start_frame"]), int(rtm["seg_end_frame"]), float(rtm["fps"])
                mpz = np.load(f"{MP_POS}/{r['action_name']}/{ident}.npz")
                fi = mpz["frame_indices"]
                keep = (fi >= s) & (fi < e)
                if keep.sum() < 2:
                    raise ValueError(f"positive span {s}:{e} has {int(keep.sum())} MediaPipe frames")
                lm = mpz["poses"][keep].astype(np.float32)             # (T, 33, 4)
                expected = e - s
                got = int(keep.sum())
                if abs(got - expected) > max(2, 0.02 * expected):
                    status = f"frame_count_mismatch rtm {expected} mp {got}"
                arr17 = mp33_to_coco17(lm, img_w, img_h)
            else:
                vid, s, e, fps, _, _, n_rtm = neg_meta[ident]
                z = np.load(f"{MP_NEG_OUT}/{ident}.npz")
                lm = z["landmarks"].astype(np.float32)
                if lm.shape[0] != n_rtm:
                    status = f"frame_count_mismatch rtm {n_rtm} mp {lm.shape[0]}"
                arr17 = mp33_to_coco17(lm, img_w, img_h)
            seq = window(resample_30(arr17, fps)).astype(np.float32)
        except Exception as ex:                                        # noqa: BLE001
            status = f"failed: {ex}"
            seq = np.zeros((WINDOW, 17, 3), np.float32)
        if status != "ok":
            gate["failures"].append({"identifier": ident, "reason": status})
        kp, sc = seq[..., :2], seq[..., 2]
        q = quality_features(kp, sc)
        valid = sc > 0
        if valid.any():
            ranges["x_min"].append(float(kp[valid][:, 0].min() / img_w))
            ranges["x_max"].append(float(kp[valid][:, 0].max() / img_w))
            ranges["y_min"].append(float(kp[valid][:, 1].min() / img_h))
            ranges["y_max"].append(float(kp[valid][:, 1].max() / img_h))
        index_rows.append({**r, "n_frames": WINDOW, "byte_offset": offset,
                           "q_lowconf": f"{q[0]:.6f}", "q_missing": f"{q[1]:.6f}", "q_frozen": f"{q[2]:.6f}",
                           "q_energy": f"{q[3]:.6f}", "q_arm_lowconf": f"{q[4]:.6f}"})
        kp_all.append(seq)
        offset += seq.size * 4
    with open(f"{args.out}/keypoints.f32", "wb") as f:
        for seq in kp_all:
            f.write(np.ascontiguousarray(seq, dtype=np.float32).tobytes())
    with open(f"{args.out}/index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(index_rows)

    # gate: layout and parity
    allkp = np.stack(kp_all)                                           # (N, 200, 17, 3)
    conf = allkp[..., 2]
    det = conf > 0
    nose_y, sh_y = allkp[..., 0, 1], allkp[..., 5:7, 1].mean(-1)
    both = det[..., 0] & det[..., 5] & det[..., 6]
    lsh, rsh = allkp[..., 5, 0], allkp[..., 6, 0]
    gate["layout"] = {
        "joint_order": "COCO-17 from MediaPipe 33 via " + str(COCO17_FROM_MP33),
        "frames_with_detection": float(det.any(-1).mean()),
        "nose_above_shoulders_when_detected": float((nose_y[both] < sh_y[both]).mean()) if both.any() else None,
        "left_right_shoulders_distinct": float((np.abs(lsh - rsh)[det[..., 5] & det[..., 6]] > 1e-3).mean()),
        "normalised_range_x": [float(np.percentile(ranges["x_min"], 1)), float(np.percentile(ranges["x_max"], 99))],
        "normalised_range_y": [float(np.percentile(ranges["y_min"], 1)), float(np.percentile(ranges["y_max"], 99))],
        "note": "left and right follow the subject's side as MediaPipe defines them, the same convention as COCO",
    }
    rtm_ids = [r["identifier"] for r in rows]
    gate["parity"] = {"same_identifiers_and_order": rtm_ids == [r["identifier"] for r in index_rows],
                      "all_windows_200_frames": all(r["n_frames"] == WINDOW for r in index_rows),
                      "n_failures": len(gate["failures"]),
                      "failure_fraction_negatives": sum(1 for f in gate["failures"]
                                                        if "_neg_" in f["identifier"]) / max(1, len(jobs)),
                      "windows_with_no_detection": int((~det.any(-1).any(-1)).sum())}
    gate["fallback_applies"] = gate["parity"]["failure_fraction_negatives"] > 0.05
    json.dump(gate, open(args.gate, "w"), indent=1)
    print(f"cache {args.out}: {len(index_rows)} windows; failures {len(gate['failures'])}; "
          f"fallback {gate['fallback_applies']}; no-detection windows "
          f"{gate['parity']['windows_with_no_detection']}", flush=True)


if __name__ == "__main__":
    main()
