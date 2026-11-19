"""MMASD 2D skeletons (OpenPose BODY_25) as per-clip (T, J, 3) arrays: x, y, confidence.

Source: $SM_MMASD_ROOT/2D_skeleton/2D_openpose.zip unzipped to 2D_skeleton/output/; clinical
table $SM_MMASD_ROOT/ADOS_rating.xlsx (read with the standard library).
Clip name <abbr>_<child>_D<day>_<index>_<tag>, e.g. as_20583_D16_000_i: activity abbreviation
(ACTIVITIES), 5-digit child id (matches "ID#" in the ADOS table), recording day (D1, D8, D16),
clip index, and a y / n / i tag whose meaning is undocumented (kept as metadata, never a label).
On disk 1,693 clip folders, 319 empty (all frog_pose, most drumming), leaving 1,374 clips.

OpenPose has no identities and frames usually hold 3 to 4 people. Rule "track" (default) links
detections into tracks by Hungarian matching on centroid distance / skeleton height and keeps the
track with the largest summed mean confidence; frames where it is missing are all zero. Rule
"maxconf" takes the most confident detection per frame. Video is not released, so neither rule can
be checked directly; per-clip selection diagnostics are stored. est_w / est_h are the largest x / y
seen in the clip, a scale proxy only (image size is not stored). joints="coco17" maps BODY_25 to
the 17 COCO joints so ASDPose code and safe_motion.planted apply unchanged.

Caches under $SM_MMASD_ROOT/cache/: mmasd_raw.npz (all detections) and mmasd_<rule>.npz
(selected person, flat keypoints with offsets).
"""

import csv
import glob
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = os.environ.get("SM_MMASD_ROOT", "data/mmasd")
JSON_ROOT = f"{ROOT}/2D_skeleton/output"
CACHE = f"{ROOT}/cache"
ADOS_XLSX = f"{ROOT}/ADOS_rating.xlsx"
LOW_CONF = 0.3

ACTIVITIES = {"as": "arm_swing", "bs": "body_swing", "ce": "chest_expansion", "sq": "squat",
              "dr": "drumming", "mfs": "maracas_forward_shaking", "ms": "maracas_shaking",
              "sac": "sing_and_clap", "fg": "frog_pose", "tr": "tree_pose", "tw": "twist_pose"}
THEME = {"as": "robot", "bs": "robot", "ce": "robot", "sq": "robot", "dr": "rhythm",
         "mfs": "rhythm", "ms": "rhythm", "sac": "rhythm", "fg": "yoga", "tr": "yoga",
         "tw": "yoga"}

BODY25 = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow",
          "l_wrist", "mid_hip", "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
          "r_eye", "l_eye", "r_ear", "l_ear", "l_big_toe", "l_small_toe", "l_heel",
          "r_big_toe", "r_small_toe", "r_heel"]
ARM_JOINTS_BODY25 = [2, 3, 4, 5, 6, 7]          # shoulders, elbows, wrists
# COCO-17 order: nose, l_eye, r_eye, l_ear, r_ear, l_sho, r_sho, l_elb, r_elb, l_wri, r_wri,
# l_hip, r_hip, l_knee, r_knee, l_ank, r_ank
COCO17_FROM_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
ARM_JOINTS_COCO17 = [5, 6, 7, 8, 9, 10]

LINK_MAX = 0.5          # centroid distance / skeleton height
MAX_GAP = 15            # frames a track may be unseen and still be extended
NAME_RE = re.compile(r"^([a-z]+)_(\d{5})_D(\d+)_(\d{3})_([a-z])$")


def parse_clip_name(name):
    m = NAME_RE.match(name)
    if m is None:
        raise ValueError(name)
    abbr, child, day, idx, tag = m.groups()
    return {"identifier": name, "abbr": abbr, "activity": ACTIVITIES[abbr], "theme": THEME[abbr],
            "child_id": int(child), "day": int(day), "clip_index": int(idx), "tag": tag}


# raw JSON parsing

def _read_clip(folder):
    files = sorted(glob.glob(os.path.join(folder, "*_keypoints.json")))
    dets, frames = [], []
    for f in files:
        t = int(f[-27:-15])                       # <frame:012d>_keypoints.json
        for p in json.load(open(f))["people"]:
            k = np.asarray(p["pose_keypoints_2d"], np.float32).reshape(25, 3)
            dets.append(k)
            frames.append(t)
    n_frames = (int(files[-1][-27:-15]) + 1) if files else 0
    det = np.stack(dets) if dets else np.zeros((0, 25, 3), np.float32)
    return os.path.basename(folder.rstrip("/")), n_frames, det, np.asarray(frames, np.int32)


def list_clip_folders(json_root=JSON_ROOT):
    return sorted(glob.glob(os.path.join(json_root, "*", "*/")))


def build_raw_cache(json_root=JSON_ROOT, out=f"{CACHE}/mmasd_raw.npz", workers=8, folders=None):
    folders = folders if folders is not None else list_clip_folders(json_root)
    with Pool(workers) as pool:
        res = pool.map(_read_clip, folders, chunksize=4)
    names = np.array([r[0] for r in res])
    n_frames = np.array([r[1] for r in res], np.int32)
    counts = np.array([len(r[2]) for r in res], np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    det = np.concatenate([r[2] for r in res]) if counts.sum() else np.zeros((0, 25, 3), np.float32)
    det_frame = np.concatenate([r[3] for r in res]).astype(np.int32)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, names=names, n_frames=n_frames, offsets=offsets, det=det, det_frame=det_frame)
    return out


# person selection

def _centroid_height(k):
    v = k[:, 2] > 0
    if v.sum() < 2:
        return None, None
    xy = k[v, :2]
    return np.median(xy, axis=0), max(float(xy[:, 1].max() - xy[:, 1].min()), 10.0)


def link_tracks(det, det_frame, n_frames):
    """Greedy-in-time, Hungarian-per-frame tracking. Returns track id per detection (-1 = unused)."""
    tid = -np.ones(len(det), np.int64)
    tracks = []                                   # [last_centroid, last_height, last_t]
    by_frame = [[] for _ in range(n_frames)]
    for i, t in enumerate(det_frame):
        by_frame[t].append(i)
    for t in range(n_frames):
        cand = []
        for i in by_frame[t]:
            c, h = _centroid_height(det[i])
            if c is not None:
                cand.append((i, c, h))
        if not cand:
            continue
        live = [k for k, tr in enumerate(tracks) if t - tr[2] <= MAX_GAP]
        assigned = set()
        if live:
            cost = np.array([[np.linalg.norm(c - tracks[k][0]) / max(h, tracks[k][1])
                              for k in live] for _, c, h in cand])
            r, cidx = linear_sum_assignment(cost)
            for a, b in zip(r, cidx):
                if cost[a, b] < LINK_MAX:
                    i, c, h = cand[a]
                    k = live[b]
                    tid[i] = k
                    tracks[k] = [c, h, t]
                    assigned.add(a)
        for a, (i, c, h) in enumerate(cand):
            if a not in assigned:
                tid[i] = len(tracks)
                tracks.append([c, h, t])
    return tid


def select_person(det, det_frame, n_frames, rule="track"):
    """(n_frames, 25, 3) array for one person plus selection diagnostics."""
    out = np.zeros((n_frames, 25, 3), np.float32)
    info = {"n_persistent": 0, "coverage": 0.0, "sel_shortest": -1}
    if len(det) == 0:
        return out, info
    mconf = det[:, :, 2].mean(axis=1)
    if rule == "maxconf":
        best = {}
        for i, t in enumerate(det_frame):
            if t not in best or mconf[i] > mconf[best[t]]:
                best[t] = i
        for t, i in best.items():
            out[t] = det[i]
        info["coverage"] = len(best) / n_frames
        return out, info
    if rule != "track":
        raise ValueError(rule)
    tid = link_tracks(det, det_frame, n_frames)
    ok = tid >= 0
    n_tr = int(tid.max()) + 1 if ok.any() else 0
    if n_tr == 0:
        return out, info
    score = np.bincount(tid[ok], weights=mconf[ok], minlength=n_tr)
    cover = np.bincount(tid[ok], minlength=n_tr) / n_frames
    k = int(np.argmax(score))
    for i in np.flatnonzero(tid == k):
        out[det_frame[i]] = det[i]
    # diagnostics: median skeleton height of persistent tracks (coverage >= 0.5)
    pers = np.flatnonzero(cover >= 0.5)
    heights = {}
    for j in pers:
        hs = [_centroid_height(det[i])[1] for i in np.flatnonzero(tid == j)]
        heights[j] = float(np.median(hs))
    info = {"n_persistent": int(len(pers)), "coverage": float(cover[k]),
            "sel_shortest": (int(heights[k] == min(heights.values()))
                             if k in heights and len(heights) > 1 else -1)}
    return out, info


def _select_one(args):
    det, det_frame, n_frames, rule = args
    return select_person(det, det_frame, n_frames, rule)


def build_selected_cache(rule="track", raw=f"{CACHE}/mmasd_raw.npz", out=None, workers=8):
    out = out or f"{CACHE}/mmasd_{rule}.npz"
    z = np.load(raw)
    names, n_frames, off = z["names"], z["n_frames"], z["offsets"]
    det, det_frame = z["det"], z["det_frame"]
    keep = np.flatnonzero(n_frames > 0)
    jobs = [(det[off[i]:off[i + 1]], det_frame[off[i]:off[i + 1]], int(n_frames[i]), rule)
            for i in keep]
    with Pool(workers) as pool:
        res = pool.map(_select_one, jobs, chunksize=8)
    arrs = [r[0] for r in res]
    est_w = [float(det[off[i]:off[i + 1], :, 0].max()) for i in keep]
    est_h = [float(det[off[i]:off[i + 1], :, 1].max()) for i in keep]
    np.savez(out, names=names[keep], n_frames=n_frames[keep],
             offsets=np.r_[0, np.cumsum(n_frames[keep])].astype(np.int64),
             keypoints=np.concatenate(arrs), est_w=np.array(est_w), est_h=np.array(est_h),
             n_persistent=np.array([r[1]["n_persistent"] for r in res]),
             coverage=np.array([r[1]["coverage"] for r in res]),
             sel_shortest=np.array([r[1]["sel_shortest"] for r in res]),
             n_people_mean=np.array([np.bincount(det_frame[off[i]:off[i + 1]],
                                                 minlength=n_frames[i]).mean() for i in keep]))
    return out


# loading

def load_clips(rule="track", joints="body25", cache_dir=CACHE):
    """List of row dicts with 'kp' (T, J, 3) float32 arrays, labels and child ids."""
    path = f"{cache_dir}/mmasd_{rule}.npz"
    if not os.path.exists(path):
        if not os.path.exists(f"{cache_dir}/mmasd_raw.npz"):
            build_raw_cache(out=f"{cache_dir}/mmasd_raw.npz")
        build_selected_cache(rule, raw=f"{cache_dir}/mmasd_raw.npz", out=path)
    z = np.load(path)
    kp_all, off = z["keypoints"], z["offsets"]
    classes = sorted({parse_clip_name(str(n))["activity"] for n in z["names"]})
    rows = []
    for i, n in enumerate(z["names"]):
        r = parse_clip_name(str(n))
        kp = kp_all[off[i]:off[i + 1]]
        if joints == "coco17":
            kp = kp[:, COCO17_FROM_BODY25]
        elif joints != "body25":
            raise ValueError(joints)
        r.update(kp=kp, n_frames=int(z["n_frames"][i]), label=classes.index(r["activity"]),
                 est_w=float(z["est_w"][i]), est_h=float(z["est_h"][i]),
                 n_persistent=int(z["n_persistent"][i]), coverage=float(z["coverage"][i]),
                 sel_shortest=int(z["sel_shortest"][i]), n_people_mean=float(z["n_people_mean"][i]))
        rows.append(r)
    return rows, classes


def subject_split(rows, test_fraction=0.25, seed=0, key="child_id"):
    """Child-level split: (train_rows, test_rows, test_children). No child in both."""
    children = sorted({r[key] for r in rows})
    rng = np.random.RandomState(seed)
    n_test = max(1, int(round(len(children) * test_fraction)))
    test = set(rng.choice(children, n_test, replace=False).tolist())
    tr = [r for r in rows if r[key] not in test]
    te = [r for r in rows if r[key] in test]
    assert not ({r[key] for r in tr} & {r[key] for r in te})
    return tr, te, sorted(test)


# ADOS table

def _xlsx_rows(path):
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(path)
    sst = [("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
           for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = []
    for row in sheet.find("m:sheetData", ns).findall("m:row", ns):
        cells = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", ns)
            if v is None:
                continue
            cells[col] = sst[int(v.text)] if c.get("t") == "s" else v.text
        out.append(cells)
    return out


def load_ados(path=ADOS_XLSX):
    """child_id -> dict of numeric ADOS fields (social affect, RRB, total, comparison score)."""
    rows = _xlsx_rows(path)
    header = rows[0]
    col = {}
    for c, h in header.items():
        h = h.strip().lower()
        if h.startswith("id"):
            col["id"] = c
        elif h.startswith("social affect"):
            col["sa_total"] = c
        elif h.startswith("restricted"):
            col["rrb_total"] = c
        elif h.startswith("overall total"):
            col["total"] = c
        elif h.startswith("ados comparison"):
            col["comparison"] = c
        elif h.startswith("module"):
            col["module"] = c
    out = {}
    for r in rows[1:]:
        raw = r.get(col["id"], "")
        m = re.match(r"^\s*(\d{5})", str(raw))
        if not m:
            continue
        rec = {}
        for k in ("sa_total", "rrb_total", "total", "comparison", "module"):
            try:
                rec[k] = float(r.get(col[k], "nan"))
            except ValueError:
                rec[k] = float("nan")
        out[int(m.group(1))] = rec
    return out


if __name__ == "__main__":
    import sys
    rule = sys.argv[1] if len(sys.argv) > 1 else "track"
    if not os.path.exists(f"{CACHE}/mmasd_raw.npz"):
        print("raw cache:", build_raw_cache())
    print("selected cache:", build_selected_cache(rule))
