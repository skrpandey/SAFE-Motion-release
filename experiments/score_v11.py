"""Protocol v11 (configs/protocols/transfer_v11.yaml): score every model on the external
SSBD/ESBD set (T2 clean, T3 confidence zeroed, T5 planted operators), v7 score-file format.

Models: the v7 real cells (ours_v7), the v10 real-data World I references (ours_v10) and the
published SAFE-SMM checkpoints of protocol v5 (read-only, own preprocessing). Rows: the base
project's external Stage 1 cache (550 windows of 200 frames) or, with --cache, the MediaPipe
cache of build_mediapipe_external.py (same identifiers). Planting direction follows the
ASDPose cue (World S degrades negatives more). Penultimate features under the clean condition
are saved for T6.

    conda run -n torch python experiments/score_v11.py                      # RTMPose cache
    conda run -n torch python experiments/score_v11.py --cache results/v11/mediapipe_cache --tag _mediapipe
    conda run -n torch python experiments/score_v11.py --smoke               # CPU, 2 models, 40 rows
"""

import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import asdpose as D
import score_v7 as S7
import score_intervention as S5

EXTERNAL_CACHE = os.environ.get("SM_EXTERNAL_CACHE", f"{D.SMM_ROOT}/cache/external")
OUT = "results/v11"
MECHS = ("runs", "whole", "frozen", "armruns")
V1 = yaml.safe_load(open("configs/protocols/pilot_asdpose_v1.yaml"))
V5 = "configs/protocols/pilot_asdpose_v5.yaml"
OURS = re.compile(r"^pilot_(v7_real|v10_real_I)_([AB]_[a-z]+)_s(\d)$")


def registry(smoke=False):
    models = []
    for base, fam in (("results/fits/v7/cache", "ours_v7"), ("results/fits/v10/cache", "ours_v10")):
        for path in sorted(glob.glob(f"{base}/pilot_v*_real*_[AB]_*_s?.pt")):
            m = OURS.match(os.path.basename(path)[:-3])
            if not m:
                continue
            _, cell, seed = m.groups()
            models.append({"name": f"{fam}_{cell}_s{seed}", "family": fam, "arch": "compact",
                           "path": path, "cell": cell, "seed": int(seed)})
    if not smoke:
        cfg = yaml.safe_load(open(V5))
        for m in S5.registry(cfg):
            if m["family"] == "safe_smm":
                m = dict(m, family="published", cell=None, seed=None)
                models.append(m)
    return models


def build(m):
    if m["family"] == "published":
        net = S5.build(m)
        prep = dict(m["prep"])
    else:
        net, prep = S7.build({"path": m["path"], "arch": m["arch"]})
    return net.eval(), prep


class ConfZero(Dataset):
    """T3: the confidence channel zeroed after preprocessing (models with a confidence input)."""

    def __init__(self, ds, has_conf):
        self.ds, self.has_conf = ds, has_conf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        if self.has_conf:
            x = x.clone()
            x[-1] = 0.0
        return x, y


def conditions():
    c = [("clean", None, "zero", None, None, None), ("conf0", None, "zero", None, None, None)]
    for mech in MECHS:
        for w in ("S", "I"):
            c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
    for s in (3000, 4000):
        for w in ("S", "I"):
            c.append((f"runs_{w}_mid_m{s}", "runs", "zero", w, "mid", s))
    for w in ("S", "I"):
        c.append((f"gapfill_{w}_mid_m2000", "runs", "gapfill", w, "mid", 2000))
    return c


@torch.no_grad()
def run(net, rows, prep, cache_dir, masks, mode, device, workers, conf_zero=False, feats=False,
        batch=128):
    ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=mode, cache_dir=cache_dir,
                          **prep)
    if conf_zero:
        ds = ConfZero(ds, prep.get("confidence", True))
    sc, ft = [], []
    for x, _ in DataLoader(ds, batch_size=batch, num_workers=workers):
        f = net.forward_features(x.to(device))
        sc.append(F.softmax(net.fc(f), 1)[:, 1].cpu().numpy())
        if feats:
            ft.append(f.cpu().numpy())
    return np.concatenate(sc), (np.concatenate(ft) if feats else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=EXTERNAL_CACHE)
    ap.add_argument("--tag", default="", help="suffix for the scores/features directories")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.tag not in ("", "_mediapipe"):     # a tag without the underscore would write to scoresmediapipe/
        ap.error(f"--tag must be '' or '_mediapipe', got {args.tag!r}")
    device = args.device or ("cpu" if args.smoke else ("cuda" if torch.cuda.is_available() else "cpu"))
    out = OUT + ("_smoke" if args.smoke else "")
    sdir, fdir = f"{out}/scores{args.tag}", f"{out}/features{args.tag}"
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    rows = D.load_index(f"{args.cache}/index.csv")
    if args.smoke:
        rows = rows[::14]
    models = registry(args.smoke)
    if args.only:
        models = [m for m in models if args.only in m["name"]]
    if args.smoke:
        models = models[:1] + models[-1:]
    print(f"rows {len(rows)} (pos {sum(r['binary_label'] for r in rows)}), models {len(models)}, "
          f"device {device}, cache {args.cache}", flush=True)
    masks = {}

    def mask_for(mech, w, st, seed):
        key = (mech, w, st, seed)
        if key not in masks:
            masks[key] = P.build_masks(rows, w, st, seed, mechanism=mech)[0]
        return masks[key]

    t0 = time.time()
    y = [r["binary_label"] for r in rows]
    for m in models:
        path = f"{sdir}/{m['name']}.csv"
        if os.path.exists(path):
            continue
        net, prep = build(m)
        net = net.to(device)
        cols = {}
        feats = None
        for col, mech, mode, w, st, seed in conditions():
            mk = None if mech is None else mask_for(mech, w, st, seed)
            s, f = run(net, rows, prep, args.cache, mk, mode, device, args.workers,
                       conf_zero=(col == "conf0"), feats=(col == "clean"))
            cols[col] = s
            if f is not None:
                feats = f
        np.savez_compressed(f"{fdir}/{m['name']}.npz", feats=feats,
                            identifier=np.array([r["identifier"] for r in rows]),
                            label=np.array(y), child=np.array([r["child_id"] for r in rows]))
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
            for i, r in enumerate(rows):
                wr.writerow([r["identifier"], r["child_id"], r["binary_label"], r["n_frames"],
                             int(D.passes_qc(r, V1["qc_filter"]))] + [f"{cols[c][i]:.6f}" for c in cols])
        print(f"{m['name']:34s} clean {M.auc(cols['clean'], y):.3f} conf0 {M.auc(cols['conf0'], y):.3f} "
              f"runs S/I {M.auc(cols['runs_S_mid_m2000'], y):.3f}/{M.auc(cols['runs_I_mid_m2000'], y):.3f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
