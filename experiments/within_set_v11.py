"""Protocol v11: head versus representation on the external set.

Folds: video-grouped stratified 5-fold (seed 0), saved to results/v11/folds.json and reused by
analyze_v11.py. DFR: logistic regression on each model's penultimate features
(results/v11/features/<model>.npz), scored out of fold. Within-set ceiling: TinyTCN and
CompactCTRGCN trained from scratch on the external cache with the same folds, three seeds, 25
epochs, batch 32, window-std normalisation, last epoch (a validation split inside a fold would
leak videos), scored out of fold.
    conda run -n torch python experiments/within_set_v11.py              # DFR (CPU) + training (GPU)
    conda run -n torch python experiments/within_set_v11.py --smoke      # CPU, 1 epoch, 2 folds
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D
from safe_motion.models.ctrgcn import CompactCTRGCN
from safe_motion.models.tcn import TinyTCN

EXTERNAL_CACHE = os.environ.get("SM_EXTERNAL_CACHE", f"{D.SMM_ROOT}/cache/external")
OUT = "results/v11"
N_FOLDS, EPOCHS, BATCH, LR, WD = 5, 25, 32, 1e-3, 1e-4


def make_folds(rows, out, seed=0, n_folds=N_FOLDS):
    y = np.array([r["binary_label"] for r in rows])
    g = np.array([r["child_id"] for r in rows])
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for _, te in skf.split(np.zeros(len(y)), y, g):
        folds.append(sorted({int(x) for x in g[te]}))
    json.dump({"seed": seed, "n_folds": n_folds, "group": "child_id (source video)", "folds": folds},
              open(out, "w"), indent=1)
    return folds


def load_folds(path):
    return json.load(open(path))["folds"]


def fold_mask(rows, fold_videos):
    s = set(fold_videos)
    return np.array([r["child_id"] in s for r in rows])


def dfr(rows, folds, feat_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids = [r["identifier"] for r in rows]
    for path in sorted(glob.glob(f"{feat_dir}/*.npz")):
        name = os.path.basename(path)[:-4]
        outp = f"{out_dir}/{name}.csv"
        if os.path.exists(outp):
            continue
        z = np.load(path)
        assert list(z["identifier"]) == ids, name
        X, y = z["feats"], z["label"].astype(int)
        oof = np.full(len(y), np.nan)
        for fv in folds:
            te = fold_mask(rows, fv)
            tr = ~te
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            oof[te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
        with open(outp, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["identifier", "child_id", "label", "dfr"])
            for r, s in zip(rows, oof):
                wr.writerow([r["identifier"], r["child_id"], r["binary_label"], f"{s:.6f}"])
        print(f"dfr {name:34s} oof AUC {M.auc(oof, y):.3f}", flush=True)


def train_one(arch, tr_rows, te_rows, seed, device, epochs, workers):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds = D.SegmentDataset(tr_rows, train=True, seed=seed, normalize="window-std",
                          cache_dir=EXTERNAL_CACHE)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=workers, drop_last=False)
    net = (CompactCTRGCN(c_in=3) if arch == "compact" else TinyTCN(c_in=3)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * len(dl)))
    for _ in range(epochs):
        net.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    ds_te = D.SegmentDataset(te_rows, train=False, normalize="window-std", cache_dir=EXTERNAL_CACHE)
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds_te, batch_size=128, num_workers=workers):
            out.append(F.softmax(net(x.to(device)), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def within_set(rows, folds, out_dir, device, epochs, seeds, workers, archs=("tcn", "compact")):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    y = np.array([r["binary_label"] for r in rows])
    for arch in archs:
        for seed in seeds:
            outp = f"{out_dir}/{arch}_s{seed}.csv"
            if os.path.exists(outp):
                continue
            oof = np.full(len(rows), np.nan)
            for k, fv in enumerate(folds):
                te = fold_mask(rows, fv)
                tr_rows = [r for r, m in zip(rows, te) if not m]
                te_rows = [r for r, m in zip(rows, te) if m]
                oof[te] = train_one(arch, tr_rows, te_rows, seed * 100 + k, device, epochs, workers)
            with open(outp, "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["identifier", "child_id", "label", "oof"])
                for r, s in zip(rows, oof):
                    wr.writerow([r["identifier"], r["child_id"], r["binary_label"], f"{s:.6f}"])
            print(f"within-set {arch} seed {seed}: oof AUC {M.auc(oof, y):.3f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--skip-dfr", action="store_true")
    args = ap.parse_args()
    out = OUT + ("_smoke" if args.smoke else "")
    os.makedirs(out, exist_ok=True)
    device = "cpu" if args.smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = D.load_index(f"{EXTERNAL_CACHE}/index.csv")
    if args.smoke:
        rows = rows[::5]
    fpath = f"{out}/folds.json"
    folds = load_folds(fpath) if os.path.exists(fpath) else make_folds(rows, fpath, n_folds=2 if args.smoke else N_FOLDS)
    if not args.skip_dfr:
        dfr(rows, folds, f"{out}/features", f"{out}/dfr")
    if not args.skip_training:
        t0 = time.time()
        within_set(rows, folds, f"{out}/within_set", device, 1 if args.smoke else EPOCHS,
                   [0] if args.smoke else [0, 1, 2], args.workers)
        if not args.smoke:
            with open("results/gpu_ledger.csv", "a", newline="") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d"), "v11/within_set",
                                        f"{(time.time()-t0)/3600:.3f}"])
    print("done", flush=True)


if __name__ == "__main__":
    main()
