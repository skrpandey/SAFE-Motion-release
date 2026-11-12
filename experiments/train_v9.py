"""Protocol v9 (configs/protocols/replication_v9.yaml): train one fold of one cell on an MMASD or
UTD-MHAD host and score its held-out subjects.

Training follows train_asdpose.py (v7 fast configuration: CompactCTRGCN, AdamW, cosine schedule,
best epoch on a subject-level validation split). Test conditions: clean, the matched operator in
worlds S and I at mask seeds 2000/3000/4000, every host mechanism at mid for A_none and real cells,
and gapfill for real cells. --perm K trains the permutation null (label-to-severity pairing flipped
for half of the training and validation subjects; ordinary test masks).

Output: results/v9/<host>/<mechanism>_<strength>/<world>/<cell>/seed<k>/fold<f>/
        (perm<K>/fold<f>/ for the null; real/none/<cell>/seed<k>/fold<f>/ for real cells)
        {train_log.json, test_scores.csv, features.npz}, folds in results/v9/<host>/folds.json,
        checkpoints in cache/.
    conda run -n torch python experiments/train_v9.py --host mmasd_maracas --mechanism runs \\
        --strength mid --world S --cell A_none --seed 0 --fold 0 [--amp] [--perm 3]
    ... --smoke            # CPU, a handful of clips, 1 epoch, output under results/v9_smoke
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion import planted as P
from safe_motion.data import hosts as H
from safe_motion.models.ctrgcn import CompactCTRGCN

CELLS = yaml.safe_load(open("configs/protocols/pilot_asdpose_v2.yaml"))["cells"]
MASK_SEEDS = (2000, 3000, 4000)


def out_dir(args):
    base = "results/v9" + ("_smoke" if args.smoke else "")
    if args.world is None:
        return f"{base}/{args.host}/real/none/{args.cell}/seed{args.seed}/fold{args.fold}"
    who = f"perm{args.perm}" if args.perm is not None else f"seed{args.seed}"
    return f"{base}/{args.host}/{args.mechanism}_{args.strength}/{args.world}/{args.cell}/{who}/fold{args.fold}"


def make_ds(rows, cell, host, layout, train, seed=0, planted=None, mode="zero"):
    return H.ClipDataset(rows, host["window"], layout, train=train, seed=seed, planted=planted,
                         planted_mode=mode, joint_dropout=cell["joint_dropout"],
                         coord_dropout=cell["coord_dropout"], velocity=cell["velocity"])


@torch.no_grad()
def run_eval(model, rows, cell, host, layout, device, workers, planted=None, mode="zero", batch=128):
    dl = DataLoader(make_ds(rows, cell, host, layout, train=False, planted=planted, mode=mode),
                    batch_size=batch, num_workers=workers)
    model.eval()
    sc, ft = [], []
    for x, _ in dl:
        f = model.forward_features(x.to(device))
        sc.append(F.softmax(model.fc(f), 1)[:, 1].cpu().numpy())
        ft.append(f.cpu().numpy())
    return np.concatenate(sc), np.concatenate(ft)


def conditions(args, host):
    """(column, mechanism, mode, world, strength, mask_seed); None mode = none."""
    c = []
    if args.world is not None:
        for w in ("S", "I"):
            for s in MASK_SEEDS:
                c.append((f"{args.mechanism}_{w}_{args.strength}_m{s}", args.mechanism,
                          P.MECHANISM_MODE[args.mechanism], w, args.strength, s))
    if args.world is None or args.cell == "A_none":
        for mech in host["mechanisms"]:
            for w in ("S", "I"):
                c.append((f"{mech}_{w}_mid_m2000", mech, P.MECHANISM_MODE[mech], w, "mid", 2000))
    if args.world is None:
        for w in ("S", "I"):
            c.append((f"gapfill_{w}_mid_m2000", "runs", "gapfill", w, "mid", 2000))
    seen, out = set(), []
    for x in c:
        if x[0] not in seen:
            seen.add(x[0])
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=list(H.HOSTS))
    ap.add_argument("--cell", default="A_none", choices=["A_none", "A_dropout", "B_coorddrop", "A_velocity", "A_both"])
    ap.add_argument("--world", choices=["S", "I"], default=None, help="omit for a real cell")
    ap.add_argument("--mechanism", choices=list(P.MECHANISM_MODE), default="runs")
    ap.add_argument("--strength", choices=["mid", "low"], default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--perm", type=int, default=None, help="permutation-null index K (seed 0, A_none)")
    ap.add_argument("--epochs", type=int, default=None, help="override the host's protocol value")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="CPU, a few clips, 1 epoch")
    args = ap.parse_args()
    host, layout = H.HOSTS[args.host], H.LAYOUT[H.HOSTS[args.host]["dataset"]]
    if args.world is not None:
        assert args.mechanism in host["mechanisms"], f"{args.mechanism} is not a v9 mechanism on {args.host}"
    if args.perm is not None:
        assert args.world is not None and args.cell == "A_none" and args.seed == 0
    cell = CELLS[args.cell]
    seed = args.seed if args.perm is None else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu" if args.smoke or not torch.cuda.is_available() else "cuda"
    epochs = 1 if args.smoke else (args.epochs or host["epochs"])
    base = "results/v9" + ("_smoke" if args.smoke else "")
    od = out_dir(args)
    if os.path.exists(f"{od}/train_log.json"):
        print(f"skip {od}")
        return

    rows = H.load_host(args.host)
    folds = H.make_folds(args.host, rows, seed=0, path=f"{base}/{args.host}/folds.json")
    assert 0 <= args.fold < len(folds), f"fold {args.fold} of {len(folds)}"
    tr, va, te = H.split_fold(rows, folds[args.fold], val_fraction=0.25, seed=0)
    if args.smoke:
        rng = np.random.RandomState(0)
        pick = lambda rs, n: [rs[i] for i in sorted(rng.choice(len(rs), min(n, len(rs)), replace=False))]
        tr, va, te = pick(tr, 24), pick(va, 12), pick(te, 16)
    print(f"{args.host} fold {args.fold}: train {len(tr)} val {len(va)} test {len(te)} "
          f"(subjects {sorted({r['subject'] for r in te})}); cell {args.cell} {cell}; device {device}", flush=True)

    masks_trva, mode = None, P.MECHANISM_MODE[args.mechanism]
    if args.world is not None:
        masks_trva = H.host_masks(tr + va, args.world, args.strength, 0, args.mechanism, layout, perm=args.perm)
        print(f"planted world {args.world} strength {args.strength} mechanism {args.mechanism} "
              f"operator {mode}" + (f" perm {args.perm}" if args.perm is not None else ""), flush=True)

    c_in = 2 + 2 * cell["velocity"] + 1
    model = CompactCTRGCN(c_in=c_in, num_joints=layout["n_joints"]).to(device)
    dl = DataLoader(make_ds(tr, cell, host, layout, train=True, seed=seed, planted=masks_trva, mode=mode),
                    batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False,
                    persistent_workers=args.workers > 0,
                    worker_init_fn=H.A.worker_init if H.A.TRAIN_RECIPE == "v15" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    va_y = [r["binary_label"] for r in va]
    best, log, t0 = (-1.0, None, -1), [], time.time()
    for epoch in range(epochs):
        model.train()
        tot = seen = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=args.amp and device == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            seen += len(y)
        sched.step()
        va_auc = M.auc(run_eval(model, va, cell, host, layout, device, args.workers,
                                planted=masks_trva, mode=mode)[0], va_y)
        log.append({"epoch": epoch + 1, "loss": tot / seen, "val_auc": va_auc, "min": (time.time() - t0) / 60})
        star = ""
        if va_auc > best[0]:
            best = (va_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch + 1)
            star = " *"
        print(f"epoch {epoch+1:3d} loss {tot/seen:.4f} val AUC {va_auc:.4f} ({(time.time()-t0)/60:.1f} min){star}",
              flush=True)
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best[1])

    os.makedirs(od, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    run_id = od[len(base) + 1:].replace("/", "_")
    torch.save({"state_dict": best[1], "cell": cell, "seed": seed, "val_auc": best[0], "epoch": best[2],
                "host": args.host, "num_joints": layout["n_joints"]},
               f"cache/v9_{run_id}{'_smoke' if args.smoke else ''}.pt")

    clean, feats = run_eval(model, te, cell, host, layout, device, args.workers)
    cols = {"clean": clean}
    cache = {}
    for col, mech, md, w, st, ms in conditions(args, host):
        key = (mech, w, st, ms)
        if key not in cache:
            cache[key] = H.host_masks(te, w, st, ms, mech, layout)
        cols[col] = run_eval(model, te, cell, host, layout, device, args.workers, planted=cache[key], mode=md)[0]
    np.savez_compressed(f"{od}/features.npz", feats=feats,
                        n_frames=np.array([r["n_frames"] for r in te]),
                        label=np.array([r["binary_label"] for r in te]),
                        subject=np.array([r["subject"] for r in te]))
    with open(f"{od}/test_scores.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["identifier", "subject_id", "label", "n_frames", "qc_pass"] + H.Q_COLS + list(cols))
        for i, r in enumerate(te):
            wr.writerow([r["identifier"], r["subject"], r["binary_label"], r["n_frames"], r["qc_pass"]]
                        + [f"{r[q]:.4f}" for q in H.Q_COLS] + [f"{cols[c][i]:.6f}" for c in cols])
    y = [r["binary_label"] for r in te]
    msg = f"test (fold) AUC clean {M.auc(clean, y):.4f}"
    if args.world is not None:
        msg += (f" {args.mechanism} S/I {M.auc(cols[f'{args.mechanism}_S_{args.strength}_m2000'], y):.4f}/"
                f"{M.auc(cols[f'{args.mechanism}_I_{args.strength}_m2000'], y):.4f}")
    print(msg + f" (best epoch {best[2]}, val {best[0]:.4f}, {train_min:.1f} min)", flush=True)
    json.dump({"host": args.host, "cell": args.cell, "config": cell, "world": args.world,
               "mechanism": args.mechanism if args.world else None, "strength": args.strength if args.world else None,
               "seed": seed, "perm": args.perm, "fold": args.fold, "test_subjects": folds[args.fold],
               "n_train": len(tr), "n_val": len(va), "n_test": len(te), "best_epoch": best[2],
               "best_val_auc": best[0], "train_minutes": train_min, "epochs": epochs, "log": log},
              open(f"{od}/train_log.json", "w"), indent=1)
    if not args.smoke:
        with open("results/gpu_ledger.csv", "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d"), f"v9/{run_id}", f"{(time.time() - t0) / 3600:.3f}"])


if __name__ == "__main__":
    main()
