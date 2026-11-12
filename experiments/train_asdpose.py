"""Train one ASDPose cell (TinyTCN or compact CTR-GCN) under a given protocol and score the test split.

The training loop follows asdpose-smm-detection (commit e5217a5) stage1/train.py: AdamW, cosine
schedule, best-validation-AUC epoch. Base settings come from configs/protocols/pilot_asdpose_v1.yaml,
cells from the --proto file. Writes test scores and train_log.json under results/ (path set by the protocol),
checkpoints to cache/, and appends wall time to results/gpu_ledger.csv. With --repair-protocol,
every epoch is saved and both the original and the cleaned validation selections are written.

Usage:
    conda run -n torch python experiments/train_asdpose.py --cell A_none
    conda run -n torch python experiments/train_asdpose.py --cell A_none --smoke   # CPU, tiny
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M
from safe_motion.data import asdpose as D
from safe_motion.models.tcn import TinyTCN
from safe_motion.models.ctrgcn import CompactCTRGCN
from safe_motion.models.baselines import STGCN
from safe_motion import planted as P
from safe_motion import remedies as RM

PROTO = "configs/protocols/pilot_asdpose_v1.yaml"
PROTOS = {"v1": PROTO, "v2": "configs/protocols/pilot_asdpose_v2.yaml",
          "v3": "configs/protocols/pilot_asdpose_v3.yaml",
          "v4": "configs/protocols/pilot_asdpose_v4.yaml",
          "v6": "configs/protocols/pilot_asdpose_v6.yaml",
          "v7": "configs/protocols/pilot_asdpose_v7.yaml",
          "v8": "configs/protocols/pilot_asdpose_v8.yaml",
          "v10": "configs/protocols/identification_v10.yaml",
          "v12": "configs/protocols/remedies_v12.yaml",
          "v13": "configs/protocols/armconf_v13.yaml",
          "v15conf": "configs/protocols/remedies_confonly_v15.yaml",
          "v16": "configs/protocols/confonly_extension_v16.yaml",
          "v17": "configs/protocols/confonly_assignment_v17.yaml"}
LENGTH_PROTOS = ("v7", "v8", "v10", "v12", "v13", "v15conf", "v16", "v17")   # length-controlled protocol, no tiling anywhere
# v10 adds A_blind, the quality-blind control: asdpose.blind_transform input, no confidence channel.
# It is not blind to planted severity.
EXTRA_CELLS = {"A_blind": {"velocity": False, "joint_dropout": 0.0, "coord_dropout": 0.0,
                           "pool": "full", "blind": True}}
# v12: A_none inputs plus a cue-targeted training method (safe_motion/remedies.py).
# v13: valid joint means x or y nonzero ("coords"), so confidence-only planting leaves x, y unchanged.
V13_CELLS = {"A_none": {"velocity": False, "joint_dropout": 0.0, "coord_dropout": 0.0, "pool": "full",
                        "valid_rule": "coords"}}
V12_CELLS = {name: {"velocity": False, "joint_dropout": 0.0, "coord_dropout": 0.0, "pool": "full",
                    "remedy": rem}
             for name, rem in (("C_groupdro", "groupdro"), ("C_adversarial", "adversarial"),
                               ("C_augment", "augment"))}
# v15conf: the v13 rule in both arms, Group DRO in one; requires --repair-protocol and
# SM_TRAIN_RECIPE=v15.
V15CONF_CELLS = {"A_none": dict(V13_CELLS["A_none"]),
                 "C_groupdro": {**V13_CELLS["A_none"], "remedy": "groupdro"}}
# v16: confidence-only extension. ST-GCN no-remedy cell, more CompactCTRGCN no-remedy seeds, and
# Group DRO with groups from the measured arm low-confidence fraction of the planted input.
V16_CELLS = {"A_none_stgcn": {**V13_CELLS["A_none"], "arch": "stgcn"},
             "A_none": {**V13_CELLS["A_none"], "arch": "compact"},
             "C_groupdro_obs": {**V13_CELLS["A_none"], "arch": "compact", "remedy": "groupdro_obs"}}
# v17: the v16 no-remedy cell retrained under regenerated training assignments (--world-seed 1, 2, 3).
V17_CELLS = {"A_none": dict(V16_CELLS["A_none"])}


def make_ds(rows, cell, cfg, train, seed=0, planted=None, planted_mode="zero"):
    blind = bool(cell.get("blind", False))
    return D.SegmentDataset(rows, window=cfg["data"]["window"], train=train, planted=planted,
                            planted_mode=planted_mode,
                            flip_prob=cfg["data"]["flip_prob"], seed=seed,
                            joint_dropout=cell["joint_dropout"],
                            coord_dropout=cell["coord_dropout"],
                            velocity=cell["velocity"], normalize=cfg["data"]["normalize"],
                            confidence=not blind, blind=blind,
                            valid_rule=cell.get("valid_rule", "conf"))


@torch.no_grad()
def run_eval(model, rows, cell, cfg, device, workers, batch=256, planted=None, planted_mode="zero"):
    dl = DataLoader(make_ds(rows, cell, cfg, train=False, planted=planted, planted_mode=planted_mode),
                    batch_size=batch,
                    num_workers=workers)
    model.eval()
    scores, feats = [], []
    for x, _ in dl:
        f = model.forward_features(x.to(device))
        scores.append(F.softmax(model.fc(f), 1)[:, 1].cpu().numpy())
        feats.append(f.cpu().numpy())
    return np.concatenate(scores), np.concatenate(feats)


def linear_probe(f_tr, rows_tr, f_te, rows_te, stats, n_boot, alphas=(0.1, 1, 10, 100)):
    groups = np.array([r["child_id"] for r in rows_tr])
    child_te = np.array([r["child_id"] for r in rows_te])
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    xtr, xte = (f_tr - mu) / sd, (f_te - mu) / sd
    out = {}
    boots = list(M.cluster_bootstrap_indices(child_te, n_boot, seed=0))
    for k in stats:
        ytr = np.array([r[k] for r in rows_tr])
        yte = np.array([r[k] for r in rows_te])
        m, s = ytr.mean(), ytr.std() + 1e-9
        ytr, yte = (ytr - m) / s, (yte - m) / s
        cv = {}
        for a in alphas:
            errs = []
            for tr, va in GroupKFold(5).split(xtr, ytr, groups):
                p = Ridge(alpha=a).fit(xtr[tr], ytr[tr]).predict(xtr[va])
                errs.append(((p - ytr[va]) ** 2).mean())
            cv[a] = float(np.mean(errs))
        best = min(cv, key=cv.get)
        pred = Ridge(alpha=best).fit(xtr, ytr).predict(xte)

        def r2(i):
            return 1 - ((pred[i] - yte[i]) ** 2).sum() / ((yte[i] - yte[i].mean()) ** 2).sum()
        out[k] = {"alpha": best, "r2": float(r2(np.arange(len(yte)))),
                  "r2_ci": M.ci([r2(i) for i in boots])}
    return out


def _sha(path, n=1 << 24):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def repair_finish(args, repair, rep_dir, cell, seed, best, best_clean, log, va_rows, keep, va_preds,
                  train_min, eps_trva, masks_trva, test_masks):
    """Write both selected checkpoints, per-epoch validation predictions and a provenance record.
    Test scoring is left to score_v*.py."""
    import platform
    import subprocess
    run_name = os.path.basename(rep_dir)
    for tag, b in (("orig", best), ("clean", best_clean)):
        d = f"{repair['outputs']['cache_root']}/{tag}"
        os.makedirs(d, exist_ok=True)
        torch.save({"state_dict": b[1], "cell": cell, "seed": seed, "val_auc": b[0], "epoch": b[2],
                    "selection": tag, "repair_protocol": args.repair_protocol},
                   f"{d}/{historical_ckpt_name(args, seed)}")
    np.savez_compressed(f"{rep_dir}/val_preds.npz", preds=va_preds,
                        identifier=np.array([r["identifier"] for r in va_rows]),
                        child_id=np.array([int(r["child_id"]) for r in va_rows]),
                        label=np.array([r["binary_label"] for r in va_rows]), keep_clean=keep)
    static = {}
    if eps_trva is not None:
        e = np.asarray(eps_trva)
        static = {"trva_draws": int(e.size), "trva_above_15_16": int((e > 15 / 16).sum()),
                  "trva_above_0.95": int((e > 0.95).sum())}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", None
    if not commit:                              # code copy without .git: hash passed as SM_CODE_COMMIT
        commit, dirty = os.environ.get("SM_CODE_COMMIT", "unknown"), None
    prov = {"run": run_name, "argv": sys.argv, "seed": seed, "cell": cell,
            "selected_epoch_orig": best[2], "val_auc_orig": best[0],
            "selected_epoch_clean": best_clean[2], "val_auc_clean": best_clean[0],
            "selection_changed": best[2] != best_clean[2],
            "n_val_rows": len(va_rows), "n_val_rows_clean": int(keep.sum()),
            "val_exclude_ids": repair["val_exclude_ids"], "train_minutes": train_min, "log": log,
            "static_mask_clipping": static,
            "software": {"python": platform.python_version(), "torch": torch.__version__,
                         "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                         "cudnn_benchmark": torch.backends.cudnn.benchmark,
                         "cudnn_deterministic": torch.backends.cudnn.deterministic, "numpy": np.__version__},
            "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                         "host": platform.node()},
            "code": {"commit": commit, "dirty": dirty},
            "data": {"cache_index_sha256_16MiB": _sha(f"{D.CACHE_DIR}/index.csv")},
            "train_recipe": D.TRAIN_RECIPE,
            "note": repair.get("note", "post hoc repair of a discovered validation alias; rerun with the same recipe")}
    json.dump(prov, open(f"{rep_dir}/train_log.json", "w"), indent=1, default=str)
    print(f"repair done: epoch orig {best[2]} (val {best[0]:.4f}), clean {best_clean[2]} "
          f"(val {best_clean[0]:.4f}), {train_min:.1f} min", flush=True)


def repair_run_name(args, seed):
    """Unique per historical fit: the historical checkpoint stem (includes world seed)."""
    return historical_ckpt_name(args, seed)[len("pilot_"):-len(".pt")]


def historical_ckpt_name(args, seed):
    """The cache file name the historical run used, so score_v*.py registries match unchanged."""
    if args.world:
        w = f"{args.world}_w{args.world_seed}_{args.mechanism}_{args.strength}"
    elif args.real_world:
        w = f"real_{args.real_world}"
    else:
        w = "real"
    return f"pilot_{args.proto}_{w}_{args.cell}_s{seed}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU, 200 segments, 1 epoch")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=2000, help="probe CI resamples")
    ap.add_argument("--proto", default="v1", choices=list(PROTOS))
    ap.add_argument("--seed", type=int, default=None, help="training seed (v2)")
    ap.add_argument("--permute", type=int, default=None,
                    help="v2 permutation null: shuffle train and val labels with this seed")
    ap.add_argument("--world", choices=["S", "I", "Sp"], default=None, help="planted world")
    ap.add_argument("--world-seed", type=int, default=0, help="training-mask realization (v6)")
    ap.add_argument("--strength", choices=["mid", "low"], default="mid")
    ap.add_argument("--mechanism", choices=list(P.MECHANISM_MODE), default="runs",
                    help="planted mechanism; its operator (mode) follows planted.MECHANISM_MODE (v8)")
    ap.add_argument("--arch", choices=["tcn", "compact", "stgcn"], default="tcn")
    ap.add_argument("--epochs", type=int, default=None, help="override (v4 compact fast config)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-epochs-run", type=int, default=None, help="timing check: stop early")
    ap.add_argument("--real-world", choices=["S", "I"], default=None,
                    help="v10 real_world_I: plant the mechanism in this world on the FULL split, "
                         "on top of the natural failures (no host filtering)")
    ap.add_argument("--repair-protocol", default=None,
                    help="repair_selection_v14: YAML with val_exclude_ids; saves every epoch and "
                         "selects by both the original and the cleaned validation metric")
    args = ap.parse_args()
    repair = yaml.safe_load(open(args.repair_protocol)) if args.repair_protocol else None
    if repair is not None:                      # restart: a finished repair fit is skipped
        _seed = 0 if args.seed is None else args.seed
        _done = f"{repair['outputs']['epoch_dir']}/{repair_run_name(args, _seed)}/train_log.json"
        if os.path.exists(_done) and not args.smoke:
            print(f"repair skip (done): {_done}", flush=True)
            return
    if args.real_world is not None:
        assert args.proto == "v10" and args.world is None, "--real-world is a v10 real-split option"
    cfg = yaml.safe_load(open(PROTO))          # v2 inherits data, model and train settings
    b2 = cfg["batch2"]
    if args.proto == "v1":
        cell = b2["cells"][args.cell]
        seed = b2["train"]["seed"]
    else:                                       # v2 cells are shared by v3; v10 adds A_blind
        cells = dict(yaml.safe_load(open(PROTOS["v2"]))["cells"])
        if args.proto == "v10":
            cells.update(EXTRA_CELLS)
        if args.proto == "v12":
            cells.update(V12_CELLS)
        if args.proto == "v13":
            cells = dict(V13_CELLS)
        if args.proto == "v15conf":
            cells = dict(V15CONF_CELLS)
            assert repair is not None and repair["outputs"]["cache_root"].startswith("cache/v15conf"), \
                "v15conf fits run with --repair-protocol configs/protocols/remedies_confonly_v15.yaml"
            assert D.TRAIN_RECIPE == "v15", "protocol v15conf requires SM_TRAIN_RECIPE=v15"
        if args.proto == "v16":
            cells = dict(V16_CELLS)
            assert repair is not None and repair["outputs"]["cache_root"].startswith("cache/v16"), \
                "v16 fits run with --repair-protocol configs/protocols/confonly_extension_v16.yaml"
            assert D.TRAIN_RECIPE == "v15", "protocol v16 requires SM_TRAIN_RECIPE=v15"
        if args.proto == "v17":
            cells = dict(V17_CELLS)
            assert repair is not None and repair["outputs"]["cache_root"].startswith("cache/v17"), \
                "v17 fits run with --repair-protocol configs/protocols/confonly_assignment_v17.yaml"
            assert D.TRAIN_RECIPE == "v15", "protocol v17 requires SM_TRAIN_RECIPE=v15"
            assert args.world_seed in (1, 2, 3), "protocol v17 regenerates the training assignment: --world-seed 1, 2 or 3"
        cell = cells[args.cell]
        if "arch" in cell:
            assert args.arch == cell["arch"], f"cell {args.cell} is trained with --arch {cell['arch']}"
        seed = 0 if args.seed is None else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cpu" if args.smoke else "cuda"
    epochs = 1 if args.smoke else (args.epochs or b2["train"]["epochs"])

    rows = D.load_index()
    if args.proto in LENGTH_PROTOS:             # length-controlled protocol: no tiling anywhere
        rows = [r for r in rows if r["n_frames"] >= 200]
    tr_rows, va_rows, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    te_rows = [r for r in rows if r["split"] == "test"]
    if cell["pool"] == "qc":
        tr_rows, va_rows = D.qc_pool(tr_rows, cfg["qc_filter"]), D.qc_pool(va_rows, cfg["qc_filter"])
    elif cell["pool"] == "qc_sym":
        tr_rows = [r for r in tr_rows if D.passes_qc(r, cfg["qc_filter"])]
        va_rows = [r for r in va_rows if D.passes_qc(r, cfg["qc_filter"])]
    if args.world is not None:                  # v3 planted host: qc_sym in train, val and test
        assert args.proto in ("v3", "v4", "v6", "v7", "v8", "v10", "v12", "v13", "v15conf", "v16", "v17") and cell["pool"] == "full"
        tr_rows = [r for r in tr_rows if D.passes_qc(r, cfg["qc_filter"])]
        va_rows = [r for r in va_rows if D.passes_qc(r, cfg["qc_filter"])]
        te_rows = [r for r in te_rows if D.passes_qc(r, cfg["qc_filter"])]
    if args.permute is not None:
        # v2 permutation null: labels shuffled globally in train AND val, copies only
        prng = np.random.RandomState(args.permute)
        tr_rows, va_rows = [dict(r) for r in tr_rows], [dict(r) for r in va_rows]
        for rs in (tr_rows, va_rows):
            lab = prng.permutation([r["binary_label"] for r in rs])
            for r, l in zip(rs, lab):
                r["binary_label"] = int(l)
    if args.smoke:
        rng = np.random.RandomState(0)
        pick = lambda rs, n: [rs[i] for i in sorted(rng.choice(len(rs), n, replace=False))]
        tr_rows, va_rows, te_rows = pick(tr_rows, 200), pick(va_rows, 100), pick(te_rows, 200)
    print(f"cell {args.cell} {cell}: train {len(tr_rows)} val {len(va_rows)} test {len(te_rows)}",
          flush=True)
    masks_trva = test_masks = eps_trva = None
    pmode = P.MECHANISM_MODE[args.mechanism]
    if args.world is not None:
        masks_trva, eps_trva = P.build_masks(tr_rows + va_rows, args.world, args.strength,
                                             seed=args.world_seed, mechanism=args.mechanism)
        test_masks = {w: P.build_masks(te_rows, w, args.strength, seed=1000,
                                       mechanism=args.mechanism)[0] for w in ("S", "I")}
        print(f"planted world {args.world} strength {args.strength} mechanism {args.mechanism} "
              f"operator {pmode}", flush=True)
    if args.real_world is not None:             # v10 real-data reference: planted on the full split
        masks_trva, eps = P.build_masks(tr_rows + va_rows, args.real_world, args.strength,
                                        seed=args.world_seed, mechanism=args.mechanism)
        test_masks = {w: P.build_masks(te_rows, w, args.strength, seed=1000,
                                       mechanism=args.mechanism)[0] for w in ("S", "I")}
        lab = np.array([r["binary_label"] for r in tr_rows + va_rows])
        print(f"real split, planted world {args.real_world} strength {args.strength} mechanism "
              f"{args.mechanism} operator {pmode}; severity-label correlation "
              f"{np.corrcoef(eps, lab)[0, 1]:+.3f}", flush=True)

    c_in = 2 + 2 * cell["velocity"] + (0 if cell.get("blind") else 1)
    if args.arch == "stgcn":
        model = STGCN(c_in=c_in).to(device)
    else:
        model = (CompactCTRGCN(c_in=c_in) if args.arch == "compact" else TinyTCN(c_in=c_in)).to(device)
    remedy = cell.get("remedy")
    if remedy is None:
        train_ds = make_ds(tr_rows, cell, cfg, train=True, seed=seed, planted=masks_trva,
                           planted_mode=pmode)
    else:                                       # v12: dataset also returns group and tracking stats
        assert args.world is not None, "v12 remedies run on the planted host"
        groups = (RM.severity_groups(tr_rows + va_rows, eps_trva, len(tr_rows))
                  if remedy == "groupdro" else None)
        if remedy == "groupdro_obs":            # v16: groups from the measured input statistic only
            groups, agree = RM.observed_groups(tr_rows + va_rows, masks_trva, pmode, len(tr_rows),
                                               oracle=RM.severity_groups(tr_rows + va_rows, eps_trva, len(tr_rows)))
            cell = {**cell, "group_agreement_with_severity_groups": agree}
            print(f"observed groups: agreement with severity groups {agree:.3f}", flush=True)
        pi = float(np.mean([r["binary_label"] for r in tr_rows]))
        train_ds = RM.RemedyDataset(tr_rows, augment=(remedy == "augment"), pi=pi, groups=groups,
                                    aug_seed=seed, window=cfg["data"]["window"], train=True,
                                    planted=masks_trva, planted_mode=pmode,
                                    flip_prob=cfg["data"]["flip_prob"], seed=seed,
                                    joint_dropout=0.0, coord_dropout=0.0, velocity=False,
                                    normalize=cfg["data"]["normalize"],
                                    valid_rule=cell.get("valid_rule", "conf"))   # v15conf: "coords"
    dl = DataLoader(train_ds, batch_size=args.batch_size or b2["train"]["batch_size"], shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=True,
                    worker_init_fn=D.worker_init if D.TRAIN_RECIPE == "v15" else None)
    params = list(model.parameters())
    adv = dro = None
    if remedy == "adversarial":                 # target standardisation from 512 training windows
        probe_rng = np.random.RandomState(seed)
        idx = probe_rng.choice(len(train_ds), min(512, len(train_ds)), replace=False)
        st_ = np.stack([train_ds[int(i)][3].numpy() for i in idx])
        adv = RM.AdversarialHead(model.fc.in_features, st_.mean(0), st_.std(0)).to(device)
        params += list(adv.parameters())
    if remedy in ("groupdro", "groupdro_obs"):
        dro = RM.GroupDRO(device=device)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    va_y = [r["binary_label"] for r in va_rows]
    best, log, t0 = (-1.0, None, -1), [], time.time()
    if repair is not None:                      # repair_selection_v14 (post hoc)
        excl = {int(c) for c in repair["val_exclude_ids"]}
        keep = np.array([int(r["child_id"]) not in excl for r in va_rows])
        va_y_arr = np.array(va_y)
        best_clean = (-1.0, None, -1)
        rep_dir = f"{repair['outputs']['epoch_dir']}/{repair_run_name(args, seed)}"
        os.makedirs(rep_dir, exist_ok=True)
        va_preds = []
        print(f"repair: validation rows {len(va_rows)}, cleaned {int(keep.sum())} "
              f"(excluded IDs {sorted(excl)})", flush=True)
    for epoch in range(epochs):
        model.train()
        tot = seen = 0.0
        for batch in dl:
            x, y = batch[0].to(device), batch[1].to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=args.amp):
                if remedy is None or remedy == "augment":
                    loss = crit(model(x), y)
                elif remedy in ("groupdro", "groupdro_obs"):
                    per = F.cross_entropy(model(x), y, reduction="none")
                    loss = dro.loss(per, batch[2].to(device))
                else:                           # adversarial: CE plus reversed stats regression
                    f = model.forward_features(x)
                    loss = crit(model.fc(f), y) + adv.loss(f.float(), batch[3].to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            seen += len(y)
        sched.step()
        va_s = run_eval(model, va_rows, cell, cfg, device, args.workers,
                        planted=masks_trva, planted_mode=pmode)[0]
        va_auc = M.auc(va_s, va_y)
        log.append({"epoch": epoch + 1, "loss": tot / seen, "val_auc": va_auc,
                    "min": (time.time() - t0) / 60})
        if repair is not None:                  # same predictions, cleaned membership; no extra eval
            va_auc_clean = M.auc(np.asarray(va_s)[keep], va_y_arr[keep])
            log[-1]["val_auc_clean"] = va_auc_clean
            va_preds.append(np.asarray(va_s, dtype=np.float32))
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       f"{rep_dir}/epoch{epoch + 1:02d}.pt")
            if va_auc_clean > best_clean[0]:    # same rule as the original: first epoch at the maximum
                best_clean = (va_auc_clean, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              epoch + 1)
        star = ""
        if va_auc > best[0]:
            best = (va_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    epoch + 1)
            star = " *"
        print(f"epoch {epoch+1:3d} loss {tot/seen:.4f} val AUC {va_auc:.4f} "
              f"({(time.time()-t0)/60:.1f} min){star}", flush=True)
        if args.max_epochs_run and epoch + 1 >= args.max_epochs_run:
            print("timing check stop", flush=True)
            return
    train_min = (time.time() - t0) / 60

    if repair is not None:
        repair_finish(args, repair, rep_dir, cell, seed, best, best_clean, log, va_rows, keep,
                      np.stack(va_preds), train_min, eps_trva, masks_trva, test_masks)
        return
    model.load_state_dict(best[1])
    if args.proto in LENGTH_PROTOS:
        if args.world:
            w = f"{args.world}_w{args.world_seed}_{args.mechanism}_{args.strength}"
        elif args.real_world:                   # v10 real_world_I: results/v10/real_I/<cell>/seed<k>
            w = f"real_{args.real_world}"
        else:
            w = "real"
        run_id = f"{args.proto}_{w}_{args.cell}_s{seed}"
        out_dir = f"results/{args.proto}{'_smoke' if args.smoke else ''}/{w}/{args.cell}/seed{seed}"
    elif args.proto == "v6":
        run_id = f"v6_{args.world}_w{args.world_seed}_{args.cell}_s{seed}"
        out_dir = (f"results/pilots/pilot_v6{'_smoke' if args.smoke else ''}/train/"
                   f"{args.world}_w{args.world_seed}/{args.cell}/seed{seed}")
    elif args.proto == "v4":
        tag = f"{args.arch}_{args.mechanism}_{args.strength}"
        run_id = f"v4_{tag}_{args.world}_{args.cell}_s{seed}"
        out_dir = (f"results/pilots/pilot_v4{'_smoke' if args.smoke else ''}/{tag}/"
                   f"{args.world}/{args.cell}/seed{seed}")
    elif args.proto == "v3" and args.world is not None:
        run_id = f"v3_{args.strength}_{args.world}_{args.cell}_s{seed}"
        out_dir = (f"results/pilots/pilot_v3{'_smoke' if args.smoke else ''}/planted_{args.strength}/"
                   f"{args.world}/{args.cell}/seed{seed}")
    elif args.proto == "v3":
        run_id = f"v3_real_{args.cell}_s{seed}"
        out_dir = f"results/pilots/pilot_v3{'_smoke' if args.smoke else ''}/real/{args.cell}/seed{seed}"
    elif args.proto == "v1":
        run_id = args.cell
        out_dir = f"results/pilots/pilot_batch2{'_smoke' if args.smoke else ''}/{args.cell}"
    elif args.permute is not None:
        run_id = f"v2_{args.cell}_perm{args.permute}"
        out_dir = f"results/pilots/pilot_v2{'_smoke' if args.smoke else ''}/perm/{args.cell}_p{args.permute}"
    else:
        run_id = f"v2_{args.cell}_s{seed}"
        out_dir = f"results/pilots/pilot_v2{'_smoke' if args.smoke else ''}/{args.cell}/seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    torch.save({"state_dict": best[1], "cell": cell, "seed": seed, "val_auc": best[0],
                "epoch": best[2]}, f"cache/pilot_{run_id}{'_smoke' if args.smoke else ''}.pt")

    if args.world is not None or args.real_world is not None:   # v3: score clean, S- and I-planted test
        for name, mk in (("clean", None), ("S", test_masks["S"]), ("I", test_masks["I"])):
            s_ = run_eval(model, te_rows, cell, cfg, device, args.workers, planted=mk,
                          planted_mode=pmode)[0]
            with open(f"{out_dir}/test_scores_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["identifier", "binary_label", "smm_score"])
                for r, v in zip(te_rows, s_):
                    w.writerow([r["identifier"], r["binary_label"], f"{v:.6f}"])
            print(f"test AUC {name} {M.auc(s_, [r['binary_label'] for r in te_rows]):.4f}", flush=True)
    te_s, te_f = run_eval(model, te_rows, cell, cfg, device, args.workers)
    with open(f"{out_dir}/test_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["identifier", "binary_label", "smm_score"])
        for r, s in zip(te_rows, te_s):
            w.writerow([r["identifier"], r["binary_label"], f"{s:.6f}"])
    print(f"test AUC {M.auc(te_s, [r['binary_label'] for r in te_rows]):.4f} "
          f"(best epoch {best[2]}, val {best[0]:.4f})", flush=True)

    if args.proto in ("v2", "v3", "v4", "v6", "v7", "v8", "v10", "v12"):   # v2 and v3 drop the probe
        json.dump({"cell": args.cell, "config": cell, "seed": seed, "permute": args.permute,
                   "best_epoch": best[2], "best_val_auc": best[0], "train_minutes": train_min,
                   "log": log, **({"groupdro_q": dro.q.tolist()} if dro is not None else {})},
                  open(f"{out_dir}/train_log.json", "w"), indent=1)
        if not args.smoke:
            with open("results/gpu_ledger.csv", "a", newline="") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d"), f"pilot_{args.proto}/{run_id}",
                                        f"{(time.time() - t0) / 3600:.3f}"])
        return

    # probe: features on (unaugmented) train children, evaluated on test children
    _, tr_f = run_eval(model, tr_rows, cell, cfg, device, args.workers)
    probe = linear_probe(tr_f, tr_rows, te_f, te_rows, cfg["tracking_stats"],
                         20 if args.smoke else args.n_boot)
    # random-init reference with the same input channels
    torch.manual_seed(0)
    ref = TinyTCN(c_in=c_in).to(device)
    _, ref_tr = run_eval(ref, tr_rows, cell, cfg, device, args.workers)
    _, ref_te = run_eval(ref, te_rows, cell, cfg, device, args.workers)
    ref_probe = linear_probe(ref_tr, tr_rows, ref_te, te_rows, cfg["tracking_stats"],
                             20 if args.smoke else args.n_boot)
    for k in probe:
        probe[k]["random_init_r2"] = ref_probe[k]["r2"]
        probe[k]["r2_minus_random"] = probe[k]["r2"] - ref_probe[k]["r2"]
    json.dump(probe, open(f"{out_dir}/probe.json", "w"), indent=1)
    json.dump({"cell": args.cell, "config": cell, "best_epoch": best[2], "best_val_auc": best[0],
               "train_minutes": train_min, "total_minutes": (time.time() - t0) / 60,
               "log": log}, open(f"{out_dir}/train_log.json", "w"), indent=1)
    print("probe R2 (trained / random init):",
          {k: f'{v["r2"]:.3f} / {v["random_init_r2"]:.3f}' for k, v in probe.items()}, flush=True)

    if not args.smoke:
        new = not os.path.exists("results/gpu_ledger.csv")
        with open("results/gpu_ledger.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "job", "gpu_hours"])
            w.writerow([time.strftime("%Y-%m-%d"), f"pilot_batch2/{args.cell}",
                        f"{(time.time() - t0) / 3600:.3f}"])


if __name__ == "__main__":
    main()
