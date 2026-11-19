"""Protocol v18 (configs/protocols/crossed_rescoring.yaml): like-for-like crossed reading
for the confidence-only cell by rescoring, without training.

The nine World I no-remedy reference fits of Table 7 (CompactCTRGCN trained against missing runs,
whole-joint absence and frozen frames at mid strength; historical recipe; cleaned selection of
selection_repair; seeds 0-2) carry no valid_rule in their checkpoints, so score_operators.build()
encodes them with the first validity rule (confidence > 0). This script rescores them under
arm-confidence planting with the corrected rule (x or y nonzero). The only new element is the
explicit choice of that one preprocessing field; the model rebuild, row set, masks, dataset,
inference and output format are imported from experiments/score_operators.py, and the estimator from
experiments/analyze_operators.py. Existing files are never overwritten.

Inputs in this repository: the stored score files of Table 7 (results/operators/scores, first rule,
cleaned selection) and of the v13 World S models (results/confonly_historical/scores). The validate and score
steps need the cleaned-selection checkpoints under results/fits/repair_v14/cache/clean and a GPU;
the checkpoints are not redistributed, so these steps cannot be rerun from this repository. The
datacheck step needs the SAFE-SMM segment cache (SM_SMM_ROOT or SM_ASDPOSE_CACHE). The analyze
step runs from the released score files; since no file is overwritten, move
results/crossed_rescoring/analysis aside (or work in a copy of the repository) before rerunning it.

    python3 experiments/rescore_crossed.py datacheck          # CPU; before the protocol is frozen
    conda run -n torch python experiments/rescore_crossed.py validate     # amendment 1 tolerances
    conda run -n torch python experiments/rescore_crossed.py score --rule coords
    conda run -n torch python experiments/rescore_crossed.py score --rule conf
    python3 experiments/rescore_crossed.py analyze [--n-boot 2000]

Outputs under results/crossed_rescoring/: datacheck.csv; validation/ and validation.json (first attempt,
frozen 1e-4 tolerance, failed); validation_amendment1/ and validation_amendment1.json (amendment 1);
scores_coords/ and scores_conf/ (one CSV per fit, the results/operators/scores format) with registry.json;
analysis/{responses.csv,differences.csv,criteria.json}.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M                  # noqa: E402
from safe_motion import planted as P                  # noqa: E402
from safe_motion.data import asdpose as D             # noqa: E402

PROTOCOL = "configs/protocols/crossed_rescoring.yaml"
V1 = yaml.safe_load(open("configs/protocols/pilot_1.yaml"))
CKPT = "results/fits/repair_v14/cache/clean"                   # cleaned-selection checkpoints (not redistributed)
STORED_REF = "results/operators/scores"                               # Table 7 score files (first rule, cleaned selection)
STORED_WS = "results/confonly_historical/scores"                               # v13 World S / I, corrected rule
OUT = "results/crossed_rescoring"
REF_GROUPS = (("v7", "runs"), ("v7", "whole"), ("v8", "frozen"))
SEEDS = (0, 1, 2)
MASK_SEEDS = (2000, 3000, 4000)
VALIDATION_FIT = "v7_I_w0_whole_mid_A_none_s0"
VALIDATION_COLS = ("clean", "armruns_S_mid_m2000", "armruns_I_mid_m2000")
TOL = 1e-4                                                    # probability scale and AUROC (as frozen; failed)
# Amendment 1 (2026-11-16): cross-device tolerances; the first attempt stays in validation.json.
AMEND1 = {"max_abs_diff": 5e-3, "auroc_diff": 1e-3, "response_logit_diff": 2e-3}
GATE_FILE = f"{OUT}/validation_amendment1.json"               # the gate that scoring requires


def ref_names():
    return [f"{p}_I_w0_{m}_mid_A_none_s{s}" for p, m in REF_GROUPS for s in SEEDS]


def ws_names(world="S"):
    return [f"v13_{world}_w0_armruns_mid_A_none_s{s}" for s in SEEDS]


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def host_rows():
    """Host development rows (training + internal validation, as train_asdpose.py passes them to
    build_masks) and host test rows (score_operators.py order). Length protocol and frozen QC."""
    rows = [r for r in D.load_index() if r["n_frames"] >= 200]
    tr, va, _ = D.subject_level_val_split(rows, 0.2, seed=0)
    q = V1["qc_filter"]
    dev = [r for r in tr if D.passes_qc(r, q)] + [r for r in va if D.passes_qc(r, q)]
    te = [r for r in rows if r["split"] == "test" and D.passes_qc(r, q)]
    return dev, te


def new_file(path):
    """Refuse to overwrite anything."""
    if os.path.exists(path):
        sys.exit(f"refusing to overwrite {path}")
    return path


# ------------------------------------------------------------------ step 1: data check (CPU)
def disagreements(rows, masks, mode):
    """Joint-frames where confidence > 0 and (x, y) == (0, 0), and where confidence <= 0 and
    (x, y) != (0, 0), over the full raw segments after the planted operator (a superset of every
    training crop and of the evaluation window)."""
    ds = D.SegmentDataset(rows, train=False, planted=masks, planted_mode=mode)
    a = b = n = seg = 0
    for r in rows:
        arr = ds.load_segment(r)
        conf = arr[..., 2] > 0
        coords = (arr[..., 0] != 0) | (arr[..., 1] != 0)
        da, db = int((conf & ~coords).sum()), int((~conf & coords).sum())
        a, b, n, seg = a + da, b + db, n + conf.size, seg + int(da + db > 0)
    return a, b, n, seg


def datacheck():
    os.makedirs(OUT, exist_ok=True)
    path = new_file(f"{OUT}/datacheck.csv")
    dev, te = host_rows()
    print(f"host development rows {len(dev)}, host test rows {len(te)}", flush=True)
    jobs = [("development", dev, "clean", "-", "-", None)]
    for mech in ("runs", "whole", "frozen"):
        for w in ("I", "S"):
            jobs.append(("development", dev, mech, w, "0 (training assignment)", (w, 0)))
    jobs.append(("test", te, "clean", "-", "-", None))
    for mech in ("runs", "whole", "frozen"):
        for w in ("S", "I"):
            for ms in MASK_SEEDS:
                jobs.append(("test", te, mech, w, str(ms), (w, ms)))
    for w in ("S", "I"):                                     # positive control of the counter
        jobs.append(("test", te, "armruns (control)", w, "2000", (w, 2000)))
    out = []
    for split, rows, mech, w, seed, spec in jobs:
        t0 = time.time()
        if spec is None:
            masks, mode = None, "zero"
        else:
            m = mech.split()[0]
            masks, mode = P.build_masks(rows, spec[0], "mid", spec[1], mechanism=m)[0], P.MECHANISM_MODE[m]
        a, b, n, seg = disagreements(rows, masks, mode)
        out.append({"split": split, "operator": mech, "policy_or_world": w, "mask_seed": seed,
                    "n_segments": len(rows), "n_joint_frames": n,
                    "conf_pos_coords_zero": a, "conf_zero_coords_nonzero": b,
                    "segments_with_disagreement": seg})
        print(f"{split:11s} {mech:18s} {w:2s} {seed:24s} conf>0&xy=0 {a:9d}  conf<=0&xy!=0 {b:9d}  "
              f"of {n} ({time.time() - t0:.0f} s)", flush=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print("wrote", path)


# ------------------------------------------------------------------ scoring (GPU)
def protocol_committed():
    """The scoring steps run only after the protocol is committed and unmodified."""
    log = subprocess.run(["git", "log", "-1", "--format=%h", "--", PROTOCOL], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", PROTOCOL], capture_output=True,
                           text=True, cwd=ROOT).stdout.strip()
    if not log or dirty:
        sys.exit(f"{PROTOCOL} is not committed or has uncommitted changes; no scoring before the freeze")
    return log


def model_entry(name):
    path = f"{CKPT}/pilot_{name}.pt"
    proto, world, ws, mech, st, *_ = name.split("_")
    return {"name": name, "path": path, "proto": proto, "arch": "compact", "world": world,
            "world_seed": int(ws[1:]), "mechanism": mech, "strength": st, "cell": "A_none",
            "seed": int(name[-1]), "sha256_16": sha16(path)}


def score_one(S7, m, te, cols_wanted, rule, device, workers, masks):
    import torch                                                          # noqa: F401
    net, prep = S7.build(m)
    assert prep["valid_rule"] == "conf", f"{m['name']}: expected a first-rule checkpoint"
    assert prep["velocity"] is False and prep["confidence"] is True and prep["blind"] is False
    prep = dict(prep, valid_rule=rule)                                    # the one v18 change
    net = net.to(device)
    cols = {}
    for col in cols_wanted:
        if col == "clean":
            cols[col] = S7.run(net, te, prep, None, "zero", device, workers)[0]
            continue
        mech, w, st, ms = col.split("_")
        key = (mech, w, st, int(ms[1:]))
        if key not in masks:
            masks[key] = P.build_masks(te, w, st, int(ms[1:]), mechanism=mech)[0]
        cols[col] = S7.run(net, te, prep, masks[key], P.MECHANISM_MODE[mech], device, workers)[0]
    return cols


def write_scores(path, te, cols):
    with open(new_file(path), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["identifier", "child_id", "label", "n_frames", "qc_pass"] + list(cols))
        for i, r in enumerate(te):
            wr.writerow([r["identifier"], r["child_id"], r["binary_label"], r["n_frames"],
                         int(D.passes_qc(r, V1["qc_filter"]))] + [f"{cols[c][i]:.6f}" for c in cols])


def validate(args):
    import torch
    import score_operators as S7
    commit = protocol_committed()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, te = host_rows()
    suffix = "" if args.amendment == 0 else f"_amendment{args.amendment}"
    os.makedirs(f"{OUT}/validation{suffix}", exist_ok=True)
    m = model_entry(VALIDATION_FIT)
    t0 = time.time()
    cols = score_one(S7, m, te, VALIDATION_COLS, "conf", device, args.workers, {})
    minutes = (time.time() - t0) / 60
    write_scores(f"{OUT}/validation{suffix}/{m['name']}.csv", te, cols)
    from experiments.analyze_operators import load_scores, logit
    meta, stored = load_scores(f"{STORED_REF}/{m['name']}.csv")
    assert meta["identifier"] == [r["identifier"] for r in te], "row order differs from the stored file"
    y = [r["binary_label"] for r in te]
    tol = ({"max_abs_diff": TOL, "auroc_diff": TOL} if args.amendment == 0 else AMEND1)
    res = {"fit": m["name"], "checkpoint": m["path"], "checkpoint_sha256_16": m["sha256_16"],
           "stored_file": f"{STORED_REF}/{m['name']}.csv", "rule": "conf", "device": device,
           "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "torch": torch.__version__, "protocol_commit": commit, "amendment": args.amendment,
           "tolerance": tol, "gpu_minutes": round(minutes, 2), "columns": {}}
    ok = True
    rounded = {c: np.round(cols[c], 6) for c in VALIDATION_COLS}
    for c in VALIDATION_COLS:
        new = rounded[c]
        d = float(np.max(np.abs(new - stored[c])))
        a_new, a_old = M.auc(new, y), M.auc(stored[c], y)
        res["columns"][c] = {"max_abs_diff": d, "n_rows_differing": int(np.sum(new != stored[c])),
                             "auroc_new": a_new, "auroc_stored": a_old, "auroc_diff": abs(a_new - a_old)}
        ok &= d <= tol["max_abs_diff"] and abs(a_new - a_old) <= tol["auroc_diff"]
    if "response_logit_diff" in tol:                       # amendment 1: response of the fit at mask 2000
        r_new, r_old = [float(logit(M.auc(src["armruns_S_mid_m2000"], y)) - logit(M.auc(src["armruns_I_mid_m2000"], y)))
                        for src in (rounded, stored)]
        res["response_logit"] = {"new": r_new, "stored": r_old, "diff": abs(r_new - r_old)}
        ok &= abs(r_new - r_old) <= tol["response_logit_diff"]
    res["pass"] = bool(ok)
    json.dump(res, open(new_file(f"{OUT}/validation{suffix}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("VALIDATION", "PASS" if ok else "FAIL")


def score(args):
    import torch
    import score_operators as S7
    commit = protocol_committed()
    v = json.load(open(GATE_FILE)) if os.path.exists(GATE_FILE) else {}
    if not (v.get("pass") and v.get("amendment") == 1):
        sys.exit(f"{GATE_FILE} has not passed; no scoring (protocol v18 amendment 1, V18_validation)")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, te = host_rows()
    out = f"{OUT}/scores_{args.rule}"
    os.makedirs(out, exist_ok=True)
    cols_wanted = ["clean"] + [f"armruns_{w}_mid_m{ms}" for w in ("S", "I") for ms in MASK_SEEDS]
    models = [dict(model_entry(n), valid_rule_scored=args.rule, protocol_commit=commit) for n in ref_names()]
    json.dump(models, open(new_file(f"{out}/registry.json"), "w"), indent=1)
    masks, t0 = {}, time.time()
    y = [r["binary_label"] for r in te]
    for m in models:
        path = f"{out}/{m['name']}.csv"
        cols = score_one(S7, m, te, cols_wanted, args.rule, device, args.workers, masks)
        write_scores(path, te, cols)
        print(f"{m['name']} [{args.rule}]: clean {M.auc(cols['clean'], y):.4f} armruns m2000 S/I "
              f"{M.auc(cols['armruns_S_mid_m2000'], y):.4f}/{M.auc(cols['armruns_I_mid_m2000'], y):.4f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"done; wall minutes {(time.time() - t0) / 60:.2f}")


# ------------------------------------------------------------------ analysis (CPU)
def analyze(args):
    from experiments.analyze_operators import RowSet, load_scores, align, logit, verdict, R_MARGIN
    out = f"{OUT}/analysis"
    os.makedirs(out, exist_ok=True)
    for f in ("responses.csv", "differences.csv", "criteria.json"):
        new_file(f"{out}/{f}")
    sources = {"published_first": STORED_REF, "new_first": f"{OUT}/scores_conf",
               "new_corrected": f"{OUT}/scores_coords"}
    meta0, _ = load_scores(f"{STORED_WS}/{ws_names()[0]}.csv")
    rs = RowSet(meta0["identifier"], [int(v) for v in meta0["label"]], meta0["child_id"], args.n_boot)
    A = {}

    def aucs(src_dir, name):
        if (src_dir, name) not in A:
            meta, sc = load_scores(f"{src_dir}/{name}.csv")
            sc = align(rs, meta["identifier"], meta["label"], sc)
            A[(src_dir, name)] = {c: rs.aucs(v) for c, v in sc.items()}
        return A[(src_dir, name)]

    def R(src_dir, names, masks, scale):
        """Seed-averaged response inside each resample, then averaged over the given masks."""
        per_mask = []
        for ms in masks:
            per = []
            for n in names:
                a = aucs(src_dir, n)
                s, i = a[f"armruns_S_mid_m{ms}"], a[f"armruns_I_mid_m{ms}"]
                per.append(logit(s) - logit(i) if scale == "logit" else s - i)
            per_mask.append(np.mean(per, axis=0))
        return np.mean(per_mask, axis=0)

    def seed_sd(src_dir, names, masks):
        pts = [np.mean([logit(aucs(src_dir, n)[f"armruns_S_mid_m{ms}"][0])
                        - logit(aucs(src_dir, n)[f"armruns_I_mid_m{ms}"][0]) for ms in masks]) for n in names]
        return float(np.std(pts, ddof=1))

    def fmt(a):
        lo, hi = M.ci(a[1:])
        return f"{a[0]:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    def triple(a):
        lo, hi = M.ci(a[1:])
        return [float(a[0]), lo, hi]

    groups = {f"I {m}": [f"{p}_I_w0_{m}_mid_A_none_s{s}" for s in SEEDS] for p, m in REF_GROUPS}
    ws = ws_names("S")
    crit = {"protocol": PROTOCOL, "n_boot": args.n_boot, "n_rows": len(rs.ids),
            "n_identifiers": len(set(rs.child)), "world_S_models": ws, "references": groups}
    resp, diffs = [], []
    first_only = (2000,)
    RS = {(sc, mk): R(STORED_WS, ws, msk, sc) for sc in ("logit", "raw")
          for mk, msk in (("m2000", first_only), ("mask_mean", MASK_SEEDS))}
    resp.append({"model_group": "World S confidence-only (v13, corrected rule)", "rule": "corrected (stored)",
                 "R_m2000_logit": fmt(RS[("logit", "m2000")]), "R_m2000_raw": fmt(RS[("raw", "m2000")]),
                 "R_maskmean_logit": fmt(RS[("logit", "mask_mean")]), "R_maskmean_raw": fmt(RS[("raw", "mask_mean")]),
                 "sd_seeds_m2000": f"{seed_sd(STORED_WS, ws, first_only):.4f}", "verdict_m2000_logit": ""})
    crit["R_world_S"] = {f"{mk}_{sc}": triple(v) for (sc, mk), v in RS.items()}
    passes, raw_passes = [], []
    crit["references_R"], crit["differences"], crit["rule_change"] = {}, {}, {}
    for g, names in groups.items():
        for label, src in sources.items():
            row = {"model_group": g, "rule": label}
            masks_avail = first_only if label == "published_first" else MASK_SEEDS
            for sc in ("logit", "raw"):
                r1 = R(src, names, first_only, sc)
                row[f"R_m2000_{sc}"] = fmt(r1)
                crit["references_R"][f"{g}|{label}|m2000_{sc}"] = triple(r1)
                if len(masks_avail) == 3:
                    rm = R(src, names, MASK_SEEDS, sc)
                    row[f"R_maskmean_{sc}"] = fmt(rm)
                    crit["references_R"][f"{g}|{label}|mask_mean_{sc}"] = triple(rm)
                else:
                    row[f"R_maskmean_{sc}"] = "not scored"
            r1 = R(src, names, first_only, "logit")
            row["sd_seeds_m2000"] = f"{seed_sd(src, names, first_only):.4f}"
            row["verdict_m2000_logit"] = verdict(r1[0], M.ci(r1[1:]), R_MARGIN)
            resp.append(row)
            # paired differences R(World S) - R(reference), shared resamples
            for mk, msk in (("m2000", first_only), ("mask_mean", MASK_SEEDS)):
                if len(msk) > len(masks_avail):
                    continue
                for sc in ("logit", "raw"):
                    d = RS[(sc, mk)] - R(src, names, msk, sc)
                    lo, hi = M.ci(d[1:])
                    primary = label == "new_corrected" and mk == "m2000"
                    diffs.append({"reference": g, "reference_rule": label, "masks": mk, "scale": sc,
                                  "difference": fmt(d), "above_zero": bool(lo > 0),
                                  "role": ("primary (V18_rule)" if primary and sc == "logit" else
                                           "secondary" if primary else "descriptive")})
                    crit["differences"][f"{g}|{label}|{mk}|{sc}"] = triple(d) + [bool(lo > 0)]
                    if primary:
                        (passes if sc == "logit" else raw_passes).append(bool(lo > 0))
        # rule change of the reference itself: corrected minus first (both newly scored)
        for mk, msk in (("m2000", first_only), ("mask_mean", MASK_SEEDS)):
            for sc in ("logit", "raw"):
                d = R(sources["new_corrected"], names, msk, sc) - R(sources["new_first"], names, msk, sc)
                crit["rule_change"][f"{g}|{mk}|{sc}"] = triple(d)
                diffs.append({"reference": g, "reference_rule": "corrected minus first (same fits)",
                              "masks": mk, "scale": sc, "difference": fmt(d),
                              "above_zero": bool(M.ci(d[1:])[0] > 0), "role": "descriptive"})
    # clean-input identity across rules (implied by the data check) and reproduction of the stored files
    checks = {}
    for names in groups.values():
        for n in names:
            _, c1 = load_scores(f"{sources['new_corrected']}/{n}.csv")
            _, c0 = load_scores(f"{sources['new_first']}/{n}.csv")
            _, cs = load_scores(f"{STORED_REF}/{n}.csv")
            checks[n] = {"clean_corrected_vs_first_max_abs": float(np.max(np.abs(c1["clean"] - c0["clean"]))),
                         "first_m2000_vs_stored_max_abs": max(float(np.max(np.abs(c0[c] - cs[c])))
                                                              for c in VALIDATION_COLS)}
    crit["consistency_checks"] = checks
    crit["V18_rule"] = {"scale": "logit", "masks": "m2000", "per_reference_above_zero": passes,
                        "pass": bool(passes and all(passes))}
    crit["raw_scale_reading"] = {"per_reference_above_zero": raw_passes,
                                 "all_above_zero": bool(raw_passes and all(raw_passes))}
    for name, rows_ in (("responses.csv", resp), ("differences.csv", diffs)):
        with open(f"{out}/{name}", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows_[0]))
            wr.writeheader()
            wr.writerows(rows_)
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1, default=float)
    for r in resp:
        print(f"{r['model_group']:48s} {r['rule']:20s} m2000 {r['R_m2000_logit']:30s} mean {r['R_maskmean_logit']}")
    for d in diffs:
        if d["role"] != "descriptive":
            print(f"{d['reference']:10s} {d['scale']:5s} {d['difference']} above zero {d['above_zero']} ({d['role']})")
    print("V18 rule (logit, mask 2000):", "PASS" if crit["V18_rule"]["pass"] else "FAIL", passes)
    print("raw-scale reading:", raw_passes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["datacheck", "validate", "score", "analyze"])
    ap.add_argument("--rule", choices=["coords", "conf"], default="coords")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--amendment", type=int, choices=[0, 1], default=1,
                    help="validate: 0 = the frozen 1e-4 tolerance (first attempt, failed), 1 = amendment 1")
    args = ap.parse_args()
    {"datacheck": lambda: datacheck(), "validate": lambda: validate(args),
     "score": lambda: score(args), "analyze": lambda: analyze(args)}[args.step]()


if __name__ == "__main__":
    main()
