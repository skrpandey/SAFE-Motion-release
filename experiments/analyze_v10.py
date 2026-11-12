"""Analysis of protocol v10 (configs/protocols/identification_v10.yaml).

Part A: SE of the quality-blind control A_blind against v7 A_none on shared child resamples,
their ratio clipped to [0, 2], and rules A1 and A2. Part B: v7 real models against their real
World I counterparts under each operator and gapfill. Part C: mask stability of SE on v9, with a
seed / mask / seed x mask / subject variance decomposition.
Writes results/v10/analysis/{blind,real_I,masks,masks_by_seed}.csv and criteria.json.
    conda run -n torch python experiments/analyze_v10.py [--n-boot 2000] [--skip C]
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_motion import metrics as M                                   # noqa: E402
from analyze_v7 import logit, verdict, RowSet, align, load_scores, R_MARGIN   # noqa: E402

MECHS = ("runs", "whole", "frozen", "armruns")
BLIND_MECHS = ("runs", "whole")
MASK_SEEDS_V7 = (2000, 3000, 4000)
CELLS = ("A_none", "A_velocity", "A_dropout", "A_both", "B_coorddrop")
SEEDS = (0, 1, 2)
DIR_RE = re.compile(r"^(?P<host>[^/]+)/(?P<mech>runs|whole|frozen|armruns)_(?P<st>mid|low)/(?P<world>S|I)"
                    r"/A_none/seed(?P<seed>\d)/fold(?P<fold>\d+)$")


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(fmt(x) for x in v) + ")"
    return v


def write_csv(path, rows):
    keys = []
    for r in rows:                                   # union of keys, first-seen order
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys or ["empty"])
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


def rec(contrast, arr, margin=R_MARGIN, **extra):
    ci = M.ci(arr[1:])
    d = {"contrast": contrast, "point": float(arr[0]), "ci_lo": ci[0], "ci_hi": ci[1],
         "verdict": verdict(arr[0], ci, margin)}
    d.update(extra)
    return d


class Scores:
    """Score files of one row set (host or full) aligned to shared resamples."""

    def __init__(self, n_boot):
        self.n_boot, self.rs, self.S, self.A = n_boot, None, {}, {}

    def add(self, name, path):
        if not os.path.exists(path):
            return False
        meta, cols = load_scores(path)
        if self.rs is None:
            self.rs = RowSet(meta["identifier"], [int(v) for v in meta["label"]], meta["child_id"], self.n_boot)
        self.S[name] = align(self.rs, meta["identifier"], meta["label"], cols)
        return True

    def auc(self, name, col):
        key = (name, col)
        if key not in self.A:
            self.A[key] = self.rs.aucs(self.S[name][col])
        return self.A[key]

    def R(self, name, mech, st="mid", ms=2000):
        cS, cI = f"{mech}_S_{st}_m{ms}", f"{mech}_I_{st}_m{ms}"
        if cS not in self.S[name] or cI not in self.S[name]:
            return None
        return logit(self.auc(name, cS)) - logit(self.auc(name, cI))

    def R_group(self, names, mech, st="mid", ms=2000):
        per = [r for r in (self.R(n, mech, st, ms) for n in names if n in self.S) if r is not None]
        return (np.mean(per, axis=0), float(np.std([p[0] for p in per], ddof=1))) if per else (None, None)

    def clean_group(self, names):
        per = [self.auc(n, "clean") for n in names if n in self.S]
        return np.mean(per, axis=0) if per else None


# part A
def part_a(v7_dir, v10_dir, n_boot, crit):
    sc, rows = Scores(n_boot), []
    for mech in BLIND_MECHS:
        for w in ("S", "I"):
            for k in SEEDS:
                sc.add(f"none_{mech}_{w}_{k}", f"{v7_dir}/v7_{w}_w0_{mech}_mid_A_none_s{k}.csv")
                sc.add(f"blind_{mech}_{w}_{k}", f"{v10_dir}/v10_{w}_w0_{mech}_mid_A_blind_s{k}.csv")
    crit["A1_cue_attributable"], crit["A2_transform_check"] = {}, {}
    for mech in BLIND_MECHS:
        g = {(c, w): [f"{c}_{mech}_{w}_{k}" for k in SEEDS] for c in ("none", "blind") for w in ("S", "I")}
        if not any(n in sc.S for n in g[("blind", "S")]):
            continue
        se, masks = {}, {}
        for c in ("none", "blind"):
            rS, sdS = sc.R_group(g[(c, "S")], mech)
            rI, sdI = sc.R_group(g[(c, "I")], mech)
            if rS is None or rI is None:
                se[c] = None
                continue
            se[c] = {"arr": rS - rI, "R_S": rS, "R_I": rI, "sd_S": sdS, "sd_I": sdI}
            masks[c] = [float((sc.R_group(g[(c, "S")], mech, "mid", ms)[0] - sc.R_group(g[(c, "I")], mech, "mid", ms)[0])[0])
                        for ms in MASK_SEEDS_V7 if sc.R_group(g[(c, "S")], mech, "mid", ms)[0] is not None]
            cl = {w: sc.clean_group(g[(c, w)]) for w in ("S", "I")}
            rows.append(rec(f"SE({'A_none' if c == 'none' else 'A_blind'})", se[c]["arr"], mechanism=mech,
                            R_S=float(rS[0]), R_I=float(rI[0]), R_I_ci=M.ci(rI[1:]), sd_seeds_S=sdS, sd_seeds_I=sdI,
                            mask_range=(max(masks[c]) - min(masks[c])) if masks[c] else None,
                            clean_S=float(cl["S"][0]) if cl["S"] is not None else None,
                            clean_I=float(cl["I"][0]) if cl["I"] is not None else None))
        if se.get("none") is None or se.get("blind") is None:
            continue
        diff = se["none"]["arr"] - se["blind"]["arr"]
        # the interpolated control is not blind to planted severity, so these are descriptive
        # contrasts; the key "A1_cue_attributable" is kept because downstream readers use it
        rows.append(rec("SE(A_none) - SE(A_blind), standard minus interpolated interaction", diff, mechanism=mech))
        ratio = np.clip(se["blind"]["arr"] / np.where(np.abs(se["none"]["arr"]) < 1e-9, 1e-9, se["none"]["arr"]), 0, 2)
        rows.append(rec("SE(A_blind) / SE(A_none), interpolated/standard interaction ratio", ratio, mechanism=mech))
        ci_b = M.ci(se["blind"]["arr"][1:])
        half = 0.5 * float(se["none"]["arr"][0])
        crit["A1_cue_attributable"][mech] = {
            "SE_blind": [float(se["blind"]["arr"][0]), ci_b[0], ci_b[1]],
            "SE_none": float(se["none"]["arr"][0]), "threshold_half_SE_none": half,
            "share": [float(ratio[0]), *M.ci(ratio[1:])], "pass": bool(ci_b[1] < half)}
        a2 = {}
        for w in ("S", "I"):
            names = [n for n in g[("blind", w)] if n in sc.S and "gapfill_S_mid_m2000" in sc.S[n]]
            if not names:
                continue
            r = np.mean([logit(sc.auc(n, "gapfill_S_mid_m2000")) - logit(sc.auc(n, "gapfill_I_mid_m2000")) for n in names], axis=0)
            ci = M.ci(r[1:])
            rows.append(rec(f"R(A_blind trained in {w}) under gapfill", r, mechanism=mech))
            a2[w] = {"R": float(r[0]), "ci": ci, "contains_zero": bool(ci[0] <= 0 <= ci[1])}
        crit["A2_transform_check"][mech] = {"per_world": a2, "pass": bool(a2) and all(v["contains_zero"] for v in a2.values())}
    return rows


# part B
def part_b(v7_dir, v10_dir, n_boot, crit):
    sc, rows = Scores(n_boot), []
    for cell in CELLS:
        for k in SEEDS:
            sc.add(f"v7_{cell}_{k}", f"{v7_dir}/v7_real_{cell}_s{k}.csv")
            sc.add(f"v10_{cell}_{k}", f"{v10_dir}/v10_real_I_{cell}_s{k}.csv")
    summary = {}
    for cell in CELLS:
        g7, g10 = [f"v7_{cell}_{k}" for k in SEEDS], [f"v10_{cell}_{k}" for k in SEEDS]
        if not any(n in sc.S for n in g10) or not any(n in sc.S for n in g7):
            continue
        c7, c10 = sc.clean_group(g7), sc.clean_group(g10)
        rows.append(rec("AUC_clean(v7 real) - AUC_clean(real World I)", c7 - c10, margin=0.02, cell=cell,
                        operator="clean", clean_v7=float(c7[0]), clean_real_I=float(c10[0])))
        summary[cell] = {}
        for mech in list(MECHS) + ["gapfill"]:
            r7, sd7 = sc.R_group(g7, mech)
            r10, sd10 = sc.R_group(g10, mech)
            if r7 is None or r10 is None:
                continue
            rows.append(rec("R(v7 real model)", r7, cell=cell, operator=mech, sd_seeds=sd7))
            masks = [float(sc.R_group(g10, mech, "mid", ms)[0][0]) for ms in MASK_SEEDS_V7
                     if sc.R_group(g10, mech, "mid", ms)[0] is not None]
            rows.append(rec("R(real World I reference)", r10, cell=cell, operator=mech, sd_seeds=sd10,
                            mask_range=(max(masks) - min(masks)) if len(masks) > 1 else None))
            d = r7 - r10
            rows.append(rec("R(v7 real) - R(real World I)", d, cell=cell, operator=mech))
            ci = M.ci(d[1:])
            summary[cell][mech] = {"R_v7": float(r7[0]), "R_ref": float(r10[0]), "diff": [float(d[0]), ci[0], ci[1]],
                                   "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    crit["B_real_world_I"] = summary
    return rows


# part C
def discover_masks(base):
    models = defaultdict(list)
    for p in sorted(glob.glob(f"{base}/*/*/*/A_none/seed*/fold*/test_scores.csv")):
        m = DIR_RE.match(os.path.relpath(os.path.dirname(p), base))
        if m:
            g = m.groupdict()
            models[(g["host"], g["mech"], g["st"], g["world"], int(g["seed"]))].append(os.path.dirname(p))
    return models


def pool(dirs):
    meta, cols = defaultdict(list), defaultdict(list)
    for d in dirs:
        for r in csv.DictReader(open(f"{d}/test_scores.csv")):
            for k in ("identifier", "subject_id", "label"):
                meta[k].append(r[k])
            for k, v in r.items():
                if k not in ("identifier", "subject_id", "label", "n_frames", "qc_pass") and not k.startswith("q_"):
                    cols[k].append(float(v))
    return dict(meta), {k: np.array(v) for k, v in cols.items()}


def part_c(masks_dir, n_boot, crit):
    models = discover_masks(masks_dir)
    rows, by_seed = [], []
    hosts = sorted({k[0] for k in models})
    for host in hosts:
        keys = [k for k in models if k[0] == host]
        rowsets, S, A = None, {}, {}
        for key in keys:
            meta, cols = pool(models[key])
            if rowsets is None:
                rowsets = RowSet(meta["identifier"], [int(v) for v in meta["label"]], meta["subject_id"], n_boot)
            if set(meta["identifier"]) != set(rowsets.ids):
                continue                                   # incomplete model: skip
            S[key] = align(rowsets, meta["identifier"], meta["label"], cols)

        def R(key, ms):
            _, mech, st, _, _ = key
            cS, cI = f"{mech}_S_{st}_m{ms}", f"{mech}_I_{st}_m{ms}"
            if cS not in S.get(key, {}) or cI not in S[key]:
                return None
            for c in (cS, cI):
                if (key, c) not in A:
                    A[(key, c)] = rowsets.aucs(S[key][c])
            return logit(A[(key, cS)]) - logit(A[(key, cI)])

        for (mech, st) in sorted({(k[1], k[2]) for k in S}):
            seeds = sorted({k[4] for k in S if k[1] == mech and k[2] == st and k[3] == "S"}
                           & {k[4] for k in S if k[1] == mech and k[2] == st and k[3] == "I"})
            mss = sorted({int(c.rsplit("_m", 1)[1]) for k in S if k[1] == mech and k[2] == st
                          for c in S[k] if c.startswith(f"{mech}_S_{st}_m")})
            if not seeds or not mss:
                continue
            grid = {}                                       # (seed, ms) -> SE array
            RS, RI = {}, {}
            for k in seeds:
                for ms in mss:
                    rS, rI = R((host, mech, st, "S", k), ms), R((host, mech, st, "I", k), ms)
                    if rS is None or rI is None:
                        continue
                    grid[(k, ms)] = rS - rI
                    RS[(k, ms)], RI[(k, ms)] = rS, rI
            if not grid:
                continue
            se_ms = {ms: np.mean([grid[(k, ms)] for k in seeds if (k, ms) in grid], axis=0) for ms in mss}
            rs_ms = {ms: np.mean([RS[(k, ms)] for k in seeds if (k, ms) in RS], axis=0) for ms in mss}
            ri_ms = {ms: np.mean([RI[(k, ms)] for k in seeds if (k, ms) in RI], axis=0) for ms in mss}
            se_mean = np.mean([se_ms[ms] for ms in mss], axis=0)
            rs_mean, ri_mean = np.mean([rs_ms[ms] for ms in mss], axis=0), np.mean([ri_ms[ms] for ms in mss], axis=0)
            pts = np.array([[grid[(k, ms)][0] if (k, ms) in grid else np.nan for ms in mss] for k in seeds])
            tot = float(np.nanvar(pts))
            v_seed = float(np.nanvar(np.nanmean(pts, axis=1)))
            v_mask = float(np.nanvar(np.nanmean(pts, axis=0)))
            v_res = max(tot - v_seed - v_mask, 0.0)
            v_subj = float(np.var(se_mean[1:]))
            denom = v_seed + v_mask + v_res + v_subj
            for ms in mss:
                ci = M.ci(se_ms[ms][1:])
                by_seed.append({"host": host, "mechanism": mech, "strength": st, "mask_seed": ms,
                                "SE": float(se_ms[ms][0]), "ci_lo": ci[0], "ci_hi": ci[1],
                                "R_S": float(rs_ms[ms][0]), "R_I": float(ri_ms[ms][0]), "n_seeds": len(seeds)})
            ci_m = M.ci(se_mean[1:])
            single = se_ms[2000][0] if 2000 in se_ms else None
            rows.append({"host": host, "mechanism": mech, "strength": st, "n_masks": len(mss), "n_seeds": len(seeds),
                         "SE_v9_mask2000": float(single) if single is not None else None,
                         "SE_mean": float(se_mean[0]), "ci_lo": ci_m[0], "ci_hi": ci_m[1],
                         "SE_between_mask_sd": float(np.std([se_ms[ms][0] for ms in mss])),
                         "SE_min": float(min(se_ms[ms][0] for ms in mss)), "SE_max": float(max(se_ms[ms][0] for ms in mss)),
                         "R_S_mean": float(rs_mean[0]), "R_S_between_mask_sd": float(np.std([rs_ms[ms][0] for ms in mss])),
                         "R_I_mean": float(ri_mean[0]), "R_I_between_mask_sd": float(np.std([ri_ms[ms][0] for ms in mss])),
                         "var_seed": v_seed, "var_mask": v_mask, "var_seed_x_mask": v_res, "var_subject": v_subj,
                         "share_seed": v_seed / denom if denom else None, "share_mask": v_mask / denom if denom else None,
                         "share_seed_x_mask": v_res / denom if denom else None,
                         "share_subject": v_subj / denom if denom else None,
                         "verdict_SE_mean": verdict(se_mean[0], ci_m, R_MARGIN)})
    crit["C_mask_stability"] = {f"{r['host']}/{r['mechanism']}/{r['strength']}": {
        "SE_mean": [r["SE_mean"], r["ci_lo"], r["ci_hi"]], "between_mask_sd": r["SE_between_mask_sd"],
        "range": [r["SE_min"], r["SE_max"]], "share_mask": r["share_mask"], "share_subject": r["share_subject"],
        "sentence": (f"SE on {r['host']} under {r['mechanism']} ({r['strength']}) moves by "
                     f"{r['SE_max'] - r['SE_min']:.2f} between mask realizations (SD {r['SE_between_mask_sd']:.2f}); "
                     f"the v9 verdict was made on one (SE {r['SE_v9_mask2000']})")} for r in rows}
    return rows, by_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7-scores", default="results/v7/scores")
    ap.add_argument("--v10-scores", default="results/v10/scores")
    ap.add_argument("--masks", default="results/v10/masks")
    ap.add_argument("--out", default="results/v10/analysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip", nargs="*", default=[], choices=["A", "B", "C"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    crit = {"n_boot": args.n_boot}
    if "A" not in args.skip:
        write_csv(f"{args.out}/blind.csv", part_a(args.v7_scores, args.v10_scores, args.n_boot, crit))
    if "B" not in args.skip:
        write_csv(f"{args.out}/real_I.csv", part_b(args.v7_scores, args.v10_scores, args.n_boot, crit))
    if "C" not in args.skip:
        rows, by_seed = part_c(args.masks, args.n_boot, crit)
        write_csv(f"{args.out}/masks.csv", rows)
        write_csv(f"{args.out}/masks_by_seed.csv", by_seed)
    json.dump(crit, open(f"{args.out}/criteria.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in crit.items() if k.startswith(("A1", "A2"))}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
