"""Protocol v12 analysis (configs/protocols/remedies_v12.yaml): cue-targeted remedies.

SE(cell) and SE(cell) - SE(A_none) on the test-mask mean (masks 2000/3000/4000, mask mean and
seed mean taken inside each child resample), logit and raw-AUC scales, paired with the v7 runs
and v8 frozen A_none scores in results/v7/scores. P12_1: the contrast lowers SE iff its interval
lies below zero on both scales. P12_2: clean-AUC change per world. P12_3: C_augment World I
responses under the three operators it was not planted with (mask 2000).
Runs locally (needs the v7 score files) and writes results/v12/analysis/{effects.csv,criteria.json}:
    conda run -n torch python experiments/analyze_v12.py [--n-boot 2000]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                        # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, p_inversion  # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                         # noqa: E402

CELLS = ("C_groupdro", "C_adversarial", "C_augment")
MECH_PROTO = {"runs": "v7", "frozen": "v8"}          # where the A_none comparator lives
MECHS_ALL = ("runs", "whole", "frozen", "armruns")


def ci_str(a, d=4):
    lo, hi = M.ci(a[1:])
    return a[0], lo, hi, f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--v12", default="results/v12")
    ap.add_argument("--v7", default="results/v7")
    args = ap.parse_args()
    out = f"{args.v12}/analysis"
    reg12 = [m for m in json.load(open(f"{args.v12}/registry.json"))
             if os.path.exists(f"{args.v12}/scores/{m['name']}.csv")]
    reg7 = [m for m in json.load(open(f"{args.v7}/registry.json"))
            if m["cell"] == "A_none" and m["world"] in ("S", "I") and m.get("world_seed", 0) == 0
            and m["strength"] == "mid" and MECH_PROTO.get(m["mechanism"]) == m["proto"]
            and os.path.exists(f"{args.v7}/scores/{m['name']}.csv")]
    files = {m["name"]: f"{args.v12}/scores/{m['name']}.csv" for m in reg12}
    files.update({m["name"]: f"{args.v7}/scores/{m['name']}.csv" for m in reg7})
    loaded = {n: load_scores(p) for n, p in files.items()}
    common = set.intersection(*[set(meta["identifier"]) for meta, _ in loaded.values()])
    meta0 = next(iter(loaded.values()))[0]
    keep = [k for k, i in enumerate(meta0["identifier"]) if i in common]
    ids = [meta0["identifier"][k] for k in keep]
    rs = RowSet(ids, [int(meta0["label"][k]) for k in keep], [meta0["child_id"][k] for k in keep],
                args.n_boot)
    S = {}
    for n, (meta, sc) in loaded.items():
        pos = {i: k for k, i in enumerate(meta["identifier"])}
        order = [pos[i] for i in ids]
        S[n] = {c: v[order] for c, v in sc.items()}
    cache = {}

    def aucs(n, c):
        if (n, c) not in cache:
            cache[(n, c)] = rs.aucs(S[n][c])
        return cache[(n, c)]

    def group(models, world, mech, cell):
        return [m for m in models if m["world"] == world and m["mechanism"] == mech and m["cell"] == cell]

    def R_mask_mean(ms_models, mech):
        """Mask-mean R (logit) and raw-AUC difference arrays, seeds averaged inside resamples."""
        lg, rw = [], []
        for ms in MASK_SEEDS:
            per_lg = [logit(aucs(m["name"], f"{mech}_S_mid_m{ms}")) - logit(aucs(m["name"], f"{mech}_I_mid_m{ms}"))
                      for m in ms_models]
            per_rw = [aucs(m["name"], f"{mech}_S_mid_m{ms}") - aucs(m["name"], f"{mech}_I_mid_m{ms}")
                      for m in ms_models]
            lg.append(np.mean(per_lg, axis=0)); rw.append(np.mean(per_rw, axis=0))
        return np.mean(lg, axis=0), np.mean(rw, axis=0)

    def clean_auc(ms_models):
        return np.mean([aucs(m["name"], "clean") for m in ms_models], axis=0)

    rows, crit = [], {"n_boot": args.n_boot, "n_rows": len(ids), "P12_1": {}, "P12_2": {}, "P12_3": {}}
    for mech in ("runs", "frozen"):
        bS, bI = group(reg7, "S", mech, "A_none"), group(reg7, "I", mech, "A_none")
        if not bS or not bI:
            continue
        (bRS, bRSr), (bRI, bRIr) = R_mask_mean(bS, mech), R_mask_mean(bI, mech)
        b_se, b_se_raw = bRS - bRI, bRSr - bRIr
        rows.append({"mechanism": mech, "cell": "A_none", "contrast": "SE(A_none)",
                     "logit": ci_str(b_se)[3], "raw": ci_str(b_se_raw)[3],
                     "R_S": f"{bRS[0]:+.4f}", "R_I": f"{bRI[0]:+.4f}", "n_seeds": f"{len(bS)}/{len(bI)}"})
        for cell in CELLS:
            cS, cI = group(reg12, "S", mech, cell), group(reg12, "I", mech, cell)
            if not cS or not cI:
                continue
            (RS, RSr), (RI, RIr) = R_mask_mean(cS, mech), R_mask_mean(cI, mech)
            se, se_raw = RS - RI, RSr - RIr
            d, dr = se - b_se, se_raw - b_se_raw
            dp, dlo, dhi, ds = ci_str(d)
            rp, rlo, rhi, rstr = ci_str(dr)
            lowers = bool(dhi < 0 and rhi < 0)
            rows.append({"mechanism": mech, "cell": cell, "contrast": f"SE({cell})",
                         "logit": ci_str(se)[3], "raw": ci_str(se_raw)[3],
                         "R_S": f"{RS[0]:+.4f}", "R_I": f"{RI[0]:+.4f}", "n_seeds": f"{len(cS)}/{len(cI)}"})
            rows.append({"mechanism": mech, "cell": cell, "contrast": f"SE({cell}) - SE(A_none)",
                         "logit": ds, "raw": rstr, "R_S": "", "R_I": "", "n_seeds": "",
                         "lowers_SE_P12_1": str(lowers),
                         "p_inv": f"{p_inversion(d[1:]):.4f}", "raw_p_inv": f"{p_inversion(dr[1:]):.4f}"})
            crit["P12_1"][f"{cell}|{mech}"] = {"diff_logit": [dp, dlo, dhi], "diff_raw": [rp, rlo, rhi],
                                               "pass": lowers}
            for world, cg, bg in (("S", cS, bS), ("I", cI, bI)):
                cd = clean_auc(cg) - clean_auc(bg)
                p_, lo_, hi_, s_ = ci_str(cd)
                crit["P12_2"][f"{cell}|{mech}|{world}"] = {"clean_auc_change": [p_, lo_, hi_]}
                rows.append({"mechanism": mech, "cell": cell, "contrast": f"clean AUC({cell}) - clean AUC(A_none), World {world}",
                             "logit": "", "raw": s_, "R_S": "", "R_I": "", "n_seeds": ""})
        aI = group(reg12, "I", mech, "C_augment")
        if aI:
            for other in MECHS_ALL:
                if other == mech:
                    continue
                col_s, col_i = f"{other}_S_mid_m2000", f"{other}_I_mid_m2000"
                if col_s not in S[aI[0]["name"]] or col_s not in S[bI[0]["name"]]:
                    continue
                ra = np.mean([logit(aucs(m["name"], col_s)) - logit(aucs(m["name"], col_i)) for m in aI], axis=0)
                rb = np.mean([logit(aucs(m["name"], col_s)) - logit(aucs(m["name"], col_i)) for m in bI], axis=0)
                crit["P12_3"][f"{mech}|{other}"] = {"R_augment": ci_str(ra)[3], "R_A_none": ci_str(rb)[3],
                                                    "diff": ci_str(ra - rb)[3]}
                rows.append({"mechanism": mech, "cell": "C_augment",
                             "contrast": f"World I crossed R under {other} (mask 2000): augment - A_none",
                             "logit": ci_str(ra - rb)[3], "raw": "", "R_S": "", "R_I": "", "n_seeds": ""})
    os.makedirs(out, exist_ok=True)
    fields = ["mechanism", "cell", "contrast", "logit", "raw", "R_S", "R_I", "n_seeds", "lowers_SE_P12_1", "p_inv", "raw_p_inv"]
    with open(f"{out}/effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    crit["family"] = "six contrasts SE(cell) - SE(A_none), uncorrected, declared in remedies_v12.yaml"
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1, default=float)
    for r in rows:
        print(f"{r['mechanism']:7s} {r['contrast']:62s} logit {r['logit']:30s} raw {r['raw']}")


if __name__ == "__main__":
    main()
