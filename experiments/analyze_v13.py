"""Protocol v13 analysis (configs/protocols/armconf_v13.yaml): arm-confidence A_none cells
retrained with the coordinate-invariant valid-joint rule, against the v8 armruns A_none cells.

V13_1: the invariance gate from experiments/check_conf_invariance.py --valid-rule coords (max
absolute x,y difference must be exactly 0 on the standard path). V13_2: SE on the test-mask
mean (masks 2000/3000/4000), logit and raw-AUC scales, and the paired difference v13 minus v8.
V13_3: R(f_I) of the v13 reference with the +-0.05 margin verdict. Also the crossed responses
at mask 2000. Runs locally (needs the v8 score files) and writes
results/v13/analysis/{effects.csv,criteria.json}:
    conda run -n torch python experiments/analyze_v13.py [--n-boot 2000]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                        # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, verdict, R_MARGIN  # noqa: E402
from experiments.analyze_v7_masks import MASK_SEEDS                         # noqa: E402

MECH = "armruns"
MECHS_ALL = ("runs", "whole", "frozen", "armruns")


def ci_str(a, d=4):
    lo, hi = M.ci(a[1:])
    return a[0], lo, hi, f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def gate(path):
    if not os.path.exists(path):
        return {"status": "not run", "pass": False}
    rows = [r for r in csv.DictReader(open(path))
            if r["mechanism"] == MECH and r["path"].startswith("standard")]
    ok = bool(rows) and all(float(r["max_abs_diff_xy"]) == 0.0 for r in rows)
    return {"status": "run", "pass": ok, "n_segments": rows[0]["n_segments"] if rows else 0,
            "max_abs_diff_xy": max((float(r["max_abs_diff_xy"]) for r in rows), default=None),
            "comparisons": [r["comparison"] for r in rows]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--v13", default="results/v13")
    ap.add_argument("--v7", default="results/v7")
    args = ap.parse_args()
    out = f"{args.v13}/analysis"
    os.makedirs(out, exist_ok=True)
    g = gate(f"{out}/conf_invariance_v13.csv")
    reg13 = [m for m in json.load(open(f"{args.v13}/registry.json"))
             if os.path.exists(f"{args.v13}/scores/{m['name']}.csv")]
    reg8 = [m for m in json.load(open(f"{args.v7}/registry.json"))
            if m["proto"] == "v8" and m["cell"] == "A_none" and m["world"] in ("S", "I")
            and m.get("world_seed", 0) == 0 and m["mechanism"] == MECH and m["strength"] == "mid"
            and os.path.exists(f"{args.v7}/scores/{m['name']}.csv")]
    files = {m["name"]: f"{args.v13}/scores/{m['name']}.csv" for m in reg13}
    files.update({m["name"]: f"{args.v7}/scores/{m['name']}.csv" for m in reg8})
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

    def group(models, world):
        return [m for m in models if m["world"] == world]

    def R_mask_mean(models):
        lg, rw, per_mask = [], [], {}
        for ms in MASK_SEEDS:
            pl = [logit(aucs(m["name"], f"{MECH}_S_mid_m{ms}")) - logit(aucs(m["name"], f"{MECH}_I_mid_m{ms}"))
                  for m in models]
            pr = [aucs(m["name"], f"{MECH}_S_mid_m{ms}") - aucs(m["name"], f"{MECH}_I_mid_m{ms}") for m in models]
            lg.append(np.mean(pl, axis=0)); rw.append(np.mean(pr, axis=0))
            per_mask[ms] = ([p[0] for p in pl], float(np.mean(pl, axis=0)[0]))
        return np.mean(lg, axis=0), np.mean(rw, axis=0), per_mask

    def clean(models):
        return np.mean([aucs(m["name"], "clean") for m in models], axis=0)

    rows, crit = [], {"n_boot": args.n_boot, "n_rows": len(ids), "V13_1_invariance": g}
    res = {}
    for label, reg in (("v8", reg8), ("v13", reg13)):
        gS, gI = group(reg, "S"), group(reg, "I")
        if not gS or not gI:
            crit[f"{label}_missing"] = {"S": len(gS), "I": len(gI)}
            continue
        (RS, RSr, pmS), (RI, RIr, pmI) = R_mask_mean(gS), R_mask_mean(gI)
        se, se_raw = RS - RI, RSr - RIr
        seed_pts = [np.mean([pmS[ms][0][k] for ms in MASK_SEEDS]) - np.mean([pmI[ms][0][k] for ms in MASK_SEEDS])
                    for k in range(min(len(gS), len(gI)))]
        per_mask_se = {ms: pmS[ms][1] - pmI[ms][1] for ms in MASK_SEEDS}
        res[label] = {"RS": RS, "RI": RI, "se": se, "se_raw": se_raw, "clean_S": clean(gS), "clean_I": clean(gI),
                      "gS": gS, "gI": gI}
        rows.append({"proto": label, "contrast": "SE(A_none), armruns, mask mean",
                     "logit": ci_str(se)[3], "raw": ci_str(se_raw)[3],
                     "R_S": ci_str(RS)[3], "R_I": ci_str(RI)[3],
                     "R_I_margin_verdict": verdict(RI[0], M.ci(RI[1:]), R_MARGIN),
                     "per_mask": "; ".join(f"m{ms} {v:+.4f}" for ms, v in per_mask_se.items()),
                     "mask_range": f"{max(per_mask_se.values()) - min(per_mask_se.values()):.4f}",
                     "sd_seeds": f"{np.std(seed_pts, ddof=1):.4f}", "n_seeds": f"{len(gS)}/{len(gI)}",
                     "clean_S": ci_str(clean(gS))[3], "clean_I": ci_str(clean(gI))[3]})
        crit[f"V13_2_SE_{label}"] = {"logit": list(ci_str(se)[:3]), "raw": list(ci_str(se_raw)[:3]),
                                     "per_mask": per_mask_se, "sd_seeds": float(np.std(seed_pts, ddof=1))}
        crit[f"V13_3_reference_{label}"] = {"R_I": list(ci_str(RI)[:3]),
                                            "margin_verdict": verdict(RI[0], M.ci(RI[1:]), R_MARGIN)}
    if "v8" in res and "v13" in res:
        d, dr = res["v13"]["se"] - res["v8"]["se"], res["v13"]["se_raw"] - res["v8"]["se_raw"]
        rows.append({"proto": "v13 - v8", "contrast": "SE(v13) - SE(v8), paired on shared resamples",
                     "logit": ci_str(d)[3], "raw": ci_str(dr)[3], "R_S": "", "R_I": "", "R_I_margin_verdict": "",
                     "per_mask": "", "mask_range": "", "sd_seeds": "", "n_seeds": "", "clean_S": "", "clean_I": ""})
        crit["V13_2_difference"] = {"logit": list(ci_str(d)[:3]), "raw": list(ci_str(dr)[:3])}
        for w in ("S", "I"):
            cd = res["v13"][f"clean_{w}"] - res["v8"][f"clean_{w}"]
            crit[f"clean_auc_change_{w}"] = list(ci_str(cd)[:3])
            rows.append({"proto": "v13 - v8", "contrast": f"clean AUC change, World {w}", "logit": "",
                         "raw": ci_str(cd)[3], "R_S": "", "R_I": "", "R_I_margin_verdict": "", "per_mask": "",
                         "mask_range": "", "sd_seeds": "", "n_seeds": "", "clean_S": "", "clean_I": ""})
        # crossed responses of both v13 cells against v8, mask 2000
        crit["crossed"] = {}
        for w in ("S", "I"):
            for other in MECHS_ALL:
                cs, ci_ = f"{other}_S_mid_m2000", f"{other}_I_mid_m2000"
                if any(cs not in S[m["name"]] for m in res["v13"][f"g{w}"] + res["v8"][f"g{w}"]):
                    continue
                r13 = np.mean([logit(aucs(m["name"], cs)) - logit(aucs(m["name"], ci_)) for m in res["v13"][f"g{w}"]], axis=0)
                r8 = np.mean([logit(aucs(m["name"], cs)) - logit(aucs(m["name"], ci_)) for m in res["v8"][f"g{w}"]], axis=0)
                crit["crossed"][f"{w}|{other}"] = {"R_v13": ci_str(r13)[3], "R_v8": ci_str(r8)[3], "diff": ci_str(r13 - r8)[3]}
                rows.append({"proto": f"v13 vs v8, World {w}", "contrast": f"crossed R under {other} (mask 2000)",
                             "logit": ci_str(r13)[3], "raw": "", "R_S": "", "R_I": ci_str(r8)[3],
                             "R_I_margin_verdict": "", "per_mask": f"diff {ci_str(r13 - r8)[3]}", "mask_range": "",
                             "sd_seeds": "", "n_seeds": "", "clean_S": "", "clean_I": ""})
    fields = ["proto", "contrast", "logit", "raw", "R_S", "R_I", "R_I_margin_verdict", "per_mask",
              "mask_range", "sd_seeds", "n_seeds", "clean_S", "clean_I"]
    with open(f"{out}/effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    json.dump(crit, open(f"{out}/criteria.json", "w"), indent=1, default=float)
    print("V13_1 gate:", g)
    for r in rows:
        print(f"{r['proto']:18s} {r['contrast']:55s} logit {r['logit']:30s} raw {r['raw']}")


if __name__ == "__main__":
    main()
