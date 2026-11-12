"""Post hoc procedure-level and assumption checks from saved score files, written to results/v7/analysis.

procedure_intervals.csv: conditional versus crossed intervals (identifier, test-mask and seed
resampling) for the eight remedy contrasts and mid minus low strength. variance_components.csv: SD
of MWI over index-paired seeds, over masks, and over bootstrap resamples. parallel_trends.csv: logit
AUROC change from clean under policies I and S per world, and World S minus World I.
natural_arm_split.csv: R in segments below and above the median natural arm low-confidence fraction.
worldseed_realizations.csv: R(f_S) for training-mask realizations 0 to 2 under missing runs.
Also writes paper/tables/{parallel_trends,procedure_intervals}.tex. Run inside a release tree.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import metrics as M                                  # noqa: E402
from safe_motion.data import asdpose as D                             # noqa: E402
from experiments.analyze_v7 import RowSet, load_scores, logit, align   # noqa: E402

NB = 2000
MASKS = (2000, 3000, 4000)
OUT = "results/v7/analysis"
HEAD = [("v7", "runs", "mid"), ("v7", "runs", "low"), ("v7", "whole", "mid"), ("v8", "frozen", "mid"), ("v13", "armruns", "mid")]
NAMES = {"runs": "missing runs", "whole": "whole-joint absence", "frozen": "frozen frames", "armruns": "arm confidence"}
CONTRASTS = [("v7", "runs", "A_velocity"), ("v7", "runs", "A_dropout"), ("v7", "runs", "A_both"), ("v7", "runs", "B_coorddrop"),
             ("v7", "whole", "A_dropout"), ("v7", "whole", "B_coorddrop"), ("v8", "frozen", "A_dropout"), ("v8", "armruns", "A_dropout")]
CELLNAME = {"A_velocity": "velocity", "A_dropout": "whole-joint dropout", "A_both": "velocity + dropout", "B_coorddrop": "coordinate dropout"}


def fmt(a, d=3):
    lo, hi = M.ci(a[1:])
    return f"{a[0]:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def pct(x, d=3):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    reg = []
    for proto_dir in ("results/v7", "results/v13"):
        for m in json.load(open(f"{proto_dir}/registry.json")):
            if m["world"] in ("S", "I") and os.path.exists(f"{proto_dir}/scores/{m['name']}.csv"):
                reg.append(dict(m, path=f"{proto_dir}/scores/{m['name']}.csv"))
    first = load_scores(reg[0]["path"])[0]
    rs = RowSet(first["identifier"], [int(v) for v in first["label"]], first["child_id"], NB)
    S = {}

    def scores(m):
        if m["name"] not in S:
            meta, sc = load_scores(m["path"])
            S[m["name"]] = align(rs, meta["identifier"], meta["label"], sc)
        return S[m["name"]]

    cache = {}

    def aucs(m, col, idx=None):
        key = (m["name"], col, None if idx is None else id(idx))
        if key not in cache:
            s = scores(m)[col]
            cache[key] = rs.aucs(s) if idx is None else idx.aucs(s[idx.sel])
        return cache[key]

    def group(proto, mech, st, cell, world, ws=0):
        g = sorted([m for m in reg if m["proto"] == proto and m["mechanism"] == mech and m["strength"] == st
                    and m["cell"] == cell and m["world"] == world and int(m.get("world_seed", 0)) == ws], key=lambda m: m["seed"])
        return g

    def Rarr(m, mech, st):
        """(3 masks, NB+1) logit R of one model."""
        return np.stack([logit(aucs(m, f"{mech}_S_{st}_m{ms}")) - logit(aucs(m, f"{mech}_I_{st}_m{ms}")) for ms in MASKS])

    rng = np.random.RandomState(20261110)
    mask_draw = rng.randint(0, 3, size=(NB, 3))

    def cell_R(proto, mech, st, cell):
        out = {}
        for w in ("S", "I"):
            g = group(proto, mech, st, cell, w)
            assert len(g) == 3, (proto, mech, st, cell, w, len(g))
            out[w] = [Rarr(m, mech, st) for m in g]
            out["draw" + w] = rng.randint(0, 3, size=(NB, 3))
        return out

    def cond(R):
        return np.mean([r.mean(0) for r in R["S"]], 0) - np.mean([r.mean(0) for r in R["I"]], 0)

    def crossed(R):
        v = np.empty(NB)
        for b in range(NB):
            md = mask_draw[b]
            v[b] = (np.mean([R["S"][k][md, b + 1].mean() for k in R["drawS"][b]])
                    - np.mean([R["I"][k][md, b + 1].mean() for k in R["drawI"][b]]))
        return v

    os.makedirs(OUT, exist_ok=True)
    # 1. procedure-level intervals
    rows = []
    base = {}
    for proto, mech, cell in CONTRASTS:
        key = (proto, mech)
        if key not in base:
            base[key] = cell_R(proto, mech, "mid", "A_none")
        R0, R1 = base[key], cell_R(proto, mech, "mid", cell)
        d, dx = cond(R1) - cond(R0), crossed(R1) - crossed(R0)
        lo, hi = M.ci(d[1:])
        xlo, xhi = np.percentile(dx, [2.5, 97.5])
        rows.append({"contrast": f"{CELLNAME[cell]} minus none", "mechanism": NAMES[mech], "rule": "original" if mech == "armruns" else "",
                     "point": f"{d[0]:+.3f}", "conditional": f"[{lo:+.3f}, {hi:+.3f}]", "seeds_masks_resampled": f"[{xlo:+.3f}, {xhi:+.3f}]",
                     "conditional_excludes_zero": str(lo > 0 or hi < 0), "crossed_excludes_zero": str(xlo > 0 or xhi < 0)})
    Rm, Rl = cell_R("v7", "runs", "mid", "A_none"), cell_R("v7", "runs", "low", "A_none")
    d, dx = cond(Rm) - cond(Rl), crossed(Rm) - crossed(Rl)
    lo, hi = M.ci(d[1:]); xlo, xhi = np.percentile(dx, [2.5, 97.5])
    rows.append({"contrast": "mid minus low strength", "mechanism": "missing runs", "rule": "", "point": f"{d[0]:+.3f}",
                 "conditional": f"[{lo:+.3f}, {hi:+.3f}]", "seeds_masks_resampled": f"[{xlo:+.3f}, {xhi:+.3f}]",
                 "conditional_excludes_zero": str(lo > 0 or hi < 0), "crossed_excludes_zero": str(xlo > 0 or xhi < 0)})
    with open(f"{OUT}/procedure_intervals.csv", "w", newline="") as f:
        f.write("# post hoc: conditional vs seeds-and-masks-resampled intervals, logit MWI\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = [r"\begin{tabular}{llccc}", r"\toprule", r"Contrast & Mechanism & Change in MWI & Conditional 95\% & Seeds, masks resampled \\", r"\midrule"]
    for r in rows:
        mech = r["mechanism"] + (" (original rule)" if r["rule"] else "")
        L.append(f"{r['contrast']} & {mech} & {r['point']} & {r['conditional']} & {r['seeds_masks_resampled']} \\\\".replace("-", "$-$").replace("$-$joint", "-joint").replace("$-$to", "-to"))
    L += [r"\bottomrule", r"\end{tabular}"]
    open("paper/tables/procedure_intervals.tex", "w").write(
        "% generated by experiments/procedure_checks.py; do not edit\n" + "\n".join(L) + "\n")
    print("procedure intervals:"); [print(" ", r) for r in rows]

    # 2. variance components
    vc = []
    for proto, mech, st in HEAD:
        R = cell_R(proto, mech, st, "A_none")
        seed_pts = [float(rS[:, 0].mean() - rI[:, 0].mean()) for rS, rI in zip(R["S"], R["I"])]
        mask_pts = [float(np.mean([r[j, 0] for r in R["S"]]) - np.mean([r[j, 0] for r in R["I"]])) for j in range(3)]
        c = cond(R)
        vc.append({"cell": f"{NAMES[mech]}, {st}", "MWI": f"{c[0]:+.3f}", "sd_seed_ddof1": f"{np.std(seed_pts, ddof=1):.3f}",
                   "sd_mask_ddof1": f"{np.std(mask_pts, ddof=1):.3f}", "sd_bootstrap": f"{np.std(c[1:]):.3f}",
                   "seed_level_points": "; ".join(f"{v:+.3f}" for v in seed_pts)})
    with open(f"{OUT}/variance_components.csv", "w", newline="") as f:
        f.write("# post hoc: spread of the conditional MWI over seeds (index-paired), masks and identifier resamples\n")
        w = csv.DictWriter(f, fieldnames=list(vc[0])); w.writeheader(); w.writerows(vc)
    print("variance components:"); [print(" ", r) for r in vc]

    # 3. parallel trends
    pt = []
    for proto, mech, st in HEAD:
        res = {}
        for w in ("S", "I"):
            g = group(proto, mech, st, "A_none", w)
            clean = np.mean([logit(aucs(m, "clean")) for m in g], 0)
            pS = np.mean([np.mean([logit(aucs(m, f"{mech}_S_{st}_m{ms}")) for ms in MASKS], 0) for m in g], 0)
            pI = np.mean([np.mean([logit(aucs(m, f"{mech}_I_{st}_m{ms}")) for ms in MASKS], 0) for m in g], 0)
            res[w] = (clean, pI, pS)
        dS = (res["S"][2] - res["S"][0]) - (res["I"][2] - res["I"][0])      # gain under policy S, World S minus World I
        dI = (res["S"][1] - res["S"][0]) - (res["I"][1] - res["I"][0])      # change under policy I, World S minus World I
        rec = {"cell": f"{NAMES[mech]}, {st}"}
        for w in ("S", "I"):
            c, i, s = res[w]
            rec[f"W{w}_clean"] = f"{c[0]:.3f}"
            rec[f"W{w}_policyI_minus_clean"] = fmt(i - c)
            rec[f"W{w}_policyS_minus_clean"] = fmt(s - c)
        rec["policyS_change_WS_minus_WI"] = fmt(dS)
        rec["policyI_change_WS_minus_WI"] = fmt(dI)
        rec["MWI_check"] = f"{(dS - dI)[0]:+.3f}"
        pt.append(rec)
    with open(f"{OUT}/parallel_trends.csv", "w", newline="") as f:
        f.write("# post hoc: mean logit AUROC changes from clean input under each test policy, per training world; "
                "MWI = (policy S change, World S minus World I) - (policy I change, World S minus World I)\n")
        w = csv.DictWriter(f, fieldnames=list(pt[0])); w.writeheader(); w.writerows(pt)
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{World S model} & \multicolumn{2}{c}{World I model} & \multicolumn{2}{c}{World S minus World I} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Cell & policy I & policy S & policy I & policy S & policy I & policy S \\", r"\midrule"]
    for r in pt:
        L.append(" & ".join([r["cell"], r["WS_policyI_minus_clean"], r["WS_policyS_minus_clean"], r["WI_policyI_minus_clean"],
                             r["WI_policyS_minus_clean"], r["policyI_change_WS_minus_WI"], r["policyS_change_WS_minus_WI"]]).replace("-", "$-$").replace("$-$joint", "-joint") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open("paper/tables/parallel_trends.tex", "w").write("% generated by experiments/procedure_checks.py; do not edit\n" + "\n".join(L) + "\n")
    print("parallel trends:"); [print(" ", r) for r in pt]

    # 4. natural arm confidence split
    q = {r["identifier"]: r["q_arm_lowconf"] for r in D.load_index()}
    qa = np.array([q[i] for i in rs.ids])
    med = float(np.median(qa))
    split = []
    for lab, sel in (("below median", qa <= med), ("above median", qa > med)):
        idx = np.flatnonzero(sel)
        sub = RowSet([rs.ids[k] for k in idx], rs.y[idx], rs.child[idx], NB)
        sub.sel = idx
        for w in ("S", "I"):
            g = group("v13", "armruns", "mid", "A_none", w)
            R = np.mean([np.mean([logit(aucs(m, f"armruns_S_mid_m{ms}", sub)) - logit(aucs(m, f"armruns_I_mid_m{ms}", sub))
                                  for ms in MASKS], 0) for m in g], 0)
            split.append({"subset": f"natural arm low-confidence {lab} ({med:.3f})", "world": w, "segments": int(sel.sum()),
                          "R": fmt(R)})
    with open(f"{OUT}/natural_arm_split.csv", "w", newline="") as f:
        f.write("# post hoc: own-operator response of the corrected-rule arm-confidence models within host test "
                "segments split at the median natural arm low-confidence fraction\n")
        w = csv.DictWriter(f, fieldnames=list(split[0])); w.writeheader(); w.writerows(split)
    print("natural arm split:"); [print(" ", r) for r in split]

    # 5. training-mask realizations
    ws = []
    gI = group("v7", "runs", "mid", "A_none", "I")
    rI = np.mean([Rarr(m, "runs", "mid").mean(0) for m in gI], 0)
    for wseed in (0, 1, 2):
        for m in group("v7", "runs", "mid", "A_none", "S", wseed):
            r = Rarr(m, "runs", "mid").mean(0)
            ws.append({"model": m["name"], "training_mask_realization": wseed, "seed": m["seed"], "R_fS": fmt(r), "MWI_with_WorldI_mean": fmt(r - rI)})
    with open(f"{OUT}/worldseed_realizations.csv", "w", newline="") as f:
        f.write("# post hoc: World S models under missing runs, mid strength; realization 0 has seeds 0-2, "
                "realizations 1 and 2 seed 0 only\n")
        w = csv.DictWriter(f, fieldnames=list(ws[0])); w.writeheader(); w.writerows(ws)
    print("training-mask realizations:"); [print(" ", r) for r in ws]


if __name__ == "__main__":
    main()
