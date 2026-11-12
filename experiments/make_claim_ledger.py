"""Claim ledger: every quantitative claim of the abstract, contributions and conclusion, with the
source file and the value read from a release tree. Values that differ in the --retired tree
(original selection rule) are listed in a closing section. Writes paper/claim_ledger.md.
    python experiments/make_claim_ledger.py --tree build/release_v15_clean
Protocol v16 claims are read from --v16 (results/v16/analysis by default), outside the tree.
"""
import argparse
import csv
import hashlib
import json
import os
import re


def rows(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def one(rs, **kw):
    hit = [r for r in rs if all(r[k] == v for k, v in kw.items())]
    assert len(hit) == 1, (kw, len(hit))
    return hit[0]


def nums(s):
    return [float(x) for x in re.findall(r"[+-]?\d+\.\d+", s)]


def ledger(tree, v16=None):
    em = rows(f"{tree}/results/v7/analysis/effects_masks.csv")
    e12 = rows(f"{tree}/results/v12/analysis/effects.csv")
    e13 = rows(f"{tree}/results/v13/analysis/effects.csv")
    gm = rows(f"{tree}/results/v7/analysis/groups_masks.csv")
    t2 = rows(f"{tree}/results/v11/analysis/t2_transfer.csv")
    gap = rows(f"{tree}/results/v7/analysis/gap_table_b4.csv")
    v13 = [r for r in e13 if r["proto"] == "v13"][0]
    L = []

    def add(claim, where, value, src):
        L.append((claim, where, value, src))

    r = one(em, proto="v7", mechanism="runs", strength="mid", contrast="SE(A_none)")
    add("MWI, missing runs, mid", "Table 4, V-D", f"{float(r['point']):+.3f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]", "results/v7/analysis/effects_masks.csv")
    r = one(em, proto="v7", mechanism="runs", strength="low", contrast="SE(A_none)")
    add("MWI, missing runs, low", "Table 4, V-D", f"{float(r['point']):+.3f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]", "effects_masks.csv")
    r = one(em, proto="v7", mechanism="runs", strength="mid-low", contrast="SE(mid) - SE(low), A_none")
    add("MWI(mid) - MWI(low), missing runs (strength claim, Contribution 2)", "V-D", f"{float(r['point']):+.3f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]", "effects_masks.csv")
    r = one(em, proto="v7", mechanism="whole", strength="mid", contrast="SE(A_none)")
    add("MWI, whole-joint absence", "Table 4, V-D", f"{float(r['point']):+.3f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]", "effects_masks.csv")
    g = one(gm, proto="v7", world="I", mechanism="whole", strength="mid", cell="A_none")
    add("Reference own-operator response, whole-joint absence", "V-D, Table S18",
        re.sub(r"[+-]?\d+\.\d+", lambda m: f"{float(m.group()):+.3f}", g["R_maskmean"]), "groups_masks.csv")
    r = one(em, proto="v8", mechanism="frozen", strength="mid", contrast="SE(A_none)")
    add("MWI, frozen frames", "Table 4, V-D", f"{float(r['point']):+.3f} [{float(r['ci_lo']):+.3f}, {float(r['ci_hi']):+.3f}]", "effects_masks.csv")
    add("MWI, confidence-only planting, corrected rule (headline)", "abstract, Table 2, V-A", f"{v13['logit']} logit; {v13['raw']} raw", "results/v13/analysis/effects.csv")
    add("Reference own-operator response, confidence-only (corrected rule)", "Table S4, S3b", v13["R_I"], "results/v13/analysis/effects.csv")
    add("Clean AUROC difference S - I, largest magnitude (0.024, frozen frames)", "Table 4, V-C, Limitations", one(em, proto="v8", mechanism="frozen", strength="mid", contrast="clean AUC(S) - clean AUC(I), A_none")["raw_point"] if any(r["contrast"] == "clean AUC(S) - clean AUC(I), A_none" for r in em) else "see planted_v6.tex", "effects_masks.csv / planted_v6.tex")
    for mech, lab in (("runs", "missing runs"), ("frozen", "frozen frames")):
        r = one(e12, mechanism=mech, cell="C_groupdro", contrast="SE(C_groupdro) - SE(A_none)")
        add(f"Group DRO change in MWI, {lab}", "abstract, Table 3, V-C", f"{r['logit']} logit; {r['raw']} raw", "results/v12/analysis/effects.csv")
    r = [x for x in e12 if x["mechanism"] == "frozen" and x["cell"] == "C_groupdro" and x["contrast"].startswith("clean AUC(C_groupdro) - clean AUC(A_none), World S")][0]
    add("Group DRO clean-AUROC cost, frozen frames, World S (0.023)", "Table 3, V-C", r["raw"], "results/v12/analysis/effects.csv")
    r = one(e12, mechanism="frozen", cell="C_adversarial", contrast="SE(C_adversarial) - SE(A_none)")
    add("Adversarial head change in MWI, frozen frames (nominal)", "Table 3, V-C", f"{r['logit']} logit; {r['raw']} raw", "results/v12/analysis/effects.csv")
    p15 = f"{tree}/results/v15conf/analysis/effects.csv"
    if os.path.exists(p15):                                    # protocol v15conf
        e15 = rows(p15)
        r = one(e15, contrast="MWI(A_none)")
        add("MWI, confidence-only planting, no remedy, corrected recipe (protocol v15conf)", "Table 2, Table 3, V-A, S3f", f"{r['logit']} logit; {r['raw']} raw", "results/v15conf/analysis/effects.csv")
        r = one(e15, contrast="MWI(C_groupdro)")
        add("MWI, confidence-only planting, Group DRO (protocol v15conf)", "Table 3, V-C, S3f", f"{r['logit']} logit; {r['raw']} raw", "results/v15conf/analysis/effects.csv")
        r = one(e15, contrast="MWI(C_groupdro) - MWI(A_none)")
        add("Group DRO change in MWI, confidence-only planting (P15_1)", "abstract, Table 3, V-C, Conclusion, S3f", f"{r['logit']} logit; {r['raw']} raw; lowers {r['lowers_P15_1']}", "results/v15conf/analysis/effects.csv")
        for w in ("S", "I"):
            r = one(e15, contrast=f"clean AUROC(C_groupdro) - clean AUROC(A_none), World {w}")
            add(f"Group DRO clean-AUROC change, confidence-only planting, World {w}", "Table 3, V-C", r["raw"], "results/v15conf/analysis/effects.csv")
        r = one(e15, contrast="MWI(A_none, v15conf) - MWI(A_none, v13 cleaned) [descriptive]")
        add("No-remedy cell, corrected recipe minus protocol v13 (descriptive)", "V-A, Table 2, S3f, S10", f"{r['logit']} logit; {r['raw']} raw", "results/v15conf/analysis/effects.csv")
    if v16 and os.path.exists(f"{v16}/effects.csv"):           # protocol v16
        e16 = {r["quantity"]: r for r in rows(f"{v16}/effects.csv")}
        rel = {r["group"]: r for r in rows(f"{v16}/conf_reliance.csv")}
        crit = json.load(open(f"{v16}/criteria.json"))
        src, srel, scrit = f"{v16}/effects.csv", f"{v16}/conf_reliance.csv", f"{v16}/criteria.json"
        r = e16["MWI (base9)"]
        add("MWI, confidence-only planting, nine seeds per world (protocol v16)", "abstract, Table 2, V-A, Conclusion, S3h", f"{r['logit']} logit; {r['raw']} raw", src)
        add("MWI, nine seeds, seed-level Welch interval", "V-A, S3h", e16["MWI (base9), Welch t interval over per-seed responses (logit)"]["logit"], src)
        r = e16["MWI (A)"]
        add("MWI, confidence-only planting, ST-GCN (protocol v16)", "abstract, Table 2, V-A, Conclusion, S3h", f"{r['logit']} logit; {r['raw']} raw", src)
        add("ST-GCN clean AUROC, World S minus World I", "Table 2, V-A, V-B, Limitations", r["clean_S_minus_I"], src)
        add("ST-GCN gate, mean World I clean AUROC (threshold 0.75)", "V-A, Limitations, S3h", f"{crit['G16_A_convergence']['mean_clean_auroc_world_I']:.4f}", scrit)
        r = e16["MWI(C_groupdro_obs) - MWI(A_none, v15conf seeds 0-2)"]
        add("Group DRO with input-derived groups, change in MWI", "abstract, Table 3, V-C, Conclusion, S3h", f"{r['logit']} logit; {r['raw']} raw", src)
        r = e16["MWI(C_groupdro_obs) - MWI(C_groupdro, v15conf, severity groups) [descriptive]"]
        add("Group DRO, input-derived minus severity groups (descriptive)", "V-C, S3h", f"{r['logit']} logit; {r['raw']} raw", src)
        for w in ("S", "I"):
            add(f"Group DRO with input-derived groups, clean-AUROC change, World {w}", "Table 3, S3h",
                e16[f"clean AUROC(C_groupdro_obs) - clean AUROC(A_none), World {w}"]["raw"], src)
        r = e16["MWI(C_groupdro_obs) - MWI(A_none, v15conf seeds 0-2), seeds and masks resampled (logit)"]
        rr = e16["MWI(C_groupdro_obs) - MWI(A_none, v15conf seeds 0-2), seeds and masks resampled (raw)"]
        add("Group DRO with input-derived groups, change in MWI, seeds and masks resampled (paired)", "V-C, S3h, Table S17", f"{r['logit']} logit; {rr['raw']} raw", src)
        r = e16["MWI (base3)"]
        add("MWI, confidence-only planting, corrected recipe, seeds 0-2, clean AUROC S - I", "Table 2, V-B", r["clean_S_minus_I"], src)
        add("MWI, corrected recipe, seeds 0-2, seeds and masks resampled", "Table 2", e16["MWI (base3), seeds and masks resampled (logit)"]["logit"], src)
        agree = list(crit["P16_C1_reduction"]["group_agreement"].values())
        add("Agreement of input-derived and severity groups", "V-C, S3h", f"{min(agree):.3f} to {max(agree):.3f}", scrit)
        for grp, lab in (("CompactCTRGCN, no remedy, nine seeds", "nine seeds"), ("ST-GCN, no remedy", "ST-GCN")):
            add(f"Confidence-permutation reliance S - I, clean input, {lab}", "abstract, V-B, S3h", rel[grp]["clean_S_minus_I"], srel)
            add(f"Confidence-permutation reliance S - I, policy S input, {lab}", "V-B, S3h", rel[grp]["policyS_S_minus_I"], srel)
    v17 = f"{tree}/results/v17/analysis"
    if not os.path.exists(f"{v17}/effects.csv"):
        v17 = "results/v17/analysis"                         # outside the tree, as for v16
    if os.path.exists(f"{v17}/effects.csv"):                    # protocol v17 (regenerated training assignments)
        e17 = {r["quantity"]: r for r in rows(f"{v17}/effects.csv")}
        c17 = json.load(open(f"{v17}/criteria.json"))
        for ws in (1, 2, 3):
            r = e17[f"MWI (w{ws})"]
            add(f"MWI, confidence-only, regenerated training assignment {ws} (protocol v17)", "abstract, Table 2, V-A, Conclusion, S3i",
                f"{r['logit']} logit; {r['raw']} raw; World I clean AUROC {c17['G17_convergence'][f'w{ws}']['mean_clean_auroc_world_I']:.4f}", f"{v17}/effects.csv")
        sp = c17["D17_2_assignment_spread"]["spread"]["world seeds 1-3 (logit)"]
        add("MWI across regenerated assignments 1-3, mean (SD) [min, max] (logit)", "V-A, Limitations, S3i",
            f"{sp['mean']:+.3f} ({sp['sd']:.3f}) [{sp['min']:+.3f}, {sp['max']:+.3f}]", f"{v17}/criteria.json")
        r = e17["MWI (pooled)"]
        add("MWI, nine regenerated-assignment fits per world pooled (protocol v17, secondary)", "Table 2, S3i", f"{r['logit']} logit; {r['raw']} raw", f"{v17}/effects.csv")
    for cell, lab in (("A_dropout", "whole-joint dropout"), ("B_coorddrop", "coordinate dropout")):
        for mech in ("runs", "frozen", "armruns"):
            hit = [x for x in em if x["contrast"] == f"SE({cell}) - SE(A_none)" and x["mechanism"] == mech and x["strength"] == "mid"]
            if hit:
                x = hit[0]
                add(f"{lab} minus none, {mech}", "V-D input configurations", f"{float(x['point']):+.3f} [{float(x['ci_lo']):+.3f}, {float(x['ci_hi']):+.3f}] logit; raw {float(x['raw_point']):+.3f} [{float(x['raw_ci_lo']):+.3f}, {float(x['raw_ci_hi']):+.3f}]", "effects_masks.csv")
    for grp in ("stage1_detector_v3", "stage1_detector"):
        hit = [x for x in t2 if x["group"] == grp]
        if hit:
            add(f"External AUROC, {grp}", "V-E transfer", f"{float(hit[0]['auc']):.3f} [{float(hit[0]['ci_lo']):.3f}, {float(hit[0]['ci_hi']):.3f}]", "results/v11/analysis/t2_transfer.csv")
    pts = []
    for r in gap:
        o, t = r["seed points full_qc_logit original/retrained"].split(" / ")
        pts += nums(o) + nums(t)
    add("Real-split full-to-QC gap, range over 30 fits (logit)", "V-E (raw range in text)", f"{min(pts):+.2f} to {max(pts):+.2f}", "results/v7/analysis/gap_table_b4.csv")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="build/release_v15_clean")
    ap.add_argument("--retired", default="build/release_v14_orig", help="tree whose values are listed as retired where they differ")
    ap.add_argument("--v16", default="results/v16/analysis", help="protocol v16 analysis (not part of the release trees)")
    ap.add_argument("--out", default="paper/claim_ledger.md")
    a = ap.parse_args()
    cur = ledger(a.tree, a.v16)
    old = {c: v for c, _, v, _ in ledger(a.retired)} if os.path.isdir(a.retired) else {}
    man = f"{a.tree}/MANIFEST.sha256"
    mh = hashlib.sha256(open(man, "rb").read()).hexdigest()[:16] if os.path.exists(man) else "(manifest not yet written)"
    out = [f"# Claim ledger (generated by experiments/make_claim_ledger.py from `{a.tree}`; manifest {mh})", "",
           "Every quantitative claim of the abstract, the contribution list and the conclusion, with the analysis file it is read from "
           "and the value in that file now. Regenerate after every pipeline run; do not edit by hand. The tree holds the cleaned "
           "selection rule (protocol repair_selection_v14) and the post hoc analyses added in revision.", "",
           "| Claim | Where stated | Value (current tree) | Source file |", "|---|---|---|---|"]
    for c, w, v, s in cur:
        out.append(f"| {c} | {w} | {v} | {s} |")
    retired = [(c, old[c], v) for c, _, v, _ in cur if c in old and old[c] != v]
    out += ["", f"## Retired values (original selection rule, `{a.retired}`)", "",
            "Values that differ from the current tree; the original rule is kept only as the regression check of the pipeline.", "",
            "| Claim | Retired value | Current value |", "|---|---|---|"]
    out += [f"| {c} | {o} | {v} |" for c, o, v in retired] or ["| (none) | | |"]
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out, len(cur), "claims,", len(retired), "retired")


if __name__ == "__main__":
    main()
