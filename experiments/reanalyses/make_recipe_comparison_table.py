"""Build supplement Table S14 (combined recipe comparison) from stored outputs only.

Reads results/reanalyses/r26_recipe_cells.csv (six decimals, checked against the frozen
results/recipe_comparison/analysis/effects.csv by experiments/reanalyses/r26_recipe_per_fit.py), results/recipe_comparison/analysis/
criteria.json (full precision: paired contrasts, gates, verdicts) and conf_invariance_v20.csv.
Every printed value is rounded once. The decision labels are asserted against the frozen verdicts;
the frozen verdict "no detected change" is printed as "Reduction criterion not met".
Writes paper/tables/recipe_comparison.tex.
    python3 experiments/reanalyses/make_recipe_comparison_table.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results/recipe_comparison/analysis"
CELLS = ROOT / "results/reanalyses/r26_recipe_cells.csv"
OUT = ROOT / "paper/tables/recipe_comparison.tex"
NUM = re.compile(r"[-+]?\d+\.\d+")
ORDER = ("A_both", "A_none_v15conf", "A_none_local")


def m(s):
    return s.replace("-", "$-$")


def iv(p, lo, hi, sign=True):
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return m(f"{f(p)} [{f(lo)}, {f(hi)}]")


def only(lo, hi):
    return m(f"[{lo:+.3f}, {hi:+.3f}]")


def main():
    rows = {r["model_set"]: r for r in csv.DictReader(l for l in open(CELLS) if not l.startswith("#"))}
    c = json.load(open(A / "criteria.json"))
    inv = list(csv.DictReader(open(A / "conf_invariance_v20.csv")))
    P, D = c["P20_1_P20_2"], c["D20_3"]

    def q(key, name, sign=True):
        return iv(*[float(x) for x in NUM.findall(rows[key][name])], sign=sign)

    # frozen decisions that the printed labels state
    assert P["baseline"] == "A_none_v15conf" and D["baseline"] == "A_none_local"
    assert P["verdict"] == D["verdict"] == "no detected change" and D["verdict_agrees_with_primary"]
    assert P["dTTI_logit"][1] < 0 < P["dTTI_logit"][2] and P["dTTI_raw"][2] < 0
    assert D["dTTI_logit"][1] < 0 < D["dTTI_logit"][2] and D["dTTI_raw"][2] < 0
    assert all(lo <= 0 <= hi for _, lo, hi in (P["seed_mask_sensitivity"]["logit"], P["seed_mask_sensitivity"]["raw"],
                                               D["seed_mask_sensitivity"]["logit"], D["seed_mask_sensitivity"]["raw"]))
    assert c["G20_clean"]["pass"] and not P["clean_preserved"]
    assert all(P["clean_change"][w][1] > 0.02 for w in ("S", "I"))            # both rose beyond the margin
    assert all(float(r["max_abs_diff_xy_velocity"]) == 0.0 and int(r["n_segments"]) == 1966 for r in inv)
    assert {r["mask_seed"] for r in inv} == {"2000", "3000", "4000"} and len({r["path"] for r in inv}) == 2
    rI = [float(x) for x in NUM.findall(rows["A_both"]["R_I"])]
    cd = [float(x) for x in NUM.findall(rows["A_both"]["clean_S_minus_I"])]
    assert not (-0.05 < rI[1] and rI[2] < 0.05) and not (-0.02 < cd[1] and cd[2] < 0.02)

    t = [r"\textit{(a) Model sets (three seeds per condition; per-fit, per-mask AUROC transformed, then averaged)}\\[2pt]",
         r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r" & Combined recipe & No remedy, published & No remedy, local refit \\",
         r"Quantity & (local GPU) & (primary comparator) & (sensitivity comparator) \\", r"\midrule"]
    for w in ("S", "I"):
        for p in ("S", "I"):
            t.append(" & ".join([f"Mean AUROC $({w},{p})$"] + [f"{float(rows[k][f'auroc_{w}{p}']):.4f}" for k in ORDER]) + r" \\")
    for label, name, sign in ((r"Response \R{}$(f_S)$, logit", "R_S", True), (r"Response \R{}$(f_I)$, logit", "R_I", True),
                              ("TTI, logit", "TTI_logit", True), ("TTI, raw AUROC", "TTI_raw", True),
                              ("Clean AUROC, Condition S", "clean_S", False), ("Clean AUROC, Condition I", "clean_I", False),
                              ("Clean AUROC, S $-$ I", "clean_S_minus_I", True)):
        t.append(" & ".join([label] + [q(k, name, sign) for k in ORDER]) + r" \\")
    sp, sd = P["seed_mask_sensitivity"], D["seed_mask_sensitivity"]
    t += [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(b) Paired changes in TTI (``published'' is the published no-remedy cell); the second interval of each scale also resamples seeds and masks}\\[2pt]",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r"Contrast & $\Delta$TTI, logit & seeds and masks & $\Delta$TTI, raw AUROC & seeds and masks \\", r"\midrule",
          rf"Recipe $-$ published (primary) & {iv(*P['dTTI_logit'])} & {only(*sp['logit'][1:])} & {iv(*P['dTTI_raw'])} & {only(*sp['raw'][1:])} \\",
          rf"Recipe $-$ local refit (sensitivity) & {iv(*D['dTTI_logit'])} & {only(*sd['logit'][1:])} & {iv(*D['dTTI_raw'])} & {only(*sd['raw'][1:])} \\",
          rf"Local refit $-$ published (descriptive) & {iv(*D['local_minus_published_baseline_TTI_logit'])} & & {iv(*D['local_minus_published_baseline_TTI_raw'])} & \\",
          r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(c) Paired changes in clean AUROC}\\[2pt]",
          r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
          r"Contrast & Condition S & Condition I \\", r"\midrule",
          rf"Recipe $-$ published & {iv(*P['clean_change']['S'])} & {iv(*P['clean_change']['I'])} \\",
          rf"Recipe $-$ local refit & {iv(*D['clean_change']['S'])} & {iv(*D['clean_change']['I'])} \\",
          r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(d) Gates and decisions under the rules fixed in the protocol}\\[2pt]",
          r"\begin{tabularx}{\linewidth}{@{}L{1.05}L{1.2}L{0.75}@{}}", r"\toprule",
          r"Rule & Observed & Decision \\", r"\midrule",
          r"Invariance gate (before the fits) & $x$, $y$, velocity channels: maximum difference 0 in the checked cases & pass \\",
          rf"Clean-performance gate (test-set interpretation gate) & Condition I mean clean test AUROC {c['G20_clean']['mean_clean_auroc_condition_I']:.3f} ($\geq 0.75$) & pass \\",
          r"Primary rule: reduction on both scales & logit interval includes zero; raw interval below zero & Reduction criterion not met \\",
          r"Sensitivity: the same rule against the local refit & logit interval includes zero; raw interval below zero & Reduction criterion not met \\",
          r"Clean AUROC preserved within $\pm 0.02$ & both changes above $+0.02$ & not met: clean AUROC rose \\",
          rf"Reference response within $\pm 0.05$, recipe & {q('A_both', 'R_I')} & equivalence not established \\",
          rf"Clean AUROC S $-$ I within $\pm 0.02$, recipe & {q('A_both', 'clean_S_minus_I')} & equivalence not established \\",
          rf"Largest logit reduction not excluded (primary) & {abs(P['largest_reduction_not_excluded_logit']):.3f} conditional; {abs(sp['logit'][1]):.3f} with seeds and masks resampled & \\",
          r"\bottomrule", r"\end{tabularx}", ""]
    OUT.write_text("\n".join(t))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
