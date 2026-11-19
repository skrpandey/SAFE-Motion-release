"""Build supplement Table S13 (mapped-person retraining) from results/mapped_persons/analysis.

Reads effects.csv (strings formatted once from full precision) and criteria.json (full precision);
writes paper/tables/mapped_persons.tex. Cohort counts are the
row flow of experiments/train_asdpose.py person_split_v19 followed by the QC host filter.
    python3 experiments/reanalyses/make_mapped_persons_table.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results/mapped_persons/analysis"
OUT = ROOT / "paper/tables/mapped_persons.tex"
COHORT = [("Optimization training", 5993, 1945, 4048, 135, 134),
          ("Validation", 1784, 607, 1177, 33, 33),
          ("Test (analysis rows)", 866, 223, 643, 18, 18)]


def m(s):
    return s.replace("-", "$-$")


def iv(p, lo, hi, sign=True):
    f = (lambda x: f"{x:+.3f}") if sign else (lambda x: f"{x:.3f}")
    return m(f"{f(p)} [{f(lo)}, {f(hi)}]")


def main():
    rows = {r["model_set"]: r for r in csv.DictReader(open(A / "effects.csv"))}
    c = json.load(open(A / "criteria.json"))
    v, o = rows["v19"], rows["v15conf"]
    t = [r"\textit{(a) Cohort (segments after the length rule and QC filter)}\\[2pt]",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Partition & Segments & SMM & non-SMM & Identifiers & Persons \\", r"\midrule"]
    for name, *n in COHORT:
        t.append(" & ".join([name] + [f"{x:,}".replace(",", "{,}") for x in n]) + r" \\")
    t += [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(b) Condition-by-policy means (per-fit, per-mask AUROC transformed, then averaged)}\\[2pt]",
          r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
          r" & \multicolumn{2}{c}{Mean AUROC} & \multicolumn{2}{c}{Mean logit AUROC} \\",
          r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
          r"Training & policy S & policy I & policy S & policy I \\", r"\midrule"]
    for w in ("S", "I"):
        t.append(" & ".join([f"Condition {w}"] + [f"{float(v[f'{k}_{w}{p}']):.4f}" for k in ("auroc", "logit")
                                                 for p in ("S", "I")]) + r" \\")
    s = c["seed_mask_sensitivity"]
    t += [r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(c) Responses, interactions, clean performance, and decisions}\\[2pt]",
          r"\begin{tabular}{@{}lll@{}}", r"\toprule",
          r"Quantity & Estimate [95\% interval] & Decision \\", r"\midrule",
          rf"Response \R{{}}$(f_S)$, logit & {m(v['R_S'])} & \\",
          rf"Response \R{{}}$(f_I)$, logit & {m(v['R_I'])} & equivalence within $\pm 0.05$ not established \\",
          rf"TTI, logit & {m(v['TTI_logit'])} & primary rule: above zero on both scales (pass) \\",
          rf"TTI, raw AUROC & {m(v['TTI_raw'])} & \\",
          rf"TTI, logit, seeds and masks resampled & {iv(*s['logit'])} & \\",
          rf"TTI, raw, seeds and masks resampled & {iv(*s['raw'])} & \\",
          rf"Clean AUROC, Condition S & {m(v['clean_S']).replace('+', '')} & \\",
          rf"Clean AUROC, Condition I & {m(v['clean_I']).replace('+', '')} & gate: mean $\geq 0.75$ (pass) \\",
          rf"Clean AUROC, S $-$ I & {m(v['clean_S_minus_I'])} & equivalence within $\pm 0.02$ not established \\",
          r"\bottomrule", r"\end{tabular}", "", r"\medskip",
          r"\textit{(d) Original full-development fits on the same 18 persons (descriptive)}\\[2pt]",
          r"\begin{tabular}{@{}llll@{}}", r"\toprule",
          r"Scale & Original fits & Retrained fits & Retrained minus original \\", r"\midrule",
          rf"Logit TTI & {m(o['TTI_logit'])} & {m(v['TTI_logit'])} & {iv(*c['D19_2']['v19_minus_v15conf_logit'])} \\",
          rf"Raw TTI & {m(o['TTI_raw'])} & {m(v['TTI_raw'])} & {iv(*c['D19_2']['v19_minus_v15conf_raw'])} \\",
          r"\bottomrule", r"\end{tabular}", ""]
    OUT.write_text("\n".join(t))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
