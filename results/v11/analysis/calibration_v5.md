# Calibration on the external set, post hoc extensions

Prevalence 127/550 = 0.2309; a constant predictor at the pooled prevalence has Brier 0.1776.
Fold-fitted constant predictor (training-fold prevalence, held-out folds): Brier 0.1777 to 0.1777 (identical across models up to the fold prevalences), ECE 0.0004 to 0.0004.

| quantity | temperature only | Platt (intercept + slope) |
|---|---|---|
| Brier, range over 49 models | 0.198 to 0.257 | 0.166 to 0.178 |
| ECE, range | 0.140 to 0.284 | 0.004 to 0.071 |
| models firing at 0.85 (rate > 0) | 12 of 49, max 0.018 | 2 of 49, max 0.004 |

Before any scaling: Brier 0.198 to 0.513, ECE 0.149 to 0.519.
Temperatures: 106 of 245 fold fits at the 54.6 bound; the displayed temperature is the mean of the five fold temperatures.
AUROC under temperature scaling: within-fold AUROC is unchanged (max change 0.0e+00); pooled out-of-fold AUROC changes by at most 0.0107 across models because fold-specific temperatures can reorder scores across folds.
Firing-rate aggregation, ours: none: ensembled predictions (mean of three seed scores, then threshold) fire on 0.287 of windows (the transfer table); per-seed rates are 0.269, 0.340, 0.385 (mean 0.332; the calibration tables list seeds).
