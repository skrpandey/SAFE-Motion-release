# Permutation-null floors and interval caveats (revision item R13)

Source: `results/v9/analysis/null.csv` and `effects.csv` (protocol v9), `results/v11/REPORT.md` (protocol v11). No new permutations were run (the 200-permutation rerun is roadmap item S4, not approved). Written 2026-11-03 in the revision pass; nothing recomputed beyond counting.

## v9 small-host nulls

Each null has N = 20 permuted-world shortcut effects (the world assignment permuted across subjects, A_none, seed 0; the YAML's N_PERM = 20). The empirical p is (1 + #{null >= SE}) / (1 + N), so the smallest attainable p is 1/21 = 0.0476; "p = 0.047" and "p = 0.048" in the reports are the same floor value rounded. The reported "95th percentile" of 20 values is numpy's linear interpolation between the 19th and 20th order statistics, so it sits just below the maximum of the 20 and is not a calibrated 95th percentile. "Above null" therefore means "above all but about one of 20 draws".

| host | mechanism | strength | SE | N perm | max of null | reported 95th pct. | #{null >= SE} | p | at floor |
|---|---|---|---|---|---|---|---|---|---|
| mmasd_chest_squat | armruns | mid | +0.130 | 20 | +0.344 | +0.181 | 3 | 0.1905 | no |
| mmasd_chest_squat | frozen | mid | -0.020 | 20 | +0.135 | +0.083 | 11 | 0.5714 | no |
| mmasd_chest_squat | runs | mid | +0.378 | 20 | +0.425 | +0.420 | 2 | 0.1429 | no |
| mmasd_chest_squat | whole | mid | +0.028 | 20 | +0.614 | +0.589 | 10 | 0.5238 | no |
| utd_mhad_body_vs_arm | frozen | mid | +0.236 | 20 | +0.498 | +0.384 | 5 | 0.2857 | no |
| utd_mhad_body_vs_arm | runs | mid | +0.265 | 20 | +0.669 | +0.412 | 5 | 0.2857 | no |
| utd_mhad_body_vs_arm | whole | mid | +0.651 | 20 | +0.478 | +0.386 | 0 | 0.0476 | yes |

Only one pair sits at the floor: UTD-MHAD whole-joint absence (SE 0.651 above all 20 draws, p = 1/21). With N = 20 that is the strongest statement the null can make; it cannot distinguish p = 0.048 from p = 0.005.

Low-strength cells have no permutation null in v9 (the YAML planted the null for A_none at mid only), so their "above null" column is empty by design.

## Interval coverage with 8 to 11 clusters

The v9 intervals are percentile intervals from a subject-clustered bootstrap over 11 children (MMASD) or 8 subjects (UTD-MHAD), seeds averaged inside resamples. With so few clusters the percentile method's coverage is not established and is expected to be below nominal; the manuscript should label them nominal 95 percent intervals. No BCa or cluster-t interval was computed in this pass.

The pilot permutation null of the filtered gap (ten label-shuffled TinyTCNs, protocol v2) has N = 10, so its p floor is 1/11 = 0.091; the reported p = 0.18 for the no-remedy TinyTCN is 1 of 10 draws at or above the observed gap under the (1 + k)/(1 + N) rule.

## v11 nulls (T1 and T4)

The label-permutation nulls for the within-set quality-only classifiers permute labels across all 550 windows, not within source video, while the fits themselves are video-grouped 5-fold (deviation 5 in `results/v11/REPORT.md`). Because windows from one video share tracking conditions, a within-video permutation would be the stricter null and the reported p = 0.035 (gradient boosting, RTMPose) may be understated. N = 200 there, so the floor is 1/201 = 0.005, which is the value reported for the MediaPipe quality-only AUCs.
