# Replication round v9: MMASD and UTD-MHAD

Protocol: `configs/protocols/replication_v9.yaml` (frozen f48595a; implementation notes and the
gate outcome in `configs/protocols/REGISTRY.md`). Runs 2026-11-01 13:58 to 2026-11-02 15:35 UTC
on one on-demand g5.2xlarge (an EC2 instance, us-east-1c), 25.6 h, about USD 31, after
two gate batches (an EC2 instance, an EC2 instance, about USD 0.30). 2,474 fits
(1,130 MMASD, 1,344 UTD-MHAD; 1,760 of them permutation-null fits), no failed fit, no failed
score. CompactCTRGCN, three seeds, every subject scored out of fold. Subject-clustered paired
bootstrap, 2,000 resamples, seeds averaged inside resamples. Analysis `experiments/analyze_v9.py`
ran on the instance; tables in `analysis/` (this directory links to the local S3 mirror
`results/aws/v9/results`).

Scale: R and SE on the logit-AUC scale. No equivalence margins (pre-registered: 8 to 11
subjects cannot support them). "Above null" means above the 95th percentile of 20 SEs from
models trained with the label-to-severity pairing permuted across subjects.

## Gates

| host | gate | value | outcome |
|---|---|---|---|
| mmasd_maracas (pre-registered first choice) | out-of-fold clean AUC >= 0.65 | 0.539 (11 children, 174 clips) | **failed**; replaced by the pre-registered fallback before any further cell |
| mmasd_chest_squat (fallback) | same | 0.736 (11 children, 152 clips); quality-only 0.48; length-only 0.66 | passed |
| utd_mhad_body_vs_arm | out-of-fold clean AUC <= 0.97 | 0.960 (8 subjects, 514 clips); quality-only 0.49; length-only 0.64 | passed, near the ceiling |

## Verdicts

| rule | MMASD chest_expansion vs squat | UTD-MHAD whole-body vs arm |
|---|---|---|
| R1 SE > 0, interval excluding 0, above null, every mechanism | **FAIL**. runs 0.378 [0.125, 0.993], null p95 0.420, p = 0.14; armruns 0.130 [−0.047, 0.494]; whole 0.028 [−0.415, 0.575]; frozen −0.020 [−0.113, 0.159] | **FAIL**. whole **0.651 [0.195, 1.236]**, null p95 0.386, p = 0.048 (passes); runs 0.265 [−0.235, 1.057], p = 0.29; frozen 0.236 [−0.193, 0.740], p = 0.29 |
| R2 SE(mid) − SE(low) > 0 | **PASS** 0.365 [0.041, 1.215] (low: 0.014 [−0.279, 0.148]) | **FAIL** −0.166 [−0.728, 0.683] (low: 0.430 [0.098, 0.678]) |
| R3 dropout does not lower SE (interval not entirely below 0) | PASS, 4 of 4: runs −0.234 [−0.508, 0.079]; whole −0.030; frozen +0.049; armruns −0.090 | PASS, 3 of 3: runs +0.633 [−0.289, 1.382]; whole −0.275 [−0.774, 0.166]; frozen +0.288 |
| R4 pairs where dropout raises SE (ASDPose 3 of 4) | 0 of 4 | 0 of 3; coordinate dropout raises SE under runs, +0.522 [0.177, 0.912] |
| R5 reference under own operator | runs **+0.261 [+0.115, +0.531]**; whole +0.073 [−0.302, +0.585]; frozen −0.032 [−0.197, +0.119]; armruns −0.043 [−0.406, +0.146] | runs **+0.440 [+0.139, +0.665]**; whole +0.142 [−0.143, +0.425]; frozen +0.368 [−0.058, +0.717] |
| R6 length not exploited (\|Spearman\| < 0.10 within class, all A_none and real cells) | **FAIL**, 25 of 33 flagged | **FAIL**, 22 of 24 flagged |
| R7 real MMASD cells | reported below | n/a |
| replicates (R1, R2, R3 on both hosts) | **no** | **no** |

## Reading

1. **The planted cue is learned on both hosts.** Under their own operator the World S models
   respond by 0.639 (MMASD runs), 0.704 (UTD runs) and 0.793 (UTD whole), against 0.777 and
   0.478 on ASDPose. Under frozen frames and arm confidence on MMASD the World S models barely
   respond (−0.051, +0.087): on a host where those failure forms are rare (frozen median 0)
   the operators write a cue the model does not pick up at 25 epochs on 116 training clips.
2. **The invariance reference is not invariant on these hosts.** World I models respond to
   the missing-runs operator they were trained against by +0.26 (MMASD) and +0.44 (UTD-MHAD);
   on ASDPose the same reference read +0.02. The shortcut effect, which subtracts the
   reference, is therefore 0.3 to 0.4 smaller than the cue-trained response. With 152 and 514
   test clips the reference is also unstable: the seed spread of R(f_I) is 0.18 (MMASD runs)
   and 0.19 (UTD runs) against 0.05 to 0.08 for the cue-trained models.
3. **Intervals and the null are too wide for the pre-registered direction tests.** Interval
   widths for SE are 0.8 to 1.3 logits; the permutation null has standard deviation 0.25 to
   0.32 and 95th percentiles of 0.39 to 0.59. Only an SE above about 0.6 can clear it, and one
   (host, mechanism) pair does. The round was known to be underpowered for equivalence; it
   turns out to be underpowered for direction as well, except for the largest effects.
4. **SE is unstable across test-mask realizations on these hosts.** The range over the three
   test-mask seeds is 0.70 (UTD runs), 0.42 (MMASD whole), 0.38 (UTD frozen); on ASDPose it was
   0.011. A single mask realization on 150 to 500 clips moves SE by more than the effect.
5. **No remedy lowers the shortcut effect on any host** (R3 on both; 7 of 7 host-mechanism
   pairs, as 4 of 4 on ASDPose). **The ASDPose finding that whole-joint dropout raises SE does
   not replicate**: 0 of 7 pairs, point estimates in both directions. Coordinate dropout
   raises SE under runs on UTD-MHAD (+0.52 [0.18, 0.91]).
6. **Scaling replicates on MMASD and not on UTD-MHAD.** On UTD-MHAD the low-strength cell
   gives the cleaner result (SE 0.43 [0.10, 0.68], reference −0.07) and the mid cell the
   noisier one (reference +0.44). On a host near the AUC ceiling (0.96 clean), the stronger
   planting degrades the World I model more than the World S model gains.
7. **Length.** The threshold was set on ASDPose, whose two classes are homogeneous. Here each
   class pools several actions of different typical length (UTD whole-body actions 55 to 87
   frames by action), so within-class score-length correlations (median |rho| 0.23 in the
   UTD positive class) arise through the action, not through length as an input. Length is
   not recoverable from the penultimate features (ridge R² at most 0.10, median negative).
   The rule fails as written and is reported as such; the reading is that the rule needs a
   class-homogeneity condition, to be fixed in any later protocol, not in this one.
8. **Real MMASD cells (R7).** The two cells trained on natural chest_expansion vs squat data
   respond to arm confidence by +0.14 (A_none, [−0.07, +0.47]) and +0.15 (A_dropout, [+0.02,
   +0.43]) against a reference of −0.04; to missing runs by −0.18 and −0.17 against a reference
   of +0.26; to frozen frames and whole-joint absence by about zero; to gapfill by +0.02 and
   +0.07. Opposite-sign children 2 to 6 of 11. No reliance claim is made.
9. **Crossed table.** Invariance is operator-specific here too: the MMASD armruns reference
   responds −0.31 [−0.65, −0.07] to missing runs; the UTD whole reference responds +0.55
   [+0.31, +0.83] to frozen frames and the UTD runs reference +0.40 to frozen frames.
10. **Comparators (6 real models).** Every Spearman interval spans zero except sensitivity
    pairs with each other; nothing can be said about rank agreement from 6 models.

## Consequences for the paper

- The replication is reported as failed on both hosts under the frozen rules, all cells
  included. The sentence the data support: on two further hosts with 8 to 11 subjects the
  cue-trained response is of ASDPose size, the invariance reference is host-dependent and not
  zero, and the shortcut effect is positive in point estimate for the mechanisms with natural
  counterparts, with one host-mechanism pair clearing its permutation null.
- The remedy result narrows to what held on all three hosts: no tested remedy lowers the
  shortcut effect. The increase under whole-joint dropout is an ASDPose observation.
- The method section gains a stated requirement: a matched reference must be shown to be
  stable (seed spread, mask realizations) on the host before SE is read; on small hosts it was
  not. Test sets of 8 to 11 subjects are below what the contrast needs.
- Gate and host switch, the length-rule failure and the mask instability go into the
  limitations.
