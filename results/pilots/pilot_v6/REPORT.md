# Pilot v6: validation round for the reliance statistic (from the 2026-10-25 review)

Protocol: `configs/protocols/pilot_asdpose_v6.yaml` (frozen 062a1ba). Runs 2026-10-25: five
TinyTCN host runs (0.25 GPU-h) and the scoring grid on 47 checkpoints (0.35 GPU-h). Host test:
2,440 QC-symmetric segments, 40 children. Paired child-clustered bootstrap, 10,000 resamples,
seeds averaged inside resamples. Analysis: `experiments/analyze_v6.py`; numbers in `groups.csv`,
`summary.json`, `oracles.json`, `host_probe.json`.

## Verdicts

| test | verdict | result |
|---|---|---|
| V1 invariance-trained models read zero under all three operators | **FAIL** | zero: −0.03 [−0.11, +0.06] (TinyTCN), −0.06 [−0.13, +0.01] (Compact); gapfill: +0.03, **+0.19 [+0.13, +0.27]**; conflow: **+0.25 [+0.16, +0.33]**, **−0.16 [−0.25, −0.06]**; mirror under zero: −0.10 [−0.20, +0.01], −0.09 [−0.17, −0.01] |
| V2 cue-only operator separates cue-trained from invariant | **FAIL** | cue-trained read +0.26, +0.35 (CIs above 0) as required; invariant models do not read zero (above) |
| V3 information-loss component (gapfill) | reported | clean-trained cue-free model +0.055 [+0.023, +0.090]; full-split TinyTCNs +0.08 to +0.14; Compact +0.19 to +0.20 |
| V4 direction | **PASS** | Sp-trained TinyTCN R = −0.71 [−0.83, −0.60]; SE_mirror = +1.51 [+1.30, +1.74] |
| V5 crossed strengths | **PASS** | at mid test: mid-trained SE 0.83 [0.71, 0.96], low-trained 0.61 [0.52, 0.71], difference 0.23; at low test: 0.54 vs 0.33, difference 0.21 [0.08, 0.33] |
| V6 mask realizations | **PASS** | SE over three test masks 0.83, 0.86, 0.74 (range 0.12, CI half-width 0.12); SE from two training-mask worlds 0.81, 0.84, inside the seed-0 CI [0.73, 0.99] |
| V7 clean-accuracy equivalence, margin ±0.02 | **FAIL** (one of two) | Compact −0.007 [−0.020, +0.005] inside the margin; TinyTCN −0.009 [−0.046, +0.029] too wide |

Nonlinear host probe: gradient boosting on the four tracking statistics, host train → host
validation, 0.568 [0.538, 0.600]; logistic 0.530. The host is a quality-filtered pool with weak
measured association, not a cue-free world.

## What the operators do at the tracking-statistic level (cue-only oracles)

Logistic regression on the four statistics recomputed from the planted test arrays, fit on the
host validation children under the same world: zero S 0.719, I 0.512, Sp 0.720; conflow S
0.727, I 0.472, Sp 0.716; gapfill S 0.523, I 0.519, Sp 0.521; clean 0.519. The zero and conflow
operators write a cue of the natural strength (0.714); gapfill writes none. Gradient boosting
agrees within 0.04.

## Reading

1. **The between-world shortcut effect is the validated quantity.** With the same operator in
   training and test it flips sign exactly with the planted direction (V4), orders training
   strengths correctly at either test strength (V5), and is stable across test and training
   mask realizations (V6). Its reference model reads zero under the operator it was trained
   with, for both architectures, under three test-mask seeds, on both the host and (v5) the real
   split.
2. **Invariance is operator-specific, and R(f) is an operator-specific response.** The
   World-I-trained models read zero under the zero operator and not under gapfill or conflow,
   in opposite directions for the two architectures. A model is invariant to the degradation it
   was trained with, not to tracking failure in general. R(f) for one model under one operator
   therefore means "label-asymmetric response to this operator" and nothing stronger; "R = 0
   means no dependence" is withdrawn.
3. **The reviewer's information-loss component is real and measurable.** Gapfill removes the
   masked motion without writing a cue; cue-trained models barely respond (+0.01), the
   clean-trained cue-free model responds a little (+0.055), the full-split TinyTCNs more (+0.08 to
   +0.14) and Compact most (+0.19). Smoothing a segment moves it toward the non-SMM class for
   these models, more so when the smoothing lands on negatives. This is the "difficulty"
   component that World I was supposed to remove and does not; it is now reported per model
   as R under gapfill.
4. **The cue-free model's v5 reading decomposes.** Under zero it reads +0.29, under gapfill
   +0.055, under conflow +0.03. Most of its response is to the zero pattern itself (coordinates
   and confidence both zero), which it treats as non-SMM evidence; little is information loss
   and little is the confidence cue alone. In deployment terms it still depends on tracking
   quality, through the pattern that real failures produce.
5. **Decisions move, not only AUC.** Between S and I planting 12% to 37% of host-test
   decisions at threshold 0.5 flip; cue-trained models push the two classes apart (paired logit
   change +0.67 for SMM, −0.30 for non-SMM), the invariant TinyTCN moves both classes the same
   way (+0.04, +0.04). The remedy cell "both" flips 26% of decisions and moves negatives toward
   SMM (+0.13), the reversed response v5 saw.
6. **Clean-accuracy equivalence** holds within ±0.02 for Compact and is undetermined for
   TinyTCN; the paper reports the Compact interval and drops the "costs nothing" wording.

## Consequences for the paper and for the AWS round

- The paper's validated contribution is the matched-world shortcut effect with its
  invariance-trained reference, under a stated operator. Validation now rests on direction,
  strength ordering at crossed test strengths, mask realizations and two architectures.
- The real-data section becomes "responses of published models to specified tracking failures",
  reported per operator (zero, conflow, gapfill), with the gapfill response as the
  information-loss bound and the clean-trained and invariance-trained references beside it.
  Remedy contrasts stay paired and within architecture.
- AWS round (v7): Compact remedy cells trained in both worlds (so the remedy interaction is
  identified on the published architecture), three seeds, under the length-controlled protocol;
  the real full-split cells with three seeds under the same protocol; scoring under the v6
  operator grid; length recoverability measured.
