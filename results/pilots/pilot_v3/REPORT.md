# Pilot v3 (GPU, TinyTCN, 3 seeds): planted worlds and fresh-seed confirmation

Protocol: `configs/protocols/pilot_asdpose_v3.yaml` (frozen 2d18db8). Runs 2026-10-24, 45 runs,
1.70 GPU-h; local total so far 3.50 GPU-h. Analysis: `experiments/analyze_v3.py --n-boot 10000`;
numbers in `planted.csv`, `summary.json`. Pilot numbers decide go/no-go only.

Host: ASDPose segments passing QC in both classes (no natural cue). World S plants joint-missing
runs more heavily in non-SMM segments; World I plants the same amount independently of the label.
Every model is scored on the clean host test, an S-planted test and an I-planted test.
Measured reliance R = logit(AUC_S) − logit(AUC_I).

## Planted cells (mid strength unless noted)

| world | cell | AUC clean | AUC S | AUC I | R [95% CI] |
|---|---|---|---|---|---|
| S | A_none | 0.660 | 0.746 | 0.588 | 0.723 [0.597, 0.855] |
| S | A_velocity | 0.760 | 0.804 | 0.673 | 0.689 [0.564, 0.825] |
| S | A_dropout | 0.666 | 0.751 | 0.605 | 0.679 [0.570, 0.795] |
| S | A_both | 0.757 | 0.804 | 0.677 | 0.668 [0.533, 0.807] |
| S | B_coorddrop | 0.664 | 0.745 | 0.590 | 0.710 [0.586, 0.843] |
| I | A_none | 0.669 | 0.664 | 0.674 | −0.042 [−0.113, 0.028] |
| I | A_velocity | 0.743 | 0.721 | 0.724 | −0.015 [−0.096, 0.066] |
| I | A_dropout | 0.709 | 0.710 | 0.718 | −0.039 [−0.106, 0.024] |
| I | A_both | 0.761 | 0.749 | 0.741 | 0.042 [−0.036, 0.116] |
| I | B_coorddrop | 0.705 | 0.694 | 0.702 | −0.038 [−0.111, 0.034] |
| S, low | A_none | 0.661 | 0.693 | 0.625 | 0.302 [0.227, 0.379] |
| I, low | A_none | 0.663 | 0.665 | 0.670 | −0.022 [−0.072, 0.028] |

## H2 and H3: PASS. The matched-world reliance measure recovers the planted shortcut

- Shortcut effect (R of the World S model minus R of the World I model), A_none: mid 0.765
  [0.622, 0.910], low 0.324 [0.229, 0.417]; mid minus low 0.441 [0.293, 0.592].
- Every World I model has R within noise of zero (−0.04 to +0.04), so the measure has no offset
  when no shortcut was learnable. On real data (v2) the raw QC gap had a 0.24 offset for a cue-free
  model; the matched comparison is what removes it.
- The measure scales with planted strength (H3), which is the ground-truth validation
  contribution A needs.

## H5b (decomposition of remedy gains): INCONCLUSIVE under the frozen rule. Read plainly, the
## remedies regularize and do not remove the planted shortcut

| remedy | clean gain in S | gain in I (regularization part) | shortcut part | change in R |
|---|---|---|---|---|
| A_velocity | +0.100 [+0.059, +0.145] | +0.074 [+0.042, +0.108] | +0.027 [−0.005, +0.058] | −0.034 [−0.101, +0.035] |
| A_dropout | +0.006 [−0.005, +0.019] | +0.040 [+0.023, +0.056] | −0.034 [−0.050, −0.016] | −0.044 [−0.078, −0.012] |
| A_both | +0.097 [+0.053, +0.142] | +0.092 [+0.059, +0.125] | +0.006 [−0.028, +0.040] | −0.055 [−0.125, +0.017] |
| B_coorddrop | +0.005 [−0.011, +0.021] | +0.036 [+0.021, +0.050] | −0.031 [−0.049, −0.012] | −0.013 [−0.042, +0.016] |

The frozen rule returns "inconclusive" because the dropout shortcut parts are significantly
negative rather than zero. The substance: the clean-accuracy gains of every remedy are matched or
exceeded in World I, where there is no shortcut to remove, and the shortcut stays learned under
every remedy (shortcut effect 0.63 to 0.75). Whole-joint dropout at rate 0.15 does not hide a
frame-level missing-fraction cue, and it helps less in World S than in World I.

Also: learning the shortcut barely costs clean accuracy (A_none clean 0.660 in S vs 0.669 in I).
On this host the planted shortcut is learned and used without corrupting behaviour accuracy.

## H6 (fresh real-data seeds 3–5): FAIL under the frozen rule

| real ASDPose, seeds 3–5 | standard | QC | logit gap [95% CI] |
|---|---|---|---|
| A_none | 0.806 | 0.705 | 0.551 [0.458, 0.665] |
| A_both | 0.842 | 0.773 | 0.449 [0.372, 0.541] |
| B_coorddrop | 0.818 | 0.717 | 0.574 [0.482, 0.687] |

A_both − A_none: −0.101 [−0.146, −0.067] (the reliance reduction replicates on fresh seeds).
B_coorddrop − A_none: +0.023 [+0.005, +0.039], so the frozen "contains 0" condition fails.
Regularization with the cue visible slightly raises the gap; it never lowers it.

## What the pilot now supports, and what it does not

Supported:
1. A matched-world reliance measure that is zero when no shortcut is learnable and scales with
   planted strength (H2, H3), while the raw QC gap carries a cue-free offset (v2: 0.24) and a wide
   permutation null (v2). This is a methodological result with ground truth behind it.
2. On real ASDPose, the SAFE-SMM remedies lower the QC gap (v2 3 seeds, v3 fresh 3 seeds) on an
   architecture with no graph; regularization alone does not (B_coorddrop).

Not supported:
3. "The remedies remove the shortcut" as a general mechanism. Under a planted frame-level
   missingness shortcut they only regularize. Their real-data reliance reduction is specific to
   the real artifact's form, which the planted operator does not reproduce.
4. "The shortcut corrupts training" (accuracy cost): small on the planted host.

Caveats: one weak architecture (TinyTCN, clean host AUC 0.66–0.76); one planted mechanism; the
planted host is the QC subset of one dataset.
