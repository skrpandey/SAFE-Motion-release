# Round v11: the SSBD/ESBD transfer test, reported in full

Protocol: `configs/protocols/transfer_v11.yaml` (frozen b69599d). Run 2026-11-03 on the local
machine: scoring and within-set training on the local GPU (about 54 GPU-minutes in total:
RTMPose scoring 12.8 min, MediaPipe scoring 19 min, within-set training 22 min, the last two
overlapping), MediaPipe extraction and analysis on CPU (13 min with 6 workers, 5 min). Code:
`experiments/score_v11.py`, `build_mediapipe_external.py`, `within_set_v11.py`, `analyze_v11.py`,
`make_tables_v11.py`; tests `tests/test_v11.py`. Outputs under `results/v11/` (scores,
scores_mediapipe, features, dfr, within_set, mediapipe_cache, mediapipe_gate.json, folds.json,
analysis/*.csv and criteria.json, label_noise_sample.csv, label_noise_sheets/). Paper tables
`paper/tables/v11_*.tex` are generated from the analysis files. Raw MediaPipe negative windows
at `<data root>/stimming/mediapipe-negatives/` (new files only).

External set (read-only, the base project's cache of 2026-08-20): 550 windows of 200 frames at
30 fps, 127 positives (annotated stereotypy clips) and 423 negatives cut from the same 127
source videos outside every annotated span, RTMPose COCO-17 with confidence, one child-selection
rule for both classes. Every interval is a video-clustered paired bootstrap over the 127 source
videos (2,000 resamples, seeds averaged inside resamples). Models: our five v7 real cells
(three seeds), their five v10 real-data World I references (three seeds), and the 19 published
SAFE-SMM checkpoints (v1, v2 family, v3, v3 no-confidence family, four ablations, base, two
permuted controls, ST-GCN and AGCN pairs). "ASDPose AUC" is each model's own full-split test AUC.

## T1. Is the ASDPose tracking-quality artifact release-specific?

| quality-only classifier | AUC [95% CI] | null 95th pct. | permutation p |
|---|---|---|---|
| logistic regression fit on ASDPose, applied here | 0.535 [0.489, 0.579] | | |
| within-set logistic regression (video-grouped 5-fold) | 0.557 [0.500, 0.616] | 0.560 | 0.080 |
| within-set gradient boosting | 0.560 [0.508, 0.612] | 0.554 | 0.035 |
| ASDPose standard split, reference | 0.714 | | |

**Reading rule outcome: "weaker on this set", not "release-specific".** The rule needed the
within-set upper ends below 0.60 and p above 0.05; the upper ends are 0.61 and 0.62 and the
gradient-boosting p is 0.035. The four tracking statistics carry a small association with the
label here (about 0.56 against 0.71 on ASDPose), too small to be the kind of cue ASDPose has,
but not shown to be absent.

## T2. Transfer

| model | ASDPose | external AUC [95% CI] | arm flapping | spinning | head banging | hand action (n = 9) | firing at 0.85 | false alarms at 0.85 |
|---|---|---|---|---|---|---|---|---|
| SAFE-SMM v1 | 0.843 | 0.649 [0.596, 0.700] | 0.78 | 0.63 | 0.51 | 0.60 | 0.23 | 0.18 |
| SAFE-SMM v2 (3 seeds) | 0.944 | 0.580 [0.533, 0.626] | 0.70 | 0.57 | 0.48 | 0.38 | 0.47 | 0.43 |
| SAFE-SMM v3 | 0.953 | 0.568 [0.517, 0.619] | 0.59 | 0.62 | 0.50 | 0.38 | 0.47 | 0.44 |
| v3, no confidence (3) | 0.956 | 0.563 [0.516, 0.610] | 0.65 | 0.54 | 0.50 | 0.44 | 0.28 | 0.25 |
| base, no remedies | 0.864 | 0.608 [0.551, 0.661] | 0.63 | 0.67 | 0.54 | 0.39 | 0.29 | 0.25 |
| ours (v7) none | 0.854 | 0.641 [0.591, 0.688] | 0.71 | 0.64 | 0.55 | 0.60 | 0.28 | 0.22 |
| ours (v7) velocity | 0.907 | 0.577 [0.528, 0.624] | 0.61 | 0.62 | 0.49 | 0.47 | 0.39 | 0.36 |
| ours (v7) whole-joint dropout | 0.856 | 0.625 [0.576, 0.672] | 0.70 | 0.67 | 0.49 | 0.52 | 0.28 | 0.22 |
| ours (v7) velocity + dropout | 0.908 | 0.578 [0.531, 0.621] | 0.64 | 0.62 | 0.48 | 0.40 | 0.31 | 0.28 |
| ours (v7) coordinate dropout | 0.855 | 0.628 [0.579, 0.676] | 0.70 | 0.62 | 0.56 | 0.51 | 0.24 | 0.19 |
| ours, real-data World I, none | 0.846 | 0.640 [0.594, 0.685] | 0.69 | 0.65 | 0.57 | 0.56 | 0.10 | 0.08 |
| ours, real-data World I, dropout | 0.842 | 0.638 [0.593, 0.685] | 0.71 | 0.64 | 0.55 | 0.54 | 0.12 | 0.09 |
| ST-GCN base / remedies | 0.837 / 0.903 | 0.598 / 0.586 | 0.68 / 0.68 | 0.62 / 0.63 | 0.46 / 0.43 | 0.55 / 0.43 | 0.17 / 0.42 | 0.14 / 0.39 |
| AGCN base / remedies | 0.754 / 0.790 | 0.631 / 0.606 | 0.64 / 0.67 | 0.73 / 0.71 | 0.49 / 0.43 | 0.54 / 0.40 | 0.00 / 0.04 | 0.00 / 0.03 |
| permuted controls v1 / v2 | 0.606 / 0.587 | 0.578 / 0.542 | | | | | 0.00 / 0.00 | |

Full table with intervals per category: `analysis/t2_transfer.csv`, `paper/tables/v11_transfer.tex`.

Paired contrasts (raw AUC, shared video resamples): v3 minus v1 **−0.081 [−0.134, −0.025]**;
v2 minus v1 −0.069 [−0.113, −0.025]; v3 minus base −0.040 [−0.085, +0.007]; ours velocity minus
none **−0.064 [−0.094, −0.036]**; ours velocity + dropout minus none **−0.063 [−0.095, −0.029]**;
dropout minus none −0.016 [−0.036, +0.003]; coordinate dropout minus none −0.013; each v7 cell
minus its real-data World I counterpart: none +0.001, velocity −0.033 [−0.058, −0.007], dropout
−0.014, both −0.027, coordinate dropout −0.008.

**Reading.** Transfer is weak for every model (0.54 to 0.65) and the ranking inverts: the models
with the highest ASDPose accuracy (v2, v3, our velocity cells) transfer worst, and the plainest
models (v1, our no-remedy cells, AGCN base) transfer best. The velocity channel is the common
factor of the worst transfers among our cells. Head banging is near chance for every model;
arm flapping and spinning carry what transfer there is. At the published 0.85 operating point
the published v2 and v3 flag 47 percent of all windows and 43 to 44 percent of the negatives;
the real-data World I references flag 10 to 12 percent. The released ASDPose checkpoint could
not be scored (deviation 1).

## T3. Confidence channel zeroed

Zeroing the confidence channel lowers the external AUC for most models (ours none −0.050
[−0.079, −0.023]; coordinate dropout −0.066; v1 −0.062 [−0.109, −0.015]; base −0.035) and
raises it for v3 (+0.028 [−0.015, +0.074], undetermined) and the v3-normalisation velocity
ablation (+0.042, undetermined). The base project's observation that zeroing helps v3 is
reproduced in direction but is not distinguishable from zero. Full table: `analysis/t3_confidence.csv`.

## T4. Estimator swap on both classes (MediaPipe)

Gate (`mediapipe_gate.json`): MediaPipe 0.10.9, 423 negative windows re-extracted from the
source videos at the RTMPose frame spans, 0 failures, same identifiers and order, every window
200 frames, 12 windows with no detection (the same no-person windows as RTMPose), detection in
89 percent of frames, nose above shoulders in 90 percent of detected frames, left and right
shoulders distinct in 99.998 percent; the fallback did not apply. Resampling reproduces the
RTMPose cache rule to float precision (verified on five windows).

| | RTMPose | MediaPipe |
|---|---|---|
| quality-only within-set, logistic regression | 0.557 [0.500, 0.616], p = 0.08 | **0.698 [0.649, 0.748], p = 0.005** |
| quality-only within-set, gradient boosting | 0.560 [0.508, 0.612], p = 0.035 | **0.715 [0.669, 0.761], p = 0.005** |
| low-confidence fraction, mean, negatives / positives | 0.28 / 0.25 | 0.28 / 0.35 |
| missing fraction, mean | 0.08 / 0.04 | 0.10 / 0.16 |
| arm low-confidence fraction, mean | 0.23 / 0.21 | 0.23 / 0.34 |

Model AUC under MediaPipe input, paired difference against RTMPose: every model but one drops,
by −0.05 to −0.21 (ours none 0.641 to 0.544, −0.097 [−0.143, −0.050]; v1 0.649 to 0.556; v3
0.568 to 0.518, −0.050 [−0.113, +0.009]; real-data World I none 0.640 to 0.483, −0.157). The
permuted v2 control rises from 0.542 to 0.692 [+0.086, +0.214]. Positive-class logit shifts are
−0.9 to −4.3 for every trained model. Full table: `analysis/t4_estimator.csv`.

**Reading.** Under MediaPipe the tracking statistics predict the label at 0.70 to 0.72, as
strongly as on ASDPose, with the opposite sign of association: positives are worse tracked
(higher low-confidence, missing and arm low-confidence fractions). A single-person estimator
without a child-selection rule loses or mis-selects the child more often in the stereotypy
clips than in the surrounding footage. The extractor therefore writes a quality-label
association of its own, and a label-free model that reads quality (the permuted control) gains
from it. Transfer AUCs measured under one extractor do not carry to another, and the "no cue
here" result of T1 holds for RTMPose with its child-selection rule, not for the videos as such.
Disclosed differences stand: MediaPipe tracks one person with no child selection, and its
visibility is not a detector confidence.

## T5. Responses to the planted operators, relative to the real-data reference

Mask seed 2000, R = logit AUC under S minus logit AUC under I; "ref" is the v10 real-data
World I counterpart; entries are cell / ref / paired difference; bold where the difference
excludes zero.

| cell | missing runs | whole-joint absence | frozen frames | arm confidence | gap filling |
|---|---|---|---|---|---|
| none | +0.11 / −0.06 / **+0.16 [+0.04, +0.30]** | +0.21 / +0.03 / **+0.18 [+0.06, +0.31]** | +0.31 / +0.27 / +0.04 | +0.02 / +0.01 / +0.01 | +0.16 / +0.10 / +0.06 |
| velocity | +0.17 / +0.01 / **+0.16 [+0.05, +0.28]** | +0.05 / −0.00 / +0.05 | +0.21 / +0.18 / +0.04 | −0.05 / −0.01 / **−0.04** | +0.00 / +0.08 / −0.07 |
| whole-joint dropout | +0.07 / +0.01 / +0.06 | +0.05 / +0.02 / +0.03 | +0.32 / +0.31 / +0.01 | +0.02 / −0.00 / +0.03 | +0.11 / +0.08 / +0.03 |
| velocity + dropout | +0.13 / +0.06 / +0.07 | +0.02 / +0.04 / −0.02 | +0.21 / +0.24 / −0.02 | −0.02 / +0.00 / **−0.03** | +0.03 / +0.08 / −0.05 |
| coordinate dropout | +0.05 / +0.00 / +0.05 | +0.24 / +0.11 / **+0.14 [+0.03, +0.25]** | +0.33 / +0.31 / +0.02 | +0.02 / +0.02 / +0.00 | +0.15 / +0.14 / +0.01 |

Across the three missing-runs mask seeds, R of the none cell is 0.107, 0.094, 0.095 and its
reference −0.055, −0.129, −0.127. **Reading.** The frozen-frame response is the largest here as
on ASDPose (+0.21 to +0.33) and is matched by the references (+0.18 to +0.31), so it is generic
sensitivity. Under missing runs and whole-joint absence the none, velocity and coordinate-dropout
cells respond more than their references by 0.14 to 0.18; the dropout cells do not. Per the v10
wording rule no reliance sentence follows (Part A of v10 could not establish a cue reading for
these operators); these are responses relative to the real-data reference.

## T6. Head versus representation

| | out-of-fold AUC [95% CI] |
|---|---|
| within-set ceiling, CompactCTRGCN from scratch (3 seeds) | 0.651 [0.602, 0.698]; seeds 0.636, 0.664, 0.654 |
| within-set ceiling, TinyTCN from scratch (3 seeds) | 0.629 [0.578, 0.676] |
| DFR (last layer refit on external features), v1 | 0.684 [0.631, 0.734], +0.035 over the full model |
| DFR, v3 | 0.660 [0.603, 0.714], **+0.092 [+0.027, +0.155]** |
| DFR, v2 family | 0.671, **+0.090 [+0.045, +0.137]** |
| DFR, ours none / velocity / dropout / both / coord. dropout | 0.671 / 0.621 / 0.668 / 0.645 / 0.665 (+0.03 to +0.07) |
| DFR, velocity ablation at v3 normalisation | 0.641, +0.142 (from 0.500) |
| DFR, AGCN base | 0.560, −0.071 [−0.132, −0.007] |

**Reading.** Refitting only the classifier head on the external features raises almost every
ASDPose-trained model to the level of a model trained within the set from scratch (0.62 to
0.69 against a ceiling of 0.63 to 0.65). The inversion of T2 disappears under DFR: v3's features
are as good as v1's once the head is refit. The transfer gap is therefore mostly a head and
calibration problem, not a representation problem, and the in-domain ceiling on this set is
itself low (0.65 with 127 positives). Full table: `analysis/t6_dfr.csv`.

## T7. Calibration

Temperature scaling fitted on the training folds and applied to the held-out fold; AUC was
checked unchanged in every fold for every model. Selected rows (full table
`analysis/t7_calibration.csv`):

| model | T | ECE before / after | Brier before / after | firing at 0.85 before / after | recall at 0.85 before / after |
|---|---|---|---|---|---|
| SAFE-SMM v1 | 5.5 | 0.279 / 0.255 | 0.285 / 0.232 | 0.23 / 0.00 | 0.38 / 0.03 |
| SAFE-SMM v2 | 54.6 (bound) | 0.457 / 0.277 | 0.453 / 0.252 | 0.48 / 0.00 | 0.59 / 0.00 |
| SAFE-SMM v3 | 54.6 (bound) | 0.450 / 0.276 | 0.446 / 0.252 | 0.47 / 0.00 | 0.56 / 0.00 |
| base, no remedies | 13.0 | 0.328 / 0.270 | 0.328 / 0.247 | 0.29 / 0.00 | 0.44 / 0.00 |
| ours none, seed 0 | 19.6 | 0.367 / 0.275 | 0.350 / 0.250 | 0.26 / 0.00 | 0.44 / 0.00 |

**Reading.** Every ASDPose-trained model is strongly overconfident on this set (ECE 0.28 to
0.46 before scaling; the fitted temperatures for v2 and v3 sit at the search bound of 54.6).
Scaling lowers ECE to about 0.26 to 0.28 and Brier to about 0.25, after which no model fires at
the published 0.85 operating point at all: with a prevalence of 23 percent and an AUC near 0.6,
no window reaches 0.85 once the scores are calibrated. At the fold-optimal F1 threshold the
firing rate is 0.36 to 0.75. The published operating point is not usable on this set.

## T8. Position in the video

| statistic | AUC [95% CI] | null 5th / 95th pct. | p (two-sided) |
|---|---|---|---|
| start position fraction, raw AUC | **0.282 [0.227, 0.336]** | 0.451 / 0.549 | 0.005 |
| video-grouped 5-fold logistic on position, out of fold | 0.716 | | |

**Pre-registered reading "no position cue": fails.** Positive windows start earlier in their
source video than negative windows (an AUC of 0.28 for the raw position, 0.72 once the sign is
learned). The models never see position, so this is not a cue they can use, but it says the
negatives come from later parts of the videos than the annotated behaviour, which is where
unannotated footage, and possibly unannotated stereotypy, accumulates. It is a validity threat
for the negative set, reported beside T9.

## T9. Label-noise audit: independently rated, with bounds (no human adjudication)

Procedure (recorded in `REGISTRY.md` before the sensitivity analysis was run): two independent raters rated all 100 sampled negative windows from the written rules
(`T9_AUDIT_PROMPT.md`); the 27 windows at least one of them called "yes" were re-rated by both raters under a stricter prompt (`T9_ADJUDICATION_PROMPT.md`). The protocol's manual viewing by
an author did not take place, and no model rating is presented as a human one. Rule fixed
before use (`experiments/t9_finish.py`): a flagged window is "yes" in the lower bound only if
both reruns say yes, and in the upper bound if either does; unflagged windows are "no" if both
first-pass raters said no and "unsure" otherwise. Rater files: `label_noise_answers_model{A,B}.csv`,
`label_noise_rerun27_model{A,B}.csv`. Raters A and B are independent and are not authors.

| rating | yes | no | unsure | yes fraction [Wilson 95%] |
|---|---|---|---|---|
| model A, first pass | 25 | 25 | 50 | 0.25 [0.18, 0.34] |
| model B, first pass | 6 | 67 | 27 | 0.06 [0.03, 0.13] |
| strict reruns on the 27: A / B | 14 / 7 | 1 / 13 | 12 / 7 | |
| **lower bound** (both reruns yes) | **5** | 25 | 70 | **0.05 [0.02, 0.11]** |
| **upper bound** (either rerun yes) | **16** | 25 | 59 | **0.16 [0.10, 0.24]** |

Agreement between the two first-pass raters: 47 of 100, Cohen's kappa 0.22; between the two
reruns on the 27: 9 of 27, kappa 0.09. The disagreement is concentrated in the spinning
videos, where one model reads changes of facing direction across frames as rotation and the
other as a child walking and turning. The large "unsure" share is the honest reading of eight
stills.

Sensitivity (`analysis/t9_sensitivity.csv`): removing the flagged negatives from the external
set changes the clean AUC of every model by **+0.001 to +0.011** (upper bound; lower bound
+0.001 to +0.005), with paired video-bootstrap intervals that exclude zero for most models but
never exceed 0.021. SAFE-SMM v3 moves from 0.568 to 0.576 at the upper bound; our no-remedy
cell from 0.655 to 0.662.

**Reading.** Between 5 and 16 of 100 sampled negatives show stereotypy-like movement to a
rater, with wide uncertainty and low inter-rater agreement; removing them changes no
transfer number by more than 0.011. Label noise in the negatives is present, as the base
project disclosed, and it does not explain the weak transfer. The audit is labelled
independently rated in the paper; the frames were not viewed by an author.

## Deviations from the protocol

1. The ASDPose authors' released checkpoint is not scored: the base project's `score_baseline.py`
   and its loader were not retained, and the model file's architecture is not reproducible here.
   The "v3 minus released checkpoint" contrast is replaced by v3 minus v1 and v3 minus the
   no-remedy base cell.
2. Within-set training uses the last epoch: no validation split exists inside a fold without
   leaking a video; the YAML did not fix epoch selection.
3. Temperature search is bounded at exp(±4) = 54.6; two published models hit the bound.
4. DFR was run on every model with saved features (all 49), not only the compact ones.
5. T1 and T4 permutation nulls permute labels across windows (not within video); the
   within-set fit is video-grouped, so the null measures what a label-free fit achieves under
   the same grouping.

## Consequences for the paper

- The artifact is weaker on this set under RTMPose (0.56 against 0.71), not shown absent, and
  **extractor-dependent**: MediaPipe writes a quality-label association of ASDPose size with the
  opposite sign. The sentence "the artifact is a property of the ASDPose release" is not
  supported; "the artifact depends on the extractor and the selection rule" is.
- Transfer is weak and inverted; the velocity channel is the common factor in the worst
  transfers; refitting the head closes most of the gap, so the representations transfer about
  as well as a within-set model. The published operating point does not survive calibration.
- The negative set has a position structure (T8); the independently rated label-noise audit (T9) puts
  the fraction of noisy negatives at 5 to 16 percent, with no transfer number moving by more
  than 0.011 when they are removed.
- All of T1 to T8 go into the transfer section with these readings; nothing is omitted.
