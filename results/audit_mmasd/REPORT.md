# MMASD: loader and natural-cue audit

Run 2026-10-31, CPU only. Code: `safe_motion/data/mmasd.py`, `safe_motion/audit/natural_cue.py`,
`experiments/audit_mmasd.py --n-perm-lr 200 --n-perm-gb 50`. Numbers: `audit.json`; per-clip
statistics: `clip_stats.csv`; console: `summary.txt`, `logs/audit_mmasd.log`. Tests:
`tests/test_new_datasets.py`. Exploratory audit; no protocol frozen.

## Data
- `<data root>/mmasd/2D_skeleton/2D_openpose.zip` unzipped in place (`output/`). OpenPose BODY_25,
  one JSON per frame, `person_id` always -1. Caches in `<data root>/mmasd/cache/` (build 20 s).
- Clip name `<abbr>_<child>_D<day>_<index>_<tag>`: activity, 5-digit child id, recording day
  (D1, D8, D16; the session field), clip index, tag y/n/i. The tag is not defined in README.txt,
  the GitHub README or the ICMI 2023 paper. It is kept as metadata, not as a label. Counts y 535,
  n 488, i 351, present in every class. Tracking statistics do not predict y vs n (GB 0.570, LR 0.514).
- 1,693 folders, 319 empty (all 113 frog_pose, 202 drumming, 2 body_swing, 2
  maracas_forward_shaking): 1,374 clips, 254,674 frames, 32 children, 10 classes. The paper
  reports 1,315 clips and 168 drumming; disk has 343 non-empty drumming folders.
- Children nest in themes: rhythm only 11, yoga only 10 (ids 41xxx, tree and twist only), robot
  only 6, robot and rhythm 5. Clips per child 15 to 131 (median 33.5). Length p5/p50/p95
  75/180/300 frames (25 to 30 fps per the paper); 59 percent of clips are under 200 frames, 28
  percent under 150.

| class | clips | children | p5 | p25 | median | p75 | p95 | mean |
|---|---|---|---|---|---|---|---|---|
| arm_swing | 105 | 11 | 92 | 125 | 150 | 200 | 225 | 158.6 |
| body_swing | 117 | 10 | 120 | 150 | 200 | 250 | 305 | 211.2 |
| chest_expansion | 114 | 10 | 90 | 131 | 175 | 225 | 275 | 180.8 |
| drumming | 343 | 12 | 77 | 150 | 180 | 225 | 300 | 188.1 |
| maracas_forward_shaking | 101 | 11 | 75 | 150 | 200 | 240 | 300 | 194.5 |
| maracas_shaking | 130 | 11 | 94 | 125 | 180 | 240 | 300 | 191.0 |
| sing_and_clap | 113 | 11 | 75 | 150 | 210 | 270 | 330 | 214.2 |
| squat | 102 | 11 | 60 | 100 | 200 | 275 | 349 | 197.2 |
| tree_pose | 129 | 10 | 100 | 125 | 150 | 175 | 240 | 155.2 |
| twist_pose | 120 | 10 | 100 | 125 | 150 | 180 | 250 | 161.5 |

## Person selection
A median of 3.2 people per frame (child, trainer, adult model, others). Default rule "track":
Hungarian linking of detections across frames on the centroid of confident joints (cost =
distance / skeleton height < 0.5, gap up to 15 frames). Keep the track with the largest summed
per-frame mean confidence. Frames without it are all zero, as an OpenPose miss is. Selected-track
coverage is at least 0.98 in 95 percent of clips. Limitation: video is not released, so the
choice cannot be checked. The selected track is the shortest persistent skeleton in only 33
percent of the 1,371 multi-person clips, so it is often an adult if the child is the shortest
person on screen (an assumption). Read MMASD skeletons as "the most confidently tracked
participant". MMASD+ (2024) separates child and therapist. The alternative rule "maxconf" (best
detection per frame) gives per-clip statistics with Spearman 0.86 to 0.90 (frozen 0.15) and the
same audit result.

## Tracking statistics
SAFE-SMM `quality_features` definitions, cutoff 0.3, on the COCO-17 joints taken from BODY_25
(primary; BODY_25 as a check). Over clips p5/median/p95: low-conf 0.021/0.095/0.332, missing
0.002/0.059/0.219, frozen 0/0/0.011, arm low-conf 0/0.013/0.333. Frozen at least 0.10 in 2.4
percent of clips; arm low-conf at least 0.30 in 8.1 percent. Length is unrelated to the
statistics (|Spearman| at most 0.07).

| class | low-conf | missing | frozen | arm low-conf |
|---|---|---|---|---|
| arm_swing | 0.114 | 0.067 | 0.007 | 0.123 |
| body_swing | 0.096 | 0.060 | 0.008 | 0.044 |
| chest_expansion | 0.084 | 0.059 | 0.005 | 0.047 |
| drumming | 0.141 | 0.077 | 0.009 | 0.032 |
| maracas_forward_shaking | 0.114 | 0.078 | 0.015 | 0.076 |
| maracas_shaking | 0.111 | 0.073 | 0.018 | 0.075 |
| sing_and_clap | 0.148 | 0.077 | 0.009 | 0.066 |
| squat | 0.116 | 0.070 | 0.013 | 0.048 |
| tree_pose | 0.083 | 0.050 | 0.000 | 0.055 |
| twist_pose | 0.209 | 0.129 | 0.000 | 0.207 |

## Natural-cue audit
Child-grouped StratifiedGroupKFold(8). Pooled out-of-fold probabilities, macro one-vs-rest AUC
and balanced accuracy. CI: child-clustered bootstrap of the pooled predictions (2,000
resamples). Null: labels permuted across clips (200 for LR, 50 for GB; minimum p 0.005 / 0.020).
LR is standardised logistic regression; GB is histogram gradient boosting (depth 3, 150
iterations); both use balanced class weights.

10 activities (chance AUC 0.5, balanced accuracy 0.10):

| features | model | macro AUC [95% CI] | bal. acc | null mean (95th) | p |
|---|---|---|---|---|---|
| tracking | LR | 0.604 [0.571, 0.649] | 0.197 | 0.493 (0.516) | 0.005 |
| tracking | GB | 0.657 [0.623, 0.704] | 0.221 | 0.500 (0.524) | 0.020 |
| length | LR | 0.501 [0.456, 0.555] | 0.129 | 0.493 (0.512) | 0.24 |
| length | GB | 0.549 [0.494, 0.613] | 0.153 | 0.491 (0.511) | 0.020 |
| tracking + length | LR | 0.602 [0.558, 0.660] | 0.209 | 0.495 (0.515) | 0.005 |
| tracking + length | GB | 0.670 [0.628, 0.723] | 0.267 | 0.499 (0.519) | 0.020 |

Per-class OvR AUC, tracking GB: arm_swing 0.788, body_swing 0.572, chest_expansion 0.588,
drumming 0.781, maracas_forward_shaking 0.626, maracas_shaking 0.610, sing_and_clap 0.577, squat
0.476, tree_pose 0.666, twist_pose 0.888. Length GB: arm_swing 0.630, squat 0.654, tree_pose
0.624, others 0.48 to 0.57. Checks: BODY_25 statistics 0.655 (LR) / 0.706 (GB); maxconf rule
0.613 / 0.654; theme (3 classes) tracking 0.663 / 0.689, length 0.550 / 0.562.

Within theme (children nest in themes):

| theme | children | tracking LR | tracking GB | length LR | length GB |
|---|---|---|---|---|---|
| rhythm (4) | 16 | 0.681 [0.632, 0.761] | 0.658 [0.619, 0.713] | 0.464 | 0.488 |
| robot (4) | 11 | 0.600 [0.569, 0.661] | 0.588 [0.515, 0.672] | 0.473 | 0.588 |
| yoga (2) | 10 | 0.890 [0.816, 0.947] | 0.846 [0.766, 0.926] | 0.509 | 0.522 |

Binary pairs within theme (no permutation null, exploratory):

| pair | clips | children (both) | tracking GB [CI] | tracking LR | length GB |
|---|---|---|---|---|---|
| tree_pose vs twist_pose | 249 | 10 (10) | 0.846 [0.765, 0.922] | 0.890 | 0.522 |
| drumming vs maracas_forward_shaking | 444 | 16 (7) | 0.800 [0.735, 0.871] | 0.737 | 0.495 |
| drumming vs maracas_shaking | 473 | 16 (7) | 0.796 [0.740, 0.870] | 0.759 | 0.537 |
| arm_swing vs squat | 207 | 11 (11) | 0.741 [0.596, 0.940] | 0.747 | 0.802 |
| arm_swing vs body_swing | 222 | 11 (10) | 0.733 [0.618, 0.889] | 0.805 | 0.699 |
| maracas_shaking vs sing_and_clap | 243 | 11 (11) | 0.721 [0.656, 0.807] | 0.718 | 0.460 |
| body_swing vs chest_expansion | 231 | 10 (10) | 0.682 [0.606, 0.743] | 0.627 | 0.553 |
| arm_swing vs chest_expansion | 219 | 11 (10) | 0.678 [0.529, 0.843] | 0.708 | 0.566 |
| maracas_forward_shaking vs sing_and_clap | 214 | 11 (11) | 0.628 [0.549, 0.725] | 0.722 | 0.453 |
| drumming vs sing_and_clap | 456 | 16 (7) | 0.577 [0.468, 0.720] | 0.484 | 0.466 |
| maracas_forward_shaking vs maracas_shaking | 231 | 11 (11) | 0.523 [0.436, 0.625] | 0.491 | 0.508 |
| chest_expansion vs squat | 216 | 11 (10) | 0.491 [0.388, 0.654] | 0.490 | 0.615 |
| body_swing vs squat | 219 | 11 (10) | 0.435 [0.350, 0.546] | 0.339 | 0.605 |

AUCs below 0.5 with few groups (body_swing vs squat) are a known grouped-CV artefact on a null
signal, not a reversed cue.

## ADOS (child level, descriptive only)
`ADOS_rating.xlsx` has 36 rows; 20 of the 32 children match. The unmatched 12 are the 10 yoga
children plus 40163 and 40453. Spearman of each child's mean statistic with the score, n = 20:

| score | low-conf | missing | frozen | arm low-conf | log length |
|---|---|---|---|---|---|
| comparison | +0.29 | +0.24 | +0.21 | +0.03 | −0.27 |
| social affect | +0.24 | +0.18 | +0.02 | +0.08 | −0.26 |
| RRB | +0.54 | +0.46 | +0.24 | +0.27 | −0.16 |
| overall total | +0.35 | +0.27 | +0.12 | +0.13 | −0.24 |

Two of 20 cells have uncorrected p < 0.05 (RRB with low-conf p = 0.013, RRB with missing p =
0.039). With n = 20, no correction, and a selected skeleton that may be an adult, this is not
evidence of an association.

## Implications for planted-world replication
**Natural cue: yes.** 10-class macro AUC 0.60 (LR) to 0.66 (GB) against a null near 0.50. It is
strongest where behaviour causes the failure: twist_pose has 2.5 times the low-confidence
fraction of tree_pose (pair AUC 0.85 to 0.89), and drumming vs the maracas classes gives 0.80.
Length is weak overall (0.50 to 0.55) but strong for arm_swing vs squat (0.80). This differs
from the ASDPose-QC host (quality-only AUC 0.502).

**Host task.** A binary task with both classes in each child, as on ASDPose. Clean host:
maracas_forward_shaking vs maracas_shaking (231 clips, 11 children all with both classes,
tracking 0.52 / 0.49, length 0.51). Risk: the classes differ mainly in shaking direction, so
check the clean model AUC first. Fallback: chest_expansion vs squat (tracking 0.49 / 0.49,
length 0.615; needs the v7 length control). tree_pose vs twist_pose is a behaviour-caused
natural-cue contrast case, not a clean host.

**Test subjects.** 10 to 11 children per recommended pair. A 25 percent split leaves 3 test
children (8 for the whole set), against 40 on ASDPose, where margins still failed by 0.001 to
0.003. The ±0.02 / ±0.05 equivalence margins are out of reach. Score every child out of fold
(child-grouped 5-fold or leave-one-child-out, K times the cost). Report the direction and size
of SE with CIs and permutation nulls, not equivalence verdicts. Decide this before freezing.

**Operators.** All apply after the BODY_25 to COCO-17 mapping (`COCO17_FROM_BODY25`). OpenPose
misses are x = y = confidence = 0 (missing median 0.059), so zero / whole / runs are natural
forms. armruns with confscale matches the natural low-confidence range. frozen with hold has
almost no natural counterpart (median 0; 2.4 percent of clips at 0.10 or more). BODY_25 works
through `build_masks(..., n_joints=25, arm_joints=ARM_JOINTS_BODY25)`; defaults are unchanged.
`asdpose.SegmentDataset` hard-codes 17 joints and image-height normalisation. MMASD has no image
size, so use `est_h` (the largest y in the clip) or window-std.

**Cost (data size only).** ASDPose v7 CompactCTRGCN: 9,859 train + 2,713 val windows (200 frames
x 17 joints), 25 epochs, 6.8 min per run on an A10G. All of MMASD with a 25 percent test split
is about 8 percent of that per epoch (about 0.6 min per run). One pair is about 1.4 percent,
which is overhead-dominated. With 100 epochs and K = 5 out-of-fold scoring: about 1 to 2
A10G-minutes per (cell, seed). The v7 rule "training clips at least 200 frames" would drop 54
percent of the maracas pair; a 150-frame window drops 25 percent.

**Proposed protocol (not written).** maracas_forward_shaking vs maracas_shaking; track rule;
COCO-17; window 150; child-grouped 5-fold out-of-fold scoring; three seeds; runs and armruns at
the v7 strengths; worlds S / I / Sp; child-clustered bootstrap plus permutation null. Gate:
quality-only AUC on each fold's training children within 0.5 ± 0.05 before scoring.
