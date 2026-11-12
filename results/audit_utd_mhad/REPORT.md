# UTD-MHAD: loader and natural-cue audit

Run 2026-10-31, CPU only. Code: `safe_motion/data/utd_mhad.py`, `safe_motion/audit/natural_cue.py`,
`experiments/audit_utd_mhad.py --n-perm-lr 200 --n-perm-gb 50`. Numbers: `audit.json`,
`clip_stats.csv`, `summary.txt`, `logs/audit_utd_mhad.log`. Exploratory; no protocol frozen.

## Data
- `<data root>/utd-mhad/Skeleton/a{a}_s{s}_t{t}_skeleton.mat`, `d_skel` (20, 3, T), Kinect v1, joint
  order from `Skeleton_joint_order.txt`. 861 clips; absent a8_s1_t4, a23_s6_t4, a27_s8_t4. 58,299
  frames, 8 subjects, 27 actions (31 to 32 clips each), 30 fps. Length 41 to 125 (p5/p50/p95
  50/67/93). Longest classes: pickup_throw (mean 87.0), squat (82.4), lunge (80.2). Shortest:
  sit_to_stand (55.0), stand_to_sit (57.1).
- Values are camera-space metres, z 1.8 to 4.1 m. No tracking state is stored (the Kinect
  tracked / inferred flag is absent). Zero joints 0, NaN 0, repeated frames 0, repeated
  joint-frames 0.
- Loader channels: x, y in metres (orthographic; metric size is independent of subject
  distance). Third channel = validity (1 where the joint holds a reading; all 1 here). z is kept
  in the cache. Standard split: subjects 1, 3, 5, 7 train and 2, 4, 6, 8 test (`subject_split`).
  Cache: `<data root>/utd-mhad/cache/utd_mhad_skeleton.npz`.

## Natural-cue audit (leave-one-subject-out, 8 folds)
The four statistics are exactly 0 in every clip, so they predict nothing (macro AUC 0.5,
balanced accuracy 1/27 by construction). One documented proxy was added: the mean over 19 bones
of std/mean of 3D bone length (guessed Kinect joints stretch bones). p5/p50/p95 0.044 / 0.085 /
0.137. It is partly behaviour-driven (occlusion in squats and pick-ups).

| features | model | macro AUC [95% CI] | bal. acc | null mean (95th) | p |
|---|---|---|---|---|---|
| tracking (4 stats) | LR / GB | 0.500 (constant) | 0.037 | n/a | n/a |
| length | LR | 0.649 [0.623, 0.688] | 0.075 | 0.485 (0.506) | 0.005 |
| length | GB | 0.608 [0.592, 0.631] | 0.070 | 0.485 (0.509) | 0.020 |
| bone-length CV | LR | 0.674 [0.636, 0.736] | 0.096 | 0.485 (0.506) | 0.005 |
| bone-length CV | GB | 0.639 [0.607, 0.677] | 0.101 | 0.487 (0.504) | 0.020 |
| bone CV + length | LR | 0.730 [0.705, 0.773] | 0.119 | 0.487 (0.509) | 0.005 |
| bone CV + length | GB | 0.688 [0.666, 0.714] | 0.094 | 0.489 (0.510) | 0.020 |

Per-class OvR AUC furthest from 0.5. Length GB: pickup_throw 0.804, sit_to_stand 0.757,
draw_circle_ccw 0.724, draw_circle_cw 0.717, lunge 0.715, draw_triangle 0.714. Bone CV GB:
squat 0.924, pickup_throw 0.858, swipe_right 0.726, draw_triangle 0.724, sit_to_stand 0.719,
wave 0.714. CIs come from a subject-clustered bootstrap with only 8 clusters and are optimistic.
The LOSO null mean sits slightly below 0.5 (0.485), a known leave-one-group-out effect.

## Implications for planted-world replication
**Natural cue:** none in the four tracking statistics; the release cannot carry one. Length
predicts the action (0.61 to 0.65), so length control is needed as on ASDPose. The bone-length
proxy shows tracking-error-like variation that follows the action, but it is not one of the
four statistics.

**Host task.** UTD-MHAD is a zero-cue host, a non-clinical check of the zero point and of
scaling with planted strength. Planted worlds need a binary label. Option 1: a fixed,
pre-registered split of the 27 actions into two groups, e.g. whole-body actions 21 to 27
(pickup_throw, jog, walk, sit_to_stand, stand_to_sit, lunge, squat; 222 clips) vs the 20 arm
gestures (639 clips). Option 2: a balanced random half of the classes. Risk: UTD actions are easy
for skeleton models, and a clean AUC near 1 leaves little room for reliance. Check the clean AUC
first, and prefer a harder grouping if it is at ceiling.

**Test subjects.** 8 subjects in total, 4 in the standard test split. A clustered bootstrap over 4
(or 8) clusters cannot support CI-based verdicts or equivalence margins (ASDPose had 40 test
children and margins still failed by 0.001 to 0.003). Use leave-one-subject-out scoring (all 8
scored once, 8 fits per cell-seed). Report SE with permutation nulls and per-subject spread.
Treat the result as a replication of direction and scaling.

**Operators.** Kinect gives no OpenPose-style zero joints, so "zero" (runs, whole) writes a
failure form that does not occur naturally here. That is acceptable for a clean host but should
be stated. "hold" (frozen) is meaningful (Kinect can repeat a pose), with no natural rate here
(0 repeated frames). "gapfill" works. "conflow" and "confscale" write low confidence into a
channel that is constant 1, which is a cue with no natural counterpart; avoid them or report
them as synthetic only. Use 20 joints via `build_masks(..., n_joints=20,
arm_joints=utd_mhad.ARM_JOINTS)`; ASDPose defaults are unchanged. `SegmentDataset` needs a
20-joint variant (flip order `utd_mhad.FLIP_ORDER`; window-std normalisation, since coordinates
are metres).

**Cost (data size only).** Training on 4 subjects is about 430 clips; a window of about 64 frames
covers the median clip; 20 joints. Against ASDPose v7 (12,572 windows of 200 x 17, 6.8 min per
25-epoch run): 430/12,572 x 64/200 x 20/17 ≈ 1.3 percent, so seconds of GPU compute per run,
dominated by overhead. LOSO with 8 fits is still about a minute per (cell, seed). CPU training
may be practical, but this is an estimate from data size, not a measurement.

**Proposed protocol (not written).** Binary grouping fixed in advance (whole-body vs arm
gestures, after a clean-AUC check), window 64 with length control, leave-one-subject-out, three
seeds, mechanisms runs (zero), whole, frozen (hold) at the v7 strengths, worlds S / I / Sp,
permutation null for SE. Gate: tracking statistics constant (true by construction) and length
AUC reported for the binary task.
