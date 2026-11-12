# Round v10: identification control, real-data reference, mask stability

Protocol: `configs/protocols/identification_v10.yaml` (frozen 422da37; implementation notes
and this outcome in `configs/protocols/REGISTRY.md`). Parts A and B ran 2026-11-02 17:58 to
23:45 UTC on one on-demand g5.2xlarge (an EC2 instance, us-east-1c), 5.8 h, about USD 7:
27 CompactCTRGCN fits, no failure. Part C (scoring only) failed on the instance because the v9
`folds.json` files were not in the bundle; it was run on the local GPU 2026-11-03 (342
checkpoints x 10 mask seeds x 2 worlds, 7 min; reproduction of the v9 mask-2000 columns to at
most 1.4e-4 on the probability scale). Analysis `experiments/analyze_v10.py` run locally
(the instance lacks the v7 score files it pairs with). Tables in `analysis/`; this directory
links to the local mirror `results/aws/v10/results/v10`. Paired child-clustered bootstrap,
2,000 resamples, seeds averaged inside resamples; ASDPose host test 1,966 segments, real test
split 3,834 segments, 40 children.

## A. Quality-blind control (rule A1)

| mechanism | SE(A_none), v7 | SE(A_blind) | non-cue share | A1 threshold (0.5 x SE) | verdict |
|---|---|---|---|---|---|
| missing runs, mid | 0.756 [0.618, 0.920] | 0.360 [0.282, 0.455] | 0.48 [0.40, 0.58] | 0.378 | **FAIL** (upper end 0.455 > 0.378) |
| whole-joint absence, mid | 0.483 [0.353, 0.613] | 0.479 [0.380, 0.582] | 0.99 [0.78, 1.30] | 0.241 | **FAIL**, and not interpretable (below) |

R(f_S) and R(f_I) of the blind cells: runs 0.514 and 0.154 [0.097, 0.222]; whole 0.462 and
−0.016 [−0.087, 0.073]. Clean AUC of the blind cells 0.79 to 0.82, within 0.02 of the standard
cells.

**A design flaw in the blind transform, found after the run.** The transform interpolates a
joint's coordinates across frames in which it is missing and copies the nearest observed
frame into leading and trailing gaps. A joint that is missing for the whole 200-frame window
has no observed frame, so its coordinates stay at zero. Under whole-joint absence every
planted joint is of this kind, so the blind model sees the planted pattern (a joint fixed at
the origin) as plainly as the standard model does, without needing the confidence channel.
The whole-joint result therefore says nothing about information loss; it says the transform
did not blind that mechanism. Under missing runs (mean run 15 frames, stationary fraction up
to 0.6) some joints are also missing for most of the window and long edge gaps become static
copies, which are visible too. The blind SE under runs is therefore an **upper bound** on the
information-loss and representation component, not an estimate of it.

**Reading.** At most about half of the missing-runs shortcut effect on ASDPose (0.48 [0.40,
0.58] of 0.756) can be information loss or representation difference rather than use of the
quality cue; the share for whole-joint absence could not be measured with this transform.
Arm-confidence rescaling removes no information, so its SE (0.582 [0.482, 0.692]) is
cue-attributable by construction; frozen frames cannot be blinded by interpolation. The paper
reports SE under information-removing operators as a mixed quantity with this bound, as rule
A1 requires when it fails.

**A2 transform check: FAIL, and the rule's expectation was wrong.** The blind model responds
to gapfill at mid by 0.502 [0.415, 0.611] in World S and 0.143 [0.085, 0.207] in World I. The
rule expected an interval containing zero on the reasoning that a blind model's input is
already gapfilled. That reasoning was mistaken: gapfill at test time interpolates joint-frames
that the blind transform does not treat as missing (their confidence is kept high), so it
removes motion from the input in a label-tied way, and a motion-only model should respond. The
observed response is the information-loss response of a motion-only model and is consistent
with A1. Reported as a failed rule with a mistaken premise.

## B. Real-data World I reference (no pass rule; wording rule)

Five real cells retrained on the full ASDPose split with missing-runs degradation planted
independently of the label on top of the natural failures; clean AUC within 0.01 of the v7
real cells (A_none 0.846 against 0.855, difference 0.009 [−0.003, 0.020], within margin).
Response of the v7 real model, of its reference, and the paired difference:

| cell | runs | whole | frozen | armruns | gapfill |
|---|---|---|---|---|---|
| A_none | −0.08 / −0.02 / −0.06 [−0.16, +0.03] | +0.29 / +0.09 / **+0.20 [+0.12, +0.29]** | +0.34 / +0.29 / +0.05 [−0.01, +0.11] | +0.00 / +0.11 / **−0.11 [−0.17, −0.05]** | +0.18 / +0.16 / +0.02 |
| A_velocity | +0.16 / −0.01 / **+0.16 [+0.06, +0.26]** | +0.28 / +0.19 / **+0.09 [+0.02, +0.18]** | +0.37 / +0.26 / **+0.11 [+0.02, +0.21]** | +0.02 / +0.17 / **−0.15 [−0.22, −0.09]** | +0.15 / +0.20 / −0.05 |
| A_dropout | −0.17 / −0.05 / **−0.12 [−0.22, −0.04]** | +0.08 / +0.04 / +0.04 [−0.01, +0.08] | +0.40 / +0.33 / **+0.07 [+0.01, +0.14]** | +0.02 / +0.04 / −0.02 | +0.13 / +0.16 / −0.02 |
| A_both | −0.08 / −0.02 / −0.06 [−0.16, +0.04] | +0.05 / +0.04 / +0.01 | +0.35 / +0.30 / +0.05 [−0.01, +0.12] | +0.07 / +0.05 / +0.02 | +0.14 / +0.18 / −0.04 |
| B_coorddrop | −0.01 / −0.02 / +0.01 | +0.24 / +0.12 / **+0.12 [+0.04, +0.21]** | +0.33 / +0.31 / +0.02 | −0.00 / +0.05 / −0.05 [−0.11, +0.01] | +0.14 / +0.13 / +0.01 |

Bold: paired difference excludes zero.

**Reading.** The real models' frozen-frame responses (+0.33 to +0.40) are matched by
references that saw label-independent missing runs on the same data (+0.26 to +0.33); the
differences are at most 0.11 and exclude zero for two cells. Under whole-joint absence four
cells exceed their reference by 0.09 to 0.20. Under the wording rule a reliance sentence needs
both a difference excluding zero and a cue reading from Part A for that operator; Part A could
not provide one for whole-joint absence, and for missing runs the real models respond less
than their references in two cells. **Reliance of the real ASDPose models on tracking quality
remains not identified**, now against a real-data reference. The real models' responses are
reported as responses relative to that reference.

## C. Mask stability on the v9 checkpoints (report only)

| host | mechanism | v9 single-mask SE | SE over 10 masks: mean [CI] | between-mask SD | range | excludes 0 |
|---|---|---|---|---|---|---|
| MMASD | armruns | +0.130 | +0.127 [+0.089, +0.214] | 0.054 | +0.04 to +0.23 | yes |
| MMASD | frozen | −0.020 | +0.013 [−0.002, +0.056] | 0.035 | −0.03 to +0.08 | no |
| MMASD | runs, low | +0.014 | +0.012 [−0.054, +0.089] | 0.059 | −0.08 to +0.15 | no |
| MMASD | runs, mid | +0.378 | +0.380 [+0.180, +0.833] | 0.080 | +0.24 to +0.49 | yes |
| MMASD | whole | +0.028 | +0.061 [−0.134, +0.414] | 0.184 | −0.15 to +0.45 | no |
| UTD-MHAD | frozen | +0.234 | +0.206 [+0.018, +0.396] | 0.202 | −0.14 to +0.49 | yes |
| UTD-MHAD | runs, low | +0.430 | +0.131 [−0.121, +0.326] | 0.199 | −0.19 to +0.43 | no |
| UTD-MHAD | runs, mid | +0.266 | +0.555 [+0.266, +1.157] | 0.240 | +0.17 to +0.97 | yes |
| UTD-MHAD | whole | +0.651 | +0.603 [+0.414, +0.830] | 0.126 | +0.39 to +0.78 | yes |

**Reading.** On UTD-MHAD the v9 single mask (seed 2000) sat at the bottom of its range under
missing runs at mid (0.27 against a ten-mask mean of 0.56) and at the top under missing runs at
low (0.43 against 0.13); the v9 "scaling fails" reading on that host was a mask artifact, and
the "low is cleaner" reading as well. Averaged over ten masks, the shortcut effect excludes
zero for both mechanisms with natural counterparts on MMASD and for all three on UTD-MHAD,
with between-mask standard deviations of 0.05 to 0.24 against 0.01 to 0.12 on ASDPose.
Sentence fixed by the protocol: SE on hosts of this size moves by 0.1 to 0.8 between mask
realizations; the v9 verdicts were made on one. **The v9 verdicts stand**; this is a stability
analysis, and it says that a protocol for small hosts must define SE over several mask
realizations before it is read against a null.

## Consequences

- The identification objection of the 2026-11-02 review is confirmed in part and bounded: SE
  under operators that remove motion information is a mixed quantity; on ASDPose at most about
  half of the missing-runs effect is non-cue; arm confidence gives a cue-only effect by
  construction. The method section states this; the abstract no longer calls SE "learned
  reliance" without the qualifier.
- The blind transform must handle never-observed joints (for example by replacing them with
  the window's mean skeleton position for that joint, or by dropping the joint from the graph)
  before it can test whole-joint absence; left for a later protocol, not patched here.
- The real-model section gains its real-data reference and keeps "not identified".
- The boundary section gains the mask-averaged table and the sentence above; a later
  small-host protocol defines SE over several masks and a null of matching form.
- Cost: v10 about USD 7 on AWS plus 14 local GPU-minutes. Project AWS total about USD 57;
  local GPU about 8.2 h of the 10 to 12 h cap.
