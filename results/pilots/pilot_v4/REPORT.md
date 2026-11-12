# Pilot v4: matched-world measure on CompactCTRGCN (H7); remedy form vs artifact form (H8, H9)

Protocol: `configs/protocols/pilot_asdpose_v4.yaml` (frozen 5e8e7a9). Runs 2026-10-25, 34 runs,
3.56 GPU-h logged (budget 4.0); local total 7.05 GPU-h logged. The first attempt lost 26 runs to
two external events (see "Run incident"); they were rerun unchanged and only the completed runs
are analysed. Analysis: `experiments/analyze_v4.py --n-boot 10000`.

## H7: PASS. The matched-world measure replicates on CompactCTRGCN

| strength | seeds | R, World S model | R, World I model | shortcut effect [95% CI] |
|---|---|---|---|---|
| mid | 3 | 0.625 | −0.002 [−0.066, +0.062] | 0.628 [0.499, 0.759] |
| low | 2 | 0.265 | −0.052 [−0.133, +0.025] | 0.317 [0.217, 0.426] |

Mid minus low: 0.311 [0.169, 0.448]. World I models read zero at both strengths. Same pattern as
TinyTCN (v3: 0.765 / 0.324), on the architecture SAFE-SMM used. Low strength has two seeds.

The shortcut is invisible to clean accuracy on this architecture too: clean AUC 0.792 (World S)
against 0.799 (World I) at mid, 0.795 against 0.799 at low. A model that relies heavily on the cue
is indistinguishable from a cue-free model by accuracy on clean data; only the matched measure
separates them.

## H8: FAIL under the frozen rule. The matched remedy lowers reliance but adds no shortcut-specific accuracy

Whole-joint mechanism, TinyTCN, mid, 3 seeds:

| remedy | change in R | shortcut part of clean gain | regularization part |
|---|---|---|---|
| A_dropout (matched form) | −0.113 [−0.174, −0.056] | −0.010 [−0.033, +0.013] | +0.035 [+0.017, +0.052] |
| B_coorddrop | −0.036 [−0.072, +0.003] | −0.002 [−0.017, +0.014] | +0.014 [+0.004, +0.023] |
| A_velocity | −0.146 [−0.220, −0.075] | −0.000 [−0.026, +0.025] | +0.118 [+0.074, +0.164] |

Two of three conditions hold (A_dropout lowers R; B_coorddrop's change contains 0). The third
fails: the shortcut part of A_dropout's clean gain is zero, not positive. In hindsight this
condition could not be met on this host, because learning the shortcut costs almost no clean
accuracy (A_none shortcut effect 0.690 with clean AUC 0.653 in S vs 0.653 in I), so removing it has
no accuracy to give back. That is an explanation, not a rescue: the frozen verdict is FAIL.

Velocity lowers reliance more than the matched remedy does (−0.146), and its World I model has a
small positive R (0.092 [0.023, 0.161]); velocity changes how degraded segments are scored
whatever the label.

## H9: PASS, marginally. The same remedy lowers reliance more when it matches the artifact form

Reliance change of A_dropout: −0.113 [−0.174, −0.056] on whole-joint absence against
−0.044 [−0.078, −0.012] on frame-level runs (v3); contrast −0.069 [−0.138, −0.003].

## Pilot verdict (v1 to v4)

Solid, with ground truth: matched-world reliance measurement. Zero when no shortcut is learnable,
scales with planted strength, on two architectures and two mechanisms (v3 H2/H3, v4 H7, and the
whole-joint A_none effect 0.690). The raw QC gap carries a cue-free offset and a wide null (v2).
New and quotable: heavy shortcut use leaves clean accuracy unchanged, so accuracy-based checks
cannot detect it.

Weak: the remedy-form use case. Matching the remedy to the artifact helps reliance a little more
(H9, marginal), but no remedy removes most of the shortcut (best: 0.69 to 0.55), and none produces
a shortcut-specific accuracy gain (H5b, H8).

## Run incident (first attempt, 2026-10-25 07:04 to 07:50)

26 of 34 runs failed: (1) a `conda install --force-reinstall pytorch` into the shared `torch` env
at 07:04:06, not issued by this project, made in-flight launches fail with "No module named
torch"; (2) `ollama` runners on the same GPU took up to 5.3 GB and caused CUDA OOM for
CompactCTRGCN while two queues shared the card. Failed runs wrote no results. The rerun used one
queue at a time and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; no settings changed.
Logs of the failed attempt: `logs/v4_failed_attempt1/`. GPU time of the OOM attempts before
failing was not logged (estimated under 0.5 h).
