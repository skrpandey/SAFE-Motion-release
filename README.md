# SAFE-Motion

Code, protocol files, per-segment scores and analysis outputs for the article *Auditing Planted
Tracking Cues in Skeleton-Based Stereotypy Detection with Matched Training Conditions* (manuscript
for the IEEE Open Journal of the Computer Society). The author list is in the article.

The article audits whether skeleton-based detectors of stereotypical motor movements (SMMs) learn
a planted tracking cue. The same network is trained in two conditions: Condition S, in which a
tracking degradation is associated with the label, and Condition I, in which the same degradation
is drawn independently of the label. Both are scored under both test policies, and the
training-by-test interaction (TTI) compares their responses. In code and data files the conditions
are called worlds S and I and the interaction is called MWI (earlier SE). The models detect
movement patterns consistent with annotated SMMs; they do not diagnose autism.

## Contents

| path | content |
|---|---|
| `safe_motion/` | data loaders (ASDPose, MMASD, UTD-MHAD), planting operators (`planted.py`), networks, metrics, training variants |
| `experiments/` | training, scoring and analysis scripts |
| `experiments/reanalyses/` | re-analyses of the stored scores and the builders of the tables in `paper/tables/` |
| `configs/protocols/` | protocol files of every experiment and `REGISTRY.md` (dates of the freezes, outcomes, deviations) |
| `configs/runs/` | the training command of every fit, one line per model |
| `results/` | per-segment test scores of all 236 final ASDPose models, per-window transfer scores, identity-audit decisions, and the analysis outputs the article's numbers are read from |
| `paper/tables/` | article Table 2 and supplement Tables S2, S4 (panel b), S13 and S14, generated from `results/` |

Not included: model checkpoints, training logs, feature files, pose caches and the datasets, and the
outputs of side analyses that the article does not report (for example the head-refit and
calibration analyses and the unfinished label audit of the transfer round).

## Protocol files

Every experiment was specified in a protocol file before any model was scored under it, with its
pass rules and margins in the same file. Experiments added after earlier results had been examined
were fixed before their own fits (sequential prespecification within the project, not external
preregistration). Where older protocol files say "pre-registered", they mean committed before
scoring within this project.

Public protocol files are publication exports of internally retained experiment records.
Administrative notes, private file references, and approval-message quotations have been removed;
the scientific settings, analysis rules, and chronology reported in the article are unchanged. Every
protocol file starts with an export header. Protocol, script and results-folder names inside the
protocol texts, the registry and the run lists were changed to the names used in this repository; in
some files administrative entries (status lines, cost and compute notes, a report location, a
reporting note, an evidence reference) were edited as well. The exports are not byte-identical to
the frozen originals. Commit hashes refer to the development history, which is not published.

Each experiment also has a short internal code (v7 to v20, v15conf), shown in brackets in the table
below. The codes are kept where they act as keys: in the run names of the models (for example
`v20_S_w0_armruns_mid_A_both_s0`, which is also the score file name), in the `--proto` argument of
every training command, in rule labels and cross-references inside the protocol texts, and in some
output file names. They tie every released score to its retained checkpoint and training log.

## Experiments and results

| results folder | protocol file (code) | experiment | article |
|---|---|---|---|
| `operators` | `operators.yaml`, `operators_addendum.yaml` (v7, v8), selection repair `selection_repair.yaml` | four operators, three seeds per condition, no-remedy cells and input configurations; real-split models; historical loader | Table 1, Figures 2 and 4, Section VI-C, Tables S3 and S4 |
| `confonly_historical` | `confonly_historical.yaml` (v13) | arm-confidence cells with the coordinate rule, historical loader | Table 1, Figure 3, Table S8 |
| `controls` | `controls.yaml` (v10) | interpolated control, real-data reference, mask stability | Sections VI-B and VI-C, S6 |
| `cue_remedies` | `cue_remedies.yaml` (v12) | Group DRO, adversarial head and operator augmentation, historical loader | Table 3, Table S4 |
| `confonly_primary` | `confonly_primary.yaml` (v15conf) | primary confidence-only comparison and Group DRO, corrected loader | Tables 2 and 3, Figure 3, Table S2 |
| `confonly_extension` | `confonly_extension.yaml` (v16) | six more seeds, ST-GCN, Group DRO with input-derived groups | Tables 2 and 3, Figures 3 and 4 |
| `confonly_assignments` | `confonly_assignments.yaml` (v17) | three regenerated training assignments | Section V-B, Figure 3, Table S1 |
| `crossed_rescoring` | `crossed_rescoring.yaml` (v18) | Condition I references rescored under the coordinate rule, both validation attempts | Figure 4, Section S5, Table S9 |
| `mapped_persons` | `mapped_persons.yaml` (v19) | retraining with development and test persons separated under the recovered mapping | Section V-C, Section S8, Table S13 |
| `recipe_comparison` | `recipe_comparison.yaml` (v20) | combined velocity and joint-dropout recipe in the primary cell, and a device-matched no-remedy refit | Section VI-C, Section S9, Table S14 |
| `confidence_permutation` | `confidence_permutation.yaml` | confidence-channel permutation (post hoc) | Section V-D |
| `small_datasets` | `small_datasets.yaml` (v9) | boundary tests on MMASD and UTD-MHAD | Section VI-D, Section S6, Table S10 |
| `external_transfer` | `external_transfer.yaml` (v11) | SSBD/ESBD transfer, per-window scores (RTMPose and MediaPipe) | Section VI-D, Section S6, Tables S11 and S12 |
| `identity_audit` | | identity audit of the ASDPose release: decision record, per-identifier and per-group manifests, full-search outcomes | Sections IV-A, S3 and S8 |
| `reanalyses` | `cluster_scenarios_post_hoc.yaml` (clusterings) | post hoc re-analyses of the stored scores | Sections IV-B, IV-C, V-A, V-C, Tables S2 and S5 to S7 |

The selection repair (`selection_repair.yaml`) retrained all 164 historical-loader fits with a
corrected validation set; the scores in `operators`, `controls`, `cue_remedies` and
`confonly_historical` are those of the repaired fits. `configs/protocols/data_loaders.md`
describes the historical and corrected data loaders.

Folder contents: `registry.json` (model name, condition, operator, strength, cell and seed of each
score file), `scores/` (one CSV per model: segment identifier, subject identifier, label, frame
count, QC flag, and one score per test condition), and `analysis/` (the CSV and JSON outputs the
article's numbers are read from).

## Scripts

Run every script from the repository root with `python3`.

| results folder | training | scoring | analysis |
|---|---|---|---|
| `operators` | `train_asdpose.py` | `score_operators.py` | `analyze_operators.py`, `analyze_operators_masks.py`, `variance_operators.py`, `cluster_scenarios.py`, `holm_families.py`, `natural_failure_stats.py`, `natural_cue_checks.py`, `qc_exclusion_report.py`, `qc_gap_definitions.py`, `qc_gap_lengthproto.py`, `gap_table_b4.py`, `length_audit_inputs.py`, `raw_auc_robustness.py`, `se_sample_size.py`, `host_reaudit.py`, `procedure_checks.py`, `operator_clip_counts.py`, `repair_counts.py` |
| `confonly_historical` | `train_asdpose.py` | `score_confonly_historical.py` | `analyze_confonly_historical.py` |
| `controls` | `train_asdpose.py` | `score_controls.py`, `score_controls_masks.py` | `analyze_controls.py`, `check_conf_invariance.py`, `probe_interpolation_cue.py`, `oracle_mixture.py` |
| `cue_remedies` | `train_asdpose.py` | `score_cue_remedies.py`, `score_adversary_probe.py` | `analyze_cue_remedies.py`, `analyze_adversary_probe.py`, `variance_cue_remedies.py` |
| `confonly_primary` | `train_asdpose.py` | `score_confonly_primary.py` | `analyze_confonly_primary.py`, `reanalyses/r25_corrected_four_cells.py` |
| `confonly_extension` | `train_asdpose.py` | `score_confonly_corrected.py`, `score_conf_reliance_extension.py` | `analyze_confonly_extension.py`, `analyze_conf_reliance_extension.py` |
| `confonly_assignments` | `train_asdpose.py` | `score_confonly_corrected.py --proto v17` | `analyze_confonly_assignments.py`, `assignment_severities.py` |
| `crossed_rescoring` | | `rescore_crossed.py` (`validate`, `score`) | `rescore_crossed.py analyze` |
| `mapped_persons` | `train_asdpose.py` | `score_confonly_corrected.py --proto v19` | `analyze_mapped_persons.py` |
| `recipe_comparison` | `train_asdpose.py` | `check_conf_invariance_recipe.py` (gate, before the fits), `score_confonly_corrected.py --proto v20` | `analyze_recipe_comparison.py`, `reanalyses/r26_recipe_per_fit.py` |
| `confidence_permutation` | | `score_conf_reliance.py` | `analyze_conf_reliance.py` |
| `small_datasets` | `train_small_datasets.py` | inside `train_small_datasets.py` | `analyze_small_datasets.py`, `analyze_small_datasets_variance.py` |
| `external_transfer` | `train_external_within_set.py` | `build_mediapipe_external.py`, `score_external_transfer.py` | `analyze_external_transfer.py`, `summary_numbers_external.py`, `t1_t4_video_null.py` |
| `identity_audit` | | | `asdpose_id_manifest.py`, `asdpose_id_audit_pilot.py`, `asdpose_id_audit_pairs.py`, `asdpose_id_audit_full.py`, `asdpose_id_fullsearch.py` |
| `reanalyses` | | | `reanalyses/regen_test_planting.py` (first), then `r18_verified_persons.py`, `r19_combined_uncertainty.py`, `r20_assumption_tests.py`, `r22_natural_rate.py`, `r23_cue_only_reference.py`, `r24_r17_counts.py`, `r25_corrected_four_cells.py`, `r26_recipe_per_fit.py` |

The tables in `paper/tables/` are written by `reanalyses/make_corrected_cells_tables.py` (Table 2
and Table S2), `make_cue_targeted_panel.py` (Table S4, panel b), `make_mapped_persons_table.py`
(Table S13) and `make_recipe_comparison_table.py` (Table S14). The other tables and the figures of the article report values read from the `analysis/` files listed above;
their formatting and plotting scripts are not included.

What runs from this repository alone: the analysis scripts that read `scores/` and `registry.json`.
For example,

```
python3 experiments/analyze_mapped_persons.py --out /tmp/check_mapped_persons
python3 experiments/analyze_recipe_comparison.py --out /tmp/check_recipe_comparison
SM_REVISION_OUT=/tmp/check_reanalyses python3 experiments/reanalyses/r25_corrected_four_cells.py
```

These reproduce the released `effects.csv`, `criteria.json` and four-cell export byte for byte
(compare with `results/mapped_persons/analysis`, `results/recipe_comparison/analysis` and
`results/reanalyses`). Some `criteria.json` files of other experiments also hold entries read from
training logs (selected epochs, group agreement); a rerun without the logs leaves those entries
empty, and the released files keep them. The per-fit part of `r26_recipe_per_fit.py` reads training logs, which are not included; without them
the script rewrites only `r26_recipe_cells.csv`. Scripts that read
poses (`regen_test_planting.py`, `r22_natural_rate.py`, `r23_cue_only_reference.py`,
`r24_r17_counts.py`, the invariance checks, the natural-failure and length analyses) need the
segment cache (see Data). Scoring and training need the checkpoints or the data.

## Training

Each line of a file in `configs/runs/` is the argument list of one fit of `train_asdpose.py`;
`bash experiments/train_runs.sh configs/runs/<file>` trains a list in order on one GPU. Fits with
the corrected loader need `SM_TRAIN_RECIPE=v15` (the primary, extension, assignment, mapped-person
and recipe experiments). MMASD and UTD-MHAD fits use `experiments/train_small_datasets.py`
(`configs/runs/small_datasets.txt`). The fits up to the assignment replication ran on one NVIDIA A10G GPU
(about 7 minutes per CompactCTRGCN fit); the mapped-person and recipe fits ran on one Quadro RTX 4000
(7.5 to 12.6 minutes per fit).

## Data

None of the datasets is redistributed here. Set the environment variables below, or place the
data under `data/` in the working directory.

| dataset | how to obtain | variable |
|---|---|---|
| ASDPose | from its authors (Barami et al., 2024); the segment cache (`index.csv` and `keypoints.f32`) is built by the SAFE-SMM code (`stage1/build_cache.py` in that repository) | `SM_ASDPOSE_CACHE` (the cache directory) or `SM_SMM_ROOT` (a SAFE-SMM checkout with `cache/segments`) |
| MMASD | the Google Drive link on the MMASD GitHub page (Li, Bhat and Barmaki, ICMI 2023); unzip `2D_skeleton/2D_openpose.zip` in place; `ADOS_rating.xlsx` beside it | `SM_MMASD_ROOT` |
| UTD-MHAD | the official dataset page (Chen, Jafari and Kehtarnavaz, ICIP 2015); the `Skeleton/` folder of `.mat` files | `SM_UTD_ROOT` |
| SSBD/ESBD external set | built by the SAFE-SMM code from the public SSBD and ESBD videos; the cache is not redistributed | `SM_EXTERNAL_CACHE`; `SM_STIMMING_ROOT` for the source videos and pose files of the MediaPipe comparison |

## Environment

Python 3.12, `pip install -r requirements.txt` (PyTorch 2.5.1; CUDA optional for scoring and
training, not needed for the analyses).

## License

MIT, see `LICENSE`.
