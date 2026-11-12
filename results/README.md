# Results

Folders are named by protocol version (`configs/protocols/`). Each holds `analysis/` (the CSV and
JSON files the tables are built from), usually `scores/` (per-segment test scores) and a
`REPORT.md` for the round.

## Where the paper's numbers come from

| paper | folders |
|---|---|
| Table I, Fig. 2 (MWI per mechanism) | `v7` (missing runs, whole-joint absence, frozen frames; v8 cells are stored here too), `v13` (arm confidence) |
| Table II (remedies) | `v7` (input configurations), `v12` (Group DRO, adversary, augmentation), `v15conf` (Group DRO, confidence-only) |
| QC exclusion, natural cue, multiplicity, cluster scenarios (S12 to S15) | `v7/analysis` (post hoc files; Holm for v12 in `v12/analysis`) |
| Identity audit (S3a) | `asdpose_id_audit` (decision record, manifests, match distances; scripts `experiments/asdpose_id_*.py`) |
| Confidence-channel reliance (S3f, S3h) | `posthoc/conf_reliance`, `v15conf/analysis`, `v16/conf_reliance`, `v16/analysis/conf_reliance.csv` |
| Confidence-only extension: ST-GCN, nine seeds, Group DRO with groups from the input (S3h) | `v16` (`analysis`: cleaned selection; `analysis_orig`: original selection) |
| Confidence-only assignment replication (S3i; main Table 2) | `v17` (`analysis`: cleaned selection; `analysis_orig`: original selection; `analysis/assignment_severities.csv`: Table S9) |
| Provenance of every final model (main Table 1, Table S2) | `provenance/models.csv` |
| Adversary probe (S3e) | `v12/analysis` |
| Interpolated control and mask stability (S3c, S7) | `v10` |
| Real models (S6) | `v7` (real split, filtered gap), `v10` (real-data reference) |
| Small datasets (S7) | `v9`, `audit_mmasd`, `audit_utd_mhad` |
| Transfer and calibration on SSBD/ESBD (S8, S9) | `v11` |
| Selection repair (S11) | the repaired scores in `v7`, `v10`, `v12`, `v13`; counts in `v7/analysis` |

## Development record

`pilots/` holds the pilot rounds v1 (the two batches) to v6 (small temporal network). They shaped the later
protocols and are summarised in supplement S17; the main results do not rest on them. Older protocol
files still cite their former paths (`results/pilot_*`), since protocol files are not edited after
they are committed.

`gpu_ledger.csv` logs GPU time per run.
