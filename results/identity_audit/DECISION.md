# ASDPose identity and split integrity: decision (2026-11-05)

Inputs: annotated release `dataset.pkl` via the SAFE-SMM cache index (329 IDs, 1,116 recording
groups); raw release `asdpose.zip` (914 recordings, `db.csv`: 241 child_id values, which index one
child across assessments). Scripts: `experiments/asdpose_id_audit_full.py` (pass 1, stated-time
window), `asdpose_id_manifest.py` (manifest, statuses, partitions), `asdpose_id_fullsearch.py`
(pass 2, full-length FFT search for unresolved test IDs, with 30 accepted groups as positive
controls). Outputs in this folder; hashes in `summary.json`.

## Findings
- Pass 1 (stated start time, all 914 recordings): 385 of 1,116 groups accepted, 18 ambiguous, 713
  no candidate; 185 of 329 IDs accepted (132 train, 37 internal validation, 15 test; 3 IDs have no
  length-protocol rows). 114 IDs with two or more accepted groups: 0 conflicting people.
  Accepted errors <= 7.95 px; the closest wrong-child error for accepted groups is 9.88 px.
- Pass 2 (full recordings, 25 unresolved test IDs, 147 usable queries): controls recovered 52 of 57;
  at ratio (second-child / best) >= 2, 32 of 32 controls correct. Three more test IDs resolved
  (68 -> raw 76, 116 -> raw 119, 304 -> raw 175); none of these people has any other annotated ID.
  The other 22 test IDs match no released recording at any offset.
- Aliases: one person (raw 194) under annotated 196 (training) and 322 (internal **validation**),
  same ADOS session, different segments. No person found in both training/validation and test.
- Coverage of scored rows (length-protocol real test split): 1,538 of 3,834 segments (287 of 786
  SMM) belong to the 15 pass-1 test IDs; pass 2 adds three IDs.

## Decision: unresolved (no leakage found; separation not certified)
Among the 188 annotated IDs assigned to a released recording, no person spans the released
training and test partitions; one person spans training and internal validation. For 22 of the 40
test IDs (and 141 IDs overall) no released recording matches, so person-level separation of the
complete release cannot be established from the released files. The released IDs are a
re-numbering of the raw child IDs (1 of 185 keeps its number), so 329 IDs are not evidence of 329
children. Next routes: authoritative mapping from the dataset custodians, or a released-ID analysis
stated as such. The train/validation alias can affect checkpoint selection only; no test row is
involved.
