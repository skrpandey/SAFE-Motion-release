# Frozen protocols

Each YAML here fixes an evaluation protocol before any model is scored under it.
Required header fields: `frozen: true`, `frozen_date`, `frozen_commit` (filled after the
commit that adds the file), `purpose`, and `depends_on` (other protocol files).

Planned files (see docs/PLAN.md section 4 and 5):

- `asdpose_qc.yaml`          : the SAFE-SMM thresholds, copied verbatim, never re-tuned.
- `window_level.yaml`        : fixed 200-frame windows, no tiling; length adversary gate.
- `reweighted.yaml`          : propensity model spec for the no-discard protocol.
- `cqp_grid.yaml`            : operator families, epsilon grids, matching criterion,
                               symmetric-control definition, null-calibration settings.
- `planted_worlds.yaml`      : host definition (ASDPose-QC on train and test children),
                               Beta parameters per strength, World S / World I assignment
                               seeds, Bayes AUC per strength.
- `external_cv.yaml`         : video-level 5-fold assignment for SSBD/ESBD, DFR and
                               temperature-scaling fold usage, label-noise audit sample.
