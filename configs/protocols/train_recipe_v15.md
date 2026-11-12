# Training recipe versions

Two defects in the training-time data pipeline were found after the first 164 ASDPose
fits had been run. Neither affects inference: the test-time preprocessing is the centre
crop, which both recipes compute identically, and the release pipeline's rescoring of the 30
real-split models from their checkpoints is unchanged (checked against the v14 trees).

| | `legacy` (the 164 fits of protocols v7 to v14) | `v15` (corrected; the 12 fits of protocol v15conf, both arms) |
|---|---|---|
| random crop start, segment of T frames, window W | drawn from [0, T - W): the last legal start T - W is never used | drawn from [0, T - W] |
| DataLoader worker random streams | the dataset's `RandomState(seed)` is copied unchanged into each worker, so workers draw correlated crop / flip / dropout streams | `worker_init_fn=asdpose.worker_init` seeds each worker once with (seed, worker id); workers are persistent, so each stream continues across epochs (no per-epoch reseeding) |

Selection: environment variable `SM_TRAIN_RECIPE=legacy|v15`, default `legacy`, read by
`safe_motion/data/asdpose.py` (`crop_start_high`, `worker_init`) and used by
`safe_motion/data/hosts.py`, `experiments/train_asdpose.py` and `experiments/train_v9.py`.

Rules:
- Reproducing a historical fit uses `legacy` (the default); the code path is unchanged.
- Any new fit uses `v15` and says so in its protocol file. The numerical effect of the corrections on
  trained models has not been measured; a sensitivity run would be needed before calling it small.
- Scores, features and tables regenerated from existing checkpoints are not affected by the recipe.

Update 2026-11-09 (after protocol v15conf): the 12 fits of `configs/protocols/remedies_confonly_v15.yaml` ran under `v15`
in both arms (recorded in every train_log.json). That protocol's sentence "reseeded per worker and epoch" describes the
intent loosely; the implementation seeds each persistent worker once, as in the table above. The protocol file is frozen and
is not edited; this note and the registry record the precise behaviour.
