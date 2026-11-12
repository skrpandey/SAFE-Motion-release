#!/bin/bash
# Pilot batch 2: seven TinyTCN cells in sequence (protocol pilot_asdpose_v1 + addendum 1).
cd "$(dirname "$0")/.."
for c in A_none A_both A_velocity A_dropout B_coorddrop C_qc_none C_qc_velocity; do
  [ -f results/pilots/pilot_batch2/$c/train_log.json ] && { echo "skip $c"; continue; }
  conda run --no-capture-output -n torch python experiments/train_asdpose.py --cell $c \
    > logs/pilot_batch2_$c.log 2>&1 || echo "FAILED $c"
  echo "done $c $(date +%T)"
done
