#!/bin/bash
# Pilot v2 queue (protocol pilot_asdpose_v2.yaml). Skips finished runs, so it can be restarted.
cd "$(dirname "$0")/.."
run() {  # $1 out_dir, rest: args
  local out=$1; shift
  [ -f $out/train_log.json ] && { echo "skip $out"; return; }
  local tag=$(echo $out | tr / _)
  conda run --no-capture-output -n torch python experiments/train_asdpose.py --proto v2 "$@" \
    > logs/$tag.log 2>&1 && echo "done $out $(date +%T)" || echo "FAILED $out"
}
for s in 1 2; do for c in A_none A_both B_coorddrop A_dropout A_velocity; do
  run results/pilots/pilot_v2/$c/seed$s --cell $c --seed $s; done; done
for s in 0 1 2; do for c in D_qcsym_none D_qcsym_velocity; do
  run results/pilots/pilot_v2/$c/seed$s --cell $c --seed $s; done; done
for p in $(seq 100 109); do run results/pilots/pilot_v2/perm/A_none_p$p --cell A_none --permute $p; done
echo ALL_DONE
