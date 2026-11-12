#!/bin/bash
# Pilot v3 queue (protocol pilot_asdpose_v3.yaml). Skips finished runs, so it can be restarted.
cd "$(dirname "$0")/.."
run() {  # $1 out_dir, rest: args
  local out=$1; shift
  [ -f $out/train_log.json ] && { echo "skip $out"; return; }
  local tag=$(echo $out | tr / _)
  conda run --no-capture-output -n torch python experiments/train_asdpose.py --proto v3 "$@" \
    > logs/$tag.log 2>&1 && echo "done $out $(date +%T)" || echo "FAILED $out"
}
P=results/pilots/pilot_v3
for s in 0 1 2; do for w in S I; do              # H2 core first
  run $P/planted_mid/$w/A_none/seed$s --cell A_none --world $w --strength mid --seed $s; done; done
for s in 0 1 2; do for c in A_both B_coorddrop A_dropout A_velocity; do for w in S I; do
  run $P/planted_mid/$w/$c/seed$s --cell $c --world $w --strength mid --seed $s; done; done; done
for s in 0 1 2; do for w in S I; do
  run $P/planted_low/$w/A_none/seed$s --cell A_none --world $w --strength low --seed $s; done; done
for s in 3 4 5; do for c in A_none A_both B_coorddrop; do
  run $P/real/$c/seed$s --cell $c --seed $s; done; done
echo ALL_DONE
