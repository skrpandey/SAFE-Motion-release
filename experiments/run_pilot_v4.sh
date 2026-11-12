#!/bin/bash
# Pilot v4 queues (protocol pilot_asdpose_v4.yaml). Usage: run_pilot_v4.sh a|b. Restartable.
cd "$(dirname "$0")/.."
run() {
  local out=$1; shift
  [ -f $out/train_log.json ] && { echo "skip $out"; return; }
  local tag=$(echo $out | tr / _)
  conda run --no-capture-output -n torch python experiments/train_asdpose.py --proto v4 --workers 5 "$@" \
    > logs/$tag.log 2>&1 && echo "done $out $(date +%T)" || echo "FAILED $out"
}
P=results/pilots/pilot_v4
if [ "$1" = a ]; then
  C="--arch compact --amp --batch-size 128 --epochs 25 --mechanism runs"
  for s in 0 1 2; do for w in S I; do
    run $P/compact_runs_mid/$w/A_none/seed$s --cell A_none --world $w --strength mid --seed $s $C; done; done
  for s in 0 1; do for w in S I; do
    run $P/compact_runs_low/$w/A_none/seed$s --cell A_none --world $w --strength low --seed $s $C; done; done
else
  for s in 0 1 2; do for c in A_none A_dropout B_coorddrop A_velocity; do for w in S I; do
    run $P/tcn_whole_mid/$w/$c/seed$s --cell $c --world $w --strength mid --seed $s --mechanism whole; done; done; done
fi
echo ALL_DONE_$1
