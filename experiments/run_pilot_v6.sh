#!/bin/bash
# Pilot v6: five TinyTCN host runs, then the v6 scoring grid. Restartable.
cd "$(dirname "$0")/.."
run() { local out=$1; shift; [ -f $out/train_log.json ] && { echo "skip $out"; return; }
  local tag=$(echo $out | tr / _)
  conda run --no-capture-output -n torch python experiments/train_asdpose.py --proto v6 --workers 5 "$@" > logs/$tag.log 2>&1 && echo "done $out $(date +%T)" || echo "FAILED $out"; }
P=results/pilots/pilot_v6/train
for s in 0 1 2; do run $P/Sp_w0/A_none/seed$s --cell A_none --world Sp --world-seed 0 --strength mid --seed $s; done
for w in 1 2; do run $P/S_w$w/A_none/seed0 --cell A_none --world S --world-seed $w --strength mid --seed 0; done
echo TRAIN_DONE
conda run --no-capture-output -n torch python experiments/score_v6.py --workers 5 > logs/pilot_v6_score.log 2>&1 && echo SCORE_DONE || echo SCORE_FAILED
