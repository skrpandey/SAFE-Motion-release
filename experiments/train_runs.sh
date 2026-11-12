#!/bin/bash
# Train every line of a run list in configs/runs/ in order on one GPU. A line is
# "<out_dir> <arguments>" (skipped when out_dir already holds train_log.json) or, for v15conf, the
# arguments only (the trainer skips a finished fit). Scoring and analysis are separate steps
# (score_*.py, analyze_*.py; see the README).
#   bash experiments/train_runs.sh configs/runs/queue_v12.txt
#   bash experiments/train_runs.sh configs/runs/queue_v9.txt experiments/train_v9.py
#   SM_TRAIN_RECIPE=v15 bash experiments/train_runs.sh configs/runs/queue_v15conf.txt
set -uo pipefail
cd "$(dirname "$0")/.."
LIST=$1
TRAIN=${2:-experiments/train_asdpose.py}
mkdir -p logs/runs
n=0
while read -r first rest; do
  [ -z "$first" ] && continue
  n=$((n+1)); tag=$(basename "$LIST" .txt)_$(printf %04d $n)
  if [[ $first == --* ]]; then
    args="$first $rest"
  else
    [ -f "$first/train_log.json" ] && { echo "skip $first"; continue; }
    args=$rest
  fi
  SM_EVENT_LOG="logs/runs/events_$tag.log" python3 "$TRAIN" $args > "logs/runs/$tag.log" 2>&1 \
    && echo "done $tag" || echo "FAILED $tag: $args"
done < "$LIST"
