#!/bin/bash
# Failure tests for run_release_pipeline.sh on a throwaway tree: the driver must stop on a stray
# misnamed MediaPipe folder and on a missing MediaPipe cache, and score_v11.py must reject the old
# tag. Only the throwaway tree is modified. Usage: bash experiments/test_release_pipeline.sh
set -uo pipefail
T=build/test_release_tmp
python3 experiments/build_repair_view.py --rule clean --out $T --rescore-all-ours > /dev/null || exit 1
pass=0; fail=0
check() { local want=$1 name=$2; shift 2
  if "$@" > $T.out 2>&1; then got=ok; else got=fail; fi
  if [ "$got" = "$want" ]; then pass=$((pass+1)); echo "PASS $name ($got: $(tail -1 $T.out))"
  else fail=$((fail+1)); echo "FAIL $name (expected $want, got $got): $(tail -3 $T.out)"; fi; }
run() { STOP_AFTER=preflight bash experiments/run_release_pipeline.sh $T; }

check ok   "intact tree passes preflight" run
# output guard: a results/v7 that links into a protected fixture must be refused by the preflight,
# and the fixture must stay empty
F=$(realpath -m $T.fixture); mkdir -p $F/v7/analysis
mv $T/results/v7 $T/results/v7.real && ln -s $F/v7 $T/results/v7
check ok   "results/v7 linking into a protected fixture stops the preflight at the guard" \
  bash -c "SM_PROTECTED_EXTRA=$F STOP_AFTER=preflight bash experiments/run_release_pipeline.sh $T 2>&1 | grep -q 'output guard'"
check ok   "the protected fixture was not written" bash -c "[ -z \"\$(find $F -type f)\" ]"
rm $T/results/v7 && mv $T/results/v7.real $T/results/v7 && rm -rf $F
mkdir $T/results/v11/scoresmediapipe
check fail "stray scoresmediapipe/ stops the driver" run
rmdir $T/results/v11/scoresmediapipe
mv $T/results/v11/mediapipe_cache $T/results/v11/mediapipe_cache.moved && mkdir $T/results/v11/mediapipe_cache
check fail "missing MediaPipe cache stops the driver" run
rmdir $T/results/v11/mediapipe_cache && mv $T/results/v11/mediapipe_cache.moved $T/results/v11/mediapipe_cache
check ok   "score_v11.py rejects --tag mediapipe" bash -c "cd $T && conda run -n torch python experiments/score_v11.py --tag mediapipe --only none_such 2>&1 | grep -q \"tag must be\""
check ok   "intact tree passes preflight again" run
rm -rf $T $T.out
echo "tests: $pass passed, $fail failed"
[ $fail -eq 0 ]
