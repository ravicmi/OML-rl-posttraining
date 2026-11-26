#!/bin/bash
# Run at the end of every difficulty shard job:  bash scripts/after_difficulty.sh <N>
# Whichever shard finishes last (mkdir lock) merges the N-sample pass rates and enqueues the training runs that depend on them.
N=$1; ROOT=/home/ravi.sah/oml; D=$ROOT/outputs/difficulty
for i in 0 1 2 3; do [[ -f $D/n${N}_shard$i.done ]] || exit 0; done
mkdir "$D/n${N}_merge.lock" 2>/dev/null || exit 0
cat $D/math$N.shard?.jsonl > $D/math$N.jsonl.tmp && mv $D/math$N.jsonl.tmp $D/math$N.jsonl
python3 $ROOT/src/scripts/make_queue_pr_n.py $N
[[ -s $D/math16.jsonl && -s $D/math32.jsonl ]] && python3 $ROOT/src/scripts/passrate_compare.py > $D/compare.txt 2>&1
exit 0
