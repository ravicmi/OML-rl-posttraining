#!/bin/bash
# Fill in any missing evals for a finished training run (used to recover runs whose worker died during the eval phase).
#   bash scripts/eval_missing.sh <run_name>      (run from $ROOT/src with venv active, CUDA_VISIBLE_DEVICES set)
set -u
OUT=${OUT:-/home/ravi.sah/oml/outputs}; n=$1
fin=$(cat "$OUT/runs/$n/final_path.txt")
[[ -s $OUT/eval/$n.json ]] || python -m oml.eval --model "$fin" --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 --out "$OUT/eval/$n.json" || exit 1
for ck in $(ls -d $OUT/runs/$n/stage*/checkpoint-* | sort -t- -k2 -n | head -n -1); do
  o="$OUT/eval_ckpt/${n}_$(basename $ck).json"
  [[ -s $o ]] || python -m oml.eval --model "$ck" --sets math500 --n 4 --max_tokens 1536 --out "$o" || true
done
