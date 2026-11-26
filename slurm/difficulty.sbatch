#!/bin/bash
# Score MATH-train difficulty with the starting policy (4 GPU shards), and run the n=8 baseline eval on a 5th GPU.
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --exclude=ip-10-0-240-88,ip-10-0-130-183
#SBATCH --output=/home/ravi.sah/oml/logs/%x-%j.out
set -uo pipefail
ROOT=/home/ravi.sah/oml
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false
cd "$ROOT/src"
OUT=$ROOT/outputs; mkdir -p $OUT/difficulty $OUT/eval

# --- bad-node guard: some nodes have stray processes (e.g. 2-day-old root vLLM workers) holding all GPU memory.
check_node() {  # $1 = GPUs needed, $2 = script path, rest = original args
  local need=$1 script=$2; shift 2
  local nfree; nfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '$1>70000' | wc -l)
  echo "$(hostname): $nfree free GPUs (need $need)"
  if (( nfree < need )); then
    echo "$(hostname)" >> "$ROOT/bad_nodes.txt"
    local retry=${RETRY:-0}
    if (( retry >= 8 )); then echo "too many bad nodes, giving up"; exit 1; fi
    local ex; ex=$(sort -u "$ROOT/bad_nodes.txt" | paste -sd,)
    echo "resubmitting excluding: $ex"
    sbatch --exclude="$ex" --job-name="$SLURM_JOB_NAME" --export=ALL,RETRY=$((retry+1)) "$script" "$@"
    exit 0
  fi
}
check_node 5 "$ROOT/src/slurm/difficulty.sbatch" "$@"
FREE=($(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>70000{print $1}'))
echo "$(hostname) free GPUs: ${FREE[*]}"
(( ${#FREE[@]} >= 5 )) || { echo "need 5 free GPUs"; exit 1; }
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=${FREE[$i]} python -m oml.difficulty --model Qwen/Qwen3-0.6B --train math --n 8 --max_tokens 1024 \
    --shard $i --nshards 4 --out $OUT/difficulty/math.shard$i.jsonl > $ROOT/logs/difficulty_$i.log 2>&1 &
  sleep 15
done
CUDA_VISIBLE_DEVICES=${FREE[4]} python -m oml.eval --model Qwen/Qwen3-0.6B --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 \
  --out $OUT/eval/base8_qwen3_0p6b.json > $ROOT/logs/base8.log 2>&1 &
wait
cat $OUT/difficulty/math.shard?.jsonl > $OUT/difficulty/math.jsonl
python - <<'EOF'
import json, collections
r=[json.loads(l) for l in open("/home/ravi.sah/oml/outputs/difficulty/math.jsonl")]
pr=[x["pass_rate"] for x in r]
print("problems:",len(r)," always-solved(1.0):",sum(p==1 for p in pr)," never-solved(0.0):",sum(p==0 for p in pr)," informative(0<p<1):",sum(0<p<1 for p in pr))
EOF
echo "difficulty done"
