#!/bin/bash
# Pack many small runs onto ONE whole node (8xH100): one run per free GPU, queue the rest.
# Usage: sbatch --job-name=grid1 slurm/pack_node.sbatch <jobs_file>
#   jobs_file: one run per line:  <run_name> <train|eval> <args...>
#     train line args:  key=value overrides for configs/base.yaml   (then auto-evaluated on MATH-500/GSM8K/AIME24)
#     eval  line args:  <model_path_or_hf_id>                        (baseline eval only)
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --exclude=ip-10-0-240-88,ip-10-0-130-183
#SBATCH --output=/home/ravi.sah/oml/logs/%x-%j.out
set -uo pipefail
ROOT=/home/ravi.sah/oml
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false
cd "$ROOT/src"
JOBS=$1
OUT=$ROOT/outputs

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
check_node 8 "$ROOT/src/slurm/pack_node.sbatch" "$@"
mkdir -p "$OUT/eval" "$ROOT/logs/runs"

free_gpus() {  # GPUs with > 70 GiB free and not claimed by this script
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>70000{print $1}'
}
declare -A BUSY   # gpu -> pid

run_one() {  # $1=gpu $2=name $3=kind ${@:4}=args
  local gpu=$1 name=$2 kind=$3; shift 3
  export CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + gpu * 17))
  if [[ $kind == train ]]; then
    python -m oml.train_rl --config configs/base.yaml run_name=$name "$@" > "$ROOT/logs/runs/$name.log" 2>&1 \
      && python -m oml.eval --model "$(cat $OUT/runs/$name/final_path.txt)" --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 \
           --out "$OUT/eval/$name.json" >> "$ROOT/logs/runs/$name.log" 2>&1
    # learning curve: MATH-500 at each intermediate checkpoint (the last one equals the final model)
    mkdir -p "$OUT/eval_ckpt"
    for ck in $(ls -d $OUT/runs/$name/stage*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -1); do
      python -m oml.eval --model "$ck" --sets math500 --n 4 --max_tokens 1536 --out "$OUT/eval_ckpt/${name}_$(basename $ck).json" >> "$ROOT/logs/runs/$name.log" 2>&1 || true
    done
  else
    python -m oml.eval --model "$1" --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 --out "$OUT/eval/$name.json" > "$ROOT/logs/runs/$name.log" 2>&1
  fi
  local rc=$?
  echo "[$(date +%T)] done $name (gpu $gpu, exit $rc)"
}

while read -r name kind args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  waited=0
  while true; do   # wait for a free GPU
    if (( waited == 0 )); then echo "[$(date +%T)] $(hostname): free GPUs: $(free_gpus | tr '\n' ' ')"; fi
    if (( waited > 300 )); then echo "no free GPU for 300s on $(hostname); aborting"; nvidia-smi --query-gpu=index,memory.used --format=csv; exit 1; fi
    for g in $(free_gpus); do
      if [[ -z "${BUSY[$g]:-}" ]] || ! kill -0 "${BUSY[$g]}" 2>/dev/null; then
        echo "[$(date +%T)] start $name on gpu $g"
        run_one "$g" "$name" "$kind" $args < /dev/null &
        BUSY[$g]=$!; sleep 20; continue 3   # stagger startups; re-read next job
      fi
    done
    sleep 10; waited=$((waited+10))
  done
done < "$JOBS"
wait
echo "all done"
