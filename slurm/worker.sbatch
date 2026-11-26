#!/bin/bash
# Queue worker: lands on a node, uses whichever GPUs are ACTUALLY free (>70 GiB; Slurm "idle" is unreliable here because
# other users run jobs outside Slurm), and keeps pulling runs from a shared queue until it is empty.
#   queue layout: $Q/pending/<id>  (one file per run: "<name> <train|eval|shell> <args...>")  ->  $Q/claimed/<id>  (atomic mv)
# Submit several:  for i in 1 2 3 4 5; do sbatch --job-name=worker slurm/worker.sbatch; done
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --exclude=ip-10-0-240-88,ip-10-0-130-183,ip-10-0-129-167
#SBATCH --output=/home/ravi.sah/oml/logs/%x-%j.out
set -uo pipefail
ROOT=/home/ravi.sah/oml
source "$ROOT/venv/bin/activate"
export HF_HOME=$ROOT/hf_cache PYTHONPATH=$ROOT/src TOKENIZERS_PARALLELISM=false OUT=$ROOT/outputs
cd "$ROOT/src"
Q=$ROOT/queue; DIFF=$OUT/difficulty
mkdir -p "$OUT/eval" "$OUT/eval_ckpt" "$ROOT/logs/runs" "$DIFF" "$Q/claimed" "$Q/done" "$Q/failed"
MIN_GPUS=3                      # a node with fewer free GPUs is not worth holding
RUN_SECS=$((3*3600+1200))       # don't start a run that can't finish before the job's time limit
DEADLINE=$(( $(date +%s) + 12*3600 ))

free_gpus() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>70000{print $1}'; }

# --- bad-node guard (timestamped so nodes can be retried later)
NOW=$(date +%s); touch "$ROOT/bad_nodes.txt"
nfree=$(free_gpus | wc -l); echo "$(hostname): $nfree free GPUs: $(free_gpus | tr '\n' ' ')"
if (( nfree < MIN_GPUS )); then
  echo "$(hostname) $NOW" >> "$ROOT/bad_nodes.txt"
  retry=${RETRY:-0}; (( retry >= 25 )) && { echo "giving up after $retry retries"; exit 1; }
  ex=$(awk -v n=$NOW '$2>n-7200{print $1}' "$ROOT/bad_nodes.txt" | sort -u | paste -sd,)
  echo "resubmitting excluding: $ex"
  sbatch --exclude="$ex" --job-name="$SLURM_JOB_NAME" --export=ALL,RETRY=$((retry+1)) "$ROOT/src/slurm/worker.sbatch"
  exit 0
fi

claim_next() {   # prints claimed id; skips filter-arm runs until difficulty shards exist
  local f
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    if grep -q passrate_path "$Q/pending/$f" 2>/dev/null && ! ls "$DIFF"/shard{0,1,2,3}.done >/dev/null 2>&1; then continue; fi
    if mv "$Q/pending/$f" "$Q/claimed/$f" 2>/dev/null; then echo "$f"; return 0; fi
  done
  return 1
}

run_one() {  # $1=gpu $2=id
  local gpu=$1 id=$2 t0=$(date +%s) rc=0
  read -r name kind args < "$Q/claimed/$id"
  export CUDA_VISIBLE_DEVICES=$gpu MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29500 + gpu * 17))
  local log="$ROOT/logs/runs/$name.log"
  case $kind in
    train)
      if [[ $args == *passrate_path* && ! -s $DIFF/math.jsonl ]]; then cat "$DIFF"/math.shard?.jsonl > "$DIFF/math.jsonl.tmp.$$" && mv -f "$DIFF/math.jsonl.tmp.$$" "$DIFF/math.jsonl"; fi
      python -m oml.train_rl --config configs/base.yaml run_name=$name $args > "$log" 2>&1 \
      && python -m oml.eval --model "$(cat $OUT/runs/$name/final_path.txt)" --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 --out "$OUT/eval/$name.json" >> "$log" 2>&1
      rc=$?
      if (( rc == 0 )); then   # learning curve: MATH-500 at each intermediate checkpoint (last one == final model)
        for ck in $(ls -d $OUT/runs/$name/stage*/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -n -1); do
          python -m oml.eval --model "$ck" --sets math500 --n 4 --max_tokens 1536 --out "$OUT/eval_ckpt/${name}_$(basename $ck).json" >> "$log" 2>&1 || true
        done
      fi ;;
    eval)  python -m oml.eval --model "$args" --sets math500,gsm8k,aime24 --n 8 --max_tokens 1536 --out "$OUT/eval/$name.json" > "$log" 2>&1; rc=$? ;;
    shell) bash -c "$args" > "$log" 2>&1; rc=$? ;;
  esac
  local dt=$(( $(date +%s) - t0 ))
  if (( rc == 0 )); then mv "$Q/claimed/$id" "$Q/done/$id"
  elif (( dt < 900 )) && [[ $id != *.r2 ]]; then   # died early (likely GPU grabbed by someone): retry, up to 2 times
    local base=${id%%.r*} n=0; [[ $id == *.r* ]] && n=${id##*.r}
    mv "$Q/claimed/$id" "$Q/pending/$base.r$((n+1))"
  else mv "$Q/claimed/$id" "$Q/failed/$id"; fi
  echo "[$(date +%T)] finished $name on $(hostname) gpu $gpu rc=$rc after ${dt}s"
}

declare -A BUSY; idle_since=0
while true; do
  started=0
  if (( $(date +%s) + RUN_SECS < DEADLINE )); then
    for g in $(free_gpus); do
      [[ -n "${BUSY[$g]:-}" ]] && kill -0 "${BUSY[$g]}" 2>/dev/null && continue
      id=$(claim_next) || break
      echo "[$(date +%T)] gpu $g <- $id"
      run_one "$g" "$id" < /dev/null &
      BUSY[$g]=$!; started=1; sleep 25   # stagger startups
    done
  fi
  running=0; for g in "${!BUSY[@]}"; do kill -0 "${BUSY[$g]}" 2>/dev/null && running=$((running+1)); done
  if (( running == 0 && started == 0 )); then
    pend=$(ls "$Q/pending" 2>/dev/null | wc -l)
    (( pend == 0 )) && break
    (( idle_since == 0 )) && idle_since=$(date +%s)
    (( $(date +%s) - idle_since > 1500 )) && { echo "only blocked/unrunnable jobs left ($pend); exiting"; break; }
  else idle_since=0; fi
  sleep 20
done
wait
echo "worker done on $(hostname)"
