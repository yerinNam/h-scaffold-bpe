#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"/home/jovyan/main-workspace/kyu/.venv/bin/python"}
TOKEN_DIR=${1:-"$PROJECT_DIR/artifacts/tokens"}
OUTPUT_ROOT=${2:-"$PROJECT_DIR/artifacts/lm"}
mkdir -p "$OUTPUT_ROOT" "$PROJECT_DIR/logs"

architectures=(llama-3.2-1b llama-3.1-8b)
variants=(bpe scaffold hierarchical)
vocab_sizes=(32768 65536)
gpus=(4 5 6 7)
jobs=()

for architecture in "${architectures[@]}"; do
  for vocab_size in "${vocab_sizes[@]}"; do
    for variant in "${variants[@]}"; do
      jobs+=("${architecture}|${variant}|${vocab_size}")
    done
  done
done

for ((wave_start=0; wave_start<${#jobs[@]}; wave_start+=4)); do
  pids=()
  for offset in 0 1 2 3; do
    job_index=$((wave_start + offset))
    if (( job_index >= ${#jobs[@]} )); then
      break
    fi
    IFS='|' read -r architecture variant vocab_size <<<"${jobs[$job_index]}"
    gpu=${gpus[$offset]}
    tokenizer_name="${variant}-${vocab_size}"
    run_name="${architecture}-${tokenizer_name}"
    if [[ "$architecture" == "llama-3.2-1b" ]]; then
      micro_batch=16
      checkpoint_flag=--no-gradient-checkpointing
    else
      micro_batch=16
      checkpoint_flag=--gradient-checkpointing
    fi
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" -m hscaffold_bpe.lm \
      --architecture "$architecture" \
      --train-bin "$TOKEN_DIR/${tokenizer_name}-train.bin" \
      --validation-bin "$TOKEN_DIR/${tokenizer_name}-validation.bin" \
      --output-dir "$OUTPUT_ROOT/$run_name" \
      --epochs 5 \
      --sequence-length 1024 \
      --micro-batch-size "$micro_batch" \
      --grad-accum 1 \
      "$checkpoint_flag" \
      --attention-implementation kernels-community/flash-attn \
      --optimizer adamw_bnb_8bit \
      >"$PROJECT_DIR/logs/lm-${run_name}.log" 2>&1 &
    pids+=("$!")
    echo "$run_name gpu=$gpu pid=$!"
  done
  wait "${pids[@]}"
done
