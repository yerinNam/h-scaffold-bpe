#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
COUNTS_PATH=${1:-"$PROJECT_DIR/data/tokenizer/wikipedia_en_1b.counts.pkl"}
MODEL_DIR=${2:-"$PROJECT_DIR/artifacts/tokenizers"}
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_DIR/.venv/bin/python"}
mkdir -p "$MODEL_DIR" "$PROJECT_DIR/logs"

pids=()
for variant in bpe scaffold hierarchical; do
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u -m hscaffold_bpe.cli train-all \
    "$COUNTS_PATH" \
    "$MODEL_DIR" \
    --variant "$variant" \
    --vocab-sizes 32768 65536 \
    >"$PROJECT_DIR/logs/tokenizer-${variant}.log" 2>&1 &
  pids+=("$!")
  echo "$variant pid=$!"
done
printf '%s\n' "${pids[@]}" >"$PROJECT_DIR/logs/tokenizer-training.pids"
wait "${pids[@]}"
