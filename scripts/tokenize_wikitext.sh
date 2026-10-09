#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_DIR/.venv/bin/python"}
MODEL_DIR=${1:-"$PROJECT_DIR/artifacts/tokenizers"}
TOKEN_DIR=${2:-"$PROJECT_DIR/artifacts/tokens"}
DATA_DIR=${3:-"$PROJECT_DIR/data/eval"}
mkdir -p "$TOKEN_DIR" "$PROJECT_DIR/logs"

pids=()
for vocab_size in 32768 65536; do
  for variant in bpe scaffold hierarchical; do
    run_name="${variant}-${vocab_size}"
    (
      for split in train validation test; do
        "$PYTHON_BIN" -m hscaffold_bpe.tokenize_data \
          --model "$MODEL_DIR/${run_name}.json" \
          --input "$DATA_DIR/wikitext103_${split}.txt" \
          --input-format text \
          --output "$TOKEN_DIR/${run_name}-${split}.bin"
      done
    ) >"$PROJECT_DIR/logs/tokenize-${run_name}.log" 2>&1 &
    pids+=("$!")
  done
done
printf '%s\n' "${pids[@]}" >"$PROJECT_DIR/logs/tokenize-wikitext.pids"
wait "${pids[@]}"

