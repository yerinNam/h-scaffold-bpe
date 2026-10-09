#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_PY="$PROJECT_DIR/.venv/bin/python"
B200_PY="/home/jovyan/main-workspace/kyu/.venv/bin/python"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PROJECT_DIR/artifacts/eval" "$PROJECT_DIR/logs" "$PROJECT_DIR/report"

inputs=(
  "$PROJECT_DIR/data/eval/wikitext103_test.txt"
  "$PROJECT_DIR/data/eval/wmt17_de_en_test_en.txt"
  "$PROJECT_DIR/data/eval/blimp_all.txt"
)
pids=()
for vocab in 32768 65536; do
  for variant in bpe scaffold hierarchical; do
    "$PROJECT_PY" -m hscaffold_bpe.cli evaluate \
      "$PROJECT_DIR/artifacts/tokenizers/$variant-$vocab.json" "${inputs[@]}" \
      --output "$PROJECT_DIR/artifacts/eval/$variant-$vocab.json" \
      >"$PROJECT_DIR/logs/eval-tokenizer-$variant-$vocab.log" 2>&1 &
    pids+=("$!")
  done
done
wait "${pids[@]}"

read -r -a architectures <<<"${ARCHITECTURES:-llama-3.2-1b llama-3.1-8b}"
variants=(bpe scaffold hierarchical)
vocabs=(32768 65536)
gpus=(4 5 6 7)
jobs=()
for architecture in "${architectures[@]}"; do
  for vocab in "${vocabs[@]}"; do
    for variant in "${variants[@]}"; do
      jobs+=("$architecture|$variant|$vocab")
    done
  done
done

for ((start=0; start<${#jobs[@]}; start+=4)); do
  pids=()
  for offset in 0 1 2 3; do
    index=$((start + offset))
    (( index < ${#jobs[@]} )) || break
    IFS='|' read -r architecture variant vocab <<<"${jobs[$index]}"
    run="$architecture-$variant-$vocab"
    run_dir="$PROJECT_DIR/artifacts/lm/$run"
    if [[ -f "$run_dir/test_result.json" ]]; then
      continue
    fi
    if [[ ! -d "$run_dir/final" ]]; then
      echo "missing final checkpoint: $run_dir/final" >&2
      continue
    fi
    if [[ "$architecture" == "llama-3.2-1b" ]]; then batch=8; else batch=2; fi
    CUDA_VISIBLE_DEVICES=${gpus[$offset]} "$B200_PY" -m hscaffold_bpe.lm_eval \
      --run-dir "$run_dir" \
      --test-bin "$PROJECT_DIR/artifacts/tokens/$variant-$vocab-test.bin" \
      --batch-size "$batch" \
      >"$PROJECT_DIR/logs/eval-lm-$run.log" 2>&1 &
    pids+=("$!")
  done
  if (( ${#pids[@]} )); then wait "${pids[@]}"; fi
done

"$PROJECT_PY" -m hscaffold_bpe.report \
  --eval-dir "$PROJECT_DIR/artifacts/eval" \
  --lm-dir "$PROJECT_DIR/artifacts/lm" \
  --output "$PROJECT_DIR/report/report.html" \
  --architectures "${architectures[@]}"
