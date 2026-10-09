#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_DIR"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
MODEL_DIR="$PROJECT_DIR/artifacts/ablation/tokenizers"
EVAL_DIR="$PROJECT_DIR/artifacts/ablation/eval"
mkdir -p "$MODEL_DIR" "$EVAL_DIR"

for threshold in 0.25 0.75; do
  if [[ "$threshold" == 0.25 ]]; then tag=t025; else tag=t075; fi
  if [[ ! -f "$MODEL_DIR/$tag/hierarchical-32768.json" || ! -f "$MODEL_DIR/$tag/hierarchical-65536.json" ]]; then
    "$PYTHON_BIN" -m hscaffold_bpe.cli train-all \
      data/tokenizer/wikipedia_en_1b.counts.pkl "$MODEL_DIR/$tag" \
      --variant hierarchical --vocab-sizes 32768 65536 --level2-threshold "$threshold"
  fi
done

for vocab in 32768 65536; do
  for tag in t025 t050 t075; do
    if [[ "$tag" == t050 ]]; then
      model="artifacts/tokenizers/hierarchical-$vocab.json"
    else
      model="artifacts/ablation/tokenizers/$tag/hierarchical-$vocab.json"
    fi
    if [[ ! -f "$EVAL_DIR/$tag-$vocab.json" ]]; then
      "$PYTHON_BIN" -m hscaffold_bpe.cli evaluate "$model" \
        data/eval/wikitext103_test.txt data/eval/wmt17_de_en_test_en.txt \
        data/eval/blimp_all.txt --output "$EVAL_DIR/$tag-$vocab.json" >/dev/null
    fi
  done
done

"$PYTHON_BIN" scripts/analyze_ablation.py
"$PYTHON_BIN" -m hscaffold_bpe.report --eval-dir artifacts/eval \
  --lm-dir artifacts/lm --output report/report.html --architectures llama-3.2-1b
