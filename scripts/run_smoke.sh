#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_DIR/.venv/bin/python"}
SMOKE_DIR="$PROJECT_DIR/artifacts/smoke"
mkdir -p "$SMOKE_DIR"

printf '%s\n' \
  'Arizona uses zona as a scaffold example.' \
  'Hierarchical tokenization should remain lossless.' \
  'The quick brown fox jumps over the lazy dog.' \
  >"$SMOKE_DIR/corpus.txt"

"$PYTHON_BIN" -m hscaffold_bpe.cli count \
  "$SMOKE_DIR/corpus.txt" "$SMOKE_DIR/counts.pkl" --workers 1

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
from hscaffold_bpe.pretokenize import load_counts
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from hscaffold_bpe.trainer import train_from_counts

root = Path("artifacts/smoke")
counts = load_counts(root / "counts.pkl")
for variant in ("bpe", "scaffold", "hierarchical"):
    model = train_from_counts(counts, 270, variant, show_progress=False)
    model.save(root / f"{variant}.json")
    tokenizer = HierarchicalTokenizer(model)
    text = "Arizona — café."
    assert tokenizer.decode(tokenizer.encode(text)) == text
    print(variant, len(model.tokens), tokenizer.vocab_size, tokenizer.encode(text))
PY

