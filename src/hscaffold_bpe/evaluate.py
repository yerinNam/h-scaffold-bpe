from __future__ import annotations

import statistics
import time
from pathlib import Path

from .pretokenize import PATTERN
from .tokenizer import HierarchicalTokenizer


def _measure_file(tokenizer: HierarchicalTokenizer, path: Path, warmup: int) -> dict:
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                texts.append(line)
    for text in texts[:warmup]:
        tokenizer.encode(text)

    words = 0
    tokens = 0
    bytes_seen = 0
    started = time.perf_counter()
    for text in texts:
        tokens += len(tokenizer.encode(text))
        bytes_seen += len(text.encode("utf-8"))
        words += sum(bool(match.group().strip()) for match in PATTERN.finditer(text))
    elapsed = time.perf_counter() - started
    return {
        "path": str(path),
        "words": words,
        "tokens": tokens,
        "bytes": bytes_seen,
        "fertility": tokens / max(words, 1),
        "bytes_per_token": bytes_seen / max(tokens, 1),
        "tokens_per_second": tokens / max(elapsed, 1e-12),
        "megabytes_per_second": bytes_seen / max(elapsed, 1e-12) / 1_000_000,
        "elapsed_seconds": elapsed,
    }


def evaluate_tokenizer(model_path: str, inputs: list[str], warmup: int = 100) -> dict:
    tokenizer = HierarchicalTokenizer.from_file(model_path)
    domains = [_measure_file(tokenizer, Path(path), warmup) for path in inputs]
    fertilities = [domain["fertility"] for domain in domains]
    metadata = tokenizer.model.to_dict()["metadata"]
    return {
        "model": model_path,
        "variant": tokenizer.model.variant,
        "target_vocab_size": tokenizer.model.target_vocab_size,
        "expanded_vocab_size": len(tokenizer.model.tokens),
        "level_counts": metadata["level_counts"],
        "tokenizer_training_seconds": tokenizer.model.training_seconds,
        "domains": domains,
        "fertility_mean": statistics.fmean(fertilities),
        "fertility_population_std": statistics.pstdev(fertilities),
    }
