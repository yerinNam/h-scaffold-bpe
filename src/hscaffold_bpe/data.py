from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm

DATASET_REVISIONS = {
    "wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "blimp": "877fba0801ffb7cbd8c39c1ff314a46f053f6036",
    "wmt17": "54d3aacfb5429020b9b85b170a677e4bc92f2449",
}


def _write_text_rows(dataset, destination: Path, selector) -> dict:
    rows = 0
    byte_count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in dataset:
            for text in selector(record):
                text = text.strip()
                if not text:
                    continue
                handle.write(text.replace("\n", " ") + "\n")
                rows += 1
                byte_count += len(text.encode("utf-8")) + 1
    return {"path": str(destination), "rows": rows, "bytes": byte_count}


def prepare_evaluation_data(output_dir: str | Path) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"revisions": DATASET_REVISIONS.copy(), "files": {}}

    wikitext = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        revision=DATASET_REVISIONS["wikitext"],
    )
    for split in ("train", "validation", "test"):
        path = output / f"wikitext103_{split}.txt"
        manifest["files"][f"wikitext_{split}"] = _write_text_rows(
            wikitext[split], path, lambda row: [row["text"]]
        )

    blimp_path = output / "blimp_all.txt"
    blimp_rows = 0
    blimp_bytes = 0
    with blimp_path.open("w", encoding="utf-8") as handle:
        for config in get_dataset_config_names(
            "nyu-mll/blimp", revision=DATASET_REVISIONS["blimp"]
        ):
            split = load_dataset(
                "nyu-mll/blimp",
                config,
                split="train",
                revision=DATASET_REVISIONS["blimp"],
            )
            for row in split:
                for key in ("sentence_good", "sentence_bad"):
                    text = row[key].strip()
                    handle.write(text.replace("\n", " ") + "\n")
                    blimp_rows += 1
                    blimp_bytes += len(text.encode("utf-8")) + 1
    manifest["files"]["blimp"] = {
        "path": str(blimp_path),
        "rows": blimp_rows,
        "bytes": blimp_bytes,
    }

    wmt = load_dataset(
        "wmt/wmt17",
        "de-en",
        split="test",
        revision=DATASET_REVISIONS["wmt17"],
    )
    wmt_path = output / "wmt17_de_en_test_en.txt"
    manifest["files"]["wmt17_en"] = _write_text_rows(
        wmt, wmt_path, lambda row: [row["translation"]["en"]]
    )
    manifest_path = output / "eval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _stream_source(name: str, seed: int):
    if name == "wikipedia":
        dataset = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
            revision=DATASET_REVISIONS["wikipedia"],
        )
    elif name == "c4":
        dataset = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
            revision=DATASET_REVISIONS["c4"],
        )
    else:
        raise ValueError(name)
    return dataset.shuffle(seed=seed, buffer_size=10_000)


def prepare_pretraining_mix(
    output_path: str | Path,
    token_budget: int,
    wikipedia_fraction: float = 0.05,
    seed: int = 42,
) -> dict:
    """Write a deterministic Wikipedia+C4 JSONL mix.

    The budget is measured in whitespace-delimited words because it is independent
    of the tokenizer under comparison. Exact token counts are recorded later for
    every trained tokenizer.
    """
    if not 0.0 < wikipedia_fraction < 1.0:
        raise ValueError("wikipedia_fraction must be in (0, 1)")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = {
        "wikipedia": int(token_budget * wikipedia_fraction),
        "c4": token_budget - int(token_budget * wikipedia_fraction),
    }
    stats = {
        name: {"target_words": target, "words": 0, "documents": 0, "bytes": 0}
        for name, target in targets.items()
    }
    streams = {name: iter(_stream_source(name, seed)) for name in targets}
    with output_path.open("w", encoding="utf-8") as handle:
        while any(stats[name]["words"] < targets[name] for name in targets):
            unfinished = [name for name in targets if stats[name]["words"] < targets[name]]
            source = min(
                unfinished,
                key=lambda name: stats[name]["words"] / max(targets[name], 1),
            )
            row = next(streams[source])
            text = row["text"].strip()
            if not text:
                continue
            words = len(text.split())
            record = json.dumps({"source": source, "text": text}, ensure_ascii=False)
            handle.write(record + "\n")
            stats[source]["words"] += words
            stats[source]["documents"] += 1
            stats[source]["bytes"] += len(text.encode("utf-8"))
    manifest = {
        "path": str(output_path),
        "requested_word_budget": token_budget,
        "wikipedia_fraction": wikipedia_fraction,
        "seed": seed,
        "revisions": {key: DATASET_REVISIONS[key] for key in ("wikipedia", "c4")},
        "sources": stats,
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def prepare_tokenizer_corpus(
    output_path: str | Path, word_budget: int = 1_000_000_000, seed: int = 42
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    words_seen = 0
    documents = 0
    byte_count = 0
    progress = tqdm(total=word_budget, unit="word", desc="wikipedia-tokenizer-corpus")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in _stream_source("wikipedia", seed):
            text = row["text"].strip()
            if not text:
                continue
            handle.write(text + "\n<|endoftext|>\n")
            new_words = len(text.split())
            words_seen += new_words
            progress.update(new_words)
            documents += 1
            byte_count += len(text.encode("utf-8"))
            if words_seen >= word_budget:
                break
    progress.close()
    manifest = {
        "path": str(output_path),
        "requested_word_budget": word_budget,
        "words": words_seen,
        "documents": documents,
        "bytes": byte_count,
        "seed": seed,
        "dataset": "wikimedia/wikipedia:20231101.en",
        "revision": DATASET_REVISIONS["wikipedia"],
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--output-dir", default="data/eval")
    pretraining = subparsers.add_parser("pretraining")
    pretraining.add_argument("--output", required=True)
    pretraining.add_argument("--token-budget", type=int, required=True)
    pretraining.add_argument("--wikipedia-fraction", type=float, default=0.05)
    pretraining.add_argument("--seed", type=int, default=42)
    tokenizer = subparsers.add_parser("tokenizer")
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--word-budget", type=int, default=1_000_000_000)
    tokenizer.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.__dict__.get("output_dir") is not None:
        result = prepare_evaluation_data(args.output_dir)
    elif args.__dict__.get("word_budget") is not None:
        result = prepare_tokenizer_corpus(args.output, args.word_budget, args.seed)
    else:
        result = prepare_pretraining_mix(
            args.output, args.token_budget, args.wikipedia_fraction, args.seed
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
