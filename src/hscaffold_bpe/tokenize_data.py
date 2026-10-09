from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import HierarchicalTokenizer


def tokenize_file(model_path: str, input_path: str, output_path: str, input_format: str) -> dict:
    tokenizer = HierarchicalTokenizer.from_file(model_path)
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.uint32
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    tokens = 0
    source_bytes = 0
    with Path(input_path).open(encoding="utf-8") as source, output.open("wb") as destination:
        for line in tqdm(source, desc=f"tokenize:{output.name}"):
            text = json.loads(line)["text"] if input_format == "jsonl" else line.rstrip("\n")
            if not text:
                continue
            ids = tokenizer.encode(text, add_eos=True)
            np.asarray(ids, dtype=dtype).tofile(destination)
            documents += 1
            tokens += len(ids)
            source_bytes += len(text.encode("utf-8"))
    metadata = {
        "tokenizer": model_path,
        "input": input_path,
        "input_format": input_format,
        "output": output_path,
        "dtype": np.dtype(dtype).name,
        "documents": documents,
        "tokens": tokens,
        "source_bytes": source_bytes,
        "eos_token_id": tokenizer.eos_token_id,
        "vocab_size": tokenizer.vocab_size,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-format", choices=["jsonl", "text"], default="jsonl")
    args = parser.parse_args()
    print(
        json.dumps(tokenize_file(args.model, args.input, args.output, args.input_format), indent=2)
    )


if __name__ == "__main__":
    main()
