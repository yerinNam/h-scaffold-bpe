from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from .lm import TokenStream, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a finished LM on a token binary")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--test-bin", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--attention-implementation", default="kernels-community/flash-attn")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    final_dir = run_dir / "final"
    if not final_dir.is_dir():
        raise FileNotFoundError(f"Missing final checkpoint: {final_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LM evaluation")

    started = time.perf_counter()
    model = LlamaForCausalLM.from_pretrained(
        final_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attention_implementation,
    ).to("cuda")
    stream = TokenStream(args.test_bin, args.sequence_length)
    metrics = evaluate(model, stream, args.batch_size, torch.device("cuda"))
    payload = {
        "run": run_dir.name,
        "test_binary": args.test_bin,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "attention_implementation": args.attention_implementation,
        "precision": "bf16",
        "wall_seconds": time.perf_counter() - started,
        "test": metrics,
    }
    destination = run_dir / "test_result.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
