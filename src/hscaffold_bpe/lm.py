from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import LlamaConfig, LlamaForCausalLM, get_cosine_schedule_with_warmup


def llama31_8b_config(vocab_size: int, sequence_length: int, eos_token_id: int) -> LlamaConfig:
    """Llama 3.1 8B architecture with experiment-specific vocabulary/context."""
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        rope_theta=500_000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )


def llama32_1b_config(vocab_size: int, sequence_length: int, eos_token_id: int) -> LlamaConfig:
    """Llama 3.2 1B architecture with experiment-specific vocabulary/context."""
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=131_072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=False,
        rope_theta=500_000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=eos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )


def model_config(
    architecture: str, vocab_size: int, sequence_length: int, eos_token_id: int
) -> LlamaConfig:
    if architecture == "llama-3.2-1b":
        return llama32_1b_config(vocab_size, sequence_length, eos_token_id)
    if architecture == "llama-3.1-8b":
        return llama31_8b_config(vocab_size, sequence_length, eos_token_id)
    raise ValueError(f"Unsupported architecture: {architecture}")


def _metadata(binary_path: str | Path) -> dict:
    path = Path(binary_path)
    return json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))


def next_token_loss(
    model: LlamaForCausalLM, batch: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Cross-entropy of predicting batch[:, t+1] from batch[:, :t+1], for a (B, L+1) batch.

    Do NOT call ``model(input_ids=batch[:, :-1], labels=batch[:, 1:])``: HuggingFace's causal-LM
    loss shifts ``labels`` internally, so already-shifted labels train the model to predict the token
    TWO steps ahead (it then looks "untrained" under a correct next-token evaluation).
    The loss is computed explicitly here so that training and `evaluate` use the same alignment.
    """
    inputs, labels = batch[:, :-1], batch[:, 1:]
    logits = model(input_ids=inputs).logits
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction=reduction
    )


class TokenStream:
    def __init__(self, path: str | Path, sequence_length: int):
        self.path = Path(path)
        self.meta = _metadata(path)
        self.data = np.memmap(self.path, mode="r", dtype=self.meta["dtype"])
        self.sequence_length = sequence_length

    @property
    def num_tokens(self) -> int:
        return len(self.data)

    def batch(self, first_sequence: int, batch_size: int, device: torch.device) -> torch.Tensor:
        length = self.sequence_length + 1
        sequences = []
        max_start = len(self.data) - length
        for index in range(batch_size):
            start = ((first_sequence + index) * self.sequence_length) % max_start
            sequences.append(np.asarray(self.data[start : start + length], dtype=np.int64))
        array = np.stack(sequences)
        return torch.from_numpy(array).pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model: LlamaForCausalLM,
    stream: TokenStream,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    total_nll = 0.0
    total_targets = 0
    batch_count = (stream.num_tokens - 1) // (batch_size * stream.sequence_length)
    if max_batches is not None:
        batch_count = min(batch_count, max_batches)
    for batch_index in range(batch_count):
        batch = stream.batch(batch_index * batch_size, batch_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = next_token_loss(model, batch, reduction="sum")
        total_nll += float(loss)
        total_targets += batch[:, 1:].numel()
    nll_per_token = total_nll / max(total_targets, 1)
    represented_bytes = stream.meta["source_bytes"] * total_targets / max(stream.meta["tokens"], 1)
    return {
        "nll_per_token": nll_per_token,
        "perplexity": math.exp(min(nll_per_token, 30.0)),
        "bits_per_byte": total_nll / max(represented_bytes * math.log(2.0), 1.0),
        "evaluated_tokens": total_targets,
        "estimated_evaluated_bytes": represented_bytes,
    }


def train(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the language-model experiment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train_stream = TokenStream(args.train_bin, args.sequence_length)
    validation_stream = TokenStream(args.validation_bin, args.sequence_length)
    metadata = train_stream.meta
    config = model_config(
        args.architecture,
        metadata["vocab_size"],
        args.sequence_length,
        metadata["eos_token_id"],
    )
    config._attn_implementation = args.attention_implementation

    # FP32 parameters and optimizer states, BF16 tensor-core compute. This is more
    # stable for scratch pretraining. The 8-bit optimizer keeps the FP32 master
    # parameters while making a GA=1 batch practical on a B200.
    model = LlamaForCausalLM(config).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.optimizer == "adamw_bnb_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    sequences_per_epoch = (train_stream.num_tokens - 1) // args.sequence_length
    optimizer_steps_per_epoch = sequences_per_epoch // (args.micro_batch_size * args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_fraction)),
        num_training_steps=total_steps,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    consumed_sequences = 0
    started = time.perf_counter()
    model.train()
    with log_path.open("a", encoding="utf-8") as log:
        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            step_started = time.perf_counter()
            for _ in range(args.grad_accum):
                batch = train_stream.batch(consumed_sequences, args.micro_batch_size, device)
                consumed_sequences += args.micro_batch_size
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = next_token_loss(model, batch) / args.grad_accum
                loss.backward()
                accumulated_loss += float(loss.detach())
            gradient_norm = float(clip_grad_norm_(model.parameters(), args.max_grad_norm))
            optimizer.step()
            scheduler.step()
            elapsed = time.perf_counter() - step_started
            record = {
                "step": step + 1,
                "total_steps": total_steps,
                "loss": accumulated_loss,
                "gradient_norm": gradient_norm,
                "learning_rate": scheduler.get_last_lr()[0],
                "tokens_per_second": (
                    args.micro_batch_size * args.grad_accum * args.sequence_length / elapsed
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if (step + 1) % args.log_every == 0:
                print(json.dumps(record), flush=True)
            if args.max_steps and step + 1 >= args.max_steps:
                break

    validation = evaluate(
        model, validation_stream, args.eval_batch_size, device, args.max_eval_batches
    )
    model.save_pretrained(output / "final", safe_serialization=True, max_shard_size="10GB")
    result = {
        "architecture": args.architecture,
        "train_binary": args.train_bin,
        "validation_binary": args.validation_bin,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "epochs_requested": args.epochs,
        "steps_planned": total_steps,
        "steps_completed": min(total_steps, args.max_steps or total_steps),
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation": args.grad_accum,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_implementation": args.attention_implementation,
        "optimizer": args.optimizer,
        "sequence_length": args.sequence_length,
        "precision": "bf16 compute / fp32 master parameters and optimizer",
        "wall_seconds": time.perf_counter() - started,
        "validation": validation,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--validation-bin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", choices=["llama-3.2-1b", "llama-3.1-8b"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attention-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "kernels-community/flash-attn"],
    )
    parser.add_argument(
        "--optimizer", default="adamw", choices=["adamw", "adamw_bnb_8bit"]
    )
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means the full run")
    parser.add_argument("--max-eval-batches", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
