from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate_tokenizer
from .pretokenize import count_file_parallel, load_counts, save_counts
from .trainer import train_from_counts


def _count(args: argparse.Namespace) -> None:
    special_tokens = args.special_token or ["<|endoftext|>"]
    counts = count_file_parallel(args.input, special_tokens, args.workers)
    save_counts(counts, args.output)
    print(json.dumps({"unique_pretokens": len(counts), "total_pretokens": sum(counts.values())}))


def _train(args: argparse.Namespace) -> None:
    counts = load_counts(args.counts)
    special_tokens = args.special_token or ["<|endoftext|>"]
    model = train_from_counts(
        counts,
        target_vocab_size=args.vocab_size,
        variant=args.variant,
        special_tokens=special_tokens,
        level2_threshold=args.level2_threshold,
    )
    model.save(args.output)
    print(json.dumps(model.to_dict()["metadata"], indent=2))


def _train_all(args: argparse.Namespace) -> None:
    counts = load_counts(args.counts)
    special_tokens = args.special_token or ["<|endoftext|>"]
    output_dir = Path(args.output_dir)
    snapshot_paths = {
        vocab_size: output_dir / f"{args.variant}-{vocab_size}.json"
        for vocab_size in args.vocab_sizes
    }
    model = train_from_counts(
        counts,
        target_vocab_size=max(args.vocab_sizes),
        variant=args.variant,
        special_tokens=special_tokens,
        level2_threshold=args.level2_threshold,
        snapshot_paths=snapshot_paths,
    )
    print(json.dumps(model.to_dict()["metadata"], indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    result = evaluate_tokenizer(args.model, args.inputs, warmup=args.warmup)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hscaffold")
    subparsers = parser.add_subparsers(required=True)

    count = subparsers.add_parser("count", help="pre-tokenize a corpus into frequency counts")
    count.add_argument("input")
    count.add_argument("output")
    count.add_argument("--special-token", action="append", default=[])
    count.add_argument("--workers", type=int, default=32)
    count.set_defaults(func=_count)

    train = subparsers.add_parser("train", help="train one tokenizer")
    train.add_argument("counts")
    train.add_argument("output")
    train.add_argument("--variant", choices=["bpe", "scaffold", "hierarchical"], required=True)
    train.add_argument("--vocab-size", type=int, choices=[32768, 65536], required=True)
    train.add_argument("--special-token", action="append", default=[])
    train.add_argument("--level2-threshold", type=float, default=0.5)
    train.set_defaults(func=_train)

    train_all = subparsers.add_parser(
        "train-all", help="train once and snapshot multiple vocabulary sizes"
    )
    train_all.add_argument("counts")
    train_all.add_argument("output_dir")
    train_all.add_argument("--variant", choices=["bpe", "scaffold", "hierarchical"], required=True)
    train_all.add_argument("--vocab-sizes", type=int, nargs="+", default=[32768, 65536])
    train_all.add_argument("--special-token", action="append", default=[])
    train_all.add_argument("--level2-threshold", type=float, default=0.5)
    train_all.set_defaults(func=_train_all)

    evaluate = subparsers.add_parser("evaluate", help="measure fertility and throughput")
    evaluate.add_argument("model")
    evaluate.add_argument("inputs", nargs="+")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--warmup", type=int, default=100)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
