from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def aggregate(eval_dir: str | Path, lm_dir: str | Path, output_csv: str | Path) -> list[dict]:
    rows: dict[tuple[str, int], dict] = {}
    for path in Path(eval_dir).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload["variant"], int(payload["target_vocab_size"]))
        row = rows.setdefault(key, {"variant": key[0], "vocab_size": key[1]})
        row.update(
            {
                "expanded_vocab_size": payload["expanded_vocab_size"],
                "tokenizer_training_seconds": payload["tokenizer_training_seconds"],
                "fertility_mean": payload["fertility_mean"],
                "fertility_domain_std": payload["fertility_population_std"],
            }
        )
        for domain in payload["domains"]:
            name = Path(domain["path"]).stem
            row[f"fertility_{name}"] = domain["fertility"]
            row[f"tokens_per_second_{name}"] = domain["tokens_per_second"]

    for result_path in Path(lm_dir).glob("*/result.json"):
        run_name = result_path.parent.name
        variant, vocab_text = run_name.rsplit("-", 1)
        key = (variant, int(vocab_text))
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        row = rows.setdefault(key, {"variant": key[0], "vocab_size": key[1]})
        row.update(
            {
                "parameter_count": payload["parameter_count"],
                "lm_wall_seconds": payload["wall_seconds"],
                "validation_ppl": payload["validation"]["perplexity"],
                "validation_bpb": payload["validation"]["bits_per_byte"],
            }
        )

    ordered = [rows[key] for key in sorted(rows, key=lambda item: (item[1], item[0]))]
    fields = sorted({field for row in ordered for field in row})
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="artifacts/eval")
    parser.add_argument("--lm-dir", default="artifacts/lm")
    parser.add_argument("--output", default="artifacts/results.csv")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.eval_dir, args.lm_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
