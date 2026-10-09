from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Merge:
    left: int
    right: int
    output: int
    frequency: int


@dataclass(slots=True)
class TokenizerModel:
    variant: str
    target_vocab_size: int
    tokens: dict[int, bytes]
    merges: list[Merge]
    levels: dict[int, int]
    parents: dict[int, list[tuple[int, int]]]
    special_tokens: dict[str, int]
    training_seconds: float = 0.0
    corpus_pretokens: int = 0
    level2_threshold: float = 0.5

    @property
    def visible_ids(self) -> list[int]:
        return sorted(token_id for token_id, level in self.levels.items() if level == 0)

    @property
    def model_vocab_size(self) -> int:
        return len(self.visible_ids)

    def validate(self) -> None:
        if self.variant not in {"bpe", "scaffold", "hierarchical"}:
            raise ValueError(f"Unknown variant: {self.variant}")
        missing = set(self.tokens) - set(self.levels)
        if missing:
            raise ValueError(f"Tokens without a level: {sorted(missing)[:5]}")
        if self.model_vocab_size != self.target_vocab_size:
            raise ValueError(
                f"Visible vocabulary is {self.model_vocab_size}, expected {self.target_vocab_size}"
            )
        for merge in self.merges:
            if merge.left not in self.tokens or merge.right not in self.tokens:
                raise ValueError(f"Invalid merge parents: {merge}")
            if self.tokens[merge.left] + self.tokens[merge.right] != self.tokens[merge.output]:
                raise ValueError(f"Merge bytes do not concatenate: {merge}")

    def to_dict(self) -> dict:
        return {
            "format": "hierarchical-scaffold-bpe-v1",
            "variant": self.variant,
            "target_vocab_size": self.target_vocab_size,
            "tokens": {
                str(i): base64.b64encode(value).decode("ascii") for i, value in self.tokens.items()
            },
            "merges": [
                [merge.left, merge.right, merge.output, merge.frequency] for merge in self.merges
            ],
            "levels": {str(i): level for i, level in self.levels.items()},
            "parents": {
                str(i): [[left, right] for left, right in alternatives]
                for i, alternatives in self.parents.items()
            },
            "special_tokens": self.special_tokens,
            "metadata": {
                "training_seconds": self.training_seconds,
                "corpus_pretokens": self.corpus_pretokens,
                "expanded_vocab_size": len(self.tokens),
                "merge_count": len(self.merges),
                "level2_threshold": self.level2_threshold,
                "level_counts": {
                    str(level): sum(value == level for value in self.levels.values())
                    for level in (0, 1, 2)
                },
            },
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: dict) -> TokenizerModel:
        if payload.get("format") != "hierarchical-scaffold-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        metadata = payload.get("metadata", {})
        return cls(
            variant=payload["variant"],
            target_vocab_size=int(payload["target_vocab_size"]),
            tokens={
                int(i): base64.b64decode(value.encode("ascii"))
                for i, value in payload["tokens"].items()
            },
            merges=[Merge(*map(int, row)) for row in payload["merges"]],
            levels={int(i): int(level) for i, level in payload["levels"].items()},
            parents={
                int(i): [tuple(map(int, pair)) for pair in alternatives]
                for i, alternatives in payload["parents"].items()
            },
            special_tokens={key: int(value) for key, value in payload["special_tokens"].items()},
            training_seconds=float(metadata.get("training_seconds", 0.0)),
            corpus_pretokens=int(metadata.get("corpus_pretokens", 0)),
            level2_threshold=float(metadata.get("level2_threshold", 0.5)),
        )

    @classmethod
    def load(cls, path: str | Path) -> TokenizerModel:
        with Path(path).open(encoding="utf-8") as handle:
            model = cls.from_dict(json.load(handle))
        model.validate()
        return model
