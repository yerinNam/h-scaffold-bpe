from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from tqdm import tqdm

from .model import Merge, TokenizerModel

Pair = tuple[int, int]


def _initial_state(
    counts: Mapping[bytes, int], special_tokens: list[str]
) -> tuple[
    list[tuple[int, ...]],
    list[int],
    dict[int, bytes],
    dict[bytes, int],
    Counter[Pair],
    dict[Pair, set[int]],
    Counter[int],
]:
    tokens = {value: bytes([value]) for value in range(256)}
    special_map: dict[bytes, int] = {}
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded in special_map:
            continue
        token_id = len(tokens)
        tokens[token_id] = encoded
        special_map[encoded] = token_id

    words = list(counts)
    word_vocab = [tuple(word) for word in words]
    word_frequencies = [counts[word] for word in words]
    pair_counts: Counter[Pair] = Counter()
    pair_to_words: dict[Pair, set[int]] = defaultdict(set)
    token_counts: Counter[int] = Counter()
    for word_id, (sequence, frequency) in enumerate(zip(word_vocab, word_frequencies)):
        token_counts.update(
            {token_id: frequency * amount for token_id, amount in Counter(sequence).items()}
        )
        word_pairs = Counter(zip(sequence, sequence[1:]))
        for pair, amount in word_pairs.items():
            pair_counts[pair] += frequency * amount
            pair_to_words[pair].add(word_id)
    return (
        word_vocab,
        word_frequencies,
        tokens,
        special_map,
        pair_counts,
        pair_to_words,
        token_counts,
    )


def _pop_valid(
    heap: list[tuple[int, int, int, int]],
    pair_counts: Counter[Pair],
    token_counts: Counter[int],
    levels: dict[int, int],
) -> tuple[str, int | Pair, int]:
    while heap:
        negative_frequency, kind, left, right = heapq.heappop(heap)
        frequency = -negative_frequency
        if kind == 0:
            if frequency > 0 and levels[left] > 0 and token_counts[left] == frequency:
                return "token", left, frequency
        elif frequency > 0 and pair_counts[(left, right)] == frequency:
            return "pair", (left, right), frequency
    raise RuntimeError("No mergeable pair remains before reaching target vocabulary size")


def _peek_valid(
    heap: list[tuple[int, int, int, int]],
    pair_counts: Counter[Pair],
    token_counts: Counter[int],
    levels: dict[int, int],
) -> int:
    while heap:
        negative_frequency, kind, left, right = heap[0]
        frequency = -negative_frequency
        if kind == 0:
            valid = frequency > 0 and levels[left] > 0 and token_counts[left] == frequency
        else:
            valid = frequency > 0 and pair_counts[(left, right)] == frequency
        if valid:
            return frequency
        heapq.heappop(heap)
    return 0


def _merge_sequence(
    sequence: tuple[int, ...], pair: Pair, output: int
) -> tuple[tuple[int, ...], int]:
    left, right = pair
    merged: list[int] = []
    occurrences = 0
    index = 0
    while index < len(sequence):
        if index + 1 < len(sequence) and sequence[index] == left and sequence[index + 1] == right:
            merged.append(output)
            occurrences += 1
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return tuple(merged), occurrences


def train_from_counts(
    counts: Mapping[bytes, int],
    target_vocab_size: int,
    variant: str,
    special_tokens: list[str] | None = None,
    show_progress: bool = True,
    snapshot_paths: Mapping[int, str | Path] | None = None,
    level2_threshold: float = 0.5,
) -> TokenizerModel:
    """Train BPE, Scaffold-BPE, or Hierarchical Scaffold-BPE.

    Scaffold classification follows Algorithm 1 of Lian et al. For the hierarchical
    variant, a visible child becomes level 1 when ``0.5 * f(Qhead) <= f(child) <
    f(Qhead)`` and level 2 below that boundary. Base bytes and special tokens are
    pinned at level 0 so the tokenizer always has lossless coverage.
    """
    if variant not in {"bpe", "scaffold", "hierarchical"}:
        raise ValueError("variant must be one of: bpe, scaffold, hierarchical")
    if not 0 < level2_threshold < 1:
        raise ValueError("level2_threshold must be between 0 and 1")
    special_tokens = special_tokens or ["<|endoftext|>"]
    if target_vocab_size <= 256 + len(special_tokens):
        raise ValueError("target_vocab_size must exceed the base and special vocabulary")

    started = time.perf_counter()
    (
        word_vocab,
        word_frequencies,
        tokens,
        special_map_bytes,
        pair_counts,
        pair_to_words,
        token_counts,
    ) = _initial_state(counts, special_tokens)
    bytes_to_id = {value: token_id for token_id, value in tokens.items()}
    special_map = {value.decode("utf-8"): token_id for value, token_id in special_map_bytes.items()}
    pinned = set(range(256)) | set(special_map.values())
    levels = {token_id: 0 for token_id in tokens}
    parents: dict[int, list[Pair]] = defaultdict(list)
    merges: list[Merge] = []
    visible_count = len(tokens)
    # Heap item: (-frequency, kind, left/token, right). kind=0 is a scaffold
    # token awaiting possible reactivation; kind=1 is a pair merge candidate.
    heap = [(-frequency, 1, left, right) for (left, right), frequency in pair_counts.items()]
    heapq.heapify(heap)
    progress = tqdm(
        total=target_vocab_size,
        initial=visible_count,
        desc=f"train:{variant}:{target_vocab_size}",
        disable=not show_progress,
    )
    pending_snapshots = dict(snapshot_paths or {})

    def save_snapshot_if_requested() -> None:
        destination = pending_snapshots.pop(visible_count, None)
        if destination is None:
            return
        snapshot = TokenizerModel(
            variant=variant,
            target_vocab_size=visible_count,
            tokens=tokens.copy(),
            merges=merges.copy(),
            levels=levels.copy(),
            parents={token_id: values.copy() for token_id, values in parents.items()},
            special_tokens=special_map.copy(),
            training_seconds=time.perf_counter() - started,
            corpus_pretokens=sum(counts.values()),
            level2_threshold=level2_threshold,
        )
        snapshot.validate()
        snapshot.save(destination)

    while visible_count < target_vocab_size:
        item_kind, item, selected_frequency = _pop_valid(heap, pair_counts, token_counts, levels)
        if item_kind == "token":
            token_id = int(item)
            levels[token_id] = 0
            visible_count += 1
            progress.update(visible_count - progress.n)
            save_snapshot_if_requested()
            continue
        pair = item
        assert isinstance(pair, tuple)
        left, right = pair
        output_bytes = tokens[left] + tokens[right]
        output = bytes_to_id.get(output_bytes)
        if output is None:
            output = len(tokens)
            tokens[output] = output_bytes
            bytes_to_id[output_bytes] = output
            levels[output] = 0
            visible_count += 1
        elif levels[output] != 0:
            levels[output] = 0
            visible_count += 1
            progress.update(visible_count - progress.n)
            save_snapshot_if_requested()
            continue
        parents[output].append(pair)
        merges.append(Merge(left, right, output, selected_frequency))

        total_occurrences = 0
        globally_changed_pairs: set[Pair] = set()
        affected_words = tuple(pair_to_words.get(pair, ()))
        for word_id in affected_words:
            old_sequence = word_vocab[word_id]
            new_sequence, occurrences = _merge_sequence(old_sequence, pair, output)
            if not occurrences:
                continue
            frequency = word_frequencies[word_id]
            total_occurrences += occurrences * frequency

            old_pairs = Counter(zip(old_sequence, old_sequence[1:]))
            new_pairs = Counter(zip(new_sequence, new_sequence[1:]))
            changed_pairs = set(old_pairs) | set(new_pairs)
            for changed_pair in changed_pairs:
                delta = (new_pairs[changed_pair] - old_pairs[changed_pair]) * frequency
                if delta:
                    pair_counts[changed_pair] += delta
                    globally_changed_pairs.add(changed_pair)
            for old_pair in old_pairs:
                pair_to_words[old_pair].discard(word_id)
            for new_pair in new_pairs:
                pair_to_words[new_pair].add(word_id)
            word_vocab[word_id] = new_sequence

        for changed_pair in globally_changed_pairs:
            current = pair_counts[changed_pair]
            if current > 0:
                heapq.heappush(heap, (-current, 1, *changed_pair))

        if total_occurrences <= 0:
            pair_counts[pair] = 0
            continue
        token_counts[left] -= total_occurrences
        token_counts[right] -= total_occurrences
        if left == right:
            # Both decrements above target the same Counter entry, as required.
            pass
        token_counts[output] += total_occurrences

        if variant != "bpe":
            for child in {left, right} - pinned:
                if levels[child] > 0 and token_counts[child] > 0:
                    heapq.heappush(heap, (-token_counts[child], 0, child, -1))
            qhead_frequency = _peek_valid(heap, pair_counts, token_counts, levels)
            for child in {left, right} - pinned:
                if levels[child] == 0 and token_counts[child] < qhead_frequency:
                    if (
                        variant == "hierarchical"
                        and token_counts[child] < level2_threshold * qhead_frequency
                    ):
                        levels[child] = 2
                    else:
                        levels[child] = 1
                    visible_count -= 1
                    if token_counts[child] > 0:
                        heapq.heappush(heap, (-token_counts[child], 0, child, -1))
                elif variant == "hierarchical" and levels[child] == 1:
                    if token_counts[child] < level2_threshold * qhead_frequency:
                        levels[child] = 2

        progress.update(visible_count - progress.n)
        save_snapshot_if_requested()

    progress.close()
    model = TokenizerModel(
        variant=variant,
        target_vocab_size=target_vocab_size,
        tokens=tokens,
        merges=merges,
        levels=levels,
        parents=dict(parents),
        special_tokens=special_map,
        training_seconds=time.perf_counter() - started,
        corpus_pretokens=sum(counts.values()),
        level2_threshold=level2_threshold,
    )
    model.validate()
    if pending_snapshots:
        raise ValueError(f"Snapshot sizes were not reached: {sorted(pending_snapshots)}")
    return model
