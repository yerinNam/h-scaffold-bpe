from __future__ import annotations

import pickle
import re as stdlib_re
from collections import Counter
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path

import regex

# The GPT-2 regex, shared by training and inference.
PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def iter_pretokens(text: str, special_tokens: Iterable[str] = ()) -> Iterator[bytes]:
    specials = sorted(set(special_tokens), key=len, reverse=True)
    if specials:
        splitter = stdlib_re.compile("(" + "|".join(map(stdlib_re.escape, specials)) + ")")
        special_set = set(specials)
        parts = splitter.split(text)
    else:
        special_set = set()
        parts = [text]
    for part in parts:
        if not part or part in special_set:
            continue
        for match in PATTERN.finditer(part):
            yield match.group().encode("utf-8")


def count_lines(lines: Iterable[str], special_tokens: Iterable[str] = ()) -> Counter[bytes]:
    result: Counter[bytes] = Counter()
    for line in lines:
        result.update(iter_pretokens(line, special_tokens))
    return result


def count_file(path: str | Path, special_tokens: Iterable[str] = ()) -> Counter[bytes]:
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        return count_lines(handle, special_tokens)


def _count_range(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with Path(path).open("rb") as handle:
        handle.seek(start)
        content = handle.read(end - start).decode("utf-8", errors="ignore")
    return Counter(iter_pretokens(content, special_tokens))


def count_file_parallel(
    path: str | Path, special_tokens: Iterable[str] = (), workers: int = 32
) -> Counter[bytes]:
    path = Path(path)
    if workers <= 1:
        return count_file(path, special_tokens)
    file_size = path.stat().st_size
    boundaries = [0]
    with path.open("rb") as handle:
        for index in range(1, workers):
            handle.seek(file_size * index // workers)
            handle.readline()
            boundaries.append(handle.tell())
    boundaries.append(file_size)
    tasks = [
        (str(path), start, end, tuple(special_tokens))
        for start, end in zip(boundaries, boundaries[1:])
    ]
    result: Counter[bytes] = Counter()
    with Pool(workers) as pool:
        for partial in pool.imap_unordered(_count_range, tasks):
            result.update(partial)
    return result


def save_counts(counts: Counter[bytes], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(counts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_counts(path: str | Path) -> Counter[bytes]:
    with Path(path).open("rb") as handle:
        counts = pickle.load(handle)
    if not isinstance(counts, Counter):
        counts = Counter(counts)
    return counts
