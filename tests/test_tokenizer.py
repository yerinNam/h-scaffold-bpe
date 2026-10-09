from collections import Counter

import pytest

from hscaffold_bpe.model import TokenizerModel
from hscaffold_bpe.pretokenize import iter_pretokens
from hscaffold_bpe.tokenizer import HierarchicalTokenizer
from hscaffold_bpe.trainer import train_from_counts

CORPUS = (
    "Arizona zona Arizona data database mate mating. " * 40
    + "The quick brown fox jumps over the lazy dog. " * 20
)


def _counts() -> Counter[bytes]:
    return Counter(iter_pretokens(CORPUS))


@pytest.mark.parametrize("variant", ["bpe", "scaffold", "hierarchical"])
def test_roundtrip_and_visible_vocab(tmp_path, variant):
    model = train_from_counts(_counts(), 275, variant, show_progress=False)
    model_path = tmp_path / f"{variant}.json"
    model.save(model_path)
    restored = TokenizerModel.load(model_path)
    tokenizer = HierarchicalTokenizer(restored)

    text = "Arizona zona — café!\n"
    assert tokenizer.vocab_size == 275
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert all(0 <= token_id < 300 for token_id in tokenizer.encode(text))


def test_hierarchical_has_no_scaffolds_in_output():
    model = train_from_counts(_counts(), 275, "hierarchical", show_progress=False)
    tokenizer = HierarchicalTokenizer(model)
    internal = [tokenizer._model_to_internal[index] for index in tokenizer.encode(CORPUS)]
    assert all(model.levels[token_id] == 0 for token_id in internal)


def test_scaffold_expands_internal_vocabulary():
    model = train_from_counts(_counts(), 275, "scaffold", show_progress=False)
    assert len(model.tokens) >= model.target_vocab_size
    assert sum(level == 0 for level in model.levels.values()) == model.target_vocab_size


def test_multi_vocab_snapshots(tmp_path):
    paths = {270: tmp_path / "270.json", 275: tmp_path / "275.json"}
    final = train_from_counts(
        _counts(), 275, "hierarchical", show_progress=False, snapshot_paths=paths
    )
    assert final.target_vocab_size == 275
    for size, path in paths.items():
        assert TokenizerModel.load(path).model_vocab_size == size


def test_level2_threshold_roundtrips_and_rejects_invalid_values(tmp_path):
    model = train_from_counts(
        _counts(), 275, "hierarchical", level2_threshold=0.75, show_progress=False
    )
    path = tmp_path / "threshold.json"
    model.save(path)
    assert TokenizerModel.load(path).level2_threshold == 0.75
    with pytest.raises(ValueError, match="level2_threshold"):
        train_from_counts(
            _counts(), 275, "hierarchical", level2_threshold=1.0, show_progress=False
        )


@pytest.mark.parametrize("variant", ["bpe", "scaffold", "hierarchical"])
def test_tiktoken_backend_matches_reference(variant):
    model = train_from_counts(_counts(), 275, variant, show_progress=False)
    reference = HierarchicalTokenizer(model, backend="python")
    fast = HierarchicalTokenizer(model, backend="tiktoken")
    for text in (CORPUS, "Arizona — café.\n", "aaaaaa bbb abcabc"):
        assert fast.encode(text) == reference.encode(text)
