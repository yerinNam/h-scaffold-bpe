"""Regression test for the label-shift bug (see scripts/demo_label_shift.py)."""
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from hscaffold_bpe.lm import next_token_loss


def _tiny(vocab=50):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


def test_loss_is_aligned_next_token_cross_entropy():
    model, x = _tiny(), torch.randint(0, 50, (4, 17))
    with torch.no_grad():
        logits = model(input_ids=x[:, :-1]).logits
        expected = F.cross_entropy(logits.reshape(-1, 50), x[:, 1:].reshape(-1))      # logits[i] vs x[i+1]
        assert torch.allclose(next_token_loss(model, x), expected, atol=1e-6)
        total = next_token_loss(model, x, reduction="sum")
        assert torch.allclose(total, expected * x[:, 1:].numel(), atol=1e-4)


def test_pre_shifted_labels_in_hf_loss_would_be_wrong():
    """Documents WHY the helper exists: HF shifts labels again, giving the two-steps-ahead loss."""
    model, x = _tiny(), torch.randint(0, 50, (4, 17))
    with torch.no_grad():
        hf = model(input_ids=x[:, :-1], labels=x[:, 1:]).loss
        two_ahead = F.cross_entropy(model(input_ids=x[:, :-1]).logits[:, :-1].reshape(-1, 50), x[:, 2:].reshape(-1))
        assert torch.allclose(hf, two_ahead, atol=1e-5)
        assert not torch.allclose(hf, next_token_loss(model, x), atol=1e-6)
