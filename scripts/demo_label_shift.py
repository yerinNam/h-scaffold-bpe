#!/usr/bin/env python
"""Controlled demo: what does `model(input_ids=x[:, :-1], labels=x[:, 1:])` train?

HuggingFace's LlamaForCausalLM shifts `labels` internally (logits[i] is scored against labels[i+1]).
Passing already-shifted labels therefore trains the model to predict the token TWO steps ahead.

Data: a first-order Markov chain with a known entropy rate, so every number has a ground truth.
Both models are trained identically (same init seed, data, steps) and evaluated with the SAME aligned
next-token cross-entropy that `hscaffold_bpe.lm.evaluate` uses.
"""
import math
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

V, L, STEPS, BATCH = 128, 64, 600, 32


def make_chain(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(V, V, generator=g) * 3.0          # peaked rows: strong next-token dependence
    return F.softmax(logits, dim=-1)


def sample(P, n_seq, length, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_seq, length, dtype=torch.long)
    x[:, 0] = torch.randint(0, V, (n_seq,), generator=g)
    for t in range(1, length):
        x[:, t] = torch.multinomial(P[x[:, t - 1]], 1, generator=g).squeeze(-1)
    return x


def train(variant, data, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=V, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=L + 1, tie_word_embeddings=True)
    m = LlamaForCausalLM(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    for _ in range(STEPS):
        b = data[torch.randint(0, len(data), (BATCH,), generator=g)]
        inputs, labels = b[:, :-1], b[:, 1:]
        if variant == "as_in_lm.py":                       # model(input_ids=inputs, labels=labels).loss
            loss = m(input_ids=inputs, labels=labels).loss
        else:                                              # fixed: explicit aligned cross-entropy
            logits = m(input_ids=inputs).logits
            loss = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def aligned_nll(m, x):                                    # identical to hscaffold_bpe.lm.evaluate
    inputs, labels = x[:, :-1], x[:, 1:]
    logits = m(input_ids=inputs).logits
    return float(F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1)))


if __name__ == "__main__":
    P = make_chain()
    # stationary distribution -> unigram entropy; conditional entropy rate -> best achievable next-token NLL
    pi = torch.full((V,), 1.0 / V)
    for _ in range(200):
        pi = pi @ P
    h_uni = float(-(pi * pi.clamp_min(1e-12).log()).sum())
    h_cond = float(-(pi[:, None] * P * P.clamp_min(1e-12).log()).sum())
    train_x, test_x = sample(P, 4000, L + 1, 10), sample(P, 400, L + 1, 20)
    print(f"ground truth (nats/token): unigram entropy = {h_uni:.3f} (PPL {math.exp(h_uni):.1f}), "
          f"true next-token entropy = {h_cond:.3f} (PPL {math.exp(h_cond):.1f})")
    for variant in ("as_in_lm.py", "fixed"):
        nll = aligned_nll(train(variant, train_x), test_x)
        print(f"{variant:12s}: aligned next-token test NLL = {nll:.3f} nats (PPL {math.exp(nll):8.1f})")
