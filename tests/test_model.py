"""
Model tests.

    python tests/test_model.py
    pytest tests/test_model.py

The causality test is the important one. Everything else here fails loudly. A
broken causal mask fails quietly and in the flattering direction: the model peeks
at the answer, training loss drops faster than it should, and the curves look
like success. It is the one bug in this file that could survive all the way into
a submitted report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import GPTConfig, get_config, PRESETS  # noqa: E402
from model.gpt import GPTModel, LayerNorm, GELU, CausalSelfAttention  # noqa: E402
from model.generate import generate  # noqa: E402
from data.tokenizer import get_tokenizer, fertility  # noqa: E402
from data.dataset import TokenWindowDataset, split_documents, encode_documents  # noqa: E402

torch.manual_seed(0)

CFG = GPTConfig(vocab_size=128, context_length=32, emb_dim=48, n_heads=4,
                n_layers=2, drop_rate=0.0)


def _model(cfg=CFG):
    m = GPTModel(cfg)
    m.eval()  # dropout off, otherwise every determinism check is noise
    return m


# ─── the one that matters ────────────────────────────────────────────────────

def test_attention_is_strictly_causal():
    """
    Change the token at position t. Every output at positions < t has to be
    bit-identical, because it is not allowed to see t at all.

    I test the property directly instead of inspecting the mask, so this also
    catches an off-by-one in `diagonal=`, a mask applied after softmax, or a
    fast-path that forgets is_causal.
    """
    for fast in (False, True):
        cfg = GPTConfig(**{**CFG.to_dict(), "use_fast_attention": fast})
        model = _model(cfg)
        n = cfg.context_length
        idx = torch.randint(0, cfg.vocab_size, (1, n))

        with torch.no_grad():
            base, _ = model(idx)

        for t in (1, n // 2, n - 1):
            poked = idx.clone()
            # Guarantee an actual change.
            poked[0, t] = (poked[0, t] + 1) % cfg.vocab_size
            with torch.no_grad():
                out, _ = model(poked)

            before = out[:, :t, :] - base[:, :t, :]
            assert torch.equal(out[:, :t, :], base[:, :t, :]), (
                f"fast={fast}: changing token {t} altered outputs at earlier "
                f"positions by up to {before.abs().max():.3e}. The model can see "
                "the future and its loss curve is a lie"
            )
            # And the position itself must change, or the mask is masking everything.
            assert not torch.equal(out[:, t, :], base[:, t, :]), (
                f"fast={fast}: changing token {t} did not change its own output"
            )


def test_fast_attention_matches_explicit():
    """
    The fused SDPA path has to be numerically equivalent to the explicit one.
    Otherwise the explicit implementation is documentation for code that never
    runs.
    """
    torch.manual_seed(7)
    slow_cfg = GPTConfig(**{**CFG.to_dict(), "use_fast_attention": False})
    fast_cfg = GPTConfig(**{**CFG.to_dict(), "use_fast_attention": True})

    slow, fast = _model(slow_cfg), _model(fast_cfg)
    fast.load_state_dict(slow.state_dict())

    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_length))
    with torch.no_grad():
        a, _ = slow(idx)
        b, _ = fast(idx)
    assert torch.allclose(a, b, atol=1e-5), f"max diff {(a - b).abs().max():.3e}"


# ─── shapes and wiring ───────────────────────────────────────────────────────

def test_forward_shapes_and_loss():
    model = _model()
    b, n = 3, 16
    idx = torch.randint(0, CFG.vocab_size, (b, n))
    logits, loss = model(idx, idx)
    assert logits.shape == (b, n, CFG.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_loss_at_init_is_near_uniform():
    """
    An untrained model should be maximally uncertain: loss ≈ ln(vocab_size).
    A materially lower value at step 0 means something is leaking: bad init, or
    targets misaligned with inputs.
    """
    import math
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (8, CFG.context_length))
    _, loss = model(idx, idx)
    expected = math.log(CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.7, f"loss {loss.item():.3f} vs ln(V) {expected:.3f}"


def test_rejects_sequence_longer_than_context():
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.context_length + 1))
    try:
        model(idx)
    except ValueError as e:
        assert "context_length" in str(e)
    else:
        raise AssertionError("accepted a sequence longer than the context window")


def test_parameter_count_derived_matches_measured():
    """GPTConfig.n_params() is used to plan runs before building the model."""
    for name in PRESETS:
        cfg = get_config(name, vocab_size=512)
        model = GPTModel(cfg)
        assert model.num_parameters() == cfg.n_params(), (
            f"{name}: measured {model.num_parameters()} != derived {cfg.n_params()}"
        )


def test_weight_tying_shares_storage():
    tied = GPTModel(GPTConfig(**{**CFG.to_dict(), "weight_tying": True}))
    assert tied.out_head.weight.data_ptr() == tied.tok_emb.weight.data_ptr()

    untied = GPTModel(GPTConfig(**{**CFG.to_dict(), "weight_tying": False}))
    assert untied.out_head.weight.data_ptr() != untied.tok_emb.weight.data_ptr()
    assert untied.num_parameters() > tied.num_parameters()


def test_config_rejects_indivisible_head_split():
    try:
        GPTConfig(emb_dim=100, n_heads=7)
    except ValueError:
        return
    raise AssertionError("accepted emb_dim not divisible by n_heads")


def test_optimizer_excludes_1d_params_from_decay():
    model = _model()
    opt = model.configure_optimizer(lr=1e-3, weight_decay=0.1)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


# ─── components ──────────────────────────────────────────────────────────────

def test_layernorm_normalises():
    ln = LayerNorm(16)
    x = torch.randn(4, 7, 16) * 5 + 3
    out = ln(x)
    assert out.mean(dim=-1).abs().max() < 1e-5
    assert (out.std(dim=-1, unbiased=False) - 1).abs().max() < 1e-3


def test_gelu_shape_and_sign():
    g = GELU()
    x = torch.linspace(-3, 3, 50)
    y = g(x)
    assert y.shape == x.shape
    assert y[x > 2].min() > 0            # ~identity for large positive
    assert (y[x < -2] < 0).all()         # small negative, not clipped to zero
    assert abs(g(torch.tensor([0.0])).item()) < 1e-6


def test_attention_output_shape():
    attn = CausalSelfAttention(CFG)
    x = torch.randn(2, CFG.context_length, CFG.emb_dim)
    assert attn(x).shape == x.shape


# ─── generation ──────────────────────────────────────────────────────────────

def test_generate_appends_requested_tokens():
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (2, 5))
    out = generate(model, idx, max_new_tokens=7, temperature=1.0)
    assert out.shape == (2, 12)
    assert torch.equal(out[:, :5], idx), "prompt was modified"


def test_generate_crops_to_context_window():
    """Generating past the context window must slide, not index out of bounds."""
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.context_length))
    out = generate(model, idx, max_new_tokens=5, temperature=0.8, top_k=5)
    assert out.shape[1] == CFG.context_length + 5


def test_greedy_decoding_is_deterministic():
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    a = generate(model, idx, 10, temperature=0.0)
    b = generate(model, idx, 10, temperature=0.0)
    assert torch.equal(a, b)


def test_top_k_restricts_support():
    """With k=1, sampling must collapse onto the argmax."""
    model = _model()
    idx = torch.randint(0, CFG.vocab_size, (1, 4))
    greedy = generate(model, idx, 6, temperature=0.0)
    topk1 = generate(model, idx, 6, temperature=1.0, top_k=1)
    assert torch.equal(greedy, topk1)


def test_generate_restores_training_mode():
    model = _model()
    model.train()
    generate(model, torch.randint(0, CFG.vocab_size, (1, 3)), 3)
    assert model.training, "generate() left the model in eval mode"


# ─── tokenizer and dataset ───────────────────────────────────────────────────

def test_byte_tokenizer_roundtrips_swedish():
    tok = get_tokenizer("bytes")
    for text in ["Tvättställsblandare Nordic", "handdukstork i borstad mässing",
                 "Golvbrunn 110 mm — 649 kr"]:
        assert tok.decode(tok.encode(text)) == text


def test_fertility_reports_expected_fields():
    tok = get_tokenizer("bytes")
    f = fertility(tok, ["Tvättställsblandare Nordic finns i krom."])
    assert f["tokens"] > f["words"]
    # å/ä/ö are two UTF-8 bytes, so a Swedish sentence must exceed 1 token/char.
    assert f["tokens_per_char"] > 1.0


def test_dataset_targets_are_inputs_shifted_by_one():
    ids = list(range(100))
    ds = TokenWindowDataset(ids, context_length=8, stride=8)
    x, y = ds[0]
    assert torch.equal(y[:-1], x[1:]), "targets are not next-token"
    assert x.tolist() == list(range(8))
    assert y.tolist() == list(range(1, 9))


def test_dataset_stride_controls_overlap():
    ids = list(range(100))
    assert len(TokenWindowDataset(ids, 10, stride=10)) < len(
        TokenWindowDataset(ids, 10, stride=5))


def test_dataset_refuses_corpus_smaller_than_context():
    try:
        TokenWindowDataset(list(range(5)), context_length=32)
    except ValueError as e:
        assert "at least" in str(e)
    else:
        raise AssertionError("accepted a corpus too small for one window")


def test_split_is_by_document_not_token_position():
    """
    Splitting a flat token stream leaks the first half of a document into train
    and the second half into val, flattering the validation loss.
    """
    docs = [f"dokument nummer {i}" for i in range(50)]
    train, val = split_documents(docs, val_fraction=0.2, seed=0)
    assert len(val) == 10 and len(train) == 40
    assert not (set(train) & set(val)), "a document appears in both splits"
    assert set(train) | set(val) == set(docs), "documents were lost in the split"


def test_encode_documents_separates_with_eot():
    tok = get_tokenizer("bytes")
    stream = encode_documents(["ab", "cd"], tok)
    assert stream.count(tok.eot_id) == 2
    assert stream[-1] == tok.eot_id


# ─── the loop actually learns ────────────────────────────────────────────────

def test_model_can_overfit_a_single_batch():
    """
    The end-to-end check that the whole training path works: forward, loss,
    backward, step. A model that cannot drive loss toward zero on a single batch
    it sees repeatedly has a wiring bug, and more data or compute will not fix it.
    """
    cfg = GPTConfig(vocab_size=64, context_length=16, emb_dim=64, n_heads=4,
                    n_layers=2, drop_rate=0.0)
    torch.manual_seed(0)
    model = GPTModel(cfg)
    model.train()
    opt = model.configure_optimizer(lr=3e-3, weight_decay=0.0)

    x = torch.randint(0, cfg.vocab_size, (4, cfg.context_length))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.context_length))

    _, first = model(x, y)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    assert loss.item() < first.item() * 0.2, (
        f"loss only fell {first.item():.3f} -> {loss.item():.3f}; "
        "the training path cannot learn even a memorisable batch"
    )


def test_all_parameters_receive_gradients():
    """An unused parameter means a layer is disconnected from the loss."""
    model = _model()
    model.train()
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.context_length))
    _, loss = model(idx, idx)
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert not missing, f"no gradient reached: {missing}"


# ─── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}\n      {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
