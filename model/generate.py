"""
Decoding.

The model gives a distribution over the next token, and turning that into text is
a separate decision with real consequences. Greedy decoding maximises each step
locally and reliably produces repetitive text, because the most probable next
token is often the one that continues a loop.

I implemented four strategies here. Sampling settings change the output far more
visibly than a few points of validation loss do, which is where lecture 6 opens.
"""

from __future__ import annotations

import torch

from .gpt import GPTModel


@torch.no_grad()
def generate(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressive sampling.

    temperature: 0 (or very small) = greedy. Below 1 sharpens the distribution,
                 above 1 flattens it. Applied to logits before softmax.
    top_k:       keep only the k most likely tokens.
    top_p:       nucleus sampling. Keep the smallest set whose cumulative
                 probability reaches p. The cutoff adapts to how confident the
                 model is at this step, which top-k cannot do.

    idx: (batch, seq) of context tokens. Returns context + generated.
    """
    was_training = model.training
    model.eval()
    ctx_len = model.cfg.context_length

    for _ in range(max_new_tokens):
        # Crop to the context window. The positional embedding table has
        # context_length rows, and indexing past it raises.
        idx_cond = idx[:, -ctx_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # only the last position predicts the next token

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)
                # Shift so the token that crosses the threshold is itself kept.
                # Without it, a single token with p > top_p leaves nothing to sample.
                remove = cumulative - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        if eos_id is not None and (next_id == eos_id).all():
            break

    if was_training:
        model.train()
    return idx


def generate_text(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    device: str | torch.device = "cpu",
    **kwargs,
) -> str:
    """Convenience wrapper: string in, string out."""
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = generate(model, ids, max_new_tokens, eos_id=tokenizer.eot_id, **kwargs)
    return tokenizer.decode(out[0].tolist())


def complete(
    model: GPTModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    device: str | torch.device = "cpu",
    **kwargs,
) -> str:
    """
    Generate a continuation and return only the continuation.

    I slice by token count. Slicing by the decoded prompt's character length is
    the obvious alternative and it breaks: a byte-level decode replaces an
    incomplete multi-byte sequence with U+FFFD, so the decoded prompt can differ
    in length from the prompt that went in, and the slice then eats the first
    character of the answer or leaves the last character of the prompt behind.
    Token counts are unaffected by the substitution.
    """
    prompt_ids = tokenizer.encode(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = generate(model, ids, max_new_tokens, eos_id=tokenizer.eot_id, **kwargs)
    return tokenizer.decode(out[0].tolist()[len(prompt_ids):])
