"""
Corpus -> token stream -> training batches.

The corpus is a list of documents and the model wants a flat stream of ids. I
join the documents with an end-of-text token so the model can learn where a
document stops, instead of running one product description into the next as if
the two were a single text.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


def load_corpus(path: str | Path, key: str = "text") -> list[dict]:
    """Read a JSONL corpus, keeping metadata so mixtures can be filtered by it."""
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if rows and key not in rows[0]:
        raise KeyError(f"{path} rows have no {key!r} field; got {sorted(rows[0])}")
    return rows


def encode_documents(docs: list[str], tokenizer) -> list[int]:
    """Flatten documents into one id stream, separated by end-of-text."""
    stream: list[int] = []
    for d in docs:
        stream.extend(tokenizer.encode(d))
        stream.append(tokenizer.eot_id)
    return stream


class TokenWindowDataset(Dataset):
    """
    Sliding windows over the token stream.

    Each item is (x, y) where y is x shifted one position, so the target at every
    position is the next token. One window of length L therefore supplies L
    training signals, which is why language modelling is so data-efficient per
    token.

    `stride` controls overlap. stride == context_length means no token is seen
    twice in a single epoch; a smaller stride multiplies the number of windows by
    re-using the same text, so what looks like more data is the model seeing the
    same sentences again. I default to no overlap.
    """

    def __init__(self, token_ids: list[int], context_length: int, stride: int | None = None):
        if stride is None:
            stride = context_length
        if stride <= 0:
            raise ValueError("stride must be positive")
        if len(token_ids) < context_length + 1:
            raise ValueError(
                f"corpus has {len(token_ids)} tokens, need at least "
                f"{context_length + 1} for one window. Use a shorter context or more data."
            )

        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride
        self.starts = list(range(0, len(token_ids) - context_length, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        e = s + self.context_length
        return self.data[s:e], self.data[s + 1:e + 1]


def split_documents(docs: list, val_fraction: float = 0.1, seed: int = 0):
    """
    Split by document, never by token position.

    Splitting a flat token stream puts the first half of a product description in
    train and the second half in validation, so validation loss then measures
    memorisation of a document the model has already partly seen. It inflates the
    number in the flattering direction, which is the worst direction for a bug to
    point.
    """
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(docs), generator=g).tolist()
    n_val = max(1, int(len(docs) * val_fraction)) if len(docs) > 1 else 0
    val_idx = set(order[:n_val])
    train = [d for i, d in enumerate(docs) if i not in val_idx]
    val = [d for i, d in enumerate(docs) if i in val_idx]
    return train, val


def make_dataloaders(
    corpus_path: str | Path,
    tokenizer,
    context_length: int,
    batch_size: int = 8,
    val_fraction: float = 0.1,
    stride: int | None = None,
    seed: int = 0,
    num_workers: int = 0,
    exclude_product_ids: set[str] | None = None,
):
    """
    Returns (train_loader, val_loader, stats).

    `exclude_product_ids` drops every document belonging to those products before
    training. I use it to keep the instruction-tuning holdout out of pretraining
    as well. Without it the held-out products have still been seen during
    pretraining, and the claim I can make about generalisation is narrower.
    """
    rows = load_corpus(corpus_path)
    n_before = len(rows)
    if exclude_product_ids:
        rows = [r for r in rows if r.get("product_id") not in exclude_product_ids]
        if len(rows) == n_before:
            raise ValueError(
                f"--exclude-products matched nothing in {corpus_path}. Either the "
                "corpus rows carry no product_id, or the holdout file belongs to a "
                "different catalogue. Regenerate both from the same seed."
            )
    texts = [r["text"] for r in rows]
    train_docs, val_docs = split_documents(texts, val_fraction, seed)

    train_ids = encode_documents(train_docs, tokenizer)
    val_ids = encode_documents(val_docs, tokenizer) if val_docs else []

    train_ds = TokenWindowDataset(train_ids, context_length, stride)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers,
    )

    val_loader = None
    if len(val_ids) >= context_length + 1:
        val_ds = TokenWindowDataset(val_ids, context_length, stride)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers,
        )

    stats = {
        "documents": len(texts),
        "excluded_documents": n_before - len(rows),
        "excluded_products": len(exclude_product_ids or ()),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_windows": len(train_ds),
        "val_windows": len(val_loader.dataset) if val_loader else 0,
        "context_length": context_length,
        "vocab_size": tokenizer.vocab_size,
    }
    return train_loader, val_loader, stats


# ─── instruction tuning (Lab 3) ──────────────────────────────────────────────

IGNORE_INDEX = -100  # torch's cross_entropy default; model/gpt.py needs no change


class InstructionDataset(Dataset):
    """
    Prompt-masked instruction pairs.

    Loss is computed on the answer tokens only. Training on the prompt as well
    teaches the model to generate plausible customer questions, and it dilutes
    the gradient from the answer: at a median of 136 tokens per pair, roughly
    half of every sequence is prompt.

    I tokenize the prompt and the answer separately and concatenate them, instead
    of tokenizing the joined string and counting the prefix. With a byte
    tokenizer the two agree; with a BPE they do not, because a merge can straddle
    the boundary and shift it by a token. Defining the sequence as the
    concatenation makes the mask correct by construction. An off-by-one here is
    silent: the model trains on predicting the last token of its own prompt, and
    the loss curve still looks healthy.
    """

    def __init__(self, records: list[dict], tokenizer, max_length: int):
        from data.instructions import format_prompt

        self.items: list[tuple[list[int], int]] = []
        too_long = []
        for r in records:
            prompt_ids = tokenizer.encode(format_prompt(r["question"]))
            answer_ids = tokenizer.encode(r["answer"]) + [tokenizer.eot_id]
            full = prompt_ids + answer_ids
            if len(full) > max_length + 1:
                too_long.append((len(full), r["task"]))
                continue
            self.items.append((full, len(prompt_ids)))

        if too_long:
            longest = max(t[0] for t in too_long)
            raise ValueError(
                f"{len(too_long)} of {len(records)} instruction pairs exceed "
                f"context_length {max_length} (longest {longest} tokens).\n"
                "Dropping or truncating them is NOT the fix: dropping changes the "
                "evaluation denominator between runs, and truncating corrupts the "
                "gold answer. Pretrain with --context-length "
                f"{1 << (longest - 1).bit_length()} so every pair fits identically "
                "in every configuration being compared."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        full, n_prompt = self.items[i]
        x = torch.tensor(full[:-1], dtype=torch.long)
        y = torch.tensor(full[1:], dtype=torch.long)
        # y[j] is the target for x[j], i.e. full[j+1]. Supervise only where
        # full[j+1] is an answer token: j + 1 >= n_prompt.
        y[: n_prompt - 1] = IGNORE_INDEX
        return x, y


def collate_instructions(batch, pad_id: int):
    """
    Right-pad to the longest sequence in the batch.

    The padding needs no attention mask. Attention is causal, so a real token at
    position t never attends to a pad at position > t, and every pad's target is
    IGNORE_INDEX, so it contributes no loss. Left-padding would break both of
    those properties.
    """
    n = max(x.size(0) for x, _ in batch)
    xs = torch.full((len(batch), n), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : x.size(0)] = x
        ys[i, : y.size(0)] = y
    return xs, ys


def make_instruction_loaders(
    train_records: list[dict],
    val_records: list[dict],
    tokenizer,
    context_length: int,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Returns (train_loader, val_loader, stats)."""
    from functools import partial

    train_ds = InstructionDataset(train_records, tokenizer, context_length)
    val_ds = InstructionDataset(val_records, tokenizer, context_length) if val_records else None
    collate = partial(collate_instructions, pad_id=tokenizer.eot_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, collate_fn=collate, num_workers=num_workers)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False, collate_fn=collate, num_workers=num_workers)
                  if val_ds else None)

    supervised = sum(len(f) - n for f, n in train_ds.items)
    stats = {
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds) if val_ds else 0,
        "train_tokens": sum(len(f) for f, _ in train_ds.items),
        "supervised_tokens": supervised,
        "context_length": context_length,
        "vocab_size": tokenizer.vocab_size,
    }
    return train_loader, val_loader, stats
