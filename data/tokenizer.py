"""
Tokenizers behind one interface.

Experiment 1 lives on this seam. Swapping the tokenizer had to be a flag rather
than a rewrite, because the experiment holds the model, the data and the token
budget fixed and varies only the vocabulary.

Available now:
    bytes   UTF-8 bytes. No dependencies, no out-of-vocabulary token, and a
            useful experimental floor: no vocabulary can do worse per token, so
            every other tokenizer is measured against it.
    gpt2    tiktoken's GPT-2 BPE, the English-trained baseline that Swedish
            compounds shatter under. Needs `pip install tiktoken`.

Still to come (Experiment 1's actual subject): a BPE trained on Swedish. I plan
to implement `train_bpe()` per Raschka chapter 2 / lecture 3 and add it here;
nothing else in the codebase should need to change.

The number I report is fertility, tokens per word. Lower is better, because more
text then fits in the same context and the same compute buys more content.
Swedish compounds such as tvättställsblandare and toalettpappershållare are
where an English-trained vocabulary bleeds, and `fertility()` below measures
that directly.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class Tokenizer(Protocol):
    vocab_size: int
    eot_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class ByteTokenizer:
    """
    Raw UTF-8 bytes, plus one end-of-text token.

    Every Swedish character survives: å, ä and ö are two bytes each in UTF-8, so
    they cost two tokens each. Sequences come out long, which is what I pay for
    never having an unknown token.
    """

    def __init__(self):
        self.vocab_size = 257
        self.eot_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        data = bytes(i for i in ids if i < 256)
        # Generated text can end mid-character, so a truncated multi-byte
        # sequence at the tail is expected here.
        return data.decode("utf-8", errors="replace")

    def __repr__(self):
        return f"ByteTokenizer(vocab_size={self.vocab_size})"


class TiktokenTokenizer:
    """GPT-2's BPE via tiktoken, the English-trained baseline for Experiment 1."""

    def __init__(self, encoding: str = "gpt2"):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "tiktoken is needed for the gpt2 tokenizer: pip install tiktoken\n"
                "(or use --tokenizer bytes, which has no dependencies)"
            ) from e
        self._enc = tiktoken.get_encoding(encoding)
        self.vocab_size = self._enc.n_vocab
        self.eot_id = self._enc.eot_token
        self.name = encoding

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode(list(ids))

    def __repr__(self):
        return f"TiktokenTokenizer({self.name!r}, vocab_size={self.vocab_size})"


def get_tokenizer(name: str = "bytes") -> Tokenizer:
    if name == "bytes":
        return ByteTokenizer()
    if name in ("gpt2", "tiktoken"):
        return TiktokenTokenizer("gpt2")
    raise KeyError(f"unknown tokenizer {name!r}; choose from: bytes, gpt2")


def fertility(tokenizer: Tokenizer, texts: Sequence[str]) -> dict[str, float]:
    """
    Tokens per word and per character, the Experiment 1 measurement.

    I report both. Tokens-per-word is the intuitive number; tokens-per-character
    is the one that stays comparable across languages with different word
    lengths, which matters when the subject is Swedish compounding.
    """
    n_tokens = n_words = n_chars = 0
    for t in texts:
        n_tokens += len(tokenizer.encode(t))
        n_words += len(t.split())
        n_chars += len(t)
    return {
        "tokens": n_tokens,
        "words": n_words,
        "chars": n_chars,
        "tokens_per_word": round(n_tokens / max(n_words, 1), 4),
        "tokens_per_char": round(n_tokens / max(n_chars, 1), 4),
        "chars_per_token": round(n_chars / max(n_tokens, 1), 4),
    }
