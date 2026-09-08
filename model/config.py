"""
Model configuration and size presets.

Parameter counts are computed from the architecture. I need exact numbers for the
report, and Experiment 2 compares data mixtures at identical architecture, so
"identical" has to be something the code can check instead of something I assume.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 256
    emb_dim: int = 384
    n_heads: int = 6
    n_layers: int = 6
    drop_rate: float = 0.1
    qkv_bias: bool = False

    # Tie the token embedding and the output head, as GPT-2 does. It removes
    # vocab_size * emb_dim parameters and helps small models, where the embedding
    # table would otherwise dominate the parameter count.
    weight_tying: bool = True

    # Use torch's fused scaled_dot_product_attention instead of the explicit
    # implementation. The two are mathematically identical (asserted in
    # tests/test_model.py) and the fused one is much faster on GPU. I left the
    # explicit path as the default because it shows the mechanism.
    use_fast_attention: bool = False

    def __post_init__(self):
        if self.emb_dim % self.n_heads:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

    def n_params(self, non_embedding: bool = False) -> int:
        """
        Parameter count derived from the architecture, so I can call it before
        the model is built when planning a run.

        Per block: attention (4 square projections) + MLP (two 4x projections)
                   + 2 LayerNorms.
        """
        d, v, L, ctx = self.emb_dim, self.vocab_size, self.n_layers, self.context_length

        tok_emb = v * d
        pos_emb = ctx * d

        attn = 4 * d * d + (3 * d if self.qkv_bias else 0) + d  # +d = out_proj bias
        mlp = (d * 4 * d + 4 * d) + (4 * d * d + d)
        norms = 2 * (2 * d)                                      # scale + shift each
        per_block = attn + mlp + norms

        final_norm = 2 * d
        head = 0 if self.weight_tying else (v * d)

        total = tok_emb + pos_emb + L * per_block + final_norm + head
        return total - tok_emb - pos_emb if non_embedding else total

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        tot = self.n_params()
        non_emb = self.n_params(non_embedding=True)
        return (
            f"{self.n_layers}L {self.n_heads}H {self.emb_dim}d ctx={self.context_length} "
            f"vocab={self.vocab_size}\n"
            f"{tot/1e6:.1f}M parameters ({non_emb/1e6:.1f}M non-embedding)"
        )


# Presets. The names describe scale only.
#
# I pick one by token budget. A Chinchilla-ish ~20 tokens per parameter is the
# defensible starting point, so `small` (~30M) wants roughly 600M tokens. With a
# corpus smaller than that I drop to a smaller model instead of running many
# epochs over too little data.
PRESETS: dict[str, dict] = {
    # For tests and CPU smoke runs. Trains in seconds; produces nonsense.
    "debug": dict(context_length=64, emb_dim=64, n_heads=4, n_layers=2, drop_rate=0.0),

    # Sensible on a 24 GB community-cloud card. This is my default for the report.
    "tiny": dict(context_length=256, emb_dim=384, n_heads=6, n_layers=6, drop_rate=0.1),

    # The Experiment 2 workhorse: three data mixtures at this size is still cheap.
    "small": dict(context_length=512, emb_dim=512, n_heads=8, n_layers=8, drop_rate=0.1),

    # GPT-2 small geometry. Needs a real token budget behind it.
    "base": dict(context_length=1024, emb_dim=768, n_heads=12, n_layers=12, drop_rate=0.1),
}


def get_config(preset: str = "tiny", **overrides) -> GPTConfig:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    return GPTConfig(**{**PRESETS[preset], **overrides})
