"""
A GPT-style decoder-only transformer, written from scratch.

"From scratch" means I implemented the pieces that carry the ideas: layer
normalisation, the GELU activation, causal multi-head self-attention, and the
residual structure of a transformer block. nn.Linear, nn.Embedding and the
optimiser come from torch.

Follows Raschka chapters 3–4. Lecture 2's transformer walkthrough is the same
architecture drawn as diagrams; the naming here matches the book so the report
can cite either.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class LayerNorm(nn.Module):
    """
    Layer normalisation, written out instead of nn.LayerNorm.

    Normalises across the feature dimension so each token's representation has
    zero mean and unit variance, then applies a learned scale and shift. The
    learned pair lets the network undo the normalisation where it needs to.

    Uses the biased variance (unbiased=False) to match GPT-2. With emb_dim in the
    hundreds the difference is negligible, but it matters if you load or compare
    against pretrained GPT-2 weights.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """
    The tanh approximation of GELU, as used by GPT-2.

    GELU is smooth and non-zero for small negative inputs, so gradients keep
    flowing where ReLU would cut them off.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """Position-wise MLP. Expands 4x, activates, projects back at GPT-2's ratio."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with a causal mask.

    The mask is the part I check hardest. A language model predicts token t+1
    from tokens ≤ t, so if position t can attend to t+1 the model reads the
    answer off its own input and learns nothing that generalises. The failure is
    silent and it flatters the run: nothing crashes and the training loss drops
    faster than it should, so tests/test_model.py asserts strict causality
    directly rather than trusting the mask by inspection.

    Q, K and V come from one fused Linear that is then split. Mathematically
    identical to Raschka's three projections, at one matmul instead of three.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.emb_dim = cfg.emb_dim
        self.use_fast = cfg.use_fast_attention
        self.drop_rate = cfg.drop_rate

        self.qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.attn_dropout = nn.Dropout(cfg.drop_rate)
        self.resid_dropout = nn.Dropout(cfg.drop_rate)

        # Upper-triangular mask, registered as a buffer so it moves with .to(device)
        # and is saved and loaded with the module without being a learned parameter.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1).bool(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        qkv = self.qkv(x)                                    # (b, n, 3d)
        q, k, v = qkv.split(self.emb_dim, dim=2)
        # (b, n, d) -> (b, heads, n, head_dim)
        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_fast:
            ctx = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop_rate if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Scale by sqrt(head_dim). Without it the dot products grow with
            # dimension, softmax saturates and gradients vanish. (Lecture 2,
            # "training trick #3".)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(self.mask[:n, :n], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            ctx = weights @ v

        ctx = ctx.transpose(1, 2).contiguous().view(b, n, d)
        return self.resid_dropout(self.out_proj(ctx))


class TransformerBlock(nn.Module):
    """
    One block: attention and MLP, each wrapped in a residual connection with
    pre-normalisation.

    I used pre-norm (normalise before the sublayer, add the raw input back)
    instead of the original paper's post-norm. It leaves an unobstructed additive
    path from input to output, so gradients reach the early layers without
    vanishing and deep stacks train without a warmup-dependent knife-edge.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    """
    Token embeddings + learned positional embeddings -> N transformer blocks ->
    final norm -> vocabulary logits.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        # Learned absolute positions, as in GPT-2. Lecture 2 covers the sinusoidal
        # alternative; learned is simpler and works at these context lengths.
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.drop_rate)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            # One matrix serves as both the input embedding and the output
            # projection. Saves vocab_size * emb_dim parameters, which at small
            # scale is most of the model.
            self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # Scaled init on the residual projections. Every block adds into the
        # residual stream, so without this the stream's variance grows with depth
        # and early training is unstable. (GPT-2 paper, section 2.3.)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("layers.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (batch, seq) token ids.
        Returns logits (batch, seq, vocab), and cross-entropy loss when targets given.
        """
        b, n = idx.shape
        if n > self.cfg.context_length:
            raise ValueError(
                f"sequence length {n} exceeds context_length {self.cfg.context_length}"
            )

        pos = torch.arange(n, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        logits = self.out_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Measured count. GPTConfig.n_params derives it; the tests assert they agree."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)):
        """
        AdamW with decay applied only to matrices.

        Biases and LayerNorm gains are one-dimensional, and decaying them shrinks
        learned scales and offsets toward zero for no benefit.
        """
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas,
        )
