"""
Load a checkpoint and generate text. This is the demo.

    python -m train.sample --ckpt runs/demo/ckpt_best.pt --prompt "Handdukstork Arvika"
    python -m train.sample --ckpt runs/demo/ckpt_best.pt --compare
    python -m train.sample --ckpt runs/demo/ckpt_best.pt --n 5 --temperature 1.0

`--compare` generates the same prompt under greedy, low-temperature, top-k and
nucleus decoding side by side. I added it so a reader can see for themselves that
decoding strategy changes the output far more visibly than a few points of
validation loss do. Lecture 6 opens on that comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.tokenizer import get_tokenizer
from model.config import GPTConfig
from model.gpt import GPTModel
from model.generate import generate_text


def load_model(ckpt_path: Path, device):
    """Rebuild the model from the config stored in the checkpoint."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ck["config"])
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tokenizer = get_tokenizer(ck["args"].get("tokenizer", "bytes"))
    return model, tokenizer, cfg, ck


# Each entry is (label, kwargs), ordered from most deterministic to most diverse.
STRATEGIES = [
    ("greedy            ", dict(temperature=0.0)),
    ("temperature 0.7   ", dict(temperature=0.7)),
    ("top-k 40, T=0.8   ", dict(temperature=0.8, top_k=40)),
    ("nucleus 0.9, T=1.0", dict(temperature=1.0, top_p=0.9)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--prompt", default="Handdukstork Arvika finns i")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--n", type=int, default=1, help="samples to draw")
    ap.add_argument("--compare", action="store_true", help="one sample per decoding strategy")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, cfg, ck = load_model(args.ckpt, device)

    print(f"checkpoint  {args.ckpt}  (step {ck['step']}, best val {ck['best_val']:.4f})")
    print(f"model       {cfg.summary()}")
    print(f"tokenizer   {tokenizer}")
    print(f"prompt      {args.prompt!r}\n")

    if args.compare:
        for label, kwargs in STRATEGIES:
            text = generate_text(model, tokenizer, args.prompt,
                                 max_new_tokens=args.max_new_tokens,
                                 device=device, **kwargs)
            print(f"── {label} ──")
            print(text.strip())
            print()
        return 0

    for i in range(args.n):
        text = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens, device=device,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        )
        if args.n > 1:
            print(f"── sample {i + 1} ──")
        print(text.strip())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
