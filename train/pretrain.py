"""
Pretrain the GPT.

    python -m train.pretrain --preset tiny --steps 2000
    python -m train.pretrain --preset debug --steps 50 --device cpu   # smoke test
    python -m train.pretrain --resume runs/tiny/ckpt_last.pt          # after a reclaim

Every run writes to runs/<name>/:
    ckpt_last.pt   rolling checkpoint, for resuming
    ckpt_best.pt   lowest validation loss seen
    metrics.jsonl  one row per eval, which is what the report's loss curves plot
    config.json    everything needed to reproduce the run

I built resuming in from the start, because community-cloud pods can be
reclaimed mid-run and a six-hour run that cannot resume is six hours lost.
Checkpoints carry optimizer state and the step counter, so --resume continues the
run instead of restarting with a fresh optimizer and a reset schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.dataset import make_dataloaders
from data.tokenizer import get_tokenizer
from model.config import get_config, GPTConfig
from model.gpt import GPTModel
from model.generate import generate_text


# ─── schedule ────────────────────────────────────────────────────────────────


def lr_at(step: int, *, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """
    Linear warmup then cosine decay.

    I warm up because Adam's second-moment estimate is unreliable for the first
    few hundred steps, and a full learning rate that early can wreck the model
    before the optimizer has any idea of gradient scale. Cosine decay then anneals
    toward lr_min so late training refines instead of bouncing.
    """
    if step < warmup:
        return lr_max * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr_min
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ─── evaluation ──────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> float:
    """Mean cross-entropy over the validation set. Returns inf if there is none."""
    if loader is None:
        return float("inf")
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


# ─── checkpoints ─────────────────────────────────────────────────────────────


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float, cfg: GPTConfig, args):
    # Write to a temporary file and rename. A pod reclaimed mid-write would
    # otherwise leave a truncated checkpoint that fails to load, losing the run
    # the checkpoint exists to protect.
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "config": cfg.to_dict(),
        "args": vars(args),
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device):
    return torch.load(path, map_location=device, weights_only=False)


# ─── main ────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny", help="debug | tiny | small | base")
    # Architecture overrides. Experiment 2 holds the architecture fixed while the
    # data mixture varies; a scaling sweep does the opposite. Both want these set
    # from the command line instead of by editing a preset.
    ap.add_argument("--emb-dim", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--corpus", default="data/out/corpus_sv.jsonl")
    ap.add_argument("--exclude-products", type=Path, default=None,
                    help="holdout_products.json from data.instructions. Drops those "
                         "products from pretraining too, so the instruction holdout "
                         "is genuinely unseen. Changes which claim the report can make.")
    ap.add_argument("--tokenizer", default="bytes", help="bytes | gpt2")
    ap.add_argument("--name", default=None, help="run name (default: preset-tokenizer)")
    ap.add_argument("--out", type=Path, default=Path("runs"))

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step; raises effective batch "
                         "size without more memory")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--sample-prompt", default="Handdukstork")

    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", help="torch.compile (slow first step)")
    ap.add_argument("--fast-attention", action="store_true",
                    help="fused SDPA instead of the explicit implementation")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)

    # Determinism. Seeding costs a little speed and buys a run I can reproduce.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.name or f"{args.preset}-{args.tokenizer}"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # bf16 where supported. I chose it over fp16 for the fp32 exponent range: no
    # loss scaling, and no silent overflow to NaN mid-run.
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16

    print(f"device      {device}" + ("  (bf16 autocast)" if use_amp else ""))

    tokenizer = get_tokenizer(args.tokenizer)
    print(f"tokenizer   {tokenizer}")

    overrides = {
        k: v for k, v in {
            "emb_dim": args.emb_dim,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "context_length": args.context_length,
            "drop_rate": args.dropout,
        }.items() if v is not None
    }
    cfg = get_config(args.preset, vocab_size=tokenizer.vocab_size,
                     use_fast_attention=args.fast_attention, **overrides)
    print(f"model       {cfg.summary()}"
          + (f"\n            overrides: {overrides}" if overrides else ""))

    excluded = None
    if args.exclude_products:
        holdout = json.loads(args.exclude_products.read_text(encoding="utf-8"))
        excluded = set(holdout["val_product_ids"])

    train_loader, val_loader, stats = make_dataloaders(
        args.corpus, tokenizer, cfg.context_length,
        batch_size=args.batch_size, seed=args.seed,
        exclude_product_ids=excluded,
    )
    if excluded:
        print(f"excluded    {stats['excluded_products']} held-out products "
              f"({stats['excluded_documents']} documents) from pretraining")
    print(f"data        {stats['train_tokens']:,} train tokens, "
          f"{stats['val_tokens']:,} val, {stats['train_windows']:,} windows"
          .replace(",", " "))

    tokens_per_step = args.batch_size * args.grad_accum * cfg.context_length
    total_tokens = tokens_per_step * args.steps
    epochs = total_tokens / max(stats["train_tokens"], 1)
    print(f"budget      {total_tokens:,} tokens over {args.steps} steps "
          f"({epochs:.1f} epochs)".replace(",", " "))

    # Two diagnostics from lecture 4, printed before the run starts so I see them
    # in time to change it.
    #
    # Chinchilla-optimal is roughly 20 tokens of unique data per parameter. Far
    # below that and the model is too big for the data, so it memorises. Far above
    # and it is too small to use the data available.
    tokens_per_param = stats["train_tokens"] / max(cfg.n_params(non_embedding=True), 1)
    print(f"            {tokens_per_param:.1f} unique tokens/parameter "
          f"(Chinchilla-optimal ≈ 20)")
    if tokens_per_param < 5:
        print("            ! model is large for this corpus; expect memorisation")
    elif tokens_per_param > 100:
        print("            ! corpus is large for this model; a bigger model would "
              "use it better")

    if epochs > 10:
        print(f"            ! {epochs:.0f} passes over the same text. Repeated epochs "
              "buy memorisation,\n"
              "              not generalisation. Prefer more data over more epochs.")

    model = GPTModel(cfg).to(device)
    assert model.num_parameters() == cfg.n_params(), "derived and measured param counts disagree"
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)

    start_step, best_val = 0, float("inf")
    if args.resume:
        ck = load_checkpoint(args.resume, device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step, best_val = ck["step"], ck["best_val"]
        print(f"resumed     from {args.resume} at step {start_step}")

    if args.compile:
        model = torch.compile(model)

    (run_dir / "config.json").write_text(json.dumps({
        "run": run_name, "model": cfg.to_dict(), "data": stats,
        "args": {k: str(v) for k, v in vars(args).items()},
        "tokens_per_step": tokens_per_step, "total_tokens": total_tokens,
    }, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")

    print(f"\nrun         {run_dir}\n")
    print(f"{'step':>7} {'loss':>8} {'val':>8} {'lr':>9} {'tok/s':>9}  elapsed")
    print("─" * 58)

    model.train()
    train_iter = iter(train_loader)
    t0 = time.time()
    window_t0, window_tokens = t0, 0
    running_loss, running_n = 0.0, 0

    for step in range(start_step, args.steps):
        lr = lr_at(step, warmup=args.warmup, total=args.steps,
                   lr_max=args.lr, lr_min=args.lr_min)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)

            # Divide so accumulated gradients average instead of summing.
            # Otherwise the effective learning rate scales with grad_accum.
            (loss / args.grad_accum).backward()
            running_loss += loss.item()
            running_n += 1
            window_tokens += x.numel()

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        is_last = step == args.steps - 1
        if (step + 1) % args.eval_every == 0 or is_last:
            val = evaluate(model, val_loader, device, args.eval_batches)
            train_loss = running_loss / max(running_n, 1)
            running_loss, running_n = 0.0, 0

            now = time.time()
            tok_s = window_tokens / max(now - window_t0, 1e-9)
            window_t0, window_tokens = now, 0

            print(f"{step + 1:>7} {train_loss:>8.4f} {val:>8.4f} {lr:>9.2e} "
                  f"{tok_s:>9,.0f}  {now - t0:>6.0f}s".replace(",", " "))

            metrics_file.write(json.dumps({
                "step": step + 1, "train_loss": round(train_loss, 6),
                "val_loss": None if math.isinf(val) else round(val, 6),
                # Perplexity is the number I quote in the report. It compares
                # across runs more readily than raw loss does, as long as the
                # tokenizer is held fixed. Across tokenizers it does not compare
                # at all; use bits-per-character there.
                "val_perplexity": None if math.isinf(val) else round(math.exp(min(val, 20)), 4),
                "lr": lr, "tokens_per_s": round(tok_s), "elapsed_s": round(now - t0, 1),
            }) + "\n")
            metrics_file.flush()

            if val < best_val:
                best_val = val
                save_checkpoint(run_dir / "ckpt_best.pt", model, optimizer, step + 1,
                                best_val, cfg, args)

        if (step + 1) % args.save_every == 0 or is_last:
            save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, step + 1,
                            best_val, cfg, args)

        if args.sample_every and ((step + 1) % args.sample_every == 0 or is_last):
            base = model._orig_mod if hasattr(model, "_orig_mod") else model
            text = generate_text(base, tokenizer, args.sample_prompt,
                                 max_new_tokens=80, device=device,
                                 temperature=0.8, top_k=40)
            print(f"\n  sample: {text.strip()[:300]}\n")

    metrics_file.close()
    elapsed = time.time() - t0
    print("─" * 58)
    print(f"done in {elapsed:.0f}s   best val loss {best_val:.4f}"
          + (f"   (perplexity {math.exp(min(best_val, 20)):.1f})" if best_val < 20 else ""))
    print(f"checkpoints in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
