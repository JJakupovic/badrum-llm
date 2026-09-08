"""
Instruction fine-tune the GPT (Lab 3).

    python -m data.instructions                                   # build the pairs
    python -m train.finetune --init runs/tiny-bytes/ckpt_best.pt  # from pretrained
    python -m train.finetune --scratch --like runs/tiny-bytes/ckpt_best.pt   # control

Those last two commands are Experiment 2. Same architecture, same instruction
data, same schedule, same seed; the only thing that varies is whether the weights
start from the pretrained checkpoint or from random init. If pretraining on 30k
tokens of templated Swedish buys nothing, this is what says so.

Loss is computed on answer tokens only; see data.dataset.InstructionDataset.

Every run writes runs/<name>/ with the same layout as pretraining:
ckpt_best.pt (lowest answer-loss), ckpt_last.pt, metrics.jsonl, config.json.

Fine-tuning starts a fresh optimizer. Adam's moment estimates describe the
gradient statistics of the pretraining objective over a flat token stream, while
the fine-tuning objective is a masked loss on short, padded sequences, so
carrying them over shows up as a violent first few steps. A fresh optimizer plus
warmup costs a hundred steps and removes the problem.
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

from data.dataset import make_instruction_loaders, IGNORE_INDEX
from data.instructions import format_prompt
from data.tokenizer import get_tokenizer
from model.config import get_config, GPTConfig
from model.gpt import GPTModel
from model.generate import complete
from train.pretrain import lr_at, save_checkpoint, load_checkpoint


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, int]:
    """
    Mean cross-entropy over answer tokens only, weighted by token count.

    Averaging per batch would weight a batch of short category answers the same
    as a batch of long variant lists, so the number would drift with batch
    composition rather than tracking the model.
    """
    if loader is None:
        return float("inf"), 0
    model.eval()
    total, n_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        k = int((y != IGNORE_INDEX).sum().item())
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum",
        )
        total += loss.item()
        n_tokens += k
    model.train()
    return total / max(n_tokens, 1), n_tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--init", type=Path, help="pretrained checkpoint to start from")
    src.add_argument("--scratch", action="store_true",
                     help="random init, the Experiment 2 control")

    ap.add_argument("--like", type=Path, default=None,
                    help="with --scratch: copy the architecture from this checkpoint "
                         "without its weights, so the Experiment 2 control matches the "
                         "treatment by construction; --preset would only happen "
                         "to match.")
    ap.add_argument("--preset", default="tiny",
                    help="architecture when --scratch and no --like")
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the one recorded in the checkpoint")
    ap.add_argument("--train", type=Path, default=Path("data/out/instructions_train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/out/instructions_val.jsonl"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))

    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--samples", type=int, default=3,
                    help="held-out questions to answer at the end, for eyeballing")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── architecture: inherited from the checkpoint, never re-derived ────────
    # Rebuilding it from --preset would silently succeed whenever the preset
    # happened to match, and load a shape-mismatched state dict when it did not.
    ckpt = None
    if args.init:
        ckpt = load_checkpoint(args.init, device)
        cfg = GPTConfig(**ckpt["config"])
        tok_name = args.tokenizer or ckpt.get("args", {}).get("tokenizer", "bytes")
        init_desc = f"pretrained {args.init}"
    elif args.like:
        ref = load_checkpoint(args.like, device)
        cfg = GPTConfig(**ref["config"])
        tok_name = args.tokenizer or ref.get("args", {}).get("tokenizer", "bytes")
        init_desc = f"random init, architecture copied from {args.like}"
        del ref
    else:
        tok_name = args.tokenizer or "bytes"
        cfg = None
        init_desc = "random init (scratch)"

    tokenizer = get_tokenizer(tok_name)
    # Record the resolved tokenizer name rather than the flag. --tokenizer
    # defaults to None, meaning "whatever the checkpoint used", and saving that
    # None leaves the fine-tuned checkpoint unable to say which tokenizer it was
    # trained with, so anything loading it later has to guess.
    args.tokenizer = tok_name
    if cfg is None:
        cfg = get_config(args.preset, vocab_size=tokenizer.vocab_size)
    elif cfg.vocab_size != tokenizer.vocab_size:
        print(f"tokenizer {tok_name} has vocab {tokenizer.vocab_size} but the "
              f"checkpoint was trained with {cfg.vocab_size}. These must match.")
        return 1

    if args.name:
        run_name = args.name
    elif args.init:
        run_name = f"ft-{args.init.parent.name}"
    elif args.like:
        run_name = f"ft-{args.like.parent.name}-scratch"
    else:
        run_name = f"ft-{args.preset}-scratch"
    run_dir = args.out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device      {device}")
    print(f"init        {init_desc}")
    print(f"tokenizer   {tokenizer}")
    print(f"model       {cfg.summary()}")

    train_records = read_jsonl(args.train)
    val_records = read_jsonl(args.val) if args.val.exists() else []
    train_loader, val_loader, stats = make_instruction_loaders(
        train_records, val_records, tokenizer, cfg.context_length,
        batch_size=args.batch_size,
    )
    frac = stats["supervised_tokens"] / max(stats["train_tokens"], 1)
    print(f"data        {stats['train_pairs']} train pairs / {stats['val_pairs']} val")
    print(f"            {stats['supervised_tokens']:,} supervised tokens "
          f"({frac:.0%} of {stats['train_tokens']:,}; the rest is masked prompt)"
          .replace(",", " "))

    model = GPTModel(cfg).to(device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"schedule    {args.epochs} epochs x {steps_per_epoch} steps = {total_steps} steps")

    start_step, best_val = 0, float("inf")
    if args.resume:
        rk = load_checkpoint(args.resume, device)
        model.load_state_dict(rk["model"])
        optimizer.load_state_dict(rk["optimizer"])
        start_step, best_val = rk["step"], rk["best_val"]
        print(f"resumed     from {args.resume} at step {start_step}")

    (run_dir / "config.json").write_text(json.dumps({
        "run": run_name, "stage": "finetune", "init": init_desc,
        "model": cfg.to_dict(), "data": stats,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")
    metrics_file = (run_dir / "metrics.jsonl").open("a", encoding="utf-8")

    val0, n_val_tokens = evaluate(model, val_loader, device)
    print(f"\n{'epoch':>6} {'step':>7} {'train':>8} {'val':>8} {'val ppl':>9}  elapsed")
    print("─" * 52)
    print(f"{0:>6} {0:>7} {'':>8} {val0:>8.4f} {math.exp(min(val0, 20)):>9.2f}   (before)")

    model.train()
    t0 = time.time()
    step = start_step

    for epoch in range(1, args.epochs + 1):
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            lr = lr_at(step, warmup=args.warmup, total=total_steps,
                       lr_max=args.lr, lr_min=args.lr_min)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item()
            n_batches += 1
            step += 1

        train_loss = running / max(n_batches, 1)
        val, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        print(f"{epoch:>6} {step:>7} {train_loss:>8.4f} {val:>8.4f} "
              f"{math.exp(min(val, 20)):>9.2f}  {elapsed:>6.0f}s")

        metrics_file.write(json.dumps({
            "epoch": epoch, "step": step,
            "train_loss": round(train_loss, 6),
            "val_loss": None if math.isinf(val) else round(val, 6),
            "val_perplexity": None if math.isinf(val) else round(math.exp(min(val, 20)), 4),
            "lr": lr, "elapsed_s": round(elapsed, 1),
        }) + "\n")
        metrics_file.flush()

        if val < best_val:
            best_val = val
            save_checkpoint(run_dir / "ckpt_best.pt", model, optimizer, step, best_val, cfg, args)
        save_checkpoint(run_dir / "ckpt_last.pt", model, optimizer, step, best_val, cfg, args)

    metrics_file.close()
    print("─" * 52)
    print(f"done in {time.time() - t0:.0f}s   best answer loss {best_val:.4f}"
          f"   (perplexity {math.exp(min(best_val, 20)):.2f})")

    # Qualitative check. Answer-token perplexity says how surprised the model is
    # by the right answer; it does not say whether the model would produce it.
    # Lab 4 measures that properly, and this is the thirty-second version.
    if args.samples and val_records:
        best = load_checkpoint(run_dir / "ckpt_best.pt", device)
        model.load_state_dict(best["model"])
        print("\nheld-out questions (greedy)\n" + "─" * 52)
        for r in val_records[: args.samples]:
            out = complete(model, tokenizer, format_prompt(r["question"]),
                           max_new_tokens=120, device=device, temperature=0.0)
            got = out.split("###")[0].strip() or "(empty)"
            print(f"Q  {r['question']}")
            print(f"A  {got[:200]}")
            print(f"   gold: {r['answer'][:200]}\n")

    print(f"checkpoints in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
