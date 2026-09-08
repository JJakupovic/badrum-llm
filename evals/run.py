"""
Evaluate on the held-out instruction set (Lab 4).

    python -m evals.run --ckpt runs/ft-tiny-bytes/ckpt_best.pt
    python -m evals.run --ckpt runs/ft-tiny-bytes/ckpt_best.pt --baselines
    python -m evals.run --baselines-only          # no model, just the floors

Writes `eval.json` (every prediction, so the table can be rebuilt without
re-running the model) and prints the table the report quotes.

## Why there are two baselines

An accuracy of 0.6 means nothing on its own. It is a good score if a constant
answer gets 0.2 and a bad one if a constant answer gets 0.58.

`majority` answers the most common gold value for that task, every time. It is
the floor a metric has to clear before it is measuring the model rather than the
class balance. It has already paid for itself: `option_check` originally asked
only about options a product lacked, so "Nej" scored 100%. The metric was
measuring nothing, and only the baseline showed that.

`retrieval` answers with the gold answer of the most similar *training* question,
by word overlap. It is the floor for "did the model learn anything beyond
template matching?". A fine-tuned model that cannot beat it has learned to
imitate the answer shape and to look up the wrong product. Because the split is
by product, the retrieved answer always names a different product, so this
baseline scores well wherever a task is answerable without knowing which product
was asked about.

## Decoding is greedy by default

Temperature 0 makes the number reproducible, which is what the report needs. It
does not give the nicest prose. `--temperature` and `--top-k` are there to show
how much the metric moves under sampling, which is lecture 6's point in one line
of output.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from data.instructions import format_prompt, normalise
from data.render_sv import money, sv_list
from evals.scoring import score_record

# torch is imported lazily, inside ModelPredictor and the --ckpt branch of main.
# The two baselines are pure Python, so `--baselines-only` runs on a clean
# checkout with nothing installed. That is the cheapest way to confirm that the
# scorer and the gold data agree before spending anything on a GPU, and it keeps
# the floors reproducible on machines without PyTorch.


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# ─── predictors ──────────────────────────────────────────────────────────────
#
# All three return text and go through the same scorer. A baseline that took a
# shortcut into the scorer would be measuring something different from the model.


class MajorityPredictor:
    """The most common gold value for the task, rendered as a minimal answer."""

    def __init__(self, train_records: list[dict]):
        golds: dict[str, Counter] = defaultdict(Counter)
        for r in train_records:
            golds[r["task"]][json.dumps(r["gold"]["value"], sort_keys=True)] += 1
        self.answer = {}
        for task, counter in golds.items():
            value = json.loads(counter.most_common(1)[0][0])
            gold_type = next(r["gold"]["type"] for r in train_records if r["task"] == task)
            self.answer[task] = self._render(gold_type, value)

    @staticmethod
    def _render(gold_type: str, value) -> str:
        if gold_type == "bool":
            return "Ja." if value else "Nej."
        if gold_type == "number":
            return f"Det kostar {money(value)}."
        if gold_type == "set":
            return "Den finns i " + sv_list(list(value)) + "."
        if gold_type == "call":
            return f'add_to_cart(sku="{value["sku"]}", antal={value["antal"]})'
        return str(value)

    def __call__(self, record: dict) -> str:
        return self.answer.get(record["task"], "")


class RetrievalPredictor:
    """Gold answer of the nearest training question, by word-set Jaccard."""

    def __init__(self, train_records: list[dict]):
        self.index = [(set(normalise(r["question"]).split()), r["answer"], r["task"])
                      for r in train_records]

    def __call__(self, record: dict) -> str:
        q = set(normalise(record["question"]).split())
        best, best_sim = "", -1.0
        for words, answer, task in self.index:
            # Same-task only. Cross-task retrieval answers "hur många?" with a
            # price, which measures confusion in the index instead of the floor.
            if task != record["task"]:
                continue
            sim = len(q & words) / max(len(q | words), 1)
            if sim > best_sim:
                best, best_sim = answer, sim
        return best


class ModelPredictor:
    def __init__(self, ckpt_path: Path, device, tokenizer_name=None,
                 temperature=0.0, top_k=None, max_new_tokens=160):
        from data.tokenizer import get_tokenizer
        from model.config import GPTConfig
        from model.gpt import GPTModel
        from train.pretrain import load_checkpoint

        ck = load_checkpoint(ckpt_path, device)
        self.cfg = GPTConfig(**ck["config"])
        # `or` at every step: an older checkpoint can carry tokenizer=None.
        self.tokenizer = get_tokenizer(
            tokenizer_name or ck.get("args", {}).get("tokenizer") or "bytes")
        self.model = GPTModel(self.cfg).to(device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.device, self.temperature, self.top_k = device, temperature, top_k
        self.max_new_tokens = max_new_tokens
        self.step = ck.get("step")

    def __call__(self, record: dict) -> str:
        from model.generate import complete

        kwargs = {"temperature": self.temperature}
        if self.top_k:
            kwargs["top_k"] = self.top_k
        return complete(self.model, self.tokenizer, format_prompt(record["question"]),
                        max_new_tokens=self.max_new_tokens, device=self.device, **kwargs)


# ─── running ─────────────────────────────────────────────────────────────────


def evaluate(predictor, records: list[dict], progress: str | None = None) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, r in enumerate(records, 1):
        rows.append(score_record(r, predictor(r)))
        if progress and (i % 25 == 0 or i == len(records)):
            print(f"  {progress}: {i}/{len(records)}  ({time.time() - t0:.0f}s)", flush=True)
    return rows


def summarise(rows: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    per_task = {t: {"n": len(v),
                    "score": sum(x["score"] for x in v) / len(v),
                    "exact": sum(x["exact"] for x in v) / len(v)}
                for t, v in sorted(by_task.items())}
    return {
        "per_task": per_task,
        # Macro-averaged, so every task counts equally regardless of how many
        # pairs it happens to have. Micro would let `category` (99 pairs) outvote
        # `variant_list` (43) for no reason connected to the model.
        "macro": sum(v["score"] for v in per_task.values()) / max(len(per_task), 1),
        "micro": sum(r["score"] for r in rows) / max(len(rows), 1),
        "exact": sum(r["exact"] for r in rows) / max(len(rows), 1),
        "n": len(rows),
    }


def print_table(results: dict[str, dict]) -> None:
    names = list(results)
    tasks = sorted({t for r in results.values() for t in r["per_task"]})
    w = max(14, *(len(n) for n in names))
    print(f"\n{'task':<16}{'n':>5}" + "".join(f"{n:>{w}}" for n in names))
    print("─" * (21 + w * len(names)))
    for t in tasks:
        any_n = next(r["per_task"][t]["n"] for r in results.values() if t in r["per_task"])
        cells = "".join(f"{results[n]['per_task'].get(t, {}).get('score', float('nan')):>{w}.2f}"
                        for n in names)
        print(f"{t:<16}{any_n:>5}{cells}")
    print("─" * (21 + w * len(names)))
    for label, key in (("macro avg", "macro"), ("micro avg", "micro"), ("exact match", "exact")):
        print(f"{label:<16}{'':>5}" + "".join(f"{results[n][key]:>{w}.2f}" for n in names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=None, help="fine-tuned checkpoint")
    ap.add_argument("--baselines", action="store_true", help="also run majority + retrieval")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--train", type=Path, default=Path("data/out/instructions_train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("data/out/instructions_val.jsonl"))
    ap.add_argument("--out", type=Path, default=None, help="default: next to the checkpoint")
    ap.add_argument("--limit", type=int, default=None, help="first N val pairs (quick checks)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if not args.ckpt and not args.baselines_only:
        print("give --ckpt, or --baselines-only to see the floors on their own")
        return 1

    device = None
    if args.ckpt:
        import torch
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_records = read_jsonl(args.train)
    val_records = read_jsonl(args.val)[: args.limit]
    print(f"val         {len(val_records)} pairs from "
          f"{len({r['product_id'] for r in val_records})} held-out products")

    results, predictions = {}, {}

    if args.ckpt:
        model = ModelPredictor(args.ckpt, device, args.tokenizer,
                               args.temperature, args.top_k)
        decode = "greedy" if args.temperature <= 0 else \
                 f"T={args.temperature}" + (f" top-k={args.top_k}" if args.top_k else "")
        print(f"model       {args.ckpt} (step {model.step}), {decode}")
        rows = evaluate(model, val_records, progress="model")
        results["model"], predictions["model"] = summarise(rows), rows

    if args.baselines or args.baselines_only:
        for name, predictor in (("majority", MajorityPredictor(train_records)),
                                ("retrieval", RetrievalPredictor(train_records))):
            rows = evaluate(predictor, val_records)
            results[name], predictions[name] = summarise(rows), rows

    print_table(results)

    out = args.out or (args.ckpt.parent / "eval.json" if args.ckpt else Path("eval.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": str(args.ckpt) if args.ckpt else None,
        "decoding": {"temperature": args.temperature, "top_k": args.top_k},
        "val_pairs": len(val_records),
        "summary": results,
        "predictions": predictions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if "model" in results and "majority" in results:
        gap = results["model"]["macro"] - results["majority"]["macro"]
        if gap <= 0:
            print("\n! the model does not beat answering the majority class. "
                  "Whatever the loss curve says, it has not learned the tasks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
