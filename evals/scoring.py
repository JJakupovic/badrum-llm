"""
Scoring an answer against its gold (Lab 4).

Every scorer takes the model's raw text and the gold value that
`data.instructions` computed from the catalogue. I import the primitives
(`extract_numbers`, `normalise`, `parse_call`) from there instead of
reimplementing them here. A scorer that read Swedish numbers slightly
differently from the generator would produce disagreements that look like model
errors.

Lecture 7's framing, made concrete:

    subject    the fine-tuned model's greedy continuation of a held-out prompt
    criteria   per task: one number, one set, one boolean, one call
    reference  computed from the catalogue, never written by hand
    method     exact match, set F1 or argument equality, no judge model

## What these scores do and do not say

They are containment checks. `price` counts as correct when the gold amount
appears anywhere among the numbers in the answer, so "Kranen kostar 1 535 kr
eller kanske 2 000 kr" scores 1.0. That is the usual looseness of exact-match QA
metrics, and it inflates the score rather than deflating it. I say so in one
sentence in the report and read a sample of the answers instead of trusting the
table on its own.

I report the strict full-string match alongside for that reason. When the two
numbers diverge sharply, the loose one is being gamed.
"""

from __future__ import annotations

import re

from data.instructions import extract_numbers, normalise, parse_call

# I defined the answer format and fine-tuned the model to produce it, so a parse
# failure here means the model got the format wrong. Lists always follow
# " finns ": "X finns i a, b och c."
_LIST_RE = re.compile(r"\bfinns\b(?:\s+i)?\s+(.*)$", re.DOTALL)


def extract_list(text: str) -> set[str]:
    m = _LIST_RE.search(text.strip())
    if not m:
        return set()
    tail = m.group(1).split("\n")[0].strip().rstrip(".")
    parts = []
    for chunk in tail.split(" och "):
        parts.extend(chunk.split(","))
    return {normalise(x) for x in parts if normalise(x)}


def read_bool(text: str) -> bool | None:
    t = normalise(text)
    if t.startswith("ja"):
        return True
    if t.startswith("nej"):
        return False
    return None  # neither; scored wrong, never silently coerced


def score(gold: dict, prediction: str) -> float:
    """0.0–1.0. Only `set` is fractional; everything else is right or wrong."""
    kind, value = gold["type"], gold["value"]

    if kind == "number":
        return float(value in extract_numbers(prediction))

    if kind == "bool":
        return float(read_bool(prediction) is value)

    if kind == "text":
        return float(normalise(value) in normalise(prediction))

    if kind == "call":
        return float(parse_call(prediction) == value)

    if kind == "set":
        got, want = extract_list(prediction), {normalise(v) for v in value}
        if not got or not want:
            return 0.0
        hit = len(got & want)
        precision, recall = hit / len(got), hit / len(want)
        # I score this with F1. Recall alone would reward listing every finish
        # in the catalogue, which is the failure this task exists to catch.
        return 0.0 if hit == 0 else 2 * precision * recall / (precision + recall)

    raise KeyError(f"unknown gold type {kind!r}")


def score_record(record: dict, prediction: str) -> dict:
    pred = prediction.split("###")[0].strip()
    return {
        "task": record["task"],
        "product_id": record["product_id"],
        "question": record["question"],
        "gold_answer": record["answer"],
        "prediction": pred,
        "score": score(record["gold"], pred),
        "exact": float(pred == record["answer"]),
    }
