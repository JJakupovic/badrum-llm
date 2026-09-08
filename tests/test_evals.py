"""
Checks on the Lab 4 scorer.

    python tests/test_evals.py      (no dependencies; the scorer is pure Python)

The load-bearing check is `test_gold_answers_all_score_one`. A scorer that
disagrees with its own gold data reports the generator's parsing quirks as model
error, and the report then describes a model that does not exist. All 840
answers go through the scorer that will grade them.

The rest cover the failure directions, since a scorer that always returned 1.0
would sail through the check above.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.catalogue import generate_catalogue  # noqa: E402
from data.instructions import build, split_by_product  # noqa: E402
from evals.run import MajorityPredictor, RetrievalPredictor, evaluate, summarise  # noqa: E402
from evals.scoring import extract_list, read_bool, score, score_record  # noqa: E402

PRODUCTS = generate_catalogue(120, seed=20260824)
RECORDS = build(PRODUCTS, per_product=7)
TRAIN, VAL, _ = split_by_product(RECORDS, 0.15, seed=0)


# ─── agreement with the gold data ────────────────────────────────────────────


def test_gold_answers_all_score_one():
    for r in RECORDS:
        s = score(r["gold"], r["answer"])
        assert s == 1.0, f"{r['task']}: scored {s:.2f} on its own gold\n  {r['answer']}"


def test_empty_prediction_scores_zero_everywhere():
    for r in RECORDS:
        assert score(r["gold"], "") == 0.0, r["task"]


def test_wrong_answers_score_zero():
    by_task = {r["task"]: r for r in RECORDS}
    assert score(by_task["price"]["gold"], "Den kostar 7 kr.") == 0.0
    assert score(by_task["sku"]["gold"], "Artikelnumret är XXX-YYY-ZZZ.") == 0.0
    assert score(by_task["add_to_cart"]["gold"],
                 'add_to_cart(sku="FEL-SKU", antal=99)') == 0.0
    b = by_task["option_check"]
    flipped = "Nej." if b["gold"]["value"] else "Ja."
    assert score(b["gold"], flipped) == 0.0


def test_a_non_answer_is_wrong_not_neutral():
    """"Kanske" is not a yes and must not be quietly read as one."""
    assert read_bool("Kanske, det beror på.") is None
    b = next(r for r in RECORDS if r["gold"]["type"] == "bool")
    assert score(b["gold"], "Kanske, det beror på.") == 0.0


# ─── the set metric ──────────────────────────────────────────────────────────


def test_set_score_penalises_over_listing():
    """
    The metric is F1. Recall alone would reward answering with every finish in
    the catalogue, which is the failure this task exists to detect.
    """
    r = next(x for x in RECORDS if x["task"] == "variant_list")
    everything = ("Den finns i " + ", ".join(
        list(r["gold"]["value"]) + ["mässing", "titan", "guld", "brons"]) + ".")
    s = score(r["gold"], everything)
    assert 0.0 < s < 1.0, s


def test_set_score_penalises_under_listing():
    r = next(x for x in RECORDS if x["task"] == "variant_list"
             and len(x["gold"]["value"]) >= 3)
    partial = "Den finns i " + list(r["gold"]["value"])[0] + "."
    assert 0.0 < score(r["gold"], partial) < 1.0


def test_extract_list_returns_nothing_on_prose():
    assert extract_list("Jag vet inte riktigt.") == set()
    assert extract_list("") == set()


# ─── record-level behaviour ──────────────────────────────────────────────────


def test_score_record_cuts_at_the_next_prompt():
    """A model that runs on into a fresh '### Fråga:' is graded on its answer only."""
    r = next(x for x in RECORDS if x["task"] == "category")
    row = score_record(r, r["answer"] + "\n\n### Fråga:\nVad kostar den?")
    assert row["score"] == 1.0 and row["exact"] == 1.0


def test_exact_match_is_stricter_than_score():
    r = next(x for x in RECORDS if x["task"] == "price")
    row = score_record(r, f"Priset är {r['gold']['value']} kr, ungefär.")
    assert row["score"] == 1.0 and row["exact"] == 0.0


# ─── baselines ───────────────────────────────────────────────────────────────


def test_baselines_do_not_solve_the_benchmark():
    """
    If a constant or a lookup scores highly, the tasks are measuring the class
    balance of the data. This guard caught option_check back when it only ever
    asked about options that did not exist.
    """
    for predictor in (MajorityPredictor(TRAIN), RetrievalPredictor(TRAIN)):
        macro = summarise(evaluate(predictor, VAL))["macro"]
        assert macro < 0.5, f"{type(predictor).__name__} macro {macro:.2f}"


def test_no_single_task_is_solved_by_the_majority_answer():
    rows = evaluate(MajorityPredictor(TRAIN), VAL)
    for task, stats in summarise(rows)["per_task"].items():
        assert stats["score"] < 0.8, f"{task}: majority scores {stats['score']:.2f}"


# ─── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except AssertionError as e:
            print(f"  ✗ {name}\n      {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
