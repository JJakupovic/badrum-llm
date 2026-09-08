"""
Checks on the instruction data and the loss mask.

    python tests/test_instructions.py     (torch tests skip if torch is absent)
    pytest tests/

I am defending two things here.

The first is that every gold answer is derivable from the catalogue. Lab 4 needs
no LLM-as-judge because the gold is computed instead of written by hand. If a
gold and the catalogue ever disagreed, the evaluation would measure the
generator's bugs and report them as model error.

The second is the loss mask, this lab's version of the causal-mask trap. Off by
one and the model is trained to predict the last token of its own prompt:
nothing crashes, the loss curve looks slightly better, and the number reaches the
report. I check the mask by reconstruction, decoding the supervised positions and
requiring the answer back, then mutate the check to prove it can fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.catalogue import generate_catalogue  # noqa: E402
from data.instructions import (  # noqa: E402
    A_PREFIX, Q_PREFIX, TASKS, build, extract_numbers, format_prompt,
    normalise, parse_call, split_by_product,
)
from data.tokenizer import ByteTokenizer  # noqa: E402

PRODUCTS = generate_catalogue(120, seed=20260824)
BY_ID = {p["id"]: p for p in PRODUCTS}
RECORDS = build(PRODUCTS, per_product=7)
TRAIN, VAL, VAL_IDS = split_by_product(RECORDS, 0.15, seed=0)


# ─── the data says what the catalogue says ───────────────────────────────────


def test_every_gold_matches_the_catalogue():
    """The gold is recomputed here from the product, independently of the generator."""
    for r in RECORDS:
        p, g = BY_ID[r["product_id"]], r["gold"]
        if r["task"] == "variant_count":
            assert g["value"] == len(p["variants"]), r
        elif r["task"] == "cheapest":
            assert g["value"] == min(v["price"]["amount"] for v in p["variants"]), r
        elif r["task"] == "variant_list":
            assert set(g["value"]) == {normalise(v["title"]) for v in p["variants"]}, r
        elif r["task"] == "sku":
            assert g["value"] in {v["sku"] for v in p["variants"]}, r
        elif r["task"] == "add_to_cart":
            assert g["value"]["sku"] in {v["sku"] for v in p["variants"]}, r
            assert 1 <= g["value"]["antal"] <= 3, r


def test_answer_contains_its_own_gold():
    """Self-consistency: the scorer applied to the gold answer must score 1.0."""
    for r in RECORDS:
        g, a = r["gold"], r["answer"]
        if g["type"] == "number":
            assert g["value"] in extract_numbers(a), r
        elif g["type"] == "set":
            for item in g["value"]:
                assert normalise(item) in normalise(a), r
        elif g["type"] == "bool":
            opens_yes = normalise(a).startswith("ja")
            assert opens_yes == g["value"], r
        elif g["type"] == "text":
            assert normalise(g["value"]) in normalise(a), r
        elif g["type"] == "call":
            assert parse_call(a) == g["value"], r


def test_option_check_polarity_matches_the_catalogue():
    """The probe is worthless if a "Nej" option is quietly on offer, or the reverse."""
    for r in RECORDS:
        if r["task"] != "option_check":
            continue
        p = BY_ID[r["product_id"]]
        offered = {normalise(v) for a in p["option_axes"] for v in a["values"]}
        asked = normalise(r["question"].split(" i ")[-1].rstrip("?"))
        assert (asked in offered) == r["gold"]["value"], r


def test_no_task_is_won_by_a_constant_answer():
    """
    A metric that a constant answer beats is measuring the class balance.

    This is why `stock` picks its polarity before its variant and why
    `option_check` asks about options that do exist half the time. The bound is
    loose; it is here to fail loudly if a task ever drifts back to one-sided.
    """
    from collections import Counter
    for task in {r["task"] for r in RECORDS}:
        golds = [r["gold"]["value"] for r in RECORDS
                 if r["task"] == task and r["gold"]["type"] == "bool"]
        if not golds:
            continue
        top = Counter(golds).most_common(1)[0][1] / len(golds)
        assert top < 0.75, f"{task}: answering the majority class scores {top:.0%}"


def test_variant_list_and_axis_values_do_not_overlap():
    for r in RECORDS:
        n_axes = len(BY_ID[r["product_id"]]["option_axes"])
        if r["task"] == "variant_list":
            assert n_axes == 1, r
        if r["task"] == "axis_values":
            assert n_axes >= 2, r


def test_all_ten_tasks_are_represented():
    seen = {r["task"] for r in RECORDS}
    assert len(seen) == len(TASKS), f"only {sorted(seen)}"


# ─── split ───────────────────────────────────────────────────────────────────


def test_split_is_by_product_not_by_pair():
    train_ids = {r["product_id"] for r in TRAIN}
    val_ids = {r["product_id"] for r in VAL}
    assert train_ids & val_ids == set(), sorted(train_ids & val_ids)[:5]
    assert val_ids == set(VAL_IDS)


def test_split_holds_out_a_useful_amount():
    assert 10 <= len(VAL_IDS) <= 30, len(VAL_IDS)
    assert len(VAL) > 50, len(VAL)


def test_build_is_deterministic():
    again = build(PRODUCTS, per_product=7)
    assert [r["answer"] for r in again] == [r["answer"] for r in RECORDS]


# ─── surface ─────────────────────────────────────────────────────────────────


def test_prompt_format_is_unambiguous():
    p = format_prompt("Vad kostar den?")
    assert p.startswith(Q_PREFIX) and p.endswith(A_PREFIX)
    for r in RECORDS:
        assert Q_PREFIX not in r["answer"], r
        assert A_PREFIX not in r["question"], r


def test_no_non_breaking_space_anywhere():
    """U+00A0 is invisible in every editor and breaks exact-match scoring."""
    for r in RECORDS:
        assert " " not in r["question"] + r["answer"], r


def test_prose_answers_start_capitalised():
    """add_to_cart is code and is excluded; everything else is a Swedish sentence."""
    for r in RECORDS:
        if r["task"] == "add_to_cart":
            continue
        assert r["answer"][0].isupper(), r["answer"][:40]


def test_extract_numbers_handles_both_swedish_shapes():
    assert 1535 in extract_numbers("kostar 1 535 kr.")   # space thousands separator
    assert 5 in extract_numbers("finns i fem utföranden.")  # spelled out
    assert extract_numbers("inga siffror här") == []


def test_parse_call_rejects_prose():
    assert parse_call("Jag har lagt den i varukorgen.") is None
    assert parse_call('add_to_cart(sku="ABC-1", antal=2)') == {"sku": "ABC-1", "antal": 2}


# ─── the loss mask (needs torch) ─────────────────────────────────────────────

try:
    import torch  # noqa: F401
    from data.dataset import InstructionDataset, collate_instructions, IGNORE_INDEX
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def test_mask_supervises_exactly_the_answer():
    """
    Decode the supervised positions. They have to reconstruct the answer and
    nothing else: no token of the prompt, no token of the answer missing.
    """
    if not HAVE_TORCH:
        return
    tok = ByteTokenizer()
    ds = InstructionDataset(RECORDS[:50], tok, max_length=512)
    for i, r in enumerate(RECORDS[:50]):
        x, y = ds[i]
        supervised = [int(t) for t in y if int(t) != IGNORE_INDEX]
        assert tok.decode(supervised[:-1]) == r["answer"], r["answer"][:60]
        assert supervised[-1] == tok.eot_id, "answer must end with end-of-text"


def test_the_mask_check_can_actually_fail():
    """
    Mutation test. I shift the boundary by one in each direction and require the
    reconstruction to break, because a check that cannot fail proves nothing.
    """
    if not HAVE_TORCH:
        return
    tok = ByteTokenizer()
    r = RECORDS[0]
    full = tok.encode(format_prompt(r["question"])) + tok.encode(r["answer"]) + [tok.eot_id]
    n_prompt = len(tok.encode(format_prompt(r["question"])))
    for delta in (-1, +1):
        y = torch.tensor(full[1:])
        y[: n_prompt - 1 + delta] = IGNORE_INDEX
        supervised = [int(t) for t in y if int(t) != IGNORE_INDEX]
        assert tok.decode(supervised[:-1]) != r["answer"], f"delta {delta} went undetected"


def test_padding_is_never_supervised():
    if not HAVE_TORCH:
        return
    tok = ByteTokenizer()
    ds = InstructionDataset(RECORDS[:8], tok, max_length=512)
    batch = [ds[i] for i in range(8)]
    xs, ys = collate_instructions(batch, pad_id=tok.eot_id)
    assert xs.shape == ys.shape
    for i, (x, y) in enumerate(batch):
        assert (ys[i, y.size(0):] == IGNORE_INDEX).all(), "pad target is supervised"
        assert (ys[i, : y.size(0)] == y).all(), "collate altered a real target"


def test_over_long_pairs_raise_instead_of_being_dropped():
    """Silently dropping them would change the evaluation denominator per run."""
    if not HAVE_TORCH:
        return
    try:
        InstructionDataset(RECORDS[:20], ByteTokenizer(), max_length=16)
    except ValueError as e:
        assert "context_length" in str(e)
        return
    raise AssertionError("over-long pairs were accepted")


# ─── runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not HAVE_TORCH:
        print("  (torch not installed, so the mask tests will no-op)\n")
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
