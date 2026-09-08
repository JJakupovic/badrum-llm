"""
Checks on the generated dataset.

    python tests/test_data.py        (no dependencies)
    pytest tests/                    (if you prefer)

Most of these exist because the thing they check was once wrong. The Swedish
ones matter more than they look: this text becomes training data, so a
grammatical error here is a pattern the model learns.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.catalogue import generate_catalogue, flatten_variants  # noqa: E402
from data.render_sv import render_product, sv_list, count_word, inline, money  # noqa: E402

PRODUCTS = generate_catalogue(120, seed=20260824)
VARIANTS = [r for p in PRODUCTS for r in flatten_variants(p)]
CORPUS = [d for p in PRODUCTS for d in render_product(p)]
TEXT = "\n".join(d["text"] for d in CORPUS)


# ─── structure ───────────────────────────────────────────────────────────────

def test_catalogue_is_populated():
    assert len(PRODUCTS) == 120
    assert len(VARIANTS) > 300, "expected several variants per product on average"


def test_ids_unique():
    pids = [p["id"] for p in PRODUCTS]
    vids = [v["variant_id"] for v in VARIANTS]
    assert len(pids) == len(set(pids))
    assert len(vids) == len(set(vids))


def test_every_variant_priced_and_plausible():
    for v in VARIANTS:
        assert isinstance(v["price"], (int, float)), f"{v['sku']} unpriced"
        assert 50 < v["price"] < 100_000, f"{v['sku']} implausible price {v['price']}"


def test_skus_are_ascii():
    """SKUs travel through URLs, CSVs and instruction-tuning targets."""
    bad = [v["sku"] for v in VARIANTS if not v["sku"].isascii()]
    assert not bad, f"non-ASCII SKUs: {bad[:5]}"


def test_variant_options_resolve_to_axis_names():
    for p in PRODUCTS:
        axis_names = {a["name"] for a in p["option_axes"]}
        for v in p["variants"]:
            assert set(v["options"]) == axis_names, (
                f"{p['title']}: variant options {set(v['options'])} != axes {axis_names}"
            )


def test_catalogue_contains_single_variant_products():
    """A bot that has never seen one will invent options for products with none."""
    singles = [p for p in PRODUCTS if p["variant_count"] == 1]
    assert singles, "no single-SKU products generated"
    assert len(singles) < len(PRODUCTS) * 0.4, "too many; nothing left to disambiguate"


def test_no_dimension_contradicts_a_varying_axis():
    """
    A product offered in 500/600/800 mm must not also claim "bredd 600 mm".
    Someone demoing the bot will ask "hur bred är den?" and get a contradiction.
    """
    governs = {"Bredd": "width", "Djup": "length", "Höjd": "height"}
    for p in PRODUCTS:
        dims = p.get("dimensions_mm") or {}
        for axis in p["option_axes"]:
            key = governs.get(axis["name"])
            if key and len(axis["values"]) > 1:
                assert key not in dims, (
                    f"{p['title']} varies {axis['name']} but states {key}={dims[key]}"
                )


# ─── Swedish quality ─────────────────────────────────────────────────────────

def test_sentences_start_uppercase():
    """Catches the number-word bug: 'två av tre varianter finns i lager.'"""
    bad = [
        s for d in CORPUS for s in re.split(r"(?<=[.])\s+", d["text"])
        if s and s[0].islower() and not s.startswith("•")
    ]
    assert not bad, f"lowercase sentence starts: {bad[:3]}"


def test_single_variant_products_use_singular_stock_wording():
    """'Alla utföranden finns i lager' after 'finns i ett enda utförande' is a
    self-contradiction the model would pick up."""
    singles = {p["id"] for p in PRODUCTS if p["variant_count"] == 1}
    for d in CORPUS:
        if d["product_id"] in singles and d["kind"] == "product_overview":
            assert "Alla utföranden" not in d["text"], d["text"]
            assert "Samtliga varianter" not in d["text"], d["text"]


def test_no_double_spaces_or_orphan_punctuation():
    assert "  " not in TEXT, "double space in corpus"
    assert " ." not in TEXT and " ," not in TEXT
    assert ".." not in TEXT


def test_no_html_or_placeholder_leakage():
    assert not re.search(r"<[a-zA-Z/][^>]*>", TEXT)
    for token in ("None", "undefined", "null", "{}", "NaN"):
        assert token not in TEXT, f"'{token}' leaked into corpus text"


def test_compound_nouns_present():
    """The words an English-trained BPE shatters, which is Experiment 1's subject."""
    want = ["tvättställsblandare", "handdukstork", "badrumsskåp", "golvbrunn",
            "duschblandare", "toalettpappershållare", "tvättställsskåp"]
    low = TEXT.lower()
    missing = [w for w in want if w not in low]
    assert len(missing) <= 1, f"missing compound nouns: {missing}"


def test_swedish_characters_present():
    assert len(re.findall(r"[åäö]", TEXT)) > 500


def test_multi_axis_products_do_not_enumerate_the_cross_product():
    """
    Naming every combination inline produces "Bredd: 500 mm, Finish: Krom, Bredd:
    500 mm, …". The option commas collide with the list commas and the sentence
    becomes unparseable, so multi-axis products state the axes instead.
    """
    multi = [p for p in PRODUCTS if len([a for a in p["option_axes"] if len(a["values"]) > 1]) > 1]
    assert multi, "fixture should contain multi-axis products"
    ids = {p["id"] for p in multi}
    for d in CORPUS:
        if d["product_id"] in ids and d["kind"] == "product_overview":
            # A cross-product enumeration shows up as the same axis name twice.
            for axis in ("Bredd", "Kulör", "Finish"):
                assert d["text"].count(f"{axis}:") <= 1, d["text"]


def test_gendered_articles_are_correct():
    """'en blandare' but 'ett duschset'. Wrong gender in training data is worse
    than flatter phrasing, so the catalogue carries gender explicitly."""
    by_id = {p["id"]: p for p in PRODUCTS}
    checked = 0
    for d in CORPUS:
        p = by_id[d["product_id"]]
        noun = (p.get("noun") or "").lower()
        if not noun:
            continue
        wrong = "en" if p["gender"] == "neutrum" else "ett"
        if f" är {wrong} {noun}" in d["text"]:
            raise AssertionError(f"{p['title']} ({p['gender']}): ' är {wrong} {noun}'")
        right = "ett" if p["gender"] == "neutrum" else "en"
        checked += f" är {right} {noun}" in d["text"]
    assert checked > 0, "no gendered-article sentences generated; template unreachable?"


# ─── determinism ─────────────────────────────────────────────────────────────

def test_generation_is_deterministic():
    """A corpus that drifts between runs makes every ablation result meaningless."""
    again = generate_catalogue(120, seed=20260824)
    assert [p["id"] for p in again] == [p["id"] for p in PRODUCTS]
    assert [d["text"] for p in again for d in render_product(p)] == [d["text"] for d in CORPUS]


def test_seed_actually_changes_output():
    other = generate_catalogue(120, seed=1)
    assert [p["title"] for p in other] != [p["title"] for p in PRODUCTS]


# ─── helpers ─────────────────────────────────────────────────────────────────

def test_sv_list():
    assert sv_list(["a"]) == "a"
    assert sv_list(["a", "b"]) == "a och b"
    assert sv_list(["a", "b", "c"]) == "a, b och c"


def test_count_word_gender():
    assert count_word(1, "utrum") == "en"
    assert count_word(1, "neutrum") == "ett"
    assert count_word(3) == "tre"


def test_inline_leaves_codes_alone():
    assert inline("Krom") == "krom"
    assert inline("Matt svart") == "matt svart"
    assert inline("500 mm") == "500 mm"      # measurement
    assert inline("LED runtom") == "LED runtom"  # acronym


def test_money_uses_swedish_thousands_separator():
    assert money(2495) == "2 495 kr"
    assert money(649) == "649 kr"


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
