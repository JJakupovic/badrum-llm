"""
Generate the dataset. No network, no services, no credentials.

    python -m data.generate                       # 120 products -> ./data/out
    python -m data.generate --products 400 --out ./data/out
    python -m data.generate --seed 7              # a different catalogue

Outputs (JSONL, one object per line):
    catalogue.jsonl   one row per product, variants nested
    variants.jsonl    one flat row per variant  <- grounds the instruction data
    corpus_sv.jsonl   2-3 Swedish documents per product
    stats.json        counts and sanity checks; keep this with your results
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .catalogue import generate_catalogue, flatten_variants
from .render_sv import render_product


def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanity_checks(products, variants, corpus) -> list[dict]:
    """
    Cheap invariants. I added each one after a bug reached the training run
    without anything complaining first.
    """
    checks = []

    def add(ok, msg):
        checks.append({"ok": bool(ok), "message": msg})

    priced = [v for v in variants if isinstance(v.get("price"), (int, float))]
    add(len(priced) == len(variants), f"{len(priced)}/{len(variants)} variants priced")

    if priced:
        med = sorted(v["price"] for v in priced)[len(priced) // 2]
        add(0 < med < 100_000, f"median variant price {med} SEK")

    multi = [p for p in products if p["variant_count"] > 1]
    add(len(multi) > len(products) * 0.5,
        f"{len(multi)}/{len(products)} products have >1 variant "
        "(the bot needs something to disambiguate)")

    add(all(v.get("sku") for v in variants), "every variant has a SKU")

    ids = [v["variant_id"] for v in variants]
    add(len(ids) == len(set(ids)), f"variant ids unique ({len(set(ids))}/{len(ids)})")

    text = "\n".join(d["text"] for d in corpus)
    add(bool(re.search(r"[åäö]", text)), "corpus contains å/ä/ö")

    # I got number-word capitalisation wrong at first, and produced sentences
    # like "två av tre varianter finns i lager."
    lowered = [
        s for d in corpus for s in re.split(r"(?<=[.])\s+", d["text"])
        if s and s[0].islower() and not s.startswith("•")
    ]
    add(not lowered, f"no sentence starts lowercase{'' if not lowered else ': ' + lowered[0][:60]}")

    add(not re.search(r"<[a-zA-Z/][^>]*>", text), "no HTML leaked into the corpus")

    # The tokenizer experiment turns on these compound nouns, so I check for them.
    compounds = ["tvättställsblandare", "handdukstork", "badrumsskåp", "golvbrunn",
                 "duschblandare", "toalettpappershållare"]
    found = [c for c in compounds if c in text.lower()]
    add(len(found) >= 4, f"{len(found)}/{len(compounds)} target compound nouns present")

    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--out", type=Path, default=Path("data/out"))
    args = ap.parse_args(argv)

    products = generate_catalogue(args.products, seed=args.seed)
    variants = [row for p in products for row in flatten_variants(p)]
    corpus = [doc for p in products for doc in render_product(p)]

    words = sum(len(d["text"].split()) for d in corpus)
    chars = sum(len(d["text"]) for d in corpus)
    checks = sanity_checks(products, variants, corpus)

    stats = {
        "seed": args.seed,
        "products": len(products),
        "variants": len(variants),
        "mean_variants_per_product": round(len(variants) / max(len(products), 1), 2),
        "corpus_documents": len(corpus),
        "corpus_words": words,
        "corpus_chars": chars,
        # ~4 chars/token is the usual rule of thumb. Swedish under an
        # English-trained BPE runs worse, which is what Experiment 1 measures.
        "corpus_tokens_estimate": round(chars / 4),
        "all_synthetic": True,
        "checks": checks,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "catalogue.jsonl", products)
    write_jsonl(args.out / "variants.jsonl", variants)
    write_jsonl(args.out / "corpus_sv.jsonl", corpus)
    (args.out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"products   {len(products)}")
    print(f"variants   {len(variants)}  (mean {stats['mean_variants_per_product']}/product)")
    print(f"corpus     {len(corpus)} docs, ~{words:,} words, ~{stats['corpus_tokens_estimate']:,} tokens".replace(",", " "))
    print()
    failed = 0
    for c in checks:
        print(f"  {'✓' if c['ok'] else '✗'} {c['message']}")
        failed += not c["ok"]
    print(f"\nwrote {args.out}/")

    if failed:
        print(f"\n{failed} sanity check(s) failed", file=sys.stderr)
        return 1

    print(
        "\nNote: every word above is machine-generated. It supplies domain\n"
        "vocabulary; real Swedish supplies language. Mix it into FineWeb-2 sv\n"
        "at a small ratio. That ratio is Experiment 2."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
