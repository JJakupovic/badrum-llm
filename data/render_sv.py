"""
Render products as Swedish prose for the pretraining corpus.

The catalogue as JSON teaches a language model nothing about language. To earn a
place in a pretraining mixture it has to become running text carrying the domain
vocabulary, and Swedish bathroom retail is unusually compound-heavy:
tvättställsblandare, handdukstork, toalettpappershållare, badrumsskåp,
golvbrunn. Those are the words an English-trained BPE shatters, which is what
Experiment 1 measures.

The text is templated, so its diversity ceiling is this file, and that
constrains every template below. Generating 10,000 products does not give me
10,000 products' worth of linguistic variety. Lecture 3 covers model collapse
under synthetic-heavy training directly, so I keep this a minority of the
mixture; that ratio is the knob Experiment 2 sweeps.

On gender: Swedish grammatical gender cannot be derived from the noun ("en
blandare" but "ett duschset"). The synthetic catalogue carries it explicitly on
each product, which makes the article templates below safe. Reading from a real
shop where gender is unknown, I would drop those templates rather than guess.
Confidently wrong Swedish in training data is worse than flatter phrasing.
"""

from __future__ import annotations

import hashlib
from typing import Any

NUMWORD = ["noll", "en", "två", "tre", "fyra", "fem", "sex", "sju", "åtta", "nio",
           "tio", "elva", "tolv"]


def sv_number(n: float) -> str:
    """
    Swedish thousands separator is a space: 2 495.

    I use U+0020 here deliberately. The typographically correct character is
    U+00A0, but a non-breaking space is invisible in every editor and terminal,
    so it silently breaks exact-match scoring of prices in evaluation and turns
    into a surprise token in the tokenizer.
    """
    if isinstance(n, float) and not n.is_integer():
        return f"{n:,.2f}".replace(",", " ").replace(".", ",")
    return f"{int(n):,}".replace(",", " ")


def sv_list(items: list[str]) -> str:
    """a, b och c"""
    items = [str(i) for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " och " + items[-1]


def count_word(n: int, gender: str = "utrum") -> str:
    if n == 1:
        return "ett" if gender == "neutrum" else "en"
    return NUMWORD[n] if n < len(NUMWORD) else sv_number(n)


def money(amount: float | None, currency: str = "sek") -> str | None:
    if amount is None:
        return None
    cur = (currency or "sek").lower()
    return f"{sv_number(amount)} kr" if cur == "sek" else f"{sv_number(amount)} {cur.upper()}"


def inline(value: str) -> str:
    """
    Option values arrive title-cased because that is how they render in a
    dropdown ("Krom", "Matt svart"). Mid-sentence they read wrong capitalised.
    Codes, measurements and acronyms are left alone ("500 mm", "LED runtom").
    """
    s = str(value)
    if any(ch.isdigit() for ch in s):
        return s
    if len(s) >= 2 and s[:2].isupper():
        return s
    return s[0].lower() + s[1:] if s else s


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


class _Rng:
    """
    Deterministic picker seeded from the product id.

    The corpus is an experimental input, so one that drifted between runs would
    leave the ablation results unusable.
    """

    def __init__(self, seed_str: str):
        h = hashlib.blake2b(seed_str.encode("utf-8"), digest_size=8).digest()
        self.state = int.from_bytes(h, "big") or 1

    def next(self) -> float:
        # xorshift64*
        x = self.state
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) / 2**64

    def pick(self, options: list[str]) -> str:
        return options[int(self.next() * len(options)) % len(options)]


# ─── sentence generators ─────────────────────────────────────────────────────


def _opening(p: dict[str, Any], rng: _Rng) -> str:
    cat = (p["categories"][0]["name"] if p.get("categories") else None) or p.get("type")
    gender = p.get("gender")
    noun = (p.get("noun") or "").lower()
    opts = []
    if cat:
        opts.append(f"{p['title']} hör till kategorin {cat.lower()}.")
        opts.append(f"Bland våra {cat.lower()} finns {p['title']}.")
        opts.append(f"{p['title']} hittar du bland våra {cat.lower()}.")
    if gender and noun:
        article = "ett" if gender == "neutrum" else "en"
        opts.append(f"{p['title']} är {article} {noun} från vårt sortiment.")
        opts.append(f"{p['title']} är {article} {noun} för moderna badrum.")
    if p.get("subtitle"):
        opts.append(f"{p['title']} – {p['subtitle'].lower()}.")
    opts.append(f"{p['title']} ingår i vårt badrumssortiment.")
    return rng.pick(opts)


def _variants_sentence(p: dict[str, Any], rng: _Rng) -> str:
    """
    One axis  -> name the values: "finns i krom, borstad mässing och matt svart".
    Two+ axes -> naming the cross product is unreadable, because option values are
                 themselves comma-joined and collide with the list commas. State
                 the axes; the listing block carries the enumeration.
    """
    n = p["variant_count"]
    if n <= 1:
        return rng.pick([
            f"{p['title']} finns i ett enda utförande.",
            f"Det finns bara en variant av {p['title']}.",
        ])

    axes = [a for a in p["option_axes"] if a["name"] and len(a["values"]) > 1]

    if len(axes) == 1:
        vals = sv_list([inline(v) for v in axes[0]["values"]])
        return rng.pick([
            f"{p['title']} finns i {vals}.",
            f"{p['title']} finns i {count_word(n)} varianter: {vals}.",
            f"Du kan välja mellan {vals}.",
            f"{p['title']} levereras i {vals}.",
        ])

    if len(axes) > 1:
        names = sv_list([a["name"].lower() for a in axes])
        return rng.pick([
            f"{p['title']} finns i {count_word(n)} varianter, med olika {names}.",
            f"Det finns {count_word(n)} varianter av {p['title']} – välj {names}.",
            f"{p['title']} kan kombineras i {count_word(n)} varianter beroende på {names}.",
        ])

    return f"{p['title']} finns i {count_word(n)} varianter."


def _axis_sentences(p: dict[str, Any], rng: _Rng) -> list[str]:
    """Skipped for single-axis products, since _variants_sentence lists those."""
    axes = [a for a in p["option_axes"] if a["name"] and len(a["values"]) > 1]
    if len(axes) < 2:
        return []
    out = []
    for axis in axes:
        name = axis["name"].lower()
        vals = [inline(v) for v in axis["values"]]
        lst = sv_list(vals)
        # Swedish takes "mellan X och Y" for two values, "bland X, Y och Z" above that.
        choose = f"Välj {name} mellan {lst}." if len(vals) == 2 else f"Välj {name} bland {lst}."
        out.append(rng.pick([f"{cap(name)} finns i {lst}.", choose, f"{cap(name)}: {lst}."]))
    return out


def _price_sentence(p: dict[str, Any], rng: _Rng) -> str | None:
    pr = p.get("price_range")
    if not pr:
        return None
    lo, hi = money(pr["min"], pr["currency"]), money(pr["max"], pr["currency"])
    if not lo:
        return None
    if pr["min"] == pr["max"]:
        return rng.pick([f"Priset är {lo}.", f"{p['title']} kostar {lo}.", f"Pris: {lo}."])
    return rng.pick([
        f"Priset varierar mellan {lo} och {hi} beroende på variant.",
        f"Priserna börjar på {lo} och går upp till {hi}.",
        f"{p['title']} kostar från {lo}.",
    ])


def _spec_sentences(p: dict[str, Any], rng: _Rng) -> list[str]:
    out = []
    if p.get("material"):
        out.append(rng.pick([
            f"Materialet är {p['material'].lower()}.",
            f"Tillverkad i {p['material'].lower()}.",
        ]))
    d = p.get("dimensions_mm") or {}
    parts = []
    if d.get("width"):
        parts.append(f"bredd {sv_number(d['width'])} mm")
    if d.get("height"):
        parts.append(f"höjd {sv_number(d['height'])} mm")
    if d.get("length"):
        parts.append(f"djup {sv_number(d['length'])} mm")
    if parts:
        out.append(f"Mått: {sv_list(parts)}.")
    return out


def _stock_sentence(p: dict[str, Any], rng: _Rng) -> str | None:
    known = [v for v in p["variants"] if v["inventory"]["quantity"] is not None]
    if not known:
        return None
    in_stock = [v for v in known if v["inventory"]["quantity"] > 0]

    # A one-variant product must not be described in the plural. Without this
    # guard the renderer follows "finns i ett enda utförande" with "Alla
    # utföranden finns i lager", which teaches a model to contradict itself.
    if len(known) == 1:
        return "Produkten finns i lager." if in_stock else "Produkten är slut i lager."

    if len(in_stock) == len(known):
        return rng.pick(["Samtliga varianter finns i lager.", "Alla utföranden finns i lager."])
    if not in_stock:
        return rng.pick(["Tillfälligt slut i lager.", "Produkten är för närvarande slutsåld."])
    return cap(f"{count_word(len(in_stock))} av {count_word(len(known))} varianter finns i lager.")


def _variant_lines(p: dict[str, Any]) -> list[str]:
    """Dense option/SKU/price/stock associations, one line per variant."""
    lines = []
    for v in p["variants"]:
        label = v.get("option_summary") or v.get("title")
        if not label:
            continue
        bits = [f"{p['title']}, {label}"]
        if v.get("sku"):
            bits.append(f"artikelnummer {v['sku']}")
        m = money(v["price"]["amount"], v["price"]["currency_code"]) if v.get("price") else None
        if m:
            bits.append(f"pris {m}")
        q = v["inventory"]["quantity"]
        if q is not None:
            bits.append("i lager" if q > 0 else "slut i lager")
        lines.append(", ".join(bits) + ".")
    return lines


# ─── public API ──────────────────────────────────────────────────────────────


def render_product(p: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Returns 2-3 documents per product.

    Every document is machine-generated, so `synthetic` is True throughout. The
    Medusa path differed: there the shop's own copywriting was real human text.
    I state that plainly in the report, since it sharpens the question of what
    the mixture ratio should be.
    """
    if not p.get("title"):
        return []
    rng = _Rng(p.get("id") or p["title"])
    docs: list[dict[str, Any]] = []

    overview = [
        _opening(p, rng),
        _variants_sentence(p, rng),
        *_axis_sentences(p, rng),
        _price_sentence(p, rng),
        *_spec_sentences(p, rng),
        _stock_sentence(p, rng),
    ]
    overview = [s for s in overview if s]
    if len(overview) >= 2:
        docs.append({
            "text": " ".join(overview),
            "kind": "product_overview",
            "product_id": p["id"],
            "synthetic": True,
        })

    if p.get("description") and len(p["description"]) > 80:
        docs.append({
            "text": f"{p['title']}\n\n{p['description']}",
            "kind": "product_description",
            "product_id": p["id"],
            "synthetic": True,
        })

    lines = _variant_lines(p)
    if len(lines) > 1:
        docs.append({
            "text": f"Varianter av {p['title']}:\n" + "\n".join(f"• {l}" for l in lines),
            "kind": "variant_listing",
            "product_id": p["id"],
            "synthetic": True,
        })

    return docs
