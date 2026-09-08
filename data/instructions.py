"""
Instruction data for Lab 3, generated from the catalogue.

    python -m data.instructions                    # -> data/out/instructions_*.jsonl
    python -m data.instructions --per-product 9    # every applicable task

Every pair is grounded: the answer is computed from the catalogue, never written
by hand and never produced by another model. Each pair therefore carries its own
gold answer in machine-checkable form, which is why Lab 4 costs almost nothing.
Evaluation is exact-match arithmetic, with no LLM-as-judge and no API key for the
examiner to configure.

I chose the ten tasks so the metric means something different in each:

    price           look up one number
    variant_count   count a set, which is what a language model cannot do
    variant_list    reproduce a set without inventing a member (single-axis)
    axis_values     one axis of a multi-axis product, never the cross product
    stock           a yes/no grounded in inventory
    option_check    yes or no to an option, half of which do not exist
    sku             copy an identifier exactly
    cheapest        argmin over variants, a search rather than a lookup
    category        a one-word fact
    add_to_cart     emit a structured call with the right arguments

`variant_count` and `option_check` are the two that carry the report. A model
that has read fluent Swedish product prose will happily write "finns i fem
utföranden" above a list of four, and will happily agree that a tap comes in
brass because taps generally do. Both failures are invisible in perplexity and
obvious here.

The split is by product, never by pair. Splitting pairs at random puts "vad
kostar X?" in train and "vilka utföranden har X?" in validation. The model has
then already been shown X's variant table during training, so validation
measures recall of a memorised fact and reports it as generalisation. Holding
out whole products asks the question the report wants to answer.

The held-out product ids go to `holdout_products.json` so pretraining can
exclude the same products (`train.pretrain --exclude-products`). Without that
flag the held-out products still appear in the pretraining corpus, and the
honest claim shrinks to "generalises the instruction format to products it was
never instruction-tuned on", a real result but a smaller one. Say which one you
ran.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .render_sv import _Rng as Rng, count_word, inline, money, sv_list, sv_number

# ─── wire format ─────────────────────────────────────────────────────────────
#
# One place, imported by the fine-tuner, the evaluator and the demo. If these
# three ever disagree about where the answer starts, the loss mask silently
# covers the wrong tokens and the model trains on predicting its own prompt.

Q_PREFIX = "### Fråga:\n"
A_PREFIX = "\n\n### Svar:\n"


def format_prompt(question: str) -> str:
    """The full prefix the model conditions on. Loss is masked over this span."""
    return f"{Q_PREFIX}{question}{A_PREFIX}"


# ─── answer normalisation (shared with the evaluator) ────────────────────────

_NUMWORD = {
    "noll": 0, "en": 1, "ett": 1, "två": 2, "tre": 3, "fyra": 4, "fem": 5,
    "sex": 6, "sju": 7, "åtta": 8, "nio": 9, "tio": 10, "elva": 11, "tolv": 12,
}
_NUM_RE = re.compile(r"\d[\d ]*\d|\d")


def extract_numbers(text: str) -> list[int]:
    """
    Every number in a Swedish answer, as ints.

    Two shapes have to survive: "1 535 kr" (space is the Swedish thousands
    separator, and I keep it a plain U+0020 to save this function a special case)
    and "fem utföranden" (the corpus spells small numbers as words, so the
    fine-tuned model does too). Missing the second shape would score every
    correct count as wrong.
    """
    out = [int(m.group().replace(" ", "")) for m in _NUM_RE.finditer(text)]
    for w in re.findall(r"[a-zåäö]+", text.lower()):
        if w in _NUMWORD:
            out.append(_NUMWORD[w])
    return out


def normalise(s: str) -> str:
    return " ".join(str(s).lower().split())


# ─── catalogue helpers ───────────────────────────────────────────────────────


def _price(v: dict) -> int | None:
    p = v.get("price") or {}
    return p.get("amount")


def _in_stock(v: dict) -> bool:
    inv = v.get("inventory") or {}
    if not inv.get("manages_stock", True):
        return True
    return (inv.get("quantity") or 0) > 0


def _pron(p: dict) -> str:
    """Swedish gender comes from the product record. I never guess it."""
    return "Det" if p.get("gender") == "neutrum" else "Den"


def _vphrase(v: dict) -> str:
    return inline(v["title"])


def axis_pool(products: list[dict]) -> dict[str, set[str]]:
    """Every value each axis takes anywhere in the catalogue; the option_check source."""
    pool: dict[str, set[str]] = defaultdict(set)
    for p in products:
        for a in p.get("option_axes", []):
            pool[a["name"]].update(a["values"])
    return pool


# ─── task generators ─────────────────────────────────────────────────────────
#
# Each returns zero or one record. `gold` is what the evaluator compares against:
#   number  -> gold value must appear among the numbers in the answer
#   set     -> set-F1 over the listed items
#   bool    -> the answer must open with Ja / Nej accordingly
#   text    -> normalised substring match
#   call    -> parsed add_to_cart(...) arguments must match exactly


def _rec(p, task, question, answer, gold) -> dict:
    return {
        "task": task,
        "product_id": p["id"],
        "product_title": p["title"],
        "question": question,
        "answer": answer,
        "gold": gold,
    }


def t_price(p, rng, ctx):
    vs = [v for v in p["variants"] if _price(v) is not None]
    if not vs:
        return None
    v = vs[int(rng.next() * len(vs)) % len(vs)]
    q = rng.pick([
        f"Vad kostar {p['title']} i {_vphrase(v)}?",
        f"Hur mycket kostar {p['title']} i {_vphrase(v)}?",
        f"Vilket pris har {p['title']} i {_vphrase(v)}?",
    ])
    a = f"{p['title']} i {_vphrase(v)} kostar {money(_price(v))}."
    return _rec(p, "price", q, a, {"type": "number", "value": _price(v)})


def t_variant_count(p, rng, ctx):
    n = len(p["variants"])
    q = rng.pick([
        f"Hur många utföranden finns det av {p['title']}?",
        f"Hur många varianter finns av {p['title']}?",
        f"I hur många utföranden finns {p['title']}?",
    ])
    if n == 1:
        a = f"{p['title']} finns i ett enda utförande."
    else:
        a = f"{p['title']} finns i {count_word(n, 'neutrum')} utföranden."
    return _rec(p, "variant_count", q, a, {"type": "number", "value": n})


def t_variant_list(p, rng, ctx):
    """
    Single-axis products only.

    On a multi-axis product the full variant list is the cross product: eighteen
    entries of "600 mm / Vit, 600 mm / Ek, ..." for one mirror cabinet. Nobody
    answers a customer that way, and as a training target it teaches long
    enumeration, which is where a language model loses count. Multi-axis products
    get `axis_values` instead, one axis at a time.
    """
    if len(p.get("option_axes", [])) != 1 or len(p["variants"]) < 2:
        return None
    titles = [v["title"] for v in p["variants"]]
    q = rng.pick([
        f"Vilka utföranden finns av {p['title']}?",
        f"Vilka varianter kan jag välja mellan för {p['title']}?",
        f"Vad finns {p['title']} i för utföranden?",
    ])
    a = f"{p['title']} finns i {sv_list([inline(t) for t in titles])}."
    return _rec(p, "variant_list", q, a,
                {"type": "set", "value": [normalise(t) for t in titles]})


def t_axis_values(p, rng, ctx):
    """
    One axis of a multi-axis product.

    I left the phrasing uninflected ("Vilka alternativ finns för Bredd" instead
    of "Vilka bredder") for the same reason the corpus never guesses grammatical
    gender: the axis name comes from the catalogue, its plural does not, and
    inventing one puts confidently wrong Swedish into the training data.
    """
    axes = p.get("option_axes", [])
    if len(axes) < 2:
        return None
    ax = axes[int(rng.next() * len(axes)) % len(axes)]
    q = rng.pick([
        f"Vilka alternativ finns för {ax['name']} på {p['title']}?",
        f"Vad kan jag välja för {ax['name']} på {p['title']}?",
    ])
    a = f"För {ax['name']} finns {sv_list([inline(v) for v in ax['values']])}."
    return _rec(p, "axis_values", q, a,
                {"type": "set", "value": [normalise(v) for v in ax["values"]]})


def t_stock(p, rng, ctx):
    """
    In stock or not.

    When a product has an out-of-stock variant, I ask about that one. Picking a
    variant at random gives whatever balance the catalogue happens to have, 83%
    "Ja", and on such a metric a model that answers "Ja" every time scores 83%.
    Only 55 of the 120 products carry an out-of-stock variant, so choosing the
    answer first and then preferring the harder case brings the split to roughly
    45/55 without discarding a single pair. My other option was to coin-flip the
    polarity and skip the product when it cannot supply that answer; that lands
    at 69/31 and throws pairs away. `evals.run` reports the majority-class
    baseline per task, so the residual imbalance sits next to the score.
    """
    vs = p["variants"]
    out_of_stock = [v for v in vs if not _in_stock(v)]
    matching = out_of_stock or vs
    v = matching[int(rng.next() * len(matching)) % len(matching)]
    q = rng.pick([
        f"Finns {p['title']} i {_vphrase(v)} i lager?",
        f"Har ni {p['title']} i {_vphrase(v)} på lager?",
    ])
    if _in_stock(v):
        a = f"Ja, {p['title']} i {_vphrase(v)} finns i lager."
    else:
        a = f"Nej, {p['title']} i {_vphrase(v)} är slut i lager."
    return _rec(p, "stock", q, a, {"type": "bool", "value": _in_stock(v)})


def t_option_check(p, rng, ctx):
    """
    The hallucination probe: does this product come in X?

    I draw the intended answer first and then look for a value that fits it. Half
    the time X is an option the product really offers; half the time it is one
    that another product in the catalogue offers, so it is plausible and cannot
    be ruled out on vocabulary alone. Both halves are needed. My first version
    asked only about options the product lacks, which made "Nej" a perfect
    answer: a model that had learned to refuse everything would have scored 100%
    and looked careful.
    """
    axes = p.get("option_axes", [])
    if not axes:
        return None

    missing = []
    for a in axes:
        have = set(a["values"])
        for m in sorted(ctx["pool"][a["name"]] - have):
            missing.append((m, sorted(have)))
    offered = [(v, sorted(a["values"])) for a in axes for v in a["values"]]

    want_missing = rng.next() < 0.5
    pool = missing if (want_missing and missing) else offered
    value, have = pool[int(rng.next() * len(pool)) % len(pool)]
    exists = value in have

    q = rng.pick([
        f"Finns {p['title']} i {inline(value)}?",
        f"Kan jag få {p['title']} i {inline(value)}?",
    ])
    if exists:
        a = f"Ja, {p['title']} finns i {inline(value)}."
    else:
        a = (f"Nej, {p['title']} finns inte i {inline(value)}. "
             f"{_pron(p)} finns i {sv_list([inline(h) for h in have])}.")
    return _rec(p, "option_check", q, a, {"type": "bool", "value": exists})


def t_sku(p, rng, ctx):
    vs = [v for v in p["variants"] if v.get("sku")]
    if not vs:
        return None
    v = vs[int(rng.next() * len(vs)) % len(vs)]
    q = rng.pick([
        f"Vilket artikelnummer har {p['title']} i {_vphrase(v)}?",
        f"Vad är artikelnumret för {p['title']} i {_vphrase(v)}?",
    ])
    a = f"Artikelnumret för {p['title']} i {_vphrase(v)} är {v['sku']}."
    return _rec(p, "sku", q, a, {"type": "text", "value": v["sku"]})


def t_cheapest(p, rng, ctx):
    vs = [v for v in p["variants"] if _price(v) is not None]
    if len(vs) < 2:
        return None
    v = min(vs, key=_price)
    q = rng.pick([
        f"Vilket är det billigaste utförandet av {p['title']}?",
        f"Vad kostar {p['title']} som billigast?",
    ])
    a = f"Det billigaste utförandet av {p['title']} är {_vphrase(v)}, {money(_price(v))}."
    return _rec(p, "cheapest", q, a, {"type": "number", "value": _price(v)})


def t_category(p, rng, ctx):
    cats = p.get("categories") or []
    if not cats:
        return None
    c = cats[0]["name"]
    q = rng.pick([
        f"Vilken kategori tillhör {p['title']}?",
        f"Var hittar jag {p['title']} i sortimentet?",
    ])
    a = f"{p['title']} tillhör kategorin {inline(c)}."
    return _rec(p, "category", q, a, {"type": "text", "value": c})


def t_add_to_cart(p, rng, ctx):
    """
    The task that connects this course model to the webshop it is named after.

    The answer is a call and it is graded on its arguments. The model has to pick
    the right SKU and quantity; a string that only sounds like a confirmation
    scores zero.
    """
    vs = [v for v in p["variants"] if v.get("sku")]
    if not vs:
        return None
    v = vs[int(rng.next() * len(vs)) % len(vs)]
    n = 1 + int(rng.next() * 3) % 3
    q = rng.pick([
        f"Lägg {n} st {p['title']} i {_vphrase(v)} i varukorgen.",
        f"Jag vill beställa {n} st {p['title']} i {_vphrase(v)}.",
    ])
    a = f'add_to_cart(sku="{v["sku"]}", antal={n})'
    return _rec(p, "add_to_cart", q, a,
                {"type": "call", "value": {"sku": v["sku"], "antal": n}})


TASKS = [t_price, t_variant_count, t_variant_list, t_axis_values, t_stock,
         t_option_check, t_sku, t_cheapest, t_category, t_add_to_cart]

CALL_RE = re.compile(r'add_to_cart\(\s*sku\s*=\s*"([^"]*)"\s*,\s*antal\s*=\s*(\d+)\s*\)')


def parse_call(text: str) -> dict | None:
    m = CALL_RE.search(text)
    return {"sku": m.group(1), "antal": int(m.group(2))} if m else None


# ─── build ───────────────────────────────────────────────────────────────────


def build(products: list[dict], per_product: int | None = 7) -> list[dict]:
    """
    Generate pairs, deterministically.

    `per_product` subsamples the applicable tasks so the set stays around 800
    pairs at 120 products. The subsample is seeded from the product id, so it is
    stable across runs; a training set that reshuffled between runs would make
    the Experiment 2 comparison meaningless.
    """
    ctx = {"pool": axis_pool(products)}
    out: list[dict] = []
    for p in products:
        rng = Rng("instr:" + p["id"])
        made = [r for t in TASKS if (r := t(p, rng, ctx)) is not None]
        if per_product is not None and len(made) > per_product:
            # Rotate the starting point by product so every task type stays
            # represented across the catalogue instead of always dropping the last.
            start = int(rng.next() * len(made)) % len(made)
            made = [made[(start + i) % len(made)] for i in range(per_product)]
        out.extend(made)
    return out


def split_by_product(records: list[dict], val_fraction: float = 0.15, seed: int = 0):
    """Whole products go to one side or the other. See the module docstring."""
    ids = sorted({r["product_id"] for r in records})
    rng = Rng(f"split:{seed}")
    order = sorted(ids, key=lambda i: Rng(f"{seed}:{i}").next())
    n_val = max(1, int(len(order) * val_fraction))
    val_ids = set(order[:n_val])
    train = [r for r in records if r["product_id"] not in val_ids]
    val = [r for r in records if r["product_id"] in val_ids]
    return train, val, sorted(val_ids)


def write_jsonl(path: Path, rows: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", type=Path, default=Path("data/out/catalogue.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/out"))
    ap.add_argument("--per-product", type=int, default=7,
                    help="cap on pairs per product; 0 = all applicable tasks")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not args.catalogue.exists():
        print(f"{args.catalogue} not found. Run `python -m data.generate` first.")
        return 1

    products = [json.loads(l) for l in args.catalogue.open(encoding="utf-8") if l.strip()]
    records = build(products, args.per_product or None)
    train, val, val_ids = split_by_product(records, args.val_fraction, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "instructions_train.jsonl", train)
    write_jsonl(args.out / "instructions_val.jsonl", val)
    (args.out / "holdout_products.json").write_text(
        json.dumps({"val_product_ids": val_ids, "seed": args.seed,
                    "val_fraction": args.val_fraction}, indent=2), encoding="utf-8")

    by_task: dict[str, int] = defaultdict(int)
    for r in records:
        by_task[r["task"]] += 1

    print(f"products    {len(products)}")
    print(f"pairs       {len(records)}  ({len(train)} train / {len(val)} val)")
    print(f"held out    {len(val_ids)} products, entirely")
    print()
    for t in TASKS:
        name = t.__name__[2:]
        print(f"  {name:<15} {by_task.get(name, 0):>4}")
    print(f"\nwrote {args.out}/instructions_{{train,val}}.jsonl "
          f"and holdout_products.json")

    example = train[0]
    print("\nexample\n" + "─" * 58)
    print(format_prompt(example["question"]) + example["answer"])
    print("─" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
