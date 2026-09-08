"""
Synthetic Swedish bathroom catalogue.

No network, no database, no Medusa, no API keys. I built it this way so that
anyone reading the report can clone the repo, run `python -m data.generate` and
reproduce every number without first standing up a webshop.

I kept the output shape identical to what the Medusa extractor produced, so
nothing downstream knows or cares where the products came from. If I point this
at a real shop later, only the loader changes.

What comes out is a plausible catalogue with correct Swedish morphology and the
compound nouns that make the tokenizer experiment interesting:
tvättställsblandare, handdukstork, duschblandare, badrumsskåp, golvbrunn.

It is not a pretraining corpus. Every word here is machine-generated, so the
diversity ceiling is this file; adding products does not raise it. Real Swedish
text (FineWeb-2 sv, or a Nordic Pile extract) does the language learning and
this supplies the domain vocabulary. See README on mixture ratios and model
collapse.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ─── vocabulary ──────────────────────────────────────────────────────────────

# I use Swedish place names as model names. They read naturally as product
# names and they carry å/ä/ö, which the tokenizer experiment needs.
MODEL_NAMES = [
    "Nordic", "Arvika", "Kiruna", "Lysekil", "Malmö", "Umeå", "Åre", "Boden",
    "Falun", "Gävle", "Halmstad", "Kalmar", "Luleå", "Mora", "Nyköping", "Orsa",
    "Piteå", "Ronneby", "Sigtuna", "Torsby", "Uppsala", "Visby", "Ystad", "Åmål",
    "Öland", "Björkö", "Dalarö", "Ekerö", "Fårö", "Hjo", "Idre", "Järna",
    "Katrineholm", "Lidköping", "Motala", "Nora", "Osby", "Rättvik", "Säffle",
    "Tranås", "Vadstena", "Åtvidaberg", "Ödeshög", "Borås", "Enköping", "Filipstad",
]

FINISHES_METAL = ["Krom", "Borstad mässing", "Matt svart", "Borstad stål", "Kopparfärgad"]
FINISHES_WOOD = ["Vit", "Ek", "Valnöt", "Antracit", "Ljusgrå", "Matt svart"]

# Gender is carried explicitly. Swedish grammatical gender cannot be derived
# from the noun ("en blandare" but "ett duschset"), so anything that needs an
# article reads it from here instead of guessing. See render_sv.py.
UTRUM, NEUTRUM = "utrum", "neutrum"


@dataclass
class Family:
    """A product family: one noun, one set of option axes, one price band."""

    noun: str                      # the head noun, e.g. "Tvättställsblandare"
    gender: str                    # UTRUM or NEUTRUM
    plural: str                    # "tvättställsblandare" / "duschset"
    category: str
    material: str
    price_band: tuple[int, int]
    axes: list[tuple[str, list[str]]]
    dims_mm: tuple[int, int, int]  # nominal (length/depth, width, height)
    # Holds a short nominal phrase. I first reused a description sentence here
    # and got subtitles like "kan även monteras med tejp på kakel utan att
    # borra", where "även" refers to nothing because the sentence has been torn
    # out of its paragraph.
    subtitle: str = ""
    blurbs: list[str] = field(default_factory=list)
    # Multiplier applied per option value, so variants aren't all the same price.
    premium: dict[str, float] = field(default_factory=dict)


PREMIUM_METAL = {
    "Krom": 1.0, "Borstad stål": 1.10, "Matt svart": 1.18,
    "Borstad mässing": 1.28, "Kopparfärgad": 1.32,
}
PREMIUM_WOOD = {
    "Vit": 1.0, "Ljusgrå": 1.05, "Antracit": 1.12,
    "Ek": 1.15, "Valnöt": 1.24, "Matt svart": 1.12,
}
PREMIUM_WIDTH = {
    "400 mm": 0.85, "500 mm": 1.0, "600 mm": 1.15, "700 mm": 1.28,
    "800 mm": 1.4, "900 mm": 1.55, "1000 mm": 1.7, "1200 mm": 2.0,
}

FAMILIES: list[Family] = [
    Family(
        noun="Tvättställsblandare", gender=UTRUM, plural="tvättställsblandare",
        category="Blandare", material="Mässing", price_band=(895, 2600),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(140, 55, 165),
        subtitle="Enhandsblandare med keramisk insats",
        premium=PREMIUM_METAL,
        blurbs=[
            "Enhandsblandare med keramisk insats och mjukt reglage.",
            "Blandaren har en inbyggd flödesbegränsare som håller vattenförbrukningen nere "
            "utan att strålen känns svag.",
            "Kallstartsfunktionen gör att varmvattnet inte går igång i onödan när du bara "
            "sköljer händerna.",
            "Levereras med flexibla anslutningsslangar och komplett monteringssats.",
            "Perlatorn går att skruva loss för avkalkning utan verktyg.",
            "Tio års garanti på funktion, fem år på ytbehandling.",
        ],
    ),
    Family(
        noun="Duschblandare", gender=UTRUM, plural="duschblandare",
        category="Blandare", material="Mässing", price_band=(1295, 3900),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(180, 150, 90),
        subtitle="Termostatblandare för dusch",
        premium=PREMIUM_METAL,
        blurbs=[
            "Termostatblandare med säkerhetsspärr vid 38 grader.",
            "Termostaten håller temperaturen jämn även när någon spolar i köket.",
            "Avstängningsventilen sitter separat så du kan behålla din inställda temperatur.",
            "Anslutningarna har justerbart centrumavstånd, vilket underlättar vid renovering.",
            "Ytbehandlingen är tålig mot kalk och rengöringsmedel.",
        ],
    ),
    Family(
        noun="Badkarsblandare", gender=UTRUM, plural="badkarsblandare",
        category="Blandare", material="Mässing", price_band=(1695, 4500),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(200, 160, 110),
        subtitle="Badkarsblandare med omkastare",
        premium=PREMIUM_METAL,
        blurbs=[
            "Badkarsblandare med omkastare mellan kar och handdusch.",
            "Levereras med duschslang och handdusch med tre strålkast.",
            "Termostatfunktion med spärr för säker badtemperatur.",
        ],
    ),
    Family(
        noun="Handdukstork", gender=UTRUM, plural="handdukstorkar",
        category="Handdukstorkar", material="Rostfritt stål", price_band=(1995, 5200),
        axes=[("Bredd", ["500 mm", "600 mm", "750 mm"]), ("Finish", ["Krom", "Matt svart", "Vit"])],
        dims_mm=(120, 600, 1200),
        subtitle="Elektrisk handdukstork med timer",
        premium={**PREMIUM_WIDTH, "Krom": 1.0, "Matt svart": 1.12, "Vit": 1.0},
        blurbs=[
            "Elektrisk handdukstork som torkar handdukar snabbt och håller badrummet fritt från fukt.",
            "Den inbyggda timern gör att du kan ställa in torktid per dygn.",
            "IP44-klassad och avsedd för fast installation.",
            "Effekten ligger mellan 150 och 350 watt beroende på storlek.",
            "Går att komplettera med torrvärmepatron för drift sommartid.",
        ],
    ),
    Family(
        noun="Spegelskåp", gender=NEUTRUM, plural="spegelskåp",
        category="Badrumsskåp", material="Melamin", price_band=(2990, 8900),
        axes=[("Bredd", ["600 mm", "800 mm", "1000 mm"]), ("Kulör", FINISHES_WOOD)],
        dims_mm=(150, 800, 700),
        subtitle="Spegelskåp med LED-belysning",
        premium={**PREMIUM_WIDTH, **PREMIUM_WOOD},
        blurbs=[
            "Spegelskåp med integrerad LED-list och eluttag på insidan.",
            "Dörrarna är dubbelsidigt speglade och skåpet levereras färdigmonterat.",
            "Belysningen är dimbar och har en sensor som stänger av efter en stund.",
            "Hyllplanen i härdat glas går att flytta i höjdled.",
            "Fuktsäkrad konstruktion avsedd för våtrum.",
        ],
    ),
    Family(
        noun="Tvättställsskåp", gender=NEUTRUM, plural="tvättställsskåp",
        category="Badrumsskåp", material="Spånskiva", price_band=(3490, 11500),
        axes=[("Bredd", ["600 mm", "800 mm", "1000 mm", "1200 mm"]), ("Kulör", FINISHES_WOOD)],
        dims_mm=(460, 800, 550),
        subtitle="Tvättställsskåp med mjukstängande lådor",
        premium={**PREMIUM_WIDTH, **PREMIUM_WOOD},
        blurbs=[
            "Tvättställsskåp med mjukstängande lådor och tvättställ i komposit.",
            "Lådorna är fullt utdragbara och tål 25 kilo belastning.",
            "Vattenlåset är platsbesparande så att den övre lådan kan användas fullt ut.",
            "Väggmonteras med medföljande beslag, inga ben behövs.",
        ],
    ),
    Family(
        noun="Duschset", gender=NEUTRUM, plural="duschset",
        category="Duschset", material="Mässing", price_band=(2495, 7900),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(400, 250, 1100),
        subtitle="Takdusch med termostatblandare",
        premium=PREMIUM_METAL,
        blurbs=[
            "Komplett duschset med termostatblandare, takdusch och handdusch.",
            "Takduschen har antikalkmunstycken som torkas rena med fingret.",
            "Duschstången är höj- och sänkbar i steglös inställning.",
            "Går att montera på befintliga anslutningar vid renovering.",
        ],
    ),
    Family(
        noun="Duschvägg", gender=UTRUM, plural="duschväggar",
        category="Duschväggar", material="Härdat glas", price_band=(2295, 8500),
        axes=[("Bredd", ["700 mm", "800 mm", "900 mm", "1000 mm"]), ("Profil", ["Krom", "Matt svart", "Vit"])],
        dims_mm=(60, 900, 2000),
        subtitle="Duschvägg i härdat säkerhetsglas",
        premium={**PREMIUM_WIDTH, "Krom": 1.0, "Matt svart": 1.15, "Vit": 1.02},
        blurbs=[
            "Duschvägg i sex millimeter härdat säkerhetsglas.",
            "Glaset har en smutsavvisande ytbehandling som gör det lätt att torka av.",
            "Profilerna är justerbara, vilket tar upp mindre ojämnheter i väggen.",
            "Kan monteras för både höger- och vänsterhängning.",
        ],
    ),
    Family(
        noun="Toalettstol", gender=UTRUM, plural="toalettstolar",
        category="Toaletter", material="Porslin", price_band=(2790, 9500),
        axes=[("Utförande", ["Golvstående", "Vägghängd"]), ("Spolknapp", ["Krom", "Vit", "Matt svart"])],
        dims_mm=(670, 360, 800),
        subtitle="Spolkantlös toalettstol med mjukstängande sits",
        premium={"Golvstående": 1.0, "Vägghängd": 1.22, "Krom": 1.0, "Vit": 0.98, "Matt svart": 1.1},
        blurbs=[
            "Toalettstol med dubbelspolning och mjukstängande sits.",
            "Den spolkantlösa konstruktionen gör porslinet enklare att hålla rent.",
            "Sitsen lyfts av utan verktyg vid rengöring.",
            "Uppfyller kraven för snålspolande installation.",
        ],
    ),
    Family(
        noun="Tvättställ", gender=NEUTRUM, plural="tvättställ",
        category="Tvättställ", material="Porslin", price_band=(895, 4200),
        axes=[("Bredd", ["400 mm", "500 mm", "600 mm", "800 mm"])],
        dims_mm=(450, 600, 165),
        subtitle="Tvättställ i vitt porslin",
        premium=PREMIUM_WIDTH,
        blurbs=[
            "Tvättställ i vitt porslin med plats för blandare i mitten.",
            "Kan monteras på konsol eller ovanpå ett tvättställsskåp.",
            "Bräddavloppet är dolt för ett renare uttryck.",
        ],
    ),
    Family(
        noun="Golvbrunn", gender=UTRUM, plural="golvbrunnar",
        category="Golvbrunnar", material="Polypropen", price_band=(549, 1890),
        axes=[("Dimension", ["75 mm", "110 mm", "150 mm"])],
        dims_mm=(150, 150, 90),
        subtitle="Golvbrunn med klämring för tätskikt",
        premium={"75 mm": 0.9, "110 mm": 1.0, "150 mm": 1.25},
        blurbs=[
            "Golvbrunn med klämring för tätskikt, uppfyller kraven i BBV.",
            "Vattenlåset går att lyfta ur för rensning.",
            "Levereras med provplugg för täthetsprovning.",
        ],
    ),
    Family(
        noun="Badrumsspegel", gender=UTRUM, plural="badrumsspeglar",
        category="Speglar", material="Glas", price_band=(995, 4900),
        axes=[("Bredd", ["500 mm", "600 mm", "800 mm", "1000 mm"]), ("Belysning", ["Utan belysning", "LED runtom", "LED ovan"])],
        dims_mm=(40, 600, 800),
        subtitle="Badrumsspegel med antiduggfunktion",
        premium={**PREMIUM_WIDTH, "Utan belysning": 1.0, "LED runtom": 1.45, "LED ovan": 1.3},
        blurbs=[
            "Badrumsspegel med fasad kant och dold upphängning.",
            "Antiduggfunktionen håller spegeln klar efter duschen.",
            "Belysningen har färgtemperatur anpassad för sminkning.",
        ],
    ),
    Family(
        noun="Handdukshängare", gender=UTRUM, plural="handdukshängare",
        category="Badrumstillbehör", material="Mässing", price_band=(249, 1290),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(60, 600, 70),
        subtitle="Handdukshängare för väggmontage",
        premium=PREMIUM_METAL,
        blurbs=[
            "Handdukshängare för väggmontage, levereras med skruv och plugg.",
            "Kan även monteras med tejp på kakel utan att borra.",
        ],
    ),
    Family(
        noun="Toalettpappershållare", gender=UTRUM, plural="toalettpappershållare",
        category="Badrumstillbehör", material="Mässing", price_band=(199, 990),
        axes=[("Finish", FINISHES_METAL)],
        dims_mm=(70, 150, 90),
        subtitle="Toalettpappershållare med lock",
        premium=PREMIUM_METAL,
        blurbs=[
            "Toalettpappershållare med lock som håller rullen på plats.",
            "Enkel att montera på både kakel och gips.",
        ],
    ),
    Family(
        noun="Duschkabin", gender=UTRUM, plural="duschkabiner",
        category="Duschkabiner", material="Härdat glas", price_band=(6900, 18900),
        axes=[("Storlek", ["800×800 mm", "900×900 mm", "1000×800 mm"]), ("Profil", ["Krom", "Matt svart"])],
        dims_mm=(900, 900, 2000),
        subtitle="Duschkabin med skjutdörrar",
        premium={"800×800 mm": 1.0, "900×900 mm": 1.18, "1000×800 mm": 1.24, "Krom": 1.0, "Matt svart": 1.15},
        blurbs=[
            "Komplett duschkabin med skjutdörrar i härdat glas.",
            "Kabinen är förmonterad vilket kortar installationstiden.",
            "Magnetlisten sluter tätt och hindrar vatten från att rinna ut.",
        ],
    ),
]

# ─── generation ──────────────────────────────────────────────────────────────


def _ascii_fold(s: str) -> str:
    """
    å/ä/ö -> a/a/o.

    SKUs must be ASCII. They travel through URLs, CSV exports, order systems and
    (the reason it matters here) the instruction-tuning targets, where the model
    has to reproduce them character for character. A SKU like 'TVÄ-HJO-KRO' is a
    latent encoding bug and a harder token sequence to emit than it needs to be.
    """
    return s.translate(_TRANSLIT_SKU)


_TRANSLIT_SKU = str.maketrans({
    "å": "a", "ä": "a", "ö": "o", "Å": "A", "Ä": "A", "Ö": "O",
    "é": "e", "è": "e", "ü": "u", "×": "X",
})


def _sku_token(value: str) -> str:
    """Compress an option value into a SKU fragment: 'Borstad mässing' -> 'BMA'."""
    cleaned = "".join(c for c in _ascii_fold(value) if c.isalnum() or c.isspace())
    words = cleaned.split()
    if not words:
        return "STD"
    if len(words) == 1:
        w = words[0]
        return w[:4].upper() if w[0].isdigit() else w[:3].upper()
    return "".join(w[0] for w in words[:3]).upper()


# Option axes that govern a physical dimension. When a product varies along one
# of these, the product-level dimension is meaningless and must not be stated:
# a "Tvättställ Umeå" offered in 500/600/800 mm cannot also be described as
# "bredd 600 mm". Anyone demoing the bot hits that contradiction by asking
# "hur bred är den?".
AXIS_GOVERNS_DIMENSION = {
    "Bredd": ("width",),
    "Djup": ("length",),
    "Höjd": ("height",),
    "Storlek": ("width", "length"),
    "Dimension": ("width", "length", "height"),
}


def _round_price(x: float) -> int:
    """Retail prices land on 45/95 endings."""
    base = int(round(x / 10.0)) * 10
    return base - 5 if base % 100 in (0, 10, 20, 30, 40) else base + 5


def _cross(axes: list[tuple[str, list[str]]]) -> list[dict[str, str]]:
    combos: list[dict[str, str]] = [{}]
    for name, values in axes:
        combos = [{**c, name: v} for c in combos for v in values]
    return combos


def generate_catalogue(n_products: int = 120, seed: int = 20260824) -> list[dict[str, Any]]:
    """
    Build `n_products` products across the families above.

    Deterministic given `seed`. The corpus is an experimental input, so one that
    shifted between runs would leave every ablation result unusable.
    """
    rng = random.Random(seed)
    products: list[dict[str, Any]] = []
    used_names: set[tuple[str, str]] = set()

    for i in range(n_products):
        fam = FAMILIES[i % len(FAMILIES)]

        # Unique model name per family.
        for _ in range(200):
            model = rng.choice(MODEL_NAMES)
            if (fam.noun, model) not in used_names:
                used_names.add((fam.noun, model))
                break
        else:
            model = f"{rng.choice(MODEL_NAMES)} {i}"

        title = f"{fam.noun} {model}"
        handle = _slug(title)
        pid = f"prod_{i:04d}"

        base_price = rng.randint(*fam.price_band)

        # Not every product carries every option value. Real catalogues have
        # gaps, and a bot that never has to say "den finns inte i mässing" is
        # not being tested on anything. Roughly one product in eight here is a
        # single-SKU item with no choice to make; a bot that has never seen one
        # will happily invent options for a product that has none.
        single_sku = rng.random() < 0.12

        axes: list[tuple[str, list[str]]] = []
        for name, values in fam.axes:
            if single_sku:
                keep = [rng.choice(values)]
            elif len(values) <= 2:
                keep = values
            else:
                keep = rng.sample(values, rng.randint(2, len(values)))
            axes.append((name, [v for v in values if v in keep]))  # preserve catalogue order

        combos = _cross(axes)
        variants = []
        for j, combo in enumerate(combos):
            mult = 1.0
            for v in combo.values():
                mult *= fam.premium.get(v, 1.0)
            price = _round_price(base_price * mult)

            sku_bits = (
                [_ascii_fold(fam.noun)[:3].upper(), _ascii_fold(model)[:3].upper()]
                + [_sku_token(v) for v in combo.values()]
            )
            qty = rng.choice([0, 0, 1, 2, 3, 5, 8, 12, 18, 25, 40])

            variants.append({
                "id": f"variant_{pid}_{j}",
                "sku": "-".join(sku_bits),
                "title": " / ".join(combo.values()) or "Standard",
                "options": combo,
                "option_summary": ", ".join(f"{k}: {v}" for k, v in combo.items()) or None,
                "price": {
                    "amount": price,
                    "amount_with_tax": round(price * 1.25, 2),
                    "amount_without_tax": price,
                    "original_amount": price,
                    "currency_code": "sek",
                    "is_sale": False,
                    "source": "synthetic",
                },
                "inventory": {"quantity": qty, "manages_stock": True, "allow_backorder": False},
                "barcode": None,
                "weight_g": None,
            })

        n_blurbs = min(len(fam.blurbs), rng.randint(2, 4))
        blurbs = rng.sample(fam.blurbs, n_blurbs)
        description = " ".join(blurbs)

        # Suppress any product-level dimension that a variant axis governs, so
        # the catalogue never states a width the variants contradict. Only an
        # axis that actually varies creates that contradiction; a single-SKU
        # product pinned to one width can state its width safely.
        governed: set[str] = set()
        for name, values in axes:
            if len(values) > 1:
                governed.update(AXIS_GOVERNS_DIMENSION.get(name, ()))

        length, width, height = fam.dims_mm
        dims = {"length": length, "width": width, "height": height}
        dims = {k: v for k, v in dims.items() if k not in governed}

        prices = [v["price"]["amount"] for v in variants]

        products.append({
            "id": pid,
            "handle": handle,
            "title": title,
            "subtitle": fam.subtitle,
            "description": description,
            "material": fam.material,
            "type": "Badrumsprodukt",
            "gender": fam.gender,          # for the renderer; absent from Medusa
            "noun": fam.noun,
            "collection": {"id": "col_01", "title": "Badrum 2026"},
            "categories": [{"id": _slug(fam.category), "name": fam.category, "handle": _slug(fam.category)}],
            "tags": ["badrum", _slug(fam.category)],
            "option_axes": [{"name": n, "values": v} for n, v in axes if v],
            "dimensions_mm": dims or None,
            "variants": variants,
            "variant_count": len(variants),
            "price_range": {"min": min(prices), "max": max(prices), "currency": "sek"},
            "in_stock": any(v["inventory"]["quantity"] > 0 for v in variants),
            "thumbnail": None,
            "images": [],
            "_source_version": "synthetic",
        })

    return products


def flatten_variants(product: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per variant. This is what grounds the synthetic instruction data."""
    out = []
    for v in product["variants"]:
        out.append({
            "variant_id": v["id"],
            "product_id": product["id"],
            "product_title": product["title"],
            "product_handle": product["handle"],
            "sku": v["sku"],
            "options": v["options"],
            "option_summary": v["option_summary"],
            "price": v["price"]["amount"],
            "currency": v["price"]["currency_code"],
            "in_stock": v["inventory"]["quantity"] > 0,
            "quantity": v["inventory"]["quantity"],
            "categories": [c["name"] for c in product["categories"]],
            "collection": product["collection"]["title"] if product["collection"] else None,
        })
    return out


_TRANSLIT = str.maketrans({"å": "a", "ä": "a", "ö": "o", "Å": "a", "Ä": "a", "Ö": "o", "×": "x"})


def _slug(s: str) -> str:
    s = s.translate(_TRANSLIT).lower()
    return "".join(c if c.isalnum() else "-" for c in s).strip("-").replace("--", "-")
