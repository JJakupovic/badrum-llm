# badrum-llm

A Swedish bathroom-retail LLM, built from scratch for *Understanding and Building
Large Language Models* (LiU VT26).

No services, no API keys, no database. Clone it and run:

```bash
python -m data.generate                          # catalogue + Swedish corpus
python -m data.instructions                      # 840 instruction pairs
python tests/test_data.py                        # 21 checks
python tests/test_model.py                       # 26 checks
python tests/test_instructions.py                # 18 checks
python tests/test_evals.py                       # 11 checks
python -m evals.run --baselines-only             # the floors, no torch needed
python -m train.pretrain --preset debug --context-length 256 --steps 120 --device cpu
python -m train.finetune --init runs/debug-bytes/ckpt_best.pt --epochs 3 --device cpu
python -m evals.run --ckpt runs/ft-debug-bytes/ckpt_best.pt --baselines
```

That's the whole setup. I wanted the examiner to be able to reproduce every number
in the report without standing up a webshop first.

`FINDINGS.md` has the results and what I think they mean.

## Why this project

I am building a webshop for bathroom products in [Medusa](https://medusajs.com/),
and I wanted the course project to run against it rather than against something
invented for the occasion. That gave me a domain I already know, a catalogue with
real structure, and an end goal I actually want: an assistant that can answer
"vilka utföranden finns?" and then put the right variant in the cart.

It also decided the data model. In Medusa a product owns variants, and variants
own option values, so "which variants exist" is a query rather than something the
model has to infer. Everything in `data/catalogue.py` mirrors that shape:
products with option axes, variants carrying a price and an inventory count, and
gaps in option coverage so some combinations do not exist. The Node extractor I
wrote against the live Store API emits the same shape, so swapping the synthetic
catalogue for the real one changes a loader and nothing else. That extractor
belongs to the product track and is kept out of this repository, since it is not
part of what the examination asks for.

The project has two tracks and they need different things.

**The course track** is this repository: a small GPT trained from scratch,
instruction-tuned on the catalogue, evaluated. It is what the examination asks
for.

**The product track** is the assistant that will actually ship: a hosted
instruction-tuned model with retrieval over the catalogue and tool calls into the
Store API. The reason those cannot be the same thing is in `FINDINGS.md` under
Lab 2. A model that writes fluent Swedish still cannot count its own list, so
variant counts, prices and stock have to come from a query with the model
choosing the arguments. Finding that out on a model I built myself is more
convincing than reading it.

## Use of AI assistance

I used an AI assistant (Claude) while building this project. What that covered:

- **Design discussion.** Talking through what to include and what to cut, how to
  frame the two experiments, and whether a given measurement actually showed what
  I thought it showed.
- **Drafting code and documentation.** A large part of the code and of the
  Markdown in this repository was drafted with the assistant and then reviewed,
  run and corrected by me.
- **Writing tests.** Several of those tests caught real problems in my own design.
  The task that scored 100% for answering "Nej", described in `FINDINGS.md`, is
  the clearest example.

The decisions are mine. Dropping the Medusa integration for the course track,
leaving the Swedish web-crawl mixing out of scope, choosing model scale and
pretraining benefit as the two experiments, and every judgement in `FINDINGS.md`
about what the numbers mean. I have run everything in this repository and can
explain and defend any part of it.

## Status

| Lab | Part | State |
|---|---|---|
| 1 | Data pre-processing pipeline | done: catalogue, corpus, instruction pairs. Web-crawl mixing is out of scope, see below |
| 2 | Pre-train a GPT-style LLM | done: model, training loop, checkpointing, sampling |
| 3 | Fine-tune | done: 840 instruction pairs, prompt-masked loss, scratch control |
| 4 | Evaluate | done: task accuracy on 10 tasks, two baselines, no judge model |

Experiment 1 is a model-scale sweep. Experiment 2 is `--init` against
`--scratch --like` in `train.finetune`: identical architecture, data, schedule and
seed, differing only in whether the weights start pretrained. Both run from one
script:

```bash
bash run_experiments.sh                  # GPU if available
bash run_experiments.sh --device cpu     # overnight on a laptop
```

It generates the data, runs every stage, and writes `RESULTS.md` with the tables.
Each stage is skipped if its checkpoint already exists, so an interrupted run
continues where it stopped. `RUNBOOK.md` has the details and the timings.

The tokenizer-fertility comparison in `data/tokenizer.py` is future work. It would
need a BPE trainer written from scratch and a full retrain of everything
downstream, which is more than the examination asks for.

## Why the data is synthetic

Given that the whole point was to build against my own shop, the obvious move is
to pull the real catalogue from its backend. I wrote the extractor to do exactly
that and then did not use it for the course track.

A real shop adds a service to install, seed and keep running, and gives back data
I have less control over. I can't ask a live catalogue for a clean distribution of
option axes or a fixed random seed. The synthetic generator gives me
reproducibility and a catalogue shaped to exercise the cases the bot actually has
to handle.

The output shape is identical to the extractor's, so swapping the source later
changes one loader.

## What it generates

```
data/out/
  catalogue.jsonl   one row per product, variants nested
  variants.jsonl    one flat row per variant, grounds the instruction data
  corpus_sv.jsonl   2-3 Swedish documents per product
  stats.json        counts + sanity checks, keep this alongside your results
```

120 products, 548 variants and about 18k words by default. `--products 400` for
more, `--seed N` for a different catalogue.

## The thing to be honest about

Every word of this corpus is machine-generated. Its linguistic variety is bounded
by `render_sv.py`, not by the product count: generating 10 000 products doesn't
give me 10 000 products' worth of variety. Lecture 3 covers model collapse under
synthetic-heavy training directly.

So this isn't a pretraining corpus on its own. It supplies domain vocabulary;
real Swedish text would supply language. The mixture would look like:

```
FineWeb-2 (sv)  →  filter  →  dedup  ─┐
                                      ├─→  mix at ratio R  →  tokenize  →  train
corpus_sv.jsonl (no dedup)  ──────────┘
```

Deduplicate before mixing. Templated text is near-duplicate by construction, so a
MinHash pass over the combined corpus deletes almost all the domain data and you
spend an evening working out where it went.

I'm not doing that mixing in this repo, and that's a scope decision rather than an
oversight. Downloading, filtering and deduplicating a Swedish web crawl is the
largest remaining piece of Lab 1 and the examination doesn't ask for it. What the
fully synthetic corpus gives me instead is a measurement: a validation perplexity
of 1.3 on the first run says the templated text is close to memorisable, which is
lecture 3's argument with a number attached. The mixing ratio stays a config value
so the experiment is one flag away for anyone who wants it.

## Swedish decisions

The corpus is training data, so a grammatical error here is a pattern the model
learns. Three choices that came out of reading the generated output rather than
from theory:

**Gender is stored, never inferred.** Swedish grammatical gender isn't derivable
from the noun: *en blandare* but *ett duschset*. Each product family carries its
gender explicitly so the article templates are safe. Reading from a real shop,
where gender is unknown, I'd drop those templates rather than guess.

**Multi-axis products don't enumerate the cross product.** Option values are
themselves comma-joined, so naming every combination inline produces
`Bredd: 500 mm, Finish: Krom, Bredd: 500 mm, …` where the commas collide and the
sentence can't be parsed. Multi-axis products state their axes; the variant
listing carries the enumeration.

**Prices use a regular space, not U+00A0.** A non-breaking space is
typographically correct in Swedish and invisible in every editor, so it silently
breaks exact-match scoring of prices and becomes a strange token. A test asserts
it.

## Catalogue realism, on purpose

- **Single-SKU products** (~13%). A model that has never seen one will invent
  options for products that have none.
- **Out-of-stock variants.** *"Finns den i lager?"* is a real user turn.
- **Gaps in option coverage.** Not every product carries every finish, so the bot
  sometimes has to say *"den finns inte i mässing"*.
- **No dimension contradicts a varying axis.** A product offered in 500/600/800 mm
  never also claims *bredd 600 mm*. Someone demoing this will ask *"hur bred är
  den?"*, and a contradiction there is the demo everyone remembers.

## Lab 3: instruction tuning

```bash
python -m data.instructions                                    # -> data/out/instructions_*.jsonl
python -m train.finetune --init runs/tiny-bytes/ckpt_best.pt   # treatment
python -m train.finetune --scratch --like runs/tiny-bytes/ckpt_best.pt   # control
```

Ten task types, each asking for something different. `price` and `sku` are
lookups, `variant_count` is a count, `variant_list` and `axis_values` reproduce a
set, `option_check` needs a yes or a no about an option that half the time doesn't
exist, `cheapest` is an argmin, and `add_to_cart` emits a structured call graded on
its arguments.

Two properties do most of the work.

Every pair carries its own gold answer in a form a program can check, because the
answers are computed from the catalogue. That's what makes Lab 4 arithmetic rather
than an LLM-as-judge: nothing to configure, no API key, no second model.

The split is by product, never by pair. Splitting pairs at random puts *"vad
kostar X?"* in train and *"vilka utföranden har X?"* in validation, so validation
measures recall of a fact the model was shown in the training half and reports it
as generalisation. `holdout_products.json` records the held-out ids so pretraining
can exclude the same products.

Loss is computed on answer tokens only. That mask is the same shape of problem as
the causal mask: off by one and the model trains on predicting the last token of
its own prompt, nothing crashes, and the loss curve looks slightly better.
`tests/test_instructions.py` checks it by reconstruction, decoding the supervised
positions and requiring the answer back, then shifts the boundary in both
directions to confirm the check can fail.

## Lab 4: evaluation

```bash
python -m evals.run --baselines-only                                   # floors only
python -m evals.run --ckpt runs/ft-tiny-bytes/ckpt_best.pt --baselines
```

Writes `eval.json` beside the checkpoint with every prediction, so the table can
be rebuilt without re-running the model.

Lecture 7's framing, made concrete:

| | |
|---|---|
| subject | the model's greedy continuation of a held-out prompt |
| criteria | per task: one number, one set, one boolean, one call |
| reference | computed from the catalogue |
| method | exact match, set F1, argument equality. No judge model |

The two baselines aren't decoration. An accuracy of 0.6 is good if a constant
answer scores 0.2 and bad if it scores 0.58, and the number alone doesn't say
which.

`majority` answers the most common gold value for the task every time. It earned
its place immediately: `option_check` originally asked only about options a
product didn't have, so "Nej" scored 100% and the task measured nothing. The
baseline is what showed me. Both boolean tasks now draw their answer before
finding a case to fit it, and the majority class sits near 58%.

`retrieval` answers with the gold answer of the most similar training question of
the same task, by word overlap. Because the split is by product that answer always
names a different product, so this baseline scores well where a task can be
answered without knowing which product was asked about. On the current held-out
set it reaches 0.71 on `category` and 0.79 on `option_check`, which I want to know
before quoting a model score on either.

Current floors over 126 held-out pairs, macro-averaged so a task with 102 pairs
doesn't outvote one with 48:

| | majority | retrieval |
|---|---|---|
| macro avg | 0.25 | 0.30 |
| exact match | 0.00 | 0.01 |

Two caveats for the report. The scores are containment checks: `price` counts as
correct if the gold amount appears anywhere in the answer, which inflates. The
strict full-string match is reported next to it, and when the two diverge sharply
the loose one is being gamed. Decoding is greedy by default because a reported
number has to be reproducible; `--temperature` and `--top-k` show how far the
metric moves, which is lecture 6 in one line of output.

## The model

A decoder-only transformer following Raschka ch. 3-4. `LayerNorm`, `GELU`, causal
multi-head self-attention and the residual block are all implemented here. Only
nn.Linear, nn.Embedding and the optimiser come from torch.

```bash
python -m train.pretrain --preset tiny --steps 4000
python -m train.pretrain --resume runs/tiny-bytes/ckpt_last.pt    # after a reclaim
python -m train.sample --ckpt runs/tiny-bytes/ckpt_best.pt --compare
```

Presets: `debug` (0.1M, seconds), `tiny` (~11M), `small` (~30M), `base` (GPT-2
small geometry). Architecture is overridable from the CLI. Experiment 1 sweeps it
while everything else is held fixed and Experiment 2 holds it fixed while the
initialisation varies, and neither should mean editing a preset.

Choices I can defend in the report:

- **Pre-norm.** Leaves an unobstructed additive path from input to output, so
  gradients reach the early layers and deep stacks train without a
  warmup-dependent knife edge.
- **Weight tying** between the token embedding and the output head. At this scale
  the embedding table would otherwise be most of the model.
- **Scaled init on residual projections** (`0.02/√(2·n_layers)`). Every block adds
  into the residual stream, and without this its variance grows with depth.
- **Warmup then cosine decay.** Adam's second-moment estimate is unreliable for
  the first few hundred steps, and a full learning rate there can wreck the model
  before the optimiser has any sense of gradient scale.
- **No weight decay on 1-D parameters.** Decaying LayerNorm gains and biases
  shrinks learned scales toward zero for no benefit.
- **bf16 over fp16.** Same exponent range as fp32, so there's no loss scaling and
  no silent overflow to NaN six hours into a run.

### The causal mask is the bug that hides

If position *t* can attend to *t+1*, the model reads the answer it's being trained
to predict. Training loss then drops faster than it should and the curves look
like success. `tests/test_model.py` asserts the property directly by poking the
token at position *t* and requiring every output before *t* to come back
bit-identical, rather than eyeballing the mask. Both the explicit and the fused
attention paths are checked and asserted numerically equivalent.

I verified the test against deliberately broken masks (`diagonal=2`,
`is_causal=False`) and both mutants fail it. A test that can't fail tells me
nothing.

## What a first run showed

1.9M params, byte tokenizer, ~386k tokens, ~10 epochs, CPU, 9 minutes:

```
step   200   loss 2.3675   val 1.2214
step   600   loss 0.3929   val 0.3415
step  1200   loss 0.1669   val 0.2289    perplexity 1.3
```

A perplexity of 1.3 isn't a good model. It's evidence that the corpus is close to
memorisable: templated text has very low entropy, so the model reproduces
templates. That's the model-collapse argument from lecture 3 with a number
attached, and the empirical case for mixing in real Swedish.

The generated Swedish is fluent and the failure modes are instructive:

```
top-k 40    …finns i fyra varianter: krom, borstad mässing, matt svart
             och borstad stål.                              ← says four, lists four
nucleus 0.9 …finns i fem varianter: krom, borstad mässing,
             borstad stål och kopparfärgad.                 ← says five, lists four
```

Fluent Swedish, wrong arithmetic. A language model has no mechanism for counting
its own list, and I don't think more pretraining fixes it. This is the concrete
argument for the tool layer in the product track: the variant count has to come
from a query.

## RunPod

```bash
bash runpod/setup.sh
```

Checks the GPU, warns if `/workspace` isn't a real network volume, installs
requirements, generates the dataset, runs the tests.

Attach a network volume before you deploy the pod. Without one, `/workspace` dies
with the pod and takes your checkpoints with it. Storage is billed while the pod
is stopped, which is the point: a few GB is cents a month and it means you can
stop the GPU between runs.

Stop the pod when a run finishes. Billing is per second and an idle GPU costs what
a busy one does.

For a model in the tens of millions of parameters, 24 GB is plenty. Community-cloud
RTX A5000, RTX 3090 and L4 are all in the $0.27-0.50/hr range and any of them will
do the pretraining runs. An A100 80 GB (~$1.39/hr) only earns its price if you go
much larger or want the Experiment 1 runs finished sooner. Community cloud is
cheaper and can be reclaimed, so checkpoint often.

## Layout

```
data/catalogue.py    product families, option axes, price logic
data/render_sv.py    Swedish prose templates
data/tokenizer.py    byte + GPT-2 BPE behind one interface; fertility()
data/dataset.py      corpus → token windows; instruction pairs + prompt mask
data/instructions.py 10 task types, gold answers, split by product     ← Lab 3
data/generate.py     CLI + sanity checks
model/config.py      presets, derived parameter counts
model/gpt.py         the model
model/generate.py    greedy / temperature / top-k / nucleus decoding
train/pretrain.py    training loop, checkpoint/resume, metrics
train/finetune.py    instruction tuning; --init vs --scratch      ← Experiment 2
train/sample.py      load a checkpoint and generate
evals/scoring.py     one scorer per gold type, no judge model     ← Lab 4
evals/run.py         model + majority + retrieval, one table
tests/               76 checks total, stdlib runner or pytest
runpod/setup.sh      one-shot pod setup
```

`data/` has no third-party dependencies at all, and neither does
`evals/scoring.py`. Only the model and training code need torch, which is why
`python -m evals.run --baselines-only` and three of the four test files run on a
clean checkout with nothing installed.

## The two splits

Both of them exist because the obvious version inflates the number in the
flattering direction.

The pretraining split is **by document, not by token position**. Splitting a flat
token stream puts the first half of a product description in train and the second
half in validation, so validation loss measures memorisation of text the model has
already partly seen.

The instruction split is **by product, not by pair**, for the same reason one
level up: a product whose price question is in train and whose stock question is
in validation has already shown the model its variant table.

Tests assert both.
