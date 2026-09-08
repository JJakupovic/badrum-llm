# Findings

Notes for the report: what I built, what I measured, and what the measurements
turned out to mean. Everything here is reproducible from a clean clone; the
commands are in `README.md`.

As of 8 September 2026 all four lab parts are built and both experiments have
run on a GPU at 400 products. The numbers below are from that run unless a line
says otherwise.

"What the experiments showed" is the result. "A pattern I noticed" is the part I
didn't expect when I started.

Background in one paragraph: I am building a bathroom-products webshop in Medusa,
and I wanted the course project to run against that rather than against a made-up
domain. Medusa's data model puts variants and their option values directly under
the product, so the catalogue here mirrors that shape and the questions I ask the
model are the ones a customer would actually ask my shop. That is also why the
counting failure in Lab 2 matters commercially and not only academically.

---

## Lab 1: the corpus

400 products, 1945 variants, 982 documents, 329 723 training tokens and 32 628
validation. Generated from a seed, so it's deterministic. No network, no
services, 21 checks. I used the 120-product default while building and switched
to 400 for the reported run, because 120 left only 6 to 16 validation pairs per
task.

I originally pulled the catalogue from a Medusa instance, and dropped that for
the course track. A real shop adds a service to install, seed and keep running,
and gives back data I have less control over: I can't ask a live catalogue for a
clean distribution of option axes or a fixed random seed. The synthetic generator
was already there as a test fixture, so I promoted it.

### The corpus is close to memorisable

My first pretraining run was 1.9M parameters with the byte tokenizer, about
386k tokens, roughly 10 epochs, nine minutes on CPU. It reached a validation loss
of 0.2289, which is a perplexity of 1.3. The reported run says the same thing at
a larger scale: **1.46 at 10.8M parameters and 1.27 at 25.5M**.

That number looked like a success for about a minute. A perplexity of 1.3 means
there's almost no uncertainty left in the text: given a prefix, the next token is
essentially determined. Templated prose has very little entropy in it, and a
model with enough capacity just absorbs the whole thing.

So the result isn't a good model. It's a measurement of my corpus, and it's the
empirical version of the model-collapse argument from lecture 3. It also settles
how I describe the data everywhere else: the synthetic corpus gives the model
domain vocabulary. Real Swedish would have to give it language.

### Four bugs I only found by reading the output

All four passed every schema check I had. I found them by reading the generated
Swedish.

1. A non-breaking space (U+00A0) in every price. Invisible in every editor and
   terminal I use. It breaks exact-match scoring of prices and becomes a strange
   token in the tokenizer. Now U+0020, with a test.
2. A product-level dimension contradicting a varying axis: a product offered in
   500/600/800 mm that also claimed "bredd 600 mm". "Hur bred är den?" is the
   first thing anyone asks in a demo.
3. Plural stock wording on single-variant products. "Alla utföranden finns i
   lager" directly after "finns i ett enda utförande".
4. Non-ASCII SKUs ("TVÄ-HJO-KRO"). SKUs end up in URLs, CSVs and fine-tuning
   targets.

Schema checks wouldn't have caught any of these. If I'm generating training text
I have to read some of it.

---

## Lab 2: the model

Hand-written LayerNorm, GELU, causal multi-head attention and residual block.
Only `nn.Linear`, `nn.Embedding` and the optimiser come from torch. Presets at
0.1M, 10.8M, 25.6M and 86.0M parameters. 26 checks.

The bug I was most careful about is the causal mask. If position *t* can attend to
*t+1* the model reads its own answer, training loss falls faster than it should,
and the curves look better than they are. Nothing crashes. So the test pokes token
*t* and requires every output before *t* to come back bit-identical, and I ran it
against two deliberately broken masks (`diagonal=2` and `is_causal=False`) to
confirm it actually fails when it should.

### The model can't count

Sampling with nucleus decoding, the model produced:

> "...finns i **fem** varianter:" followed by a list of **four**.

Fluent, grammatical, and wrong. I don't think more pretraining fixes this. A
language model has no mechanism for counting a list it is in the middle of
generating.

This is the argument for the tool-calling design in the product track, and now I
have an observed failure to point at instead of an assertion. Variant counts,
prices and stock levels are database facts, and they should come from a query with
the model choosing the arguments.

---

## Lab 3: instruction tuning

2800 pairs from the 400 products, 2380 train and 420 validation, 60 products
held out completely and also excluded from pretraining. Ten task types, 18
checks. The per-task counts below are from the 120-product build; the reported
run has roughly 42 validation pairs per task.

| task | pairs | val | what it asks for |
|---|---|---|---|
| variant_count | 102 | 15 | count a set |
| stock | 99 | 15 | a yes/no grounded in inventory |
| sku | 95 | 13 | copy an identifier |
| option_check | 94 | 14 | yes or no to an option |
| price | 91 | 16 | one number |
| category | 91 | 14 | a one-word fact |
| add_to_cart | 90 | 14 | a structured call |
| cheapest | 80 | 12 | an argmin |
| variant_list | 50 | 6 | reproduce a set (single-axis) |
| axis_values | 48 | 7 | one axis of a multi-axis product |

Median pair length is 135 byte-tokens, p95 is 178, max is 214.

Every pair carries its own gold answer in a form a program can check, because the
answers are computed from the catalogue rather than written by hand. That decision
is what made Lab 4 cheap: the evaluation is arithmetic, with no judge model and no
API key anywhere.

### I wrote a training target that taught the wrong thing

My first version of the variant-listing task enumerated the full cross product on
multi-axis products. For one mirror cabinet that came out as eighteen entries of
"600 mm / Vit, 600 mm / Ek, ..." running to 434 tokens.

Nobody answers a customer that way, which was the reason I noticed. The worse
problem is that it trains the model on long enumeration, which is where the
counting failure from Lab 2 lives. I had written a task that drills the weakness
I was trying to demonstrate.

I split it into a single-axis list and a per-axis question. The longest pair fell
from 434 to 214 tokens, which turned out to matter for a second reason: everything
now fits inside the 256-token context of the `tiny` preset, so the model-scale
sweep doesn't have to vary context length as well. A phrasing decision removed a
confound I hadn't thought about yet.

### The prompt mask has the same shape as the causal mask problem

Loss is computed on answer tokens only. At a median of 135 tokens per pair about
half of every sequence is prompt, and if the mask is off by one the model trains
on predicting the last token of its own prompt. Nothing crashes and the curve
looks slightly better.

I check it by reconstruction: decode the supervised positions and require the
answer back byte for byte, then shift the boundary by one in each direction and
require the reconstruction to break.

I also tokenize the prompt and the answer separately and concatenate them rather
than tokenizing the joined string and counting the prefix. With the byte tokenizer
the two are identical. With a BPE they aren't, because a merge can straddle the
boundary and move it by a token.

---

## Lab 4: evaluation

Ten tasks, five gold types, two baselines, 11 checks. In lecture 7's vocabulary:
the subject is the model's greedy continuation of a held-out prompt, the criteria
are per task, the reference is computed from the catalogue, and the method is
exact match, set F1 or argument equality. No judge model.

### A task that scored 100% for saying no

One of my tasks began as `absent_option`. It asked only about options a product
did **not** have, so every gold answer was "Nej". A model that had learned nothing
except to refuse everything would have scored 100% on it and looked careful.

The loss curve said nothing. The instruction data passed all eighteen of its own
checks. What exposed it was running a constant answer through the same scorer. A
second task had a milder version of the same problem: 83% "Ja", because variants
are usually in stock.

What fixed both was drawing the answer first and then looking for a case to fit
it. The stock question now asks about an out-of-stock variant whenever the product
has one. Only 55 of 120 products do, so preferring the harder case lands at about
45/55 without discarding any pairs. I tried the obvious alternative first,
coin-flipping the polarity and skipping when the product couldn't supply it, and
it gives 69/31 and throws pairs away.

Both boolean tasks now sit near 58%. Two tests fail if either drifts back.

### Baseline floors

420 pairs from 60 products, macro-averaged so that one task doesn't outvote
another on pair count alone.

| | majority | retrieval |
|---|---|---|
| macro average | 0.22 | 0.36 |
| exact match | 0.00 | 0.01 |

`majority` answers the most common gold value for the task every time.
`retrieval` answers with the gold answer of the most similar training question of
the same task, by word overlap.

### What the retrieval floor says about my tasks

Because the split is by product, the answer retrieval finds always belongs to a
different product. So a task where retrieval scores well is one that can be
answered without knowing which product was asked about.

| task | floor | why |
|---|---|---|
| option_check | 0.79 | plausible options repeat across products |
| category | 0.71 | similar product names share a category |
| variant_list | 0.58 | finish sets are largely shared |
| axis_values | 0.51 | axis values come from a small pool |
| price, sku, cheapest, add_to_cart, variant_count | 0.00 | product-specific |

I want these numbers in front of me before I quote any model score. If the model
reaches 0.75 on `category` it has done nothing. The same score on `price` would
mean something.

---

## What the experiments showed

Both ran on a RunPod GPU: 400 products, 700 steps, held-out products excluded
from pretraining. Full tables in `RESULTS.md`.

### The tasks split in two, and the model behaves completely differently on each

Because the 60 held-out products were kept out of pretraining as well as
fine-tuning, the model has never seen them in any form. That turned the
evaluation into a cleaner test than I designed it to be, because some of my
tasks can be answered from the question alone and some cannot.

**Derivable from the question.** The SKU is a deterministic function of the
product noun, the model name and the option values, all of which appear in the
question. `DUS-ORS-9M-KRO` is "Duschvägg Orsa" and "900 mm / Krom" chopped up by
a fixed rule. Category follows from the noun.

| task | tiny | small | majority | retrieval |
|---|---|---|---|---|
| category | 1.00 | 1.00 | 0.15 | 0.87 |
| add_to_cart | 0.92 | 1.00 | 0.00 | 0.00 |
| sku | 0.86 | 0.96 | 0.00 | 0.00 |

**Requires knowing the specific product**, which the model never saw:

| task | tiny | small | majority | retrieval |
|---|---|---|---|---|
| variant_list | 0.71 | 0.78 | 0.56 | 0.66 |
| axis_values | 0.69 | 0.78 | 0.24 | 0.67 |
| stock | 0.65 | 0.57 | 0.37 | 0.45 |
| option_check | 0.64 | 0.60 | 0.53 | **0.72** |
| variant_count | 0.30 | 0.20 | **0.34** | 0.18 |
| price | 0.00 | 0.00 | 0.00 | 0.02 |
| cheapest | 0.00 | 0.00 | 0.00 | 0.02 |

The model clearly beats both floors on three tasks, and they are exactly the
three where the answer is computable from the question. On the fact-dependent
tasks it sits at or below a lexical nearest-neighbour lookup: worse than
retrieval on `option_check`, worse than a constant answer on `variant_count`.

This is the Track B argument with numbers rather than an anecdote. The model
learned the form of the answers and the deterministic rules. It learned nothing
about facts it could not derive, and under this split it could not have. Prices,
stock and variant counts have to come from a query.

### Experiment 2: pretraining bought one specific capability

Macro 0.58 from the pretrained checkpoint against 0.34 from random init, same
architecture, same data, same schedule, same seed. The control also falls below
the retrieval floor of 0.36, so without pretraining the fine-tuned model does not
beat a lookup.

The interesting part is where the gap sits. On the fact-dependent tasks the two
are roughly level. The difference is almost entirely:

| task | pretrained | scratch |
|---|---|---|
| sku | 0.86 | **0.00** |
| add_to_cart | 0.92 | **0.00** |

So pretraining did not broadly improve the model. It made the compositional
character-level rule learnable at all. Fine-tuning alone on 2380 pairs never
found it.

### Experiment 1: task accuracy saturates before perplexity does

| | params | pretrain ppl | answer ppl | macro | exact match |
|---|---|---|---|---|---|
| debug | 0.1M | 9.99 | 6.63 | 0.00 | 0.00 |
| tiny | 10.8M | 1.46 | 1.05 | 0.58 | 0.46 |
| small | 25.5M | 1.27 | 1.05 | 0.59 | 0.47 |

From 0.1M to 10.8M everything changes. From 10.8M to 25.5M perplexity keeps
improving and task accuracy does not, and the per-task differences run in both
directions with none of them clearing noise. Perplexity is still measuring
something real about the language modelling; it has just stopped predicting
whether the model can do the task.

This is what I set out to ask in Experiment 1, and the answer is no.

## What these numbers do not say

**The macro average is a bad summary here.** At 0.59 it largely measures how many
of my ten tasks happen to be rule-derivable. Read as "the model is 59%
competent" it is misleading, and the table is mine, so that is my problem to fix.
The per-task split is the result.

**`price` and `cheapest` at 0.00 are not failures.** The information is absent by
construction under this split. Those two are unknowable, not wrong.

**`stock` at 0.65 is probably not skill.** The majority baseline scores 0.37,
which means validation is skewed the other way, so answering validation's
majority class alone gets about 0.63. The model is at 0.65.

**About 42 validation pairs per task.** Anything under roughly 15 points is
noise, which covers the whole of small against tiny.

---

## Things I'm not confident about

**The corpus is entirely machine-generated.** The linguistic variety is bounded
by my template file, not by the number of products. Generating 10 000 products
wouldn't give me 10 000 products' worth of variety, and mixing in real Swedish is
the obvious next step rather than scaling the generator.

**Held-out products were excluded from pretraining as well as fine-tuning**, so
the model never saw them in any form. That is the stronger of the two setups and
it is the one I ran. It also means the fact-dependent tasks are unanswerable by
construction, which is what makes the split in the results section so clean, and
it means these numbers say nothing about how the model would do on products it
had read about but not been tuned on. That would be a separate run.

**Forty-two validation pairs per task is better than the 6 to 16 I had at 120
products, and still not many.** One answer moves a task by more than two points.
I would not defend any single-task comparison of small against tiny.

**Greedy decoding throughout.** It makes the numbers reproducible. It also means
I have not measured how much of the gap between models survives sampling, which
lecture 6 suggests could be substantial.

---

## A pattern I noticed

I've now found six bugs in this project, and every one of them made a result look
better than it was.

| | bug | how it showed up |
|---|---|---|
| 1 | causal mask lets *t* see *t+1* | training loss falls faster |
| 2 | train/val split by token position | validation measures memorisation of a half-seen document |
| 3 | prompt-loss mask off by one | model trains on predicting its own prompt |
| 4 | instruction split by pair | validation recalls a fact from the training half |
| 5 | a task with only one possible answer | a constant scores 100% |
| 6 | containment scoring on prices | a hedged answer with two numbers still scores 1.0 |

None of them crash. Four of the six make a curve look better rather than worse,
which means the usual signal that something is wrong (a number moving the wrong
way) is exactly the signal I wouldn't get. I don't think this is a coincidence: a
bug that makes the loss worse gets found on the next run, so the ones that survive
long enough to be interesting are the ones that flatter.

Every countermeasure I've written has been mutated afterwards to check it can
actually fail. I got that habit from the causal-mask test and I've been applying
it since. A test that can't fail tells me nothing, and I'd rather find that out
deliberately than discover it in the report.

---
