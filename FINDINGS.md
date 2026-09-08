# Findings

Notes for the report: what I built, what I measured, and what the measurements
turned out to mean. Everything here is reproducible from a clean clone; the
commands are in `README.md`.

As of 8 September 2026 all four lab parts are built and tested. The two
experiments are set up but I haven't run them at a scale worth reporting yet, so
every number below is either a property of the data, a baseline, or a smoke run
that I've labelled as such.

The section I'd read first is "A pattern I noticed", near the end. It's the part
I didn't expect when I started.

---

## Lab 1: the corpus

120 products, 548 variants (4.57 per product), 344 documents, roughly 29 700
tokens. Generated from a seed, so it's deterministic. No network, no services,
21 checks.

I originally pulled the catalogue from a Medusa instance, and dropped that for
the course track. A real shop adds a service to install, seed and keep running,
and gives back data I have less control over: I can't ask a live catalogue for a
clean distribution of option axes or a fixed random seed. The synthetic generator
was already there as a test fixture, so I promoted it.

### The corpus is close to memorisable

My first pretraining run was 1.9M parameters with the byte tokenizer, about
386k tokens, roughly 10 epochs, nine minutes on CPU. It reached a validation loss
of 0.2289, which is a perplexity of 1.3.

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

840 pairs from the 120 products, 714 train and 126 validation, 18 products held
out completely. Ten task types, 18 checks.

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

126 pairs from 18 products, macro-averaged so that a task with 102 pairs doesn't
outvote one with 48.

| | majority | retrieval |
|---|---|---|
| macro average | 0.25 | 0.30 |
| micro average | 0.23 | 0.27 |
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

## Things I'm not confident about

**The held-out set is too small for per-task claims.** 126 pairs across ten tasks
gives per-task *n* between 6 and 16, so one answer moves the smallest task by 17
percentage points. I can already see it: the stock question's majority floor reads
0.73 on validation even though the corrected global balance is 58%, purely because
*n* is 15 there. Before the run I report I should raise the product count or the
validation fraction. Generation is instant, so this costs nothing. The aggregate
numbers are fine as they are.

**Held-out products are still in the pretraining corpus** unless I pretrain with
the exclusion flag. Without it, the claim I can honestly make is that the model
generalises the instruction format to products it was never instruction-tuned on,
which is smaller than generalising to unseen products. The held-out ids are
written to disk so this is a one-flag decision, and the report has to say which
way I ran it.

**Every word of the corpus is machine-generated.** The linguistic variety is
bounded by my template file, not by the number of products. Generating 10 000
products wouldn't give me 10 000 products' worth of variety.

---

## What I haven't run yet

Both experiments are built and neither has been run at a reportable scale.

**Experiment 1, model scale.** Three sizes against validation loss and task
accuracy. Config sweeps, no new code. What I want to know is whether task accuracy
follows perplexity at all on a corpus this templated, or whether it flattens out
while perplexity keeps falling.

**Experiment 2, does pretraining help?** Identical architecture, data, schedule
and seed, with the only difference being whether the weights start pretrained. I
think this is genuinely open. On a corpus with a perplexity of 1.3 it isn't
obvious that pretraining carries anything the instruction set doesn't already
contain.

A smoke run at the smallest preset (0.1M parameters, nine seconds of training
each) gave answer-token perplexity of 38.7 pretrained against 109.8 from scratch.
That's a plumbing check rather than a result. At that scale the model's actual
output is the string `"ar"`. But the comparison does discriminate, so the harness
works.

---

## How I'd structure the report

1. The system: data pipeline, model, instruction tuning, evaluation. One page.
2. The corpus is close to memorisable. Perplexity 1.3, and what it says about
   synthetic data. (LE3)
3. The model can't count. Five versus four, and why counts have to come from a
   query. (LE5, LE6)
4. Experiment 1: does task accuracy follow perplexity? (LE4)
5. Experiment 2: does pretraining help? (LE5)
6. Measuring a small model honestly. The six bugs, the baselines, and what the
   retrieval floor says about the tasks. (LE7)
7. Limitations, in full.

Section 6 is the one I'd protect if the report runs long. Sections 1 to 5 are
things a reader could get from the course material. Section 6 is the part where I
have something of my own to say.
