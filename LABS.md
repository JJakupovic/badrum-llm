# Where each examined part lives

The examination asks for a from-scratch LLM in four lab parts, at least two
experiments, and a short report. This file maps each part to the files that
implement it, the tests that check it, and the command that runs it.

Everything below runs from the repository root. `data/` and `evals/scoring.py`
are stdlib-only, so the data generation, the scorer and three of the four test
files work on a clean checkout with nothing installed. Only the training code
needs torch.

## Reproduce everything

```bash
python -m data.generate                          # catalogue + Swedish corpus
python -m data.instructions                      # 840 instruction pairs
python tests/test_data.py                        # 21 checks
python tests/test_model.py                       # 26 checks (needs torch)
python tests/test_instructions.py                # 18 checks
python tests/test_evals.py                       # 11 checks
python -m evals.run --baselines-only             # baseline floors, no torch needed
```

Add the training stages for a full end-to-end run on CPU in a few minutes:

```bash
python -m train.pretrain --preset debug --context-length 256 --steps 120 --device cpu
python -m train.finetune --init runs/debug-bytes/ckpt_best.pt --epochs 3 --device cpu
python -m evals.run --ckpt runs/ft-debug-bytes/ckpt_best.pt --baselines
```

The `debug` preset is 0.1M parameters and produces nonsense. It exists so the
whole pipeline can be verified quickly. Swap `--preset tiny` for a real run.

---

## Lab 1: data pre-processing pipeline

**Files**

| file | what it does |
|---|---|
| `data/catalogue.py` | product families, option axes, price and stock logic |
| `data/render_sv.py` | Swedish prose templates |
| `data/generate.py` | CLI and sanity checks |
| `data/tokenizer.py` | byte and GPT-2 BPE behind one interface, plus `fertility()` |
| `data/dataset.py` | corpus to token windows, document-level train/val split |

**Run**

```bash
python -m data.generate            # -> data/out/*.jsonl + stats.json
python -m data.generate --products 400 --seed 7
```

**Tests**: `tests/test_data.py`, 21 checks.

**What to look at.** The train/val split is by document rather than by token
position (`data/dataset.py::split_documents`). The nine sanity checks in
`data/generate.py::sanity_checks` each exist because the thing they check was
once wrong; four of those bugs were only visible by reading the generated
Swedish, and they are listed in `FINDINGS.md`.

Deliberately out of scope: mixing in a real Swedish web crawl. `README.md`
explains that decision and what the fully synthetic corpus measures instead.

---

## Lab 2: pre-train a GPT-style LLM

**Files**

| file | what it does |
|---|---|
| `model/config.py` | size presets, parameter counts derived from the architecture |
| `model/gpt.py` | the model: LayerNorm, GELU, causal attention, residual block |
| `model/generate.py` | greedy, temperature, top-k and nucleus decoding |
| `train/pretrain.py` | training loop, LR schedule, checkpoint and resume, metrics |
| `train/sample.py` | load a checkpoint and generate |

Only `nn.Linear`, `nn.Embedding` and the optimiser come from torch. Everything
else in `model/gpt.py` is written out.

**Run**

```bash
python -m train.pretrain --preset tiny --steps 4000
python -m train.pretrain --resume runs/tiny-bytes/ckpt_last.pt
python -m train.sample --ckpt runs/tiny-bytes/ckpt_best.pt --compare
```

Each run writes `runs/<name>/` with `ckpt_best.pt`, `ckpt_last.pt`,
`metrics.jsonl` and `config.json`.

**Tests**: `tests/test_model.py`, 26 checks.

**What to look at.** `test_attention_is_strictly_causal` asserts the causal
property directly by poking the token at position *t* and requiring every output
before *t* to come back bit-identical. I verified it against two deliberately
broken masks, and both fail it.

---

## Lab 3: fine-tune

**Files**

| file | what it does |
|---|---|
| `data/instructions.py` | 10 task types, gold answers, split by product |
| `data/dataset.py` | `InstructionDataset`, the prompt loss mask, collation |
| `train/finetune.py` | instruction tuning, `--init` and `--scratch --like` |

**Run**

```bash
python -m data.instructions                                    # 840 pairs
python -m train.finetune --init runs/tiny-bytes/ckpt_best.pt
```

**Tests**: `tests/test_instructions.py`, 18 checks.

**What to look at.** Loss is computed on answer tokens only.
`test_mask_supervises_exactly_the_answer` decodes the supervised positions and
requires the answer back byte for byte, and
`test_the_mask_check_can_actually_fail` shifts the boundary by one in each
direction to confirm the check fails when it should.

The train/validation split holds out whole products rather than individual
pairs. `data/out/holdout_products.json` records which ones.

---

## Lab 4: evaluate

**Files**

| file | what it does |
|---|---|
| `evals/scoring.py` | one scorer per gold type: number, set, bool, text, call |
| `evals/run.py` | the model plus two baselines, one table, `eval.json` |

**Run**

```bash
python -m evals.run --baselines-only                                   # floors only
python -m evals.run --ckpt runs/ft-tiny-bytes/ckpt_best.pt --baselines
```

**Tests**: `tests/test_evals.py`, 11 checks.

**What to look at.** There is no judge model anywhere. Gold answers are computed
from the catalogue, so scoring is exact match, set F1 or argument equality.

Two baselines run through the identical scorer: `majority` answers the most
common gold value for the task every time, and `retrieval` answers with the gold
answer of the nearest training question. Current floors over the 126 held-out
pairs are 0.25 and 0.30 macro. The majority baseline is what caught a task that
scored 100% for answering "Nej", which `FINDINGS.md` describes.

---

## The two experiments

**Experiment 1, model scale.** Same data, same schedule, three model sizes.

```bash
for p in debug tiny small; do
  python -m train.pretrain --preset $p --context-length 256 --steps 4000 --name pre-$p
  python -m train.finetune --init runs/pre-$p/ckpt_best.pt --name ft-$p
  python -m evals.run --ckpt runs/ft-$p/ckpt_best.pt --baselines
done
```

**Experiment 2, does pretraining help?** Identical architecture, data, schedule
and seed. The only difference is where the weights start.

```bash
python -m train.finetune --init runs/pre-tiny/ckpt_best.pt              --name ft-pretrained
python -m train.finetune --scratch --like runs/pre-tiny/ckpt_best.pt    --name ft-scratch
python -m evals.run --ckpt runs/ft-pretrained/ckpt_best.pt --baselines
python -m evals.run --ckpt runs/ft-scratch/ckpt_best.pt    --baselines
```

`--scratch --like` copies the architecture out of a checkpoint without its
weights, so the control is identical to the treatment by construction rather
than by remembering to pass matching flags.

---

## The report

`FINDINGS.md` holds the results so far, what I think they mean, and the things
I am not confident about. The write-up itself is still to come.
