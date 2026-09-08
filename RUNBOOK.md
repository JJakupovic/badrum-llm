# Runbook: testing Labs 1-4

Three levels. Level 1 needs nothing installed and takes half a minute. Level 2
needs torch and takes about five minutes. Level 3 is the real training run that
produces the numbers the report quotes.

All commands run from the repository root. On Windows use `python`; on
Linux or macOS use `python3`.

## Setup

```powershell
cd "C:\Users\jakjas\OneDrive - Jonkoping University\PhD\Courses\LLM\badrum-llm"

python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
# .venv\Scripts\activate.bat        # cmd
# source .venv/bin/activate         # Git Bash / Linux / macOS

pip install -r requirements.txt
pip install torch                    # not in requirements.txt: RunPod images already have it
```

Level 1 works without any of that, including without the venv.

---

# Level 1: no torch, 30 seconds

This covers Lab 1 in full, the data half of Lab 3, and the scorer of Lab 4.

```bash
python -m data.generate
python -m data.instructions
python tests/test_data.py
python tests/test_instructions.py
python tests/test_evals.py
python -m evals.run --baselines-only
```

**What you should see**

| command | time | expected |
|---|---|---|
| `data.generate` | 0.3 s | `products 120`, `variants 548`, and nine ✓ lines |
| `data.instructions` | 0.1 s | `pairs 840  (714 train / 126 val)`, `held out 18 products` |
| `test_data.py` | 0.1 s | `21/21 passed` |
| `test_instructions.py` | 2.4 s | `18/18 passed` |
| `test_evals.py` | 0.1 s | `11/11 passed` |
| `evals.run --baselines-only` | 0.1 s | `macro avg  0.25  0.30` |

`test_instructions.py` prints `(torch not installed, so the mask tests will
no-op)` at Level 1. That is expected. Four of its checks need torch and are
skipped; Level 2 runs them for real.

**Check determinism.** Run the generator twice and compare. Nothing should
change:

```bash
python -m data.generate && cp data/out/stats.json /tmp/a.json
python -m data.generate && diff /tmp/a.json data/out/stats.json && echo "identical"
```

---

# Level 2: with torch, about five minutes on CPU

This covers Lab 2, the training half of Lab 3, and Lab 4 end to end.

```bash
python tests/test_model.py

python -m train.pretrain --preset debug --context-length 256 --steps 120 \
    --batch-size 8 --device cpu --name pre-debug --sample-every 0

python -m train.finetune --init runs/pre-debug/ckpt_best.pt \
    --epochs 3 --device cpu --name ft-debug

python -m evals.run --ckpt runs/ft-debug/ckpt_best.pt --baselines
```

**What you should see**

| command | time | expected |
|---|---|---|
| `test_model.py` | **98 s** | `26/26 passed` |
| `train.pretrain` | 18 s | ends `best val loss 3.96`, writes `runs/pre-debug/` |
| `train.finetune` | 21 s | `44 596 supervised tokens (44% of 100 737)`, ends near perplexity 38.7 |
| `evals.run` | 79 s | a three-column table, and the warning below |

`test_model.py` takes over a minute and prints nothing while it runs. It is not
hung. It builds several model configurations and checks the causal property on
each.

The evaluation ends with:

> `! the model does not beat answering the majority class.`

**That is the correct result at this scale.** The `debug` preset is 0.1M
parameters trained for 18 seconds and its output is the string `"ar"`. The
warning firing here is evidence the harness discriminates. If it did *not* fire
on a model this bad, the scorer would be broken.

**Verify the mask test can actually fail.** This is the check worth showing
someone, and it takes three seconds:

```bash
python -c "import sys; sys.path.insert(0,'.'); import tests.test_model as t; t.test_attention_is_strictly_causal(); print('causal: PASS')"
```

Then break it on purpose. In `model/gpt.py`, change the mask construction to
`diagonal=2` and run the line again. It must fail. Change it back.

---

# Level 3: the real run

Level 2 proves the pipeline works. It does not produce a model worth reporting.
For that, generate a bigger catalogue and train the `tiny` preset.

**Generate more data first.** The 120-product default leaves only 6 to 16
validation pairs per task, which is too few for per-task numbers. 400 products
fixes it and costs nothing:

```bash
python -m data.generate --products 400
python -m data.instructions
```

That gives 1945 variants, about 428k byte-tokens, and 2800 instruction pairs
with 420 in validation across 60 held-out products.

**Pick the step budget from the corpus, not from a round number.** At batch 16
and context 256 each step is 4096 tokens, so 700 steps is about 7.4 passes over
385k training tokens. Much beyond ten passes buys memorisation. `train.pretrain`
prints the epoch count and warns before it starts, so read that line.

```bash
python -m train.pretrain --preset tiny --steps 700 --name pre-tiny
python -m train.finetune --init runs/pre-tiny/ckpt_best.pt --name ft-pretrained
python -m evals.run --ckpt runs/ft-pretrained/ckpt_best.pt --baselines
```

**Where to run it.** On CPU I measured 1081 tokens/second for the `tiny` preset,
which is about 3.8 seconds per step, so 700 steps is roughly 45 minutes and the
whole sequence including evaluation is two to three hours. On a GPU it is minutes.

### On RunPod

1. Push whatever is committed locally, since the pod clones from GitHub.
2. **Create a network volume first**, in the same region you will deploy in.
   20 GB is ample. Without one, `/workspace` dies with the pod and takes every
   checkpoint with it.
3. Deploy a pod on that volume with a **PyTorch** template. An RTX A5000, 3090 or
   L4 is plenty for models this size, at roughly $0.27 to $0.50 an hour. An A100
   is a waste here.
4. Open the pod's web terminal and run:

   ```bash
   cd /workspace
   git clone https://github.com/JJakupovic/badrum-llm.git
   cd badrum-llm
   bash runpod/setup.sh
   ```

   `setup.sh` checks the GPU, warns if `/workspace` is not a real volume or the
   repo is outside it, installs requirements without touching torch, generates
   400 products, and runs three test files.

5. Then the experiments:

   ```bash
   bash run_experiments.sh --exclude-holdout
   ```

6. Bring the results back and stop the pod:

   ```bash
   cat RESULTS.md
   git add RESULTS.md && git commit -m "Add experiment results" && git push
   ```

Stop the pod the moment the run finishes. Billing is per second and an idle GPU
costs what a busy one does. Community-cloud pods can be reclaimed mid-run; every
stage checkpoints and a rerun skips finished stages, so rerunning the same
command continues rather than restarting.

**The one decision to make before this run.** Held-out products are still in the
pretraining corpus unless you exclude them:

```bash
python -m train.pretrain --preset tiny --steps 700 --name pre-tiny \
    --exclude-products data/out/holdout_products.json
```

With the flag you can claim the model generalises to products it never saw at
all. Without it the claim is narrower: it generalises the instruction format to
products it was never instruction-tuned on. Both are honest. The report has to
say which one you ran.

## Both experiments, one command

```bash
bash run_experiments.sh                      # GPU if available, 400 products, 700 steps
bash run_experiments.sh --device cpu         # overnight on a laptop
bash run_experiments.sh --sizes "debug tiny" # skip the slow one
bash run_experiments.sh --exclude-holdout    # the stronger generalisation claim
```

This generates the data, runs Experiment 1 across the sizes, runs the Experiment 2
control, evaluates everything, and writes `RESULTS.md`. Every stage is skipped if
its checkpoint already exists, so an interrupted run continues rather than
restarting. Delete `runs/<name>/` to force a stage to redo. The full log goes to
`experiments.log`.

On CPU, budget two to three hours for `debug` and `tiny`, and considerably longer
if you include `small`. On a RunPod GPU the whole thing is minutes.

To rebuild the tables without retraining anything:

```bash
python -m evals.collect
```

It reads `runs/*/config.json`, `runs/*/metrics.jsonl` and `runs/*/eval.json`, so
`RESULTS.md` cannot disagree with the runs that produced it. I wrote it because
copying numbers between a dozen JSON files by hand is how a transposed digit
reaches a report.

## The same thing stage by stage

```bash
# Experiment 1: model scale
for p in debug tiny small; do
  python -m train.pretrain --preset $p --context-length 256 --steps 700 --name pre-$p
  python -m train.finetune --init runs/pre-$p/ckpt_best.pt --name ft-$p
  python -m evals.run --ckpt runs/ft-$p/ckpt_best.pt --baselines
done

# Experiment 2: does pretraining help
python -m train.finetune --init runs/pre-tiny/ckpt_best.pt           --name ft-pretrained
python -m train.finetune --scratch --like runs/pre-tiny/ckpt_best.pt --name ft-scratch
python -m evals.run --ckpt runs/ft-pretrained/ckpt_best.pt --baselines
python -m evals.run --ckpt runs/ft-scratch/ckpt_best.pt    --baselines
```

Each run leaves `metrics.jsonl` for the loss curves and `eval.json` for the
tables. Keep `data/out/stats.json` next to them so the corpus that produced the
numbers is identifiable.

---

# If something fails

| symptom | cause |
|---|---|
| `No module named data` | you are not in the repository root |
| `No module named torch` | Level 1 commands work anyway; install torch for Level 2 |
| `data/out/catalogue.jsonl not found` | run `python -m data.generate` first |
| `N instruction pairs exceed context_length` | pretrained at a context shorter than 256; the error names the size to use |
| `unknown tokenizer None` | a checkpoint from before the tokenizer name was recorded; pass `--tokenizer bytes` |
| `test_model.py` seems to hang | it takes 98 seconds and prints nothing until the end |
| evaluation prints the majority-class warning | correct for `debug`; a real concern for `tiny` |
