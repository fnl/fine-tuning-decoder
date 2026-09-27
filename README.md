# fine-tuning-decoder

Fine-tune a small decoder LLM (Qwen3-4B-Instruct, QLoRA via Unsloth/TRL) for
document-level event extraction on MUC-4, scored with a port of the GTT
evaluation. The design and its rationale are in
[`docs/DESIGN.md`](docs/DESIGN.md), the vocabulary in
[`CONTEXT.md`](CONTEXT.md).

The model reads one news document and emits its events as JSON: an event
type (attack, bombing, kidnapping, arson, robbery, forced work stoppage)
plus five roles (PerpInd, PerpOrg, Target, Victim, Weapon), each a list of
entities, each entity a list of coreferent mentions copied from the text.

```json
[{"incident_type":"bombing","PerpInd":[["guerrillas","the rebels"]],"Target":[["bridge"]]}]
```

## Prerequisites

What you need before anything below works, and what each is for:

| what | needed for | where it goes |
|---|---|---|
| Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/) | everything local: data prep, scorer, tests | your machine |
| network access to the Hugging Face Hub | the Qwen tokenizer (tests, `generate`) and the `fnl-es/muc4-chat` dataset | nothing to configure: both download on first use into the Hugging Face cache (`~/.cache/huggingface`); `data.prepare` also needs the tokenizer to count input tokens, and `generate` and `train` load the dataset named by the YAML's `dataset` key |
| a [Weights & Biases](https://wandb.ai) account and its API key | logging a baseline or training run (`eval --wandb`) | `WANDB_API_KEY` env var locally; Colab Secret `WANDB_API_KEY` |
| a [Hugging Face token](https://huggingface.co/settings/tokens) (write scope) | publishing the dataset (`prepare --push`) and the adapters (`train`) | `HF_TOKEN` env var or `hf auth login` locally; Colab Secret `HF_TOKEN` |
| a Google account with [Colab](https://colab.research.google.com) | the GPU baselines and training (a free-tier T4 suffices for both) | — |

Scoring without `--wandb`, the data prep without `--push` and the whole test
suite need no key at all. Keys are read from the environment only; never put
keys in a config, a notebook cell or a commit (notebook outputs are stripped
by a git filter).

## Setup

Everything runs on CPU.

```bash
uv sync
uv run nbstripout --install    # once per clone: git filter that strips notebook outputs
export WANDB_API_KEY=...       # only for `eval --wandb`
export HF_TOKEN=...            # only for `prepare --push` (or: hf auth login)
```

Checks, as in [`AGENTS.md`](AGENTS.md):

```bash
uv run pytest                       # ~3 s; downloads the Qwen tokenizer on first run
uv run pytest -m "not tokenizer"    # offline
uv run mypy src
uv run ruff check .
```

## Preparing the data

```bash
uv run python -m data.prepare           # local only
uv run python -m data.prepare --push    # also publish to fnl-es/muc4-chat
```

This downloads the GTT-preprocessed MUC-4 corpus into `data/raw/` (skipped
when cached), turns every document into a chat example and writes
`data/prepared/{train,dev,test}.jsonl`, printing per split how many
documents were kept, dropped (train documents over the input budget) or
truncated (dev/test documents cut to fit it).

An example is one JSON object per line:

| field | content |
|---|---|
| `docid` | corpus document id |
| `messages` | `system` (the fixed prompt), `user` (document text), `assistant` (the JSON target) |
| `n_input_tokens` | tokens of system + user rendered with the Qwen chat template |
| `truncated` | whether the document text was cut to fit the input budget |

`--push` needs a Hub login (`HF_TOKEN` or `hf auth login`). The published
dataset loads by name:

```python
from datasets import load_dataset
ds = load_dataset("fnl-es/muc4-chat")   # splits: train, dev, test
```

## Running an evaluation

Predictions are a JSONL file of `{"docid": ..., "output": ...}` rows, where
`output` is the raw model text. Gold is the prepared JSONL of the split.

```bash
uv run python -m eval --pred predictions.jsonl --gold data/prepared/dev.jsonl
```

This prints the GTT-style table (P/R/F1 for the event type, each role and
the micro average) followed by the diagnostics: relevance accuracy,
event-count accuracy, event-type accuracy, parse-failure rate and how many
documents needed the greedy alignment fallback.

To sanity-check the scorer, score the gold against itself (everything 100 %)
or the always-empty baseline:

```bash
uv run python -c "
import json
for r in map(json.loads, open('data/prepared/dev.jsonl')):
    print(json.dumps({'docid': r['docid'], 'output': r['messages'][2]['content']}))" > /tmp/pred.jsonl
uv run python -m eval --pred /tmp/pred.jsonl --gold data/prepared/dev.jsonl
```

From Python, which is what the training and baseline code does:

```python
from pathlib import Path

from data.prepare import SYSTEM_PROMPT, parse_target
from eval import read_gold, score

golds = read_gold(Path("data/prepared/dev.jsonl"))
preds, failures = {}, set()
for docid, output in outputs.items():  # docid -> raw model text
    preds[docid], ok = parse_target(output)
    if not ok:
        failures.add(docid)
result = score(preds, golds, failures)  # plain dict, ready for wandb.log
print(result["micro_avg"]["f1"], result["diagnostics"])
```

`parse_target` tolerates prose or a markdown fence around the JSON and
salvages the complete events of a cut-off output; an output with no
recoverable event is a parse failure and scores as `[]`.

## Bootstrapping a score

One score on 200 documents says little about how far it would move on 200
others. `bootstrap` puts a percentile bootstrap interval around the micro-F1:
it scores every document once, then 10,000 times draws the split's documents
with replacement and recomputes the micro-F1 from their summed counts (the
metric aligns events within each document, so the counts add up). It takes the
same predictions and gold as `eval`:

```bash
uv run python -m bootstrap --pred outputs/qwen3-4b-r16-eval/test.jsonl \
    --gold data/prepared/test.jsonl --below 53.0 50.2
```

```text
micro F1 53.91%, 95% interval 48.84% to 58.89% (10000 resamples)
resamples below 53.0: 36.0%
resamples below 50.2: 7.3%
```

`--below` takes F1 scores in percent, such as published results, and prints the
share of resamples under each. `--n`, `--seed` and `--level` set the number of
resamples, the random seed (fixed at 0, so the output is reproducible) and the
coverage of the interval.

Read the interval as a lower bound on the uncertainty. It covers which
documents are in the split, not training randomness, so it says nothing about
another seed. The `--below` shares are not significance tests, because the
published figures carry sampling noise of their own.

Predictions generated on Colab are not on your machine, but every
`eval --wandb` run logs its output directory as the artifact `<name>-<split>`:

```bash
uv run wandb artifact get flowing/muc4-event-extraction/qwen3-4b-r16-eval-test:latest \
    --root outputs/qwen3-4b-r16-eval
```

From Python:

```python
from bootstrap import bootstrap_f1

interval = bootstrap_f1(preds, golds)  # the same dicts that score() takes
print(interval.point, interval.low, interval.high)
```

## Running a baseline

A baseline is one experiment YAML in `configs/` (`qwen3-4b-zero-shot`,
`qwen3-4b-3-shot`, `always-empty`) run in two steps: `generate` renders every
dev document with the Qwen chat template (splicing in the YAML's exemplars as
demonstration turns), hands the inputs to the engine named in the YAML (`vllm`
on a GPU, `constant` for `[]`) and writes `outputs/<name>/dev.jsonl` plus a
`meta.json` sidecar (git SHA, dataset revision, engine version, GPU, counts of
truncated documents and cut-off outputs); `eval --wandb` scores the outputs
and logs one W&B run with the YAML and the sidecar as config, every scorer
number as a flat metric, the output directory as an artifact and a browsable
`predictions` table, printing the run URL last.

The always-empty baseline runs locally and proves the whole path on CPU:

```bash
uv run python -m generate --config configs/always-empty.yaml
uv run python -m eval --config configs/always-empty.yaml \
    --pred outputs/always-empty/dev.jsonl --gold data/prepared/dev.jsonl --wandb
```

`--wandb` needs `WANDB_API_KEY` in the environment; without the flag the
scorer prints the table as before. `generate --limit 5` runs the first five
documents as a smoke test, and `generate --split test` runs the test split
instead of dev (writing `outputs/<name>/test.jsonl`).

The two GPU baselines run on a free-tier T4 from the notebook:

1. In Colab, open the key icon in the left sidebar and add the secrets
   `WANDB_API_KEY` and `HF_TOKEN` with notebook access enabled (the notebook
   fails loudly if either is missing).
2. [Open the notebook in Colab](https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/baselines.ipynb)
   and pick a T4 runtime (Runtime → Change runtime type).
3. Optionally set `REF` in the clone cell to a commit, branch or tag.
4. "Run all" twice: the first pass installs vLLM and restarts the runtime;
   the second pass skips the install and runs everything else.

One cell per baseline ends with its W&B run URL.

## Fine-tuning

A fine-tune is one experiment YAML in `configs/` run by `train`. Milestone 4's
is `qwen3-4b-r16`: a QLoRA (r=16, NF4 base) fine-tune of
`Qwen/Qwen3-4B-Instruct-2507` on the whole train split, 3 epochs, 246 steps.
It takes the first `train_docs` train documents (all of them when the key is
absent), masks every prompt token so the loss covers only the JSON target and
its end-of-turn token, and trains a LoRA adapter with Unsloth and TRL. Prompt
and target are rendered with the base model's official chat template, not
Unsloth's copy (which would teach an empty thinking block), and a completion
guard stops the run naming the document if a completion is ever anything but
the target and its end-of-turn token. Training examples over the sequence
budget (`max_seq_len`) are dropped and counted, not truncated.

Every `eval_every` epochs (an *eval point*) it generates for the first
`dev_docs` dev documents with the model under training, scores them with the
scorer above and saves a checkpoint that is pushed to the Hub repository named
by `adapter` (the adapter only, never merged), together with a resumable
`last-checkpoint/`. One W&B run carries:

- the training loss (`train/*`);
- at every eval point, the dev scores (`dev/*`, the same keys as a baseline run
  under a prefix, because 50 documents are a training signal and not a result),
  the count of cut-off outputs (`dev/diagnostics/n_cut_off`) and a browsable
  `predictions` table, so a parse-failure spike can be diagnosed from the run;
- as config, the YAML under `experiment`, every derived number under `derived`
  and the library versions under `versions` (nested, because the trainer's own
  W&B integration overwrites top-level keys named like its settings, such as
  `warmup_ratio`);
- in the summary, the pushed `adapter_revision`, the commit an evaluation pins.

The run prints its W&B run id at the start and its URL last.

It needs a GPU and the notebook-installed training stack, so it runs on a
free-tier T4 (≈ 2.25 h for the 4B run). The install cell pins that stack to
the versions every GPU run so far used (unsloth 2026.9.11, unsloth_zoo
2026.9.7, trl 0.24.0, transformers 5.5.0, peft 0.20.0, bitsandbytes 0.50.2), so
that a run and its resume execute the same code; torch stays Colab's.

1. Add the Colab Secrets `WANDB_API_KEY` and `HF_TOKEN` (write scope) as for
   the baselines.
2. [Open the notebook in Colab](https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/train.ipynb)
   and pick a T4 runtime.
3. Optionally set `REF` in the clone cell to a commit, branch or tag.
4. "Run all" once: nothing restarts the runtime. A 4B `--limit 8` check runs
   before the real run.

The commands the notebook runs:

```bash
python -m train --config configs/qwen3-4b-r16.yaml --limit 8
python -m train --config configs/qwen3-4b-r16.yaml
```

`--limit N` replaces both subset sizes with N and makes the run a pipeline
check: it saves locally and never pushes, so it cannot overwrite a real
adapter. The run prints the kept-token count and the decoded completion of the
first training batch before the first step, the direct evidence that the mask
reaches the trainer: it should be the target JSON followed by `<|im_end|>`, with
no `<think>`, and the step-1 loss should be near 1.5.

A fresh run refuses to start if its adapter repo already holds a
`last-checkpoint/`, naming the repo, so a second "Run all" can never destroy a
run waiting to be resumed. A *rerun* needs a new adapter name, or a person
deleting the repo.

What would mean a broken pipeline is a parse-failure rate above the zero-shot
baseline's 7.5 %, all-`[]` predictions, or a step-1 loss far from 1.5. The
smoke run (`qwen3-0.6b-smoke`) proved the loop in milestone 3; its YAML stays as
the cheap end-to-end check.

## Resuming a run

A free-tier session guarantees nothing, so a disconnect during the run is the
normal case. A resume continues the interrupted run from its adapter repo's
`last-checkpoint/` (adapter, optimizer, scheduler, RNG, step and data position)
as the same W&B run:

1. Open a new runtime and run the notebook's setup cells (through the secrets).
2. Set `RUN_ID` in the resume cell to the id the run printed at its start (it
   is also in the adapter card's "Training run" link) and run that cell. It
   skips itself while `RUN_ID` is empty, so "Run all" never resumes anything.
3. Repeat until the run finishes.

```bash
python -m train --config configs/qwen3-4b-r16.yaml --resume <wandb_run_id>
```

The resume takes the dataset revision from the run it resumes, never the Hub's
latest. It refuses, with a `ResumeError` naming the reason:

- an adapter repo without `last-checkpoint/` (names the repo);
- a finished run (its checkpoint is at the last step: it would train nothing);
- a YAML that differs from the run's `experiment` config (names every differing
  key: eval and save points would desynchronise);
- a running torch other than the run's `versions.torch` (names both: torch is
  the one package the install cell does not pin).

`--resume` and `--limit` are mutually exclusive, because a `--limit` run pushes
no checkpoint. Newer code is allowed, so a bug fix can rescue a run: the config
keeps the original `git_sha`, and each resume appends `{"step", "git_sha"}` to
a `resumes` list in the run summary. After heavy use, expect Colab to withhold
a GPU for a while; the checkpoint waits on the Hub.

## Evaluating an adapter

The finished adapter is scored over the dev split like a baseline, from
`baselines.ipynb`, by vLLM over the fp16 official base: the baselines' own
engine, rendering and greedy sampling, so the adapter is the only variable.
Its eval YAML is a baseline YAML with three more keys:

```yaml
adapter: fnl-es/qwen3-4b-muc4-lora-r16
adapter_revision: <the training run's summary.adapter_revision>
training_run: <the training run's W&B id>
```

`generate` refuses an `adapter` without the other two. It downloads only the
`adapter_*` files at `adapter_revision` (never `last-checkpoint/`), so a later
push to the repo cannot change what was scored, and serves them as a LoRA.
`eval --wandb` logs the YAML as run config, so the eval run names the adapter
commit and the training run it scored. The YAML is written after training, and
its generate → eval cell in `baselines.ipynb` lands in the same commit.

## Reproducing the results

The test-split table in [`docs/RESULTS.md`](docs/RESULTS.md) comes from four
runs, each an `eval --wandb` run in `muc4-event-extraction`:

| run | where | W&B run | test micro-F1 |
|---|---|---|---:|
| always-empty | local CPU | `kyngxn5r` | 0.0 |
| zero-shot | Colab T4 | `1zg4m43j` | 21.4 |
| 3-shot | Colab T4 | `dnnt66g6` | 20.0 |
| QLoRA fine-tune | Colab T4 | `pqizcifd` | 53.9 |

Score the test split only for results. Every choice (prompt, exemplars,
config, adapter) was made on dev, and the test split was run once per system,
after milestone 4 had fixed all of them.

1. **Gold.** `uv run python -m data.prepare` writes
   `data/prepared/test.jsonl` locally; the notebook loads the same split from
   `fnl-es/muc4-chat`.
2. **Always-empty**, locally:

   ```bash
   uv run python -m generate --config configs/always-empty.yaml --split test
   uv run python -m eval --config configs/always-empty.yaml \
       --pred outputs/always-empty/test.jsonl --gold data/prepared/test.jsonl --wandb
   ```

3. **The GPU runs**, from
   [`baselines.ipynb`](https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/baselines.ipynb)
   on a T4. Do not use "Run all": it would also re-run the dev baselines and
   the dev adapter evaluation and log them as new W&B runs. Instead:
   1. Run the setup cells up to and including the gold cell, which writes
      `dev.jsonl` and `test.jsonl`. The install cell restarts the runtime, so
      run them a second time after it does.
   2. Run the three cells under *Test split (milestone 6)*: zero-shot, 3-shot
      and the fine-tune, about 5 to 10 minutes each.
4. **The interval**: download the fine-tune's predictions and bootstrap them
   (see *Bootstrapping a score*).

The fine-tune row scores the published adapter, not a new one:
`configs/qwen3-4b-r16-eval.yaml` pins `adapter_revision`, the commit its
training run pushed. Retraining with `configs/qwen3-4b-r16.yaml` (see
*Fine-tuning*) produces a different adapter with a different score; the
spread between training seeds has not been measured.

Decoding is greedy, so the same code, library versions and GPU type should
reproduce the table closely. It has not been checked to be bit-exact: vLLM's
batching can change floating-point results, and with them a few outputs. Set
`REF` in the clone cell to the commit a run's `git_sha` records to run the
exact code.

## Layout

| path | purpose |
|---|---|
| `src/data/prepare.py` | corpus → canonical events → chat examples → JSONL / Hub; `parse_target` |
| `src/generate.py` | render inputs (+ exemplars) → engine (vLLM [+ LoRA] \| in-process \| constant) → outputs JSONL + `meta.json` |
| `src/train.py` | mask prompts → LoRA fine-tune → dev score at every eval point → adapter on the Hub; resume |
| `src/eval.py` | GTT scorer port + diagnostics; CLI, optionally logging one W&B run |
| `src/bootstrap.py` | percentile bootstrap interval of micro-F1 over documents; CLI |
| `tests/` | the CPU-only test suite, including parity checks of the scorer against the original GTT evaluation |
| `configs/` | experiment definitions: each baseline, fine-tune and adapter evaluation is one YAML |
| `notebooks/` | running the GPU steps (baselines, adapter evaluation, training) on Colab |
| `docs/` | design decisions and their rationale, experiment write-ups and instructions for coding agents |

Written by running the code, all gitignored:

| path | written by | purpose |
|---|---|---|
| `data/` | `data.prepare` | caching the downloaded corpus and holding the prepared splits that scoring reads as gold |
| `outputs/` | `generate`, `train` | per-experiment results: model outputs to score, and a training run's working checkpoints |
| `wandb/` | `eval --wandb`, `train` | W&B's local staging area for syncing runs, and its logs for debugging them |
