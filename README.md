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
| network access to the Hugging Face Hub | the Qwen tokenizer (tests, `generate`) and the `fnl-es/muc4-chat` dataset | — |
| a [Weights & Biases](https://wandb.ai) account and its API key | logging a baseline or training run (`eval --wandb`) | `WANDB_API_KEY` env var locally; Colab Secret `WANDB_API_KEY` |
| a Hugging Face token (write scope) | publishing the dataset (`prepare --push`) and the adapters (`train`) | `HF_TOKEN` env var or `hf auth login` locally; Colab Secret `HF_TOKEN` |
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
documents as a smoke test.

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

A fine-tune is one experiment YAML in `configs/` (`qwen3-0.6b-smoke`) run by
`train`: it takes the first `train_docs` train documents (all of them when the
key is absent), masks every prompt token so the loss covers only the JSON
target and its end-of-turn token, and trains a LoRA adapter with Unsloth and
TRL. Every `eval_every` epochs it generates for the first `dev_docs` dev
documents with the model under training, scores them with the scorer above and
saves a checkpoint that is pushed to the Hub repository named by `adapter`
(the adapter only, never merged). One W&B run carries the training loss
(`train/*`), the dev scores (`dev/*`, the same keys as a baseline run under a
prefix, because 50 documents are a training signal and not a result), the
YAML with every derived number and the resolved library versions as config,
and at the end a `predictions` table from the last eval point. Training
examples over the sequence budget (`max_seq_len`) are dropped and counted, not
truncated. The run URL is printed last.

It needs a GPU and the notebook-installed training stack, so it runs on a
free-tier T4:

1. Add the Colab Secrets `WANDB_API_KEY` and `HF_TOKEN` (write scope) as for
   the baselines.
2. [Open the notebook in Colab](https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/train.ipynb)
   and pick a T4 runtime.
3. Optionally set `REF` in the clone cell to a commit, branch or tag.
4. "Run all" once: nothing restarts the runtime. A `--limit 8` smoke cell runs
   a handful of steps before the real run, and the last cell loads the pushed
   adapter back from the Hub onto a fresh base model and generates one dev
   document.

The command the notebook runs:

```bash
python -m train --config configs/qwen3-0.6b-smoke.yaml [--limit N]
```

`--limit N` replaces both subset sizes with N. The run prints the kept-token
count and the decoded completion of the first training batch before the first
step, the direct evidence that the mask reaches the trainer.

A near-zero dev F1 from the smoke run is a success: it proves the loop, not
the model. What would mean a broken pipeline is a parse-failure rate above the
zero-shot baseline's 7.5 %, or all-`[]` predictions together with a flat loss
curve.

## Layout

```
src/data/prepare.py   corpus -> canonical events -> chat examples -> JSONL / Hub; parse_target
src/generate.py       render inputs (+ exemplars) -> engine (vLLM | in-process | constant) -> outputs JSONL + meta.json
src/train.py          mask prompts -> LoRA fine-tune -> dev score at every eval point -> adapter on the Hub
src/eval.py           GTT scorer port + diagnostics; CLI, optionally logging one W&B run
tests/                pytest, CPU only; tests/oracle/ holds the original GTT eval.py for parity
configs/              one YAML per experiment
notebooks/            Colab: baselines.ipynb (vLLM), train.ipynb (Unsloth); outputs stripped by nbstripout
data/                 gitignored: raw/ corpus cache, prepared/ JSONL
docs/                 DESIGN.md, agent instructions, ADRs (in the future)
```
