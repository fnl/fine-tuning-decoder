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
| a Hugging Face token (write scope) | publishing the dataset (`prepare --push`) and, from milestone 3, adapters | `HF_TOKEN` env var or `hf auth login` locally; Colab Secret `HF_TOKEN` |
| a Google account with [Colab](https://colab.research.google.com) | the GPU baselines and training (free-tier T4 suffices for milestone 2) | — |

Scoring without `--wandb`, the data prep without `--push` and the whole test
suite need no key at all. Keys are read from the environment only; never put
one in a config, a notebook cell or a commit (notebook outputs are stripped
by a git filter for this reason).

## Setup

Everything here runs on CPU; generation with the base model and training
run on a Colab GPU from [`notebooks/colab.ipynb`](notebooks/colab.ipynb).

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
2. [Open the notebook in Colab](https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/colab.ipynb)
   and pick a T4 runtime (Runtime → Change runtime type).
3. Optionally set `REF` in the clone cell to a commit, branch or tag.
4. "Run all" twice: the install cell installs vLLM and restarts the runtime
   once; the second pass is a no-op up to the smoke cell.

One cell per baseline ends with its W&B run URL.

## Layout

```
src/data/prepare.py   corpus -> canonical events -> chat examples -> JSONL / Hub; parse_target
src/generate.py       render inputs (+ exemplars) -> engine (vLLM | constant) -> outputs JSONL + meta.json
src/eval.py           GTT scorer port + diagnostics; CLI, optionally logging one W&B run
tests/                pytest, CPU only; tests/oracle/ holds the original GTT eval.py for parity
configs/              one YAML per experiment
notebooks/colab.ipynb GPU baselines on Colab; outputs stripped by the nbstripout git filter
data/                 gitignored: raw/ corpus cache, prepared/ JSONL
docs/                 DESIGN.md, agent instructions, ADRs (in the future)
```
