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

## Setup

Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/). Everything here runs on
CPU; training and inference will live in a Colab notebook (later milestones).

```bash
uv sync
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
for docid, output in outputs.items():          # docid -> raw model text
    preds[docid], ok = parse_target(output)
    if not ok:
        failures.add(docid)
result = score(preds, golds, failures)          # plain dict, ready for wandb.log
print(result["micro_avg"]["f1"], result["diagnostics"])
```

`parse_target` tolerates prose or a markdown fence around the JSON and
salvages the complete events of a cut-off output; an output with no
recoverable event is a parse failure and scores as `[]`.

## Layout

```
src/data/prepare.py   corpus -> canonical events -> chat examples -> JSONL / Hub; parse_target
src/eval.py           GTT scorer port + diagnostics; CLI
tests/                pytest, CPU only; tests/oracle/ holds the original GTT eval.py for parity
configs/              one YAML per experiment
data/                 gitignored: raw/ corpus cache, prepared/ JSONL
docs/                 DESIGN.md, agent instructions, ADRs (in the future)
```
