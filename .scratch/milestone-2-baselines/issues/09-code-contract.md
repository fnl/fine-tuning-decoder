# 09 Contract of generate.py, eval.py and the baseline YAMLs

Type: grilling
Status: resolved
Blocked by: 06, 07, 08
HITL: yes

## Question

Pin the interfaces the spec will state verbatim:

- `src/generate.py`: callable core signature (examples + model handle →
  `{docid, output}` rows), CLI (`--config`, `--split`, `--limit`, `--out`),
  where the few-shot turns are assembled (here vs. `data/prepare.py`), and
  how the always-`[]` baseline is expressed (an engine `constant` in the
  same CLI vs. a separate path);
- `src/eval.py`: new `--wandb` flag and what it needs from the config;
- YAML schema shared by the three baseline configs and compatible with
  `configs/qwen3-0.6b-smoke.yaml` (keys: `model`, `dataset`, `engine`,
  `shots`/`exemplars`, `generation`, `wandb_project`);
- GPU imports lazy, CPU tests for prompt assembly and the constant engine,
  `mypy` overrides for vllm/unsloth.

## Answer

Grilled 2026-09-20; all recommendations accepted. Facts: modules are
top-level under `src/` (`python -m eval`, `from data.prepare import …`),
packaged via hatch `only-include`; `outputs/` is already git-ignored;
`uv lock --dry-run` resolves with `requires-python = ">=3.11,<3.14"`.

### `src/generate.py` (module `generate`; add to hatch `only-include`)

```python
@dataclass
class GenerationResult:
    text: str
    finish_reason: str          # "stop" | "length"

Engine = Callable[[list[str]], list[GenerationResult]]

def render_inputs(examples, exemplars, tokenizer) -> list[str]
    # per example: apply_chat_template([ex.messages[0], *shot turns, ex.messages[1]],
    #   tokenize=False, add_generation_prompt=True, enable_thinking=False)
def generate(examples, engine, tokenizer, exemplars=(), *, input_budget: int) -> list[Row]
    # Row = {"docid": str, "output": str, "cut_off": bool}; cut_off = finish_reason == "length"
    # raises ValueError naming the first docid whose rendered input > input_budget
def vllm_engine(model: str, *, max_model_len: int, max_new_tokens: int) -> Engine   # lazy import vllm/torch
def constant_engine() -> Engine   # GenerationResult("[]", "stop") for every input
```

`examples` / `exemplars` are dataset rows (`{docid, messages, n_input_tokens,
truncated}`) from `fnl-es/muc4-chat`. Few-shot turns are assembled here;
`prepare.py` stays the codec. `input_budget = max_model_len − max_new_tokens`.
Rows stay minimal; counts live in `meta.json`.

CLI: `uv run python -m generate --config configs/<name>.yaml [--split dev]
[--limit N] [--out outputs/<name>]`. Defaults `--split dev`, `--out
outputs/<name>` (YAML `name`). Loads `config.dataset` via
`datasets.load_dataset`, `--limit` = first N rows, exemplars from the `train`
split by docid. Writes `<split>.jsonl` + `meta.json` (ticket 08 fields).

Always-`[]` = `engine: constant` in the same CLI; `[]` hardcoded, no
`constant:`/`output:` key. Its YAML still names `model:` because rendering
and the budget check need the tokenizer (CPU-fine; same path as GPU runs).

### YAML schema (three baselines; compatible with the smoke config)

```yaml
name: qwen3-4b-3-shot
tags: [baseline, 3-shot]
model: Qwen/Qwen3-4B-Instruct-2507
dataset: fnl-es/muc4-chat
engine: vllm                    # vllm | constant
exemplars: [DEV-MUC3-0512, DEV-MUC3-0126, DEV-MUC3-0094]   # [] for zero-shot
generation:
  max_new_tokens: 512
  max_model_len: 3072
wandb_project: muc4-event-extraction
```
Files: `configs/qwen3-4b-zero-shot.yaml`, `configs/qwen3-4b-3-shot.yaml`,
`configs/always-empty.yaml` (`engine: constant`, `exemplars: []`, same model,
tags `[baseline, always-empty]`). Loaded as `dict[str, Any]` via
`yaml.safe_load` in each script; no `Config` class or shared module.

### `src/eval.py`

New flags `--config <yaml>` and `--wandb`; `--gold <path>` unchanged (on Colab
the notebook writes it once with
`load_dataset("fnl-es/muc4-chat")["dev"].to_json("data/prepared/dev.jsonl")`).
`--wandb` requires `--config`; behaviour per ticket 08 (`flatten`, config
stamps, artifact, Table, `run.url` printed last). Without `--wandb` nothing
changes.

### Tests, typing, packaging, docs

- Lazy `import vllm` / `import torch` inside `vllm_engine()`; mypy
  `ignore_missing_imports` for `vllm.*`, `torch.*` (and `wandb.*` if needed).
- CPU tests: `render_inputs` zero-shot + 3-shot (message order, exemplar
  turns verbatim, no `<think>`), budget violation raises with docid,
  `constant_engine`, `generate` end-to-end with a fake engine incl. `cut_off`,
  `flatten` key names, `meta.json` contents, W&B payload via a pure function
  (`wandb.init` never called in tests). Rendering tests use the `tokenizer`
  marker.
- `requires-python` → `>=3.11,<3.14` (Colab is 3.13.15; lock verified);
  DESIGN §3 comment updated.
- Docs: README "Running a baseline" (generate → eval `--wandb`, three
  configs); DESIGN §3 tree gains `generate.py`; DESIGN §5 names
  `Qwen3-4B-Instruct-2507` and drops the "check the Unsloth list" line.
