# Spec: Milestone 2 — baselines of Qwen3-4B-Instruct on dev

Status: ready-for-agent
Created: 2026-09-20
Source: `map.md` and `issues/01`–`10` in this directory; `docs/DESIGN.md` §2, §5, §6, §8, §9, §11; `CONTEXT.md`

## Problem Statement

The project has a dataset on the Hub and a scorer, but no number yet: nobody
knows how well the base model does on the dev split before any fine-tuning.
Without the zero-shot, 3-shot and always-empty baselines, milestone 4 ("full
QLoRA fine-tune, compare to 2") has nothing to compare against, and the W&B
project that every later run will report to has never been exercised.

The user wants to open a Colab notebook, run it, and end up with three runs in
W&B — each with its config, its micro-F1 and per-role F1, its diagnostics and
its raw outputs — produced by code that lives in the repo, is tested on CPU,
and whose generation core the milestone-3 evaluation callback can reuse on an
in-process model.

## Solution

A new generation script renders every dev example (optionally with three fixed
exemplar turns spliced in), hands the inputs to an *engine* (vLLM on the T4,
or a constant `[]` engine on CPU) and writes the outputs as JSONL plus a
metadata sidecar. The existing scorer gains a `--wandb` switch that turns one
scoring pass into one W&B run: the experiment's YAML as config, the flattened
scorer result as metrics, the outputs as an artifact and a browsable
predictions table. Three YAML files define the three baselines. A committed,
output-stripped notebook runs the two GPU baselines on a free-tier T4; the
always-empty baseline runs locally and proves the W&B path first.

## User Stories

1. AS A researcher, I WANT the base model's zero-shot micro-F1 on dev, SO THAT I have a floor to compare fine-tuning against.
2. AS A researcher, I WANT a 3-shot baseline with the same system prompt and three fixed exemplars, SO THAT I can see how much in-context demonstrations buy before training anything.
3. AS A researcher, I WANT the trivial always-empty baseline scored through the identical pipeline, SO THAT relevance accuracy has a reference (~46 % irrelevant documents) and the pipeline is proven end to end on CPU.
4. AS A researcher, I WANT each baseline to be one W&B run in `muc4-event-extraction`, SO THAT baselines and later fine-tunes are compared in one place.
5. AS A researcher, I WANT the run's config to hold the experiment YAML verbatim plus the git SHA, dataset revision, engine version and GPU, SO THAT any number can be traced back to exactly what produced it.
6. AS A researcher, I WANT every scorer number (precision, recall, F1 and their raw counts for the event type, each role and the micro average) and every diagnostic logged as a flat metric, SO THAT I can chart, filter and recompute without re-running.
7. AS A researcher, I WANT the count of truncated documents and of cut-off outputs logged with the metrics, SO THAT I know how much of a score is explained by budget effects.
8. AS A researcher, I WANT the raw outputs and metadata attached to the run as an artifact, SO THAT I can re-score or inspect them later without the Colab session.
9. AS A researcher, I WANT a predictions table (docid, gold, output, parse ok, cut off, event counts) in the run, SO THAT I can browse failures in the W&B UI sorted by count mismatch.
10. AS A researcher, I WANT the Colab cell to end with the run URL, SO THAT the result is one click away.
11. AS A developer, I WANT the generation core to take an engine as a plain callable, SO THAT milestone 3 can pass an in-process model without touching the rendering or bookkeeping.
12. AS A developer, I WANT the always-empty baseline to be an engine rather than a special path, SO THAT it exercises the same rendering, input-limit check, JSONL writing and scoring as the GPU runs.
13. AS A developer, I WANT inputs rendered with the official Qwen chat template and `enable_thinking=False`, SO THAT token counts agree with the dataset's and no `<think>` block ever appears, whichever checkpoint mirror serves the weights.
14. AS A developer, I WANT generation to fail loudly naming the docid if any rendered input exceeds what the engine can take, SO THAT the dataset's `truncated` flag stays the only truncation and nothing is silently cut twice.
15. AS A developer, I WANT the few-shot turns spliced verbatim from the train split by docid, SO THAT exemplar targets cannot drift from the codec that produced them.
16. AS A developer, I WANT GPU-only imports to happen lazily inside the vLLM engine, SO THAT the module imports, type-checks and tests on a CPU-only machine.
17. AS A developer, I WANT CPU tests for rendering, the input-limit check, the constant engine, the CLI's files and the W&B payload, SO THAT the pipeline is verified before a GPU is ever rented.
18. AS A developer, I WANT the scorer's CLI unchanged when `--wandb` is absent, SO THAT existing documentation, tests and habits keep working.
19. AS A developer, I WANT `--limit N` on the generation CLI, SO THAT a five-document smoke run precedes every full run in the notebook.
20. AS A Colab user, I WANT the notebook to clone the repo at a ref I type into a form field, SO THAT I can run any commit without editing code cells.
21. AS A Colab user, I WANT the install cell to be a no-op on re-run and to restart the runtime itself when it did install, SO THAT "Run all" twice is the whole procedure.
22. AS A Colab user, I WANT secrets read from Colab Secrets into the environment with a loud failure when missing, SO THAT no key is ever pasted into a cell.
23. AS A Colab user, I WANT the dev gold written from the Hub dataset by the notebook, SO THAT the scorer's `--gold` contract stays a file path everywhere.
24. AS A Colab user, I WANT one cell per GPU baseline, SO THAT re-running one baseline is one cell.
25. AS A maintainer, I WANT notebook outputs stripped by a git filter, SO THAT a notebook edited in Colab never commits outputs or secrets.
26. AS A maintainer, I WANT a CPU test that the notebook parses, has no outputs, and references only configs that exist, SO THAT notebook rot is caught by `pytest`.
27. AS A maintainer, I WANT the package installable on Colab's Python 3.13, SO THAT `pip install -e .` works in the only GPU environment we have.
28. AS A maintainer, I WANT the README to document running a baseline and opening the notebook in Colab, SO THAT the procedure survives the conversation that designed it.
29. AS A maintainer, I WANT the design doc to name the exact checkpoint and drop the stale "check the Unsloth list" note, SO THAT the doc matches the code.
30. AS A maintainer, I WANT the data codec finally tracked by git, SO THAT the notebook's clone actually contains it.

## Implementation Decisions

### Base model and engine (issues 04, 05, 06)

- Base model for baselines and later fine-tuning: **`Qwen/Qwen3-4B-Instruct-2507`** (Apache-2.0), the official repo, not the Unsloth mirror — the mirror's hybrid chat template injects `<think>` blocks into assistant turns. Its template ignores `enable_thinking`; the flag is still always passed so rendering is one convention across `prepare`, `generate` and later `train`.
- Engine: **vLLM 0.29.0 only**, pinned, installed by the notebook from the cu129 wheel index (`--extra-index-url https://wheels.vllm.ai/0.29.0/cu129` and the cu129 torch index), never in `pyproject`. Fallback order if the install or first generation breaks on the T4: `vllm==0.26.0` (keeps Colab's torch), then a new ticket for a `transformers.generate` engine — not coded speculatively.
- vLLM knobs: `dtype="half"`, `max_model_len=3072`, `gpu_memory_utilization=0.9`, `max_num_seqs=16`, `enforce_eager=True`, `seed=0`; sampling `temperature=0.0`, `max_tokens=512` (greedy must be explicit — vLLM otherwise applies the checkpoint's sampling defaults). Only `max_model_len` and `max_new_tokens` live in the YAML; the T4-specific rest are constants inside the vLLM engine constructor.
- Rendering: every conversation is pre-rendered to text with the official tokenizer's chat template (`add_generation_prompt=True`, `enable_thinking=False`) and passed to vLLM's `generate`, not `chat` — engine-agnostic and identical to how the dataset counted `n_input_tokens`.
- All inputs of a split go to the engine in one call (continuous batching + prefix caching make the shared system-plus-exemplars prefix nearly free); vLLM's own progress bar is the progress.
- The engine reports why each output stopped; `finish_reason == "length"` marks a **cut-off output**.

### Few-shot input budget (issue 06, fog graduated)

- One input budget only: the dataset's `truncated` flag from data preparation. The engine's limit on a rendered input, `max_input_tokens = max_model_len − max_new_tokens` (= 2560), is *not* a second budget (the glossary reserves *input budget* for the per-example limit): the generation core counts each rendered input's tokens and raises, naming the first offending docid, if any exceeds it. No second truncation layer, no per-document exemplar dropping. `n_truncated` is reported from the dataset flag.
- Measured (official tokenizer, dev split): longest zero-shot input 1375 tokens; longest 3-shot input 1857 tokens with the chosen exemplars; 0 documents over budget; ~700 tokens of headroom.

### Exemplars (issue 07, from the prototype `prototype_exemplars.py`)

- Three fixed train documents, spliced in this order as user/assistant turn pairs between the system prompt and the target document's user turn:
  1. `DEV-MUC3-0512` — irrelevant document: men arrested *with* bombs and attack plans, no event occurred (planned ≠ happened).
  2. `DEV-MUC3-0126` — one `attack` event with all five roles; a two-mention `PerpInd` entity; `PerpOrg` distinct from `PerpInd`.
  3. `DEV-MUC3-0094` — two `bombing` events (second device = second event); `PerpOrg` with abbreviation coreference; roles shared across events.
- Shared prefix (system prompt + three exemplars) is 665 tokens.
- Exemplar turns are taken verbatim from the train split rows by docid at generation time; nothing is re-rendered. Exemplars are ordinary train documents and are excluded from nothing.

### Generation module (issue 09)

- New top-level module `generate` beside `eval`, added to the package's included sources. Public shape (pinned by the grilling):

  ```python
  @dataclass
  class GenerationResult:
      text: str
      finish_reason: str          # "stop" | "length"

  Engine = Callable[[list[str]], list[GenerationResult]]

  def render_inputs(examples, exemplars, tokenizer) -> list[str]
  def generate(examples, engine, tokenizer, exemplars=(), *, max_input_tokens: int) -> list[Row]
      # Row = {"docid": str, "output": str, "cut_off": bool}
  def vllm_engine(model, *, max_model_len, max_new_tokens) -> Engine   # lazy vllm/torch imports
  def constant_engine() -> Engine                                      # "[]", "stop" for every input
  ```

  `examples` and `exemplars` are dataset rows (`docid, messages, n_input_tokens, truncated`). Rows carry only `docid`, `output` (the scorer's existing contract) and `cut_off`; counts live in the metadata sidecar.
- Few-shot assembly lives in `generate`; the data codec stays ignorant of shots.
- CLI: `--config <yaml>` (required), `--split` (default `dev`), `--limit N` (first N rows), `--out` (default `outputs/<name>`). Loads the dataset from the Hub, exemplars from its `train` split by docid, and writes `<split>.jsonl` plus `meta.json` with: `model`, `engine`, `engine_version` (`"vllm 0.29.0"` / `"constant"`), `git_sha`, `git_dirty`, `dataset_revision` (Hub commit of the dataset), `split`, `gpu` (device name or `"cpu"`), `n_docs`, `n_truncated`, `n_cut_off`, `wall_seconds`.
- The always-empty baseline is `engine: constant` in the same CLI; `[]` is hardcoded. Its YAML still names the model because rendering and the input-limit check need the tokenizer — deliberately the same path as the GPU runs.

### Experiment YAMLs (issue 09)

Three files: `qwen3-4b-zero-shot`, `qwen3-4b-3-shot`, `always-empty`. Keys, compatible with the existing smoke config's `name`, `model`, `dataset`, `generation.max_new_tokens`, `wandb_project`:

```yaml
name: qwen3-4b-3-shot
tags: [baseline, 3-shot]            # [baseline, zero-shot] / [baseline, always-empty]
model: Qwen/Qwen3-4B-Instruct-2507
dataset: fnl-es/muc4-chat
engine: vllm                        # vllm | constant
exemplars: [DEV-MUC3-0512, DEV-MUC3-0126, DEV-MUC3-0094]   # [] for zero-shot and always-empty
generation:
  max_new_tokens: 512
  max_model_len: 3072
wandb_project: muc4-event-extraction
```

Loaded as a plain mapping via `yaml.safe_load` in each script; no config class or shared config module until a third consumer exists.

### W&B run schema (issue 08)

- `eval --wandb` owns the run; `generate` is W&B-free. `--wandb` requires `--config`; `--gold` keeps its file-path contract; without `--wandb` nothing changes. `WANDB_API_KEY` comes from the environment only; a missing key is wandb's own failure. Entity is the key's default; project from the YAML.
- `run.config` = YAML mapping verbatim (nested) + stamped from `meta.json`: `git_sha`, `git_dirty`, `dataset_revision`, `engine_version`, `split`, `gpu`.
- Metrics: a `flatten` function joins the scorer result's nested keys with `/` (`micro_avg/f1`, `PerpInd/p_num`, `incident_type/r`, `diagnostics/relevance_acc`, … — 55 scalars) plus `diagnostics/n_truncated` and `diagnostics/n_cut_off` from the sidecar; logged once with `run.log` so it is both the summary and a step-0 point a training callback can extend.
- Artifact: the output directory (`<split>.jsonl` + `meta.json`) as `type="predictions"`, `name="<run-name>-<split>"`. Gold is not attached; `dataset_revision` is the reference.
- Table `predictions`: `docid, gold, output, parse_ok, cut_off, n_gold_events, n_pred_events` (gold = target string, output = raw text).
- Identity: `name` from YAML, `tags` from YAML verbatim, `job_type="eval"` hardcoded (milestone 3 uses `train`), no group. A re-run is a new run with the same name.
- After logging, the text table prints as today and `run.url` prints last.

### Notebook (issue 10)

`notebooks/colab.ipynb`, authored as JSON via `nbformat`, strictly milestone-2 cells, with a markdown cell on top saying "Run all twice: the install cell restarts the runtime once":

1. runtime check — `nvidia-smi`, torch version, Python version;
2. clone + checkout — `REF` form field (default `main`); clone skipped if the directory exists; `git checkout $REF`; change directory;
3. install — guarded by "vLLM not importable": the vLLM install line, then `pip install -e .`, then a kernel restart only when it installed;
4. secrets — `HF_TOKEN` (set now, used in milestone 3) and `WANDB_API_KEY` from Colab Secrets into the environment, asserting non-empty;
5. gold — write the dev split from the Hub dataset to the prepared JSONL path with `to_json`;
6. smoke — zero-shot `generate --limit 5` to a scratch output directory, then show the JSONL; no W&B;
7. zero-shot — `generate` then `eval --wandb`;
8. 3-shot — the same for the 3-shot YAML.

The always-empty baseline is run locally, not in the notebook.

### Repository housekeeping (issues 03, 09, 10)

- `.gitignore`: `data/` → `/data/` so the data codec under `src/data` is tracked (done in the working tree; commit pending).
- `requires-python` relaxed to `>=3.11,<3.14` (Colab is 3.13.15; lock verified to resolve); the design doc's comment follows.
- `nbstripout` as a dev dependency, installed as a git filter with a committed `.gitattributes` (`*.ipynb filter=nbstripout`); the per-clone `--install` step is documented in the README setup and executed in this clone during implementation.
- Docs: README gains a "Running a baseline" section (generate → eval `--wandb`, the three YAMLs, the local always-empty run) and the "Open in Colab" link; DESIGN §3's tree gains `generate.py` and the Python bound; DESIGN §5 names `Qwen3-4B-Instruct-2507` and drops the "check the Unsloth list" line.
- mypy: `ignore_missing_imports` for `vllm.*` and `torch.*` (and `wandb.*` if untyped).

## Testing Decisions

A good test drives a public seam with real data shapes and checks observable
behaviour — the text an engine receives, the rows and files written, the keys
logged — never internals. Prior art: `tests/test_prepare.py` (fixture corpus
under `tests/fixtures/corpus`, module-scoped `tokenizer` fixture with the
`tokenizer` marker for tests that need the Hub) and `tests/test_eval.py`
(hand-built gold/prediction pairs).

Seams, agreed with the user:

1. **`generate(examples, engine, tokenizer, exemplars, max_input_tokens)`** — the main seam. Examples built from the fixture corpus, a fake engine that records the inputs it received and returns canned results, the real tokenizer (`tokenizer` marker). Verifies zero-shot and 3-shot rendering (system prompt first, exemplar turns verbatim and in order, target document last, generation prompt at the end, no `<think>`), `cut_off` from the finish reason, and that an over-budget input raises naming the docid. `render_inputs` is not tested on its own.
2. **`constant_engine()`** — returns `[]` with finish reason `stop` for every input; no tokenizer.
3. **generation CLI** — one run with `engine: constant` and `--limit 3` into a temporary directory; checks the JSONL rows and every `meta.json` field. Needs the Hub dataset and tokenizer → `tokenizer` marker.
4. **`flatten` and the W&B payload builder** — pure functions over a scorer result from hand-built pairs: exact key names, the two sidecar counts, the config stamps. `wandb.init` is never called in tests; the `--wandb` branch of the scorer CLI is validated by the local always-empty run.
5. **Notebook** — loads with `nbformat`, no cell has outputs, every config path it mentions exists.

Run `uv run mypy src`, `uv run ruff check .`, and `uv run pytest` (full suite once at the end; `-m 'not tokenizer'` stays green offline).

## Out of Scope

- Running the three baselines and writing up their numbers (execution follows the spec; `RESULTS.md` is milestone 6).
- A `transformers.generate` or Unsloth engine (only if vLLM fails on the T4 — new ticket).
- Milestone 3: training loop, eval callback wiring, `train.py`, adapter pushes (`HF_TOKEN` is merely set).
- Colab Pay-As-You-Go; rented GPUs.
- A 0.6B baseline config; the smoke model stays a training-only concern.
- W&B resume/overwrite semantics, groups, sweeps.
- A shared config module or typed config class.

## Further Notes

- **Done criterion**: three runs in `muc4-event-extraction` on the 200 dev documents — `qwen3-4b-zero-shot`, `qwen3-4b-3-shot`, `always-empty` — each with config, 57 metrics, the `predictions` artifact and table; the always-empty run produced locally on CPU first.
- **Order of work that de-risks earliest**: (1) `.gitignore` fix + `src/data` commit; (2) `generate` core + constant engine + tests; (3) `eval --wandb` + `flatten` + tests, then the local always-empty run to prove the W&B path; (4) `vllm_engine`; (5) YAMLs; (6) notebook + filter + test; (7) docs.
- **Vocabulary**: *exemplar*, *engine* and the sharpened *input* are defined in `CONTEXT.md` (added 2026-09-20). The engine's per-input limit is deliberately named `max_input_tokens`, not "input budget", which the glossary reserves for the per-example limit applied at data preparation.
- **Facts worth re-checking in the notebook before trusting the T4 plan**: Colab's live torch version and driver (ticket 05's snapshot was 2026.07); no first-hand report of vLLM 0.29.0 on a Colab T4 exists — hence the fallback order.
- Ticket 01's entity name/project URL and ticket 02's GPU name were not recorded by the user; nothing depends on them.
