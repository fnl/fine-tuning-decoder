# Agents

## Project

A learning project: fine-tune a small (≤4B) decoder LLM — Qwen3-4B-Instruct via
Unsloth/TRL QLoRA on Google Colab — for document-level event extraction on the
MUC-4 dataset. Core concepts: a *document* yields 0–N *events* (an event
type + 5 roles: PerpInd, PerpOrg, Target, Victim, Weapon); each role holds
*entities*, each entity a list of coreferent *mentions*. The model emits
events as JSON; a ported GTT scorer computes F1. Vocabulary is fixed in
`CONTEXT.md` (the corpus and GTT call an event a *template*). Full decisions
in `docs/DESIGN.md`; next steps in `docs/PROJECT_PLANNING.md`.

Python ≥3.11 managed with `uv`. CPU-only locally (data prep, scorer, tests);
training and inference run on Colab/rented GPUs from `notebooks/colab.ipynb`.
GPU-only deps (unsloth, trl, bitsandbytes, vllm) are installed by the
notebook, not `pyproject.toml`.

## Workflow

1. Make changes
2. Ensure type checks, linter, and tests pass.
3. Commit changes when asked.

Run type-checker using: `uv run mypy src`
Run tests using: `uv run pytest`
Run linter using: `uv run ruff check .`

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
