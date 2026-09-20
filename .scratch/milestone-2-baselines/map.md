# Map: Milestone 2 — baselines of Qwen3-4B-Instruct on dev

Label: wayfinder:map
Created: 2026-09-20
Source: `docs/DESIGN.md` §2, §5, §6, §8, §9, §11 (milestone 2); charting interview 2026-09-20

## Destination

A spec ready for `/implement` that reaches DESIGN §11 milestone 2: zero-shot,
3-shot and always-`[]` baselines on the 200 dev docs, scored by `src/eval.py`,
each logged as one W&B run in `muc4-event-extraction`, driven from
`notebooks/colab.ipynb` on a free-tier T4. Every human setup task (W&B
account, Colab account + secrets, GitHub remote) is done inside this map, so
what follows the map is pure build.

## Notes

- Vocabulary: `CONTEXT.md` (event / event type / subset match / truncated
  document / cut-off output). Consult before every ticket.
- Skills: `grill-me` + `domain-modeling` for grilling tickets; `prototype`
  for the exemplar ticket; `to-spec` for the final ticket.
- Decisions already locked in the charting interview (do not re-open):
  - Milestone 1 is done: `fnl-es/muc4-chat` on the Hub (1299/200/200),
    `src/eval.py` scores `{docid, output}` JSONL, 596 tests green.
  - Generation lives in a new `src/generate.py` (callable core + thin CLI);
    `src/eval.py` stays CPU-only scorer and gains W&B logging.
  - 3-shot = multi-turn chat (system, 3 × user/assistant exemplars, target
    doc) with **fixed** exemplars pinned by docid in the YAML.
  - One YAML per baseline run (`qwen3-4b-zero-shot`, `qwen3-4b-3-shot`,
    `always-empty`); W&B logs config, flattened `score()` metrics, raw
    outputs JSONL as artifact, predictions Table.
  - Always-`[]` baseline runs locally on CPU and proves the W&B path first.
  - Notebook: `.ipynb` committed with outputs stripped, strictly M2 cells
    (clone@ref → pip GPU deps → Colab Secrets → generate → eval).
  - Colab: free tier only for this map. W&B: personal entity, project
    `muc4-event-extraction`. GitHub: public `fnl/fine-tuning-decoder`.
- User preferences: answers immediately with a recommendation; justify
  defaults; W&B integration visible and simple; only commit when asked.

## Decisions so far

<!-- one line per resolved ticket: gist + link -->

- [Which Qwen3-4B instruct checkpoint?](issues/04-base-model-id.md): `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, 2025-08-05; Qwen3.5-4B rejected — VLM hybrid, Unsloth advises against QLoRA). Unsloth mirror `unsloth/Qwen3-4B-Instruct-2507` is T4-proven, but take the chat template from the official repo: the mirror's hybrid template injects an empty `<think>` block into assistant turns. No `enable_thinking` on 2507 (smoke `Qwen3-0.6B` still needs it off). Tokenizer byte-identical across the family, so `n_input_tokens`/`truncated` hold. Side finding: `.gitignore` `data/` also ignores `src/data/` — fix in ticket 03.
- [Does vLLM run Qwen3-4B on a free Colab T4 today?](issues/05-vllm-on-t4.md): yes on paper — vLLM 0.29.0 from the cu129 wheel index, `dtype="half"`, `max_model_len=2560`, explicit `temperature=0.0` (else Qwen's generation_config temp 0.7 applies); T4/sm_75 is a supported floor with TRITON_ATTN. Memory fits (8 GB weights, ~12–15 concurrent docs), est. 6–12 min for 200 docs. Unsloth is not a baseline option (torch/transformers pins clash with vLLM; NF4 weights would change the baseline). Risk: no first-hand 0.29.0-on-Colab-T4 report; install replaces Colab's torch and needs a runtime restart. Fallbacks in order: `vllm==0.26.0` (pins Colab's torch 2.11), then batched `transformers.generate` (est. 10–25 min). Feeds ticket 06.

- [W&B account](issues/01-wandb-account.md): done; key in git-ignored `.envrc` via direnv; project `muc4-event-extraction`, personal entity.
- [Colab setup](issues/02-colab-setup.md): done; secrets `HF_TOKEN` + `WANDB_API_KEY` set; runtime Python 3.13.15, CUDA 12.8 — verify torch/driver in-notebook before picking the vLLM wheel index.
- [GitHub remote](issues/03-github-remote.md): `github.com/fnl/fine-tuning-decoder`, `main` @ `9214c69`. Pending: `.gitignore` fixed to `/data/` (uncommitted), `src/data/` still needs add + commit + push.

- [Inference engine](issues/06-inference-engine.md): vLLM 0.29.0 only (cu129 wheels, pinned, notebook-only), `dtype=half`, greedy, 512 new tokens, **`max_model_len=3072`** (3-shot needs it: dev max input 1375 + ~700–1000 exemplar tokens), T4 knobs hardcoded in `vllm_engine()`; pre-render with official tokenizer + `enable_thinking=False`; `Engine = Callable[[list[str]], list[GenerationResult]]` with `vllm_engine`/`constant_engine`; over-budget input fails loudly, no second truncation; finish reason recorded for cut-off outputs. Fallback if install breaks: vllm 0.26.0, then a ticket for HF generate.

- [3-shot exemplars](issues/07-exemplars.md): `DEV-MUC3-0512` (empty) → `DEV-MUC3-0126` (1 attack, all 5 roles) → `DEV-MUC3-0094` (2 bombings), spliced verbatim from the train split by docid; prefix 665 tokens, dev 3-shot input max 1857 < 2560 budget, 0 over → `max_model_len=3072` confirmed. Prototype: `prototype_exemplars.py` in this directory.

- [W&B schema](issues/08-wandb-logging-schema.md): `eval.py --wandb` owns the run; `generate.py` writes `outputs/<name>/<split>.jsonl` + `meta.json` (engine_version, git_sha/dirty, dataset_revision, gpu, n_docs/n_truncated/n_cut_off, wall_seconds). config = YAML verbatim + stamps; metrics = `flatten(score())` with `/` (55) + `diagnostics/n_truncated`, `n_cut_off`, one `run.log`; artifact `<name>-<split>` type `predictions`; Table `predictions` (docid, gold, output, parse_ok, cut_off, n_gold_events, n_pred_events); name/tags from YAML, `job_type=eval`; `run.url` printed last.

- [Code contract](issues/09-code-contract.md): `src/generate.py` — `Engine = Callable[[list[str]], list[GenerationResult]]`, `render_inputs` / `generate(examples, engine, tokenizer, exemplars, input_budget)` → rows `{docid, output, cut_off}`, `vllm_engine` / `constant_engine`; CLI `python -m generate --config --split --limit --out`; YAML keys name/tags/model/dataset/engine/exemplars/generation{max_new_tokens,max_model_len}/wandb_project; `eval.py` gains `--config`/`--wandb`, `--gold` unchanged (notebook dumps dev via `to_json`); `requires-python` → `<3.14`; README + DESIGN §3/§5 updates.

- [Notebook layout](issues/10-notebook-layout.md): 8 cells — runtime check, clone@`REF` form field, guarded vLLM + `pip install -e .` with auto-restart, secrets → env, gold dump via `to_json`, `--limit 5` smoke, zero-shot cell, 3-shot cell (each `generate && eval --wandb`). nbstripout git filter via `.gitattributes` (dev dep, `--install` per clone); nbformat CPU test (no outputs, configs exist); README Colab link.

## Not yet specified

- (none — both fogs graduated in tickets 06 and 08)

## Out of scope

- Colab Pay-As-You-Go setup — needed for milestones 4/5, not for 200 dev docs
  on a T4.
- Running the baselines and writing up numbers — execution after the spec.
- Milestone 3 (training loop, eval callback) — `generate.py`'s callable core
  is designed for it, but wiring it is a later map.
