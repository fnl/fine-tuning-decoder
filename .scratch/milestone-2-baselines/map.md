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

## Not yet specified

- Few-shot input budget: with three exemplars in context, how the truncated
  document policy (DESIGN §6: truncate input, report count) interacts with
  the engine's max context — sharpens once the engine and exemplars are chosen.
- Reproducibility fields on a run: commit SHA, config hash, engine version —
  what exactly the notebook stamps into W&B config; sharpens with the logging
  schema.

## Out of scope

- Colab Pay-As-You-Go setup — needed for milestones 4/5, not for 200 dev docs
  on a T4.
- Running the baselines and writing up numbers — execution after the spec.
- Milestone 3 (training loop, eval callback) — `generate.py`'s callable core
  is designed for it, but wiring it is a later map.
