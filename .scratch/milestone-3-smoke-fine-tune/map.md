# Map: Milestone 3 — smoke fine-tune of Qwen3-0.6B on 100 documents

Label: wayfinder:map
Created: 2026-09-22
Source: `docs/DESIGN.md` §7, §9, §10, §11 (milestone 3); charting interview 2026-09-22

## Destination

A spec ready for `/implement` that reaches DESIGN §11 milestone 3: `src/train.py`
fine-tunes `Qwen3-0.6B` on the first 100 train documents with completion-only
loss, an eval callback that runs the real scorer on 50 dev documents every ½
epoch into the same W&B run, and the LoRA adapter pushed to the Hub — driven
from a new `notebooks/train.ipynb` on a free-tier T4. Every setup task and
every fact the spec must state verbatim is settled inside this map; what
follows the map is pure build, and the smoke run itself is what `/implement`
verifies at the end.

## Notes

- Vocabulary: `CONTEXT.md` (smoke model / smoke run / adapter / experiment /
  run / engine / target / diagnostics). Consult before every ticket; the
  masking vocabulary is new and will need entries.
- Skills: `grill-me` + `domain-modeling` for grilling tickets; `prototype` for
  ticket 05; `to-spec` for ticket 12. There is **no** `research` skill in this
  install — research tickets are resolved by a subagent brief and answered in
  the ticket with sources and dates, as in the milestone-2 map.
- Prior art to copy, not re-derive: `.scratch/milestone-2-baselines/` (map,
  tickets, `spec.md`). `src/generate.py`'s `Engine` seam and `src/eval.py`'s
  `score()` / `flatten()` / W&B payload are designed to be reused here.
- Decisions locked in the charting interview (do not re-open):
  - **Destination is a spec**, not an executed run (Q1).
  - **Two notebooks** (Q2): new `notebooks/train.ipynb` for the Unsloth stack;
    `colab.ipynb` renamed `baselines.ipynb` for the vLLM stack. Unsloth and
    vLLM cannot share an environment (M2 ticket 05).
  - **Masking = our own pre-tokenisation** (Q3): `tokenize_example` renders
    prompt (`messages[:2]`, `add_generation_prompt=True`) and full
    (`messages`), asserts the token-level prefix, emits `input_ids` + `labels`
    with the prompt span `-100`; TRL consumes a processed dataset. Fallback if
    Unsloth re-tokenises: TRL's prompt-completion form (`completion_mask`),
    same arithmetic. Rejected: `assistant_only_loss` (TRL ≤0.24 cannot patch
    Qwen templates, and TRL 1.13's patched Qwen3 template trains the `<think>`
    block the inference prompt already supplies) and Unsloth's
    `train_on_responses_only` (same `<think>` skew, Unsloth-bound, untestable
    on CPU).
  - **Eval callback** (Q4) = `render_inputs` + an in-process engine + a
    `parse_target` + `score()` pass on the dev subset, logged to the same W&B
    run under a `dev/` prefix; cadence from a YAML `eval_every` in epochs.
  - **One W&B run per training run** (Q5), `job_type=train`; the M2-style
    separate eval run reappears in milestone 4.
  - **Checkpoint at every eval point and push every save** (Q6) — the Colab
    disconnect insurance milestone 3 exists to prove; explicit `adapter:` key
    in the YAML; adapter public, never merged.
  - **NF4 stays on for the smoke run** (Q7): it exercises the exact path the
    4B run will take. Knobs `quantization: nf4 | none` and a `dtype` chosen
    from the GPU.
  - **First N rows** of each split for the subsets (Q8) — the first 100 train
    documents are 40 empty / 40 one-event / 15 two / 5 three, the first 50 dev
    documents 25 empty / 25 relevant; same semantics as `generate --limit`.
  - **CPU-testable split** (Q9): pure functions (config, subset, masking, step
    arithmetic) tested with the real tokenizer; the callback tested with fakes;
    Unsloth/TRL glue imported lazily and untested locally.
  - `train.py` accepts the full-split config shape from the start so milestone
    4 is "write one YAML and run" (Q10).
- Facts established while charting (2026-09-22, do not re-derive):
  - `unsloth 2026.9.9` pins `trl!=0.19.0,>=0.18.2,<=0.24.0`,
    `transformers<=5.5.0`, `torch<2.13`; `unsloth_zoo 2026.9.7` allows
    `trl<=1.13.0`, so the intersection lands **TRL 0.24.0**.
  - TRL 0.24.0 accepts a pre-tokenised dataset: `is_processed = "input_ids" in
    column_names` skips tokenisation (`sft_trainer.py:887`), the collator uses
    a supplied `labels` column verbatim (`:160`), `labels` is in
    `_signature_columns` (`:1078`), and `truncate_dataset` slices every list
    column so `labels` stays aligned.
  - TRL 0.24.0 has no `chat_template_utils.py`; the auto-patched Qwen3
    templates exist only from TRL 1.13, out of reach under Unsloth's pin.
  - Neither `Qwen3-0.6B` nor `Qwen3-4B-Instruct-2507` carries `{% generation %}`.
  - The prompt is an exact **token-level** prefix of the full rendering for
    both models (verified on real examples, transformers 5.17): `DEV-MUC3-0001`
    632/705 tokens on 0.6B, 628/701 on 4B. An empty document's completion is 3
    tokens (`[]`, `<|im_end|>`, `\n`); prompts run 444–765 tokens.
  - Milestone 2 results to beat: zero-shot micro-F1 **18.6** (7.5 % parse
    failures), 3-shot **20.0** (5 %), always-empty 0. Published GTT ≈ 50–55.
- User preferences: answers immediately with a recommendation; wants defaults
  justified; only commit when asked.

## Decisions so far

<!-- one line per resolved ticket: gist + link -->

- [What counts as proof that the loop works?](issues/06-proof-criteria.md):
  **mechanical criteria only — no score gates the milestone.** Loss: a curve
  exists and the last quarter's mean is below the first quarter's (user's
  call). Masking: the CPU test is the real proof, plus a one-batch startup log
  of the decoded unmasked span, with first-step loss as corroboration.
  Checkpoint push: a final notebook cell loads the *pushed* adapter onto a
  fresh base model and generates one document — required, since the point is
  that the artifact survives the session. Callback: `dev/*` appears at the
  expected steps with `flatten(score())`'s key set; improvement not required.
  A near-zero micro-F1 is a **successful** smoke run; the real broken-pipeline
  smells are a parse-failure rate above the zero-shot 7.5 %, or an all-`[]`
  collapse together with a flat loss curve. **Constrains ticket 08**: 100 docs
  at effective batch 16 for 1 epoch is only 6 optimizer steps and 2 eval
  points, so the smoke needs ≥ ~18 steps and ≥ 4 eval points (recommended: 3
  epochs at effective batch 16 → 18 steps, 6 eval points; DESIGN **§9** — not §11 — names the
  smoke config as "1 epoch" and is superseded; ticket 12 corrects it).

- [The Unsloth + TRL stack and the pre-tokenised dataset](issues/01-unsloth-trl-install.md):
  **the decisive verdict is YES** — Unsloth replaces TRL's
  `_prepare_dataset` with `unsloth_zoo.dataset_utils.sft_prepare_dataset`,
  whose first branch is `if "labels" in column_names: do_tokenize = False`
  (`dataset_utils.py:2582-2588`), and the live collator is TRL 0.24's own,
  which uses the supplied `labels`. No re-tokenisation, no
  `dataset_text_field`, labels not dropped. **Q3 option (a) stands; the
  prompt-completion fallback is not needed.** Install: plain
  `pip install unsloth` (*not* Unsloth's notebook recipe — its torch→xformers
  map has no entry for 2.11 and falls back to a wheel pinned to torch 2.10),
  resolving unsloth 2026.9.9 / zoo 2026.9.7 / trl 0.24.0 / transformers 5.5.0
  / peft 0.21.0 / bnb 0.50.2. Colab's torch 2.11.0 satisfies every bound, so
  **no torch swap, no torchaudio clash, no restart** — unlike milestone 2.
  Two knock-ons for ticket 09: `max_length` is **not enforced** on a
  pre-tokenised split (Unsloth's prep replaces the truncating one), so we
  truncate or assert ourselves; and Unsloth **silently auto-enables
  `padding_free`** — masking survives, but set `padding_free=False`
  explicitly. API traps for ticket 08: `max_length` (not `max_seq_length`),
  `warmup_steps` (`warmup_ratio` deprecated in transformers 5.x),
  `report_to` defaults to `"none"`, `SFTConfig.__post_init__` sets
  `bf16 = not fp16` when bf16 is None (state `fp16=True, bf16=False`), and
  `max_length=1024` equals TRL's default so Unsloth treats it as unset.
  `get_peft_model`'s defaults are DESIGN §7 verbatim. The Unsloth mirror's
  chat template renders **byte-identical** to Qwen's for our message shape
  (only `pad_token` differs), so the milestone-2 mirror hazard does not
  recur — the `<think>` block is controlled by `enable_thinking=False`, which
  we already pass.

- [Adapter checkpointing and the Hub push](issues/02-checkpoints-and-hub-push.md):
  all stock `transformers 5.5.0` `TrainingArguments` — `SFTConfig` overrides
  none. **`hub_strategy="every_save"` does not deliver the disconnect
  insurance Q6 asked for**: it mirrors only `output_dir`'s root with
  `ignore_patterns=["checkpoint-*"]`, so no optimizer state, nothing
  resumable. Use `hub_strategy="checkpoint"` (adds the checkpoint folder as
  `last-checkpoint/`), `hub_always_push=True` (otherwise a save's push is
  *silently skipped* when the previous upload is still running on the single
  shared worker), `save_total_limit=2`, and omit `hub_token` (Colab's vault is
  read by `get_token()`). `"end"` pushes nothing — `trainer.push_to_hub()`
  after `train()` is needed regardless. Sizes: adapter 38.55 MiB (10,092,544
  fp32 params, byte-confirmed against a published Unsloth adapter), full
  checkpoint ≈ 131 MiB. Adapter only, no base weights; `Trainer.push_to_hub`
  is the entry point to use. Two gotchas: `adapter_config.json` will record
  `base_model_name_or_path: unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (the
  `-bnb-4bit` fixup is dead code behind `if False:`), and TRL rewrites
  `README.md` at every save with `library_name: transformers` and a broken
  snippet — a hand-written card is clobbered unless `create_model_card` is
  overridden. `resume_from_checkpoint` is **local-path only**: recovery means
  `snapshot_download(allow_patterns="last-checkpoint/*")` first.

- [See the mask: prototype `tokenize_example`](issues/05-mask-prototype.md):
  [`prototype_mask.py`](prototype_mask.py). The seam confirms Q3 visually —
  the 0.6B's empty `<think>` block lands on the **masked** side, because our
  inference prompt supplies it; the rejected mechanisms would have trained it.
  Decisions: (1) **truncate after the final `<|im_end|>`** — the template's
  trailing newline can never be generated, and for the 40 empty documents it
  was a third of everything they teach (120 → 80 unmasked tokens); (2) a named
  `MaskingError(ValueError)` naming the docid, not `assert` (which vanishes
  under `-O`); (3) signature `tokenize_example(example, tokenizer, *,
  stop_after_eos=True) -> {input_ids, labels}`, taking the example so the
  error can name the docid and it drops into `Dataset.map`; (4) **one code
  path serves both models — proven**: identical unmasked counts (3,459) and
  completion stats (min 2 / mean 34.6 / max 152), differing only by the four
  think-block tokens on the prompt side. **5.9 % of tokens carry loss.**

- [Generating with the model under training](issues/03-in-process-generation.md):
  `from_pretrained` already replaces `model.generate` with
  `unsloth_fast_generate`, which snapshots training state, calls
  `for_inference`, generates under `inference_mode`, and restores
  `for_training` — so **never call `for_inference` yourself** (it poisons the
  restore) and `eval()`/`no_grad()` are unnecessary. The engine owns two
  things: `tokenizer.padding_side="left"` (Unsloth sets it only *inside*
  generate, and TRL's SFT collator hard-codes right padding, so there is no
  conflict) and explicit greedy — Qwen3 ships `do_sample:true`, so pass
  `do_sample=False, temperature=None, top_p=None, top_k=None` plus an explicit
  `pad_token_id`. `use_cache` is handled end to end by Unsloth. Recommended:
  `unsloth_engine(model, tokenizer, *, max_new_tokens, batch_size)` beside
  `vllm_engine` in `generate.py`; length-sort inputs, slice the completion by
  prompt width (left padding makes this valid), derive `finish_reason` from
  eos presence. Batch 16 at 0.6B (~2.9 GB of 13.5), 4–8 at 4B — so batch size
  is a YAML knob, not a constant; est. 2–5 min per callback. Rejected:
  `compute_metrics` (teacher-forced logits: wrong semantics, ~36 GB for 50
  docs; no `predict_with_generate` outside Seq2Seq) and TRL's
  `unwrap_model_for_generation` (its exit calls
  `gradient_checkpointing_enable()`, which destroys Unsloth's wrapper).
  Silent failure mode if the model is left in training mode: gradient
  checkpointing forces `use_cache=False`, making decode quadratic — hours, not
  a crash.

- [Who owns the W&B run during training?](issues/04-wandb-run-ownership.md):
  **we do** — `train.py` calls `wandb.init` first (project/name/tags/job_type/
  config, as `eval.py:log_run` does), then `report_to="wandb"` (explicit:
  transformers 5.x defaults it to `"none"`); `WandbCallback` only inits when
  `wandb.run is None`, and nothing in TRL/Trainer calls `finish()`, so we do.
  The callback logs `{**{f"dev/{k}": v}, "train/global_step": state.global_step}`
  with **no** `step=` — HF pins every metric to `train/global_step` via
  `define_metric`, and an explicit `step=` silently drops rows. Rejected:
  `trainer.log` (mangles `dev/x` → `train/dev/x`), returning metrics from
  `on_evaluate` (the return value is assigned to `control`). `job_type`/`tags`
  come only from our `init` — `WandbCallback` never sets them. Gradients and
  model artifacts default off; keep them off. Config collision with the YAML is
  `warmup_ratio` alone. Two gotchas: Unsloth calls `wandb.finish()` on a
  *second* `train()` in one process (re-running the train cell abandons our
  configured run), and `on_evaluate` never fires without `eval_strategy`, so the
  callback hooks `on_step_end` with its own cadence. `WandbCallback` is
  byte-identical across transformers 5.5.0–5.17.0.

- [The training config schema](issues/08-training-config-schema.md): the YAML
  is **our** vocabulary, `train.py` translates to TRL's. `epochs: 3` (ticket
  06's floor), `effective_batch_size: 16` kept because it is the 4B's value,
  accumulation **derived** with a loud error when indivisible. `dtype` is
  **not** a key — a property of the GPU, detected, so milestone 5's bf16 run
  needs no YAML change; `quantization` stays a key because QLoRA-vs-LoRA is a
  real variable. `packing`/`padding_free` hardcoded `False`. `eval_every`
  drives evaluation *and* checkpointing. **Measured: 1 of 1299 train documents
  totals 2082 tokens** (input budget 1800 + a 152-token target), so over-budget
  training examples are **dropped and counted**, never asserted — an assertion
  would pass the smoke run and fail milestone 4 on one document. Milestone 4
  changes values only, no new keys.

- [The shape of a training run in W&B](issues/10-wandb-training-run-shape.md):
  one run, ours, `job_type=train`. `train/*` from TRL, `dev/*` = the 55
  `flatten(score())` keys under a prefix — kept **because** the 50-document
  score is not comparable to the 200-document baselines (ticket 06), so
  nothing averages them by accident. All 55 at each of the 6 eval points;
  `predictions` table and the proof-criteria summary at `on_train_end`. Config
  = YAML + stamps, with milestone 2's single `engine_version` becoming a
  `versions` mapping over the whole pin chain, plus every derived number
  (`total_steps`, `eval_steps`, `n_dropped`). **The adapter is not a W&B
  artifact** — the Hub is its home (DESIGN §9); the run records the repo id
  and final revision instead.

- [Contract of `train.py`](issues/09-train-code-contract.md): the fallback
  branch is dead (ticket 01). `unsloth_engine(model, tokenizer, *,
  max_new_tokens, batch_size)` joins `vllm_engine` in `generate.py`, so the
  callback is `generate()` + `parse_target` + `score()` and nothing else.
  `train.py` = `load_config`, `tokenize_example`, `select_subset`,
  `build_dataset` (drops over-budget, returns the count), `training_steps`,
  `eval_interval`, `ScoringCallback` (hooks **`on_step_end`**, since
  `on_evaluate` never fires without `eval_strategy`), `train`, `main`. GPU
  imports function-local so the module imports on CPU. Eight CPU tests in
  `tests/test_train.py`; `unsloth_engine` deliberately untested.

- [`train.ipynb` layout and repo hygiene](issues/11-train-notebook-and-repo-hygiene.md):
  8 cells and **no restart** — ticket 01 removed the "run all twice" dance, the
  `%cd` workaround and the `torchaudio` uninstall that cost milestone 2 three
  attempts. Cells: runtime check → guarded `pip install unsloth` (plain, not
  Unsloth's recipe) → clone@`REF` + `pip install -e .` → print resolved
  versions → secrets → `--limit 8` smoke → train → **load the pushed adapter
  back and generate** (ticket 06's proof). `colab.ipynb` → `baselines.ipynb`
  in this spec; `test_notebook.py` parameterised over both, with a new
  assertion that no cell calls `do_shutdown`. No gold-dump cell: `train.py`
  scores in-process from the Hub.

- [Spec](spec.md): written, `ready-for-agent`. `CONTEXT.md` gained a Training
  section; DESIGN §7 and §9 corrected in place with dated notes. **Map
  closed** — next: `/implement @.scratch/milestone-3-smoke-fine-tune/spec.md`.

## Not yet specified

- (none — every patch graduated or was ruled out of scope)

## Out of scope

- **A resume-after-disconnect path in the notebook.** Ticket 02 made it
  specifiable — `hub_strategy="checkpoint"` pushes the optimizer state, and
  recovery is `snapshot_download(allow_patterns="last-checkpoint/*")` then
  `resume_from_checkpoint` on the local path, since it never reads the Hub
  itself. Ruled out of this map all the same: milestone 3 must prove the
  adapter *survives*, which the push does; rebuilding a run from it is
  milestone 4's problem, when a run costs forty minutes instead of five.

- [Prove the Unsloth stack on a T4 inside the map](issues/07-t4-training-smoke.md)
  — **closed unrun, premise changed.** Ticket 01 settled its decisive question
  from source (`dataset_utils.py:2582-2588`) and found the install needs no
  restart, so neither justification survived. Its remaining value — peak
  memory, wall times — is produced by the first `/implement` run, which now
  opens with the `--limit 8` smoke as step 1.

- The full QLoRA fine-tune of Qwen3-4B and `configs/qwen3-4b-r16.yaml` —
  milestone 4.
- Evaluating a finished adapter M2-style (vLLM + LoRA over all 200 dev
  documents) and the fp16-base / NF4-adapter numerics question it raises —
  milestone 4.
- Colab Pay-As-You-Go, rented GPUs, bf16 LoRA — milestone 5.
- `RESULTS.md` and the written comparison — milestone 6.
- Merged / GGUF export, rank and lr sweeps, the plain-peft rewrite, the
  base-model variant, DocEE — round two (`docs/DESIGN.md` §11).
