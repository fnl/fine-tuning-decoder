# 08 The training config schema

Type: grilling
Status: resolved
Blocked by: 01, 02
HITL: yes

## Question

`configs/qwen3-0.6b-smoke.yaml` exists as a skeleton written before
`train.py` did; ticket 01 will have fixed the real TRL 0.24 field names.
Settle the schema the spec states verbatim, and which keys milestone 4's
4B config merely changes versus adds:

- Subsets: `train_docs` / `dev_docs` as "first N rows", and what their
  absence means (the full split).
- Sequence: `max_seq_len` 2048, packing off — is packing a key at all, or a
  hardcoded `False` with a comment?
- Optimisation: `epochs`, `lr`, `scheduler`, `warmup_ratio`,
  `effective_batch_size` + `per_device_batch_size` (accumulation derived, or
  stated?), and whether a `seed` is required for a reproducible smoke run.
- LoRA: the `lora:` block as it stands (r, alpha, dropout, target_modules).
- Precision: `quantization: nf4 | none` and `dtype` — chosen in the YAML, or
  detected from the GPU with the YAML only able to override?
- Cadence: `eval_every: 0.5` in epochs → steps; does the same number drive
  `save_steps`, given that checkpoints are pushed at every eval point (Q6)?
- Artifacts: the explicit `adapter:` key (`fnl-es/qwen3-0.6b-muc4-lora-smoke`),
  `wandb_project`, `name`, `tags`.
- Generation for the callback: reuse the `generation:` block of the baseline
  YAMLs (`max_new_tokens`, and whatever the in-process engine needs in place
  of `max_model_len`)?
- Validation: does `train.py` fail loudly on an unknown or missing key, and
  is the config a plain dict, a dataclass, or a TypedDict? (Milestone 2's
  spec deliberately avoided a shared config module — revisit or uphold.)

Fixes `configs/qwen3-0.6b-smoke.yaml` and the spec's config section.

## Comments

**2026-09-22, from ticket 06 (proof criteria):** the acceptance criteria
impose a floor on this config. `train_docs: 100` at `effective_batch_size: 16`
for 1 epoch is 6 optimizer steps and 2 eval points — too few for the "loss
trends downward" criterion to mean anything. Requirement: **≥ ~18 optimizer
steps and ≥ 4 eval points**. Recommended resolution: keep
`effective_batch_size: 16` (the 4B run's value) and set the smoke to 3
epochs → 18 steps, 6 eval points. Ticket 03 adds a second constraint: the
callback's generation batch size differs by model (16 at 0.6B, 4–8 at 4B), so
it must be a YAML knob, not a constant.

**2026-09-22, from ticket 01 (API names and traps):** use TRL 0.24 spellings —
`max_length` (0.18's `max_seq_length` is gone), `warmup_steps` (transformers 5.x
deprecates `warmup_ratio`; a float in [0,1) is read as a ratio), `report_to`
must be set explicitly (defaults to `"none"`). Two traps the YAML must dodge:
`SFTConfig.__post_init__` sets `bf16 = not fp16` when `bf16` is None, so state
`fp16: true, bf16: false` for the T4; and `max_length: 1024` equals TRL's own
default, which Unsloth reads as "unset" and widens to the model's maximum — so
2048 is also the safer value for that reason.

## Answer

Settled 2026-09-22 (delegated to the agent). The YAML is **our** vocabulary,
not TRL's: `train.py` translates to TRL's spellings, so the config outlives
the pinned library.

```yaml
name: qwen3-0.6b-smoke
tags: [smoke, qlora]
model: Qwen/Qwen3-0.6B
dataset: fnl-es/muc4-chat
adapter: fnl-es/qwen3-0.6b-muc4-lora-smoke
train_docs: 100              # absent = the whole split
dev_docs: 50
max_seq_len: 2048
epochs: 3
lr: 2.0e-4
scheduler: cosine
warmup_ratio: 0.03
effective_batch_size: 16
per_device_batch_size: 2
seed: 3407
quantization: nf4            # nf4 | none
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
eval_every: 0.5              # epochs; drives evaluation *and* checkpointing
generation:
  max_new_tokens: 512
  batch_size: 16
wandb_project: muc4-event-extraction
```

### Decisions

- **`epochs: 3`, not 1** — ticket 06's floor (≥ ~18 optimizer steps, ≥ 4 eval
  points). At effective batch 16 this gives 18 steps and 6 eval points.
  `effective_batch_size` stays 16 because it is the 4B run's value: the smoke
  run must exercise the knob milestone 4 uses.
- **Accumulation is derived**, not stated: `gradient_accumulation_steps =
  effective_batch_size // per_device_batch_size`, with a loud error when the
  division is not exact. Two numbers that must agree are one number.
- **`dtype` is not a key.** It is a property of the GPU, not of the
  experiment: `fp16` on sm_75, `bf16` from Ampere up. Detected in `train.py`
  — so milestone 5's bf16 LoRA run needs no YAML change, which is the point.
  `quantization` *is* a key, because QLoRA-vs-LoRA is a real experiment
  variable. Trap from ticket 01: `SFTConfig.__post_init__` sets
  `bf16 = not fp16` when `bf16` is None, so `train.py` states both explicitly.
- **`warmup_ratio` stays our key**, converted to TRL's `warmup_steps` (the
  ratio is deprecated in transformers 5.x). DESIGN §7 speaks of a 3 % warmup;
  the YAML keeps that language.
- **`max_seq_len: 2048` → TRL's `max_length`.** Also dodges ticket 01's trap:
  `max_length: 1024` equals TRL's default, which Unsloth reads as "unset".
- **`packing` and `padding_free` are not keys** — hardcoded `False` in
  `train.py` with a comment. DESIGN §7 fixes packing off, and ticket 01 found
  Unsloth silently auto-enables `padding_free`, which we refuse explicitly
  rather than inherit.
- **`eval_every: 0.5` drives both** evaluation and checkpointing
  (`save_steps == eval_steps`), because Q6 checkpoints at every eval point.
- **`seed: 3407`** — required, stated rather than inherited; it happens to be
  Unsloth's own default, so the explicit value changes nothing today and
  survives a change of default.
- **`generation.batch_size`** is a key because ticket 03 measured it as
  model-dependent (16 at 0.6B, 4–8 at 4B).
- **`tags`** added (ticket 04 flagged its absence); `adapter` explicit rather
  than derived from `name` (Q6).

### Over-budget examples: drop and count, never assert

Measured 2026-09-22: `prepare.py` budgets the *input* at 1800 tokens, but the
target adds up to 152 more, so **1 of 1299 train documents totals 2082 tokens**
— over `max_seq_len`. None of the 200 dev documents exceeds it (max 1398), and
none of the first 100 train documents does.

So an assertion would pass the smoke run and then fail milestone 4's full run
on a single document. Decision: `train.py` **drops** training examples over
`max_seq_len`, counts them, logs the count and puts it in the W&B config —
extending `prepare.py`'s existing *dropped document* concept rather than
inventing a second policy. This also covers ticket 01's finding that Unsloth
does not enforce `max_length` on a pre-tokenised split: nothing silently
truncates, because nothing over budget is there. The callback's dev path is
unaffected — `generate()` already raises on an over-budget input.

### Validation

`train.py` owns a `load_config` that requires every key above, rejects unknown
keys by name, and checks the batch-size division. Milestone 2 deliberately
avoided a shared config module; that holds — `generate.py` and `eval.py` keep
their own readers, and unifying them is round two.

### What milestone 4's config changes

Only values: `model`, `adapter`, `name`, `tags`, `per_device_batch_size`,
`generation.batch_size`, and `train_docs`/`dev_docs` **omitted** for the full
splits. No new keys — which is the Q10 requirement.
