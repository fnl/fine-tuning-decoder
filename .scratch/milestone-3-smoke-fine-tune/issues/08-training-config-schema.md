# 08 The training config schema

Type: grilling
Status: open
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
