# 08 The final evaluation run

Type: grilling
Status: resolved
Blocked by: 04
HITL: yes

## Question

How is the finished adapter scored over all 200 dev documents? Choose the
engine (vLLM + LoRA over an fp16 or bnb base, or the Unsloth in-process
engine over NF4) from 04's options, and settle: which notebook runs it, its
experiment YAML (a `qwen3-4b-r16-eval`?), `job_type=eval` with the same
metric keys as the milestone-2 baseline runs so they line up in W&B, and how
it records which adapter revision it scored and links back to the training
run.

## Answer

Resolved 2026-09-26 by grilling (all recommendations accepted).

1. **Engine: (A) vLLM, fp16 official base + LoRA** (ticket 04's option A).
   It is the baselines' engine, rendering and sampling, so the adapter is the
   only variable. The comparison note names the served base as fp16; the
   NF4-vs-fp16 difference is measured for free (item 8).
2. **De-risk first:** the build's first step is a smoke eval, a
   `configs/qwen3-0.6b-smoke-eval.yaml` serving `fnl-es/qwen3-0.6b-muc4-lora-smoke`
   at a pinned revision over `Qwen/Qwen3-0.6B`, first 50 dev docs, from
   `baselines.ipynb`. It passes if it runs and scores near the smoke run's
   16.3 (`63pxlpvn`), within ±3 plus the base difference.
3. **Fallback if vLLM LoRA fails on the T4: (C)** an `unsloth` engine in
   `generate.py` loading the NF4 base plus the pinned adapter. Everything
   downstream is unchanged, and the note then names the served base as NF4. Not (B).
4. **Pinned adapter:** the eval scores the exact commit the training run
   recorded as `adapter_revision` in its summary, never the repo's `main`.
   The engine snapshots only `adapter_*` files at that revision to a local path,
   because `LoRARequest` takes no revision.
5. **Link back:** config values only. The eval YAML carries `adapter`,
   `adapter_revision` and `training_run` (the W&B run id). `eval.py` already
   logs the whole YAML as run config, so they become stamps with no new
   stamping code. No artifact lineage, no W&B group.
6. **Code shape:** extend `generate.py`; no new module, no change to `eval.py`.
   The three keys are optional, and `adapter_revision` and `training_run` are
   required when `adapter` is set. When `adapter` is present, `vllm_engine`
   sets `enable_lora=True` and passes a `LoRARequest`. Metric keys match the
   baselines by construction; generation settings too (greedy,
   `max_new_tokens` 512, `max_model_len` 3072, `max_num_seqs` 16). CPU tests:
   config validation; snapshot patterns fetch `adapter_*`, never
   `last-checkpoint/`. The three values are copied by hand from the training
   run; a W&B lookup was rejected so that `generate.py` stays W&B-free.
7. **Notebook and YAML lifecycle:** `configs/qwen3-4b-r16-eval.yaml`, tags
   `[fine-tuned, qlora]`, is written and committed **after** training (it is the
   record of what was scored), then run from new generate → eval cells in
   `baselines.ipynb`, whose title widens. `job_type=eval`, split `dev`, all 200
   docs. A second served base in milestone 5 gets its own suffixed YAML.
8. **NF4-vs-fp16 cross-check:** ticket 09's job. Rescoring the eval JSONL's
   first 50 rows (the same docids as the training run's final eval point) is a
   local CPU step. This run only guarantees the rows exist (predictions
   artifact).
