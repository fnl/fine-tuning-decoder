# 04 Who owns the W&B run during training: TRL or us?

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

Milestone 2 settled that `eval.py --wandb` owns its run, calls `wandb.init`
itself and logs a flat metric per scorer number (`.scratch/milestone-2-baselines/issues/08-wandb-logging-schema.md`).
Training is different: TRL logs the loss itself through `report_to="wandb"`.
Establish:

1. Whether `SFTTrainer`/`Trainer` with `report_to="wandb"` calls
   `wandb.init` itself, or picks up a run we started first — and which
   order is supported (the `WandbCallback` in `transformers`).
2. How our eval callback logs into that same run so `dev/micro_avg/f1` lines
   up with `train/loss` on the step axis: `wandb.log(..., step=state.global_step)`
   vs. `trainer.log(...)` vs. returning metrics from `on_evaluate`. What
   breaks when two loggers write the same step.
3. How the run is named and tagged from our YAML (`run_name` in the args vs.
   `WANDB_NAME`), and how to set `job_type=train`.
4. Whether logging a `wandb.Table` of predictions at the end of training is
   allowed in a run TRL owns, and whether an artifact (the adapter) can be
   attached to it.
5. What `transformers`' `WandbCallback` logs on its own that we would be
   duplicating (config, gradients, the model as an artifact — and how to
   turn the latter off).

Answer in this file with sources and dates. Feeds ticket 10.

## Answer

Resolved 2026-09-22. Sources are real downloaded source unless marked otherwise.
Paths below are relative to the sdist/wheel root; the copies read were
`transformers-5.5.0` (sdist, PyPI, downloaded 2026-09-22),
`trl-0.24.0` (sdist), `unsloth-2026.9.9` (wheel), `unsloth_zoo-2026.9.7` (sdist),
and `wandb 0.30.0` (installed in this repo's venv — the version the repo already
depends on). `WandbCallback` is **byte-identical** between transformers 5.5.0 and
5.17.0 (diffed 2026-09-22), so nothing here is version-fragile inside the pin.

### 1. Who calls `wandb.init` — and which order is supported

`Trainer.__init__` turns `report_to` into callbacks
(`src/transformers/trainer.py:550`, `get_reporting_integration_callbacks`,
`integrations/integration_utils.py:2590`) and appends *our* callbacks after the
default ones (`trainer.py:559`). `SFTTrainer`/`SFTConfig` add nothing: `SFTConfig`
subclasses `TrainingArguments` and never mentions `report_to` or `run_name`
(`trl/trainer/sft_config.py`, grep 2026-09-22 — zero hits). TRL touches wandb only
to read `wandb.run.url` for the model card (`trl/trainer/base_trainer.py:78`).

`WandbCallback.setup()` (`integration_utils.py:710`) does:

```python
if self._wandb.run is None:                      # :760
    self._wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), **init_args)
# add config parameters (run may have been created manually)   # :765 (verbatim comment)
self._wandb.config.update(combined_dict or {}, allow_val_change=True)   # :766
```

**Both orders are supported and the "we init first" order is the one the code
explicitly anticipates** (the `:765` comment). If no run exists it creates one
itself, from `WANDB_PROJECT` + `args.run_name` only — no tags, no `job_type`, no
entity. `setup()` is called from `on_train_begin` (`:825`) and defensively from
`on_log` (`:880`) and `on_predict`.

**Recommendation: `train.py` calls `wandb.init(...)` itself before constructing
`SFTTrainer`**, exactly as `eval.py:log_run` does, and passes
`report_to="wandb"` in `SFTConfig`. We then get M2's identity fields
(`project`/`name`/`tags`/`job_type`/`config`) and TRL/HF adopts the open run.
Note `report_to` defaults to `"none"` in transformers 5.x
(`training_args.py:1028`; also stated on the HF callbacks doc page, fetched
2026-09-22: "By default, `TrainingArguments.report_to` is set to `\"none\"`") —
so it *must* be set explicitly. Unsloth's patched `UnslothSFTConfig` also writes
`report_to = 'none'` as its generated default (`unsloth/models/rl.py:2615`).

**Who finishes the run: we do.** Nothing in `Trainer`, TRL or `WandbCallback`
calls `wandb.finish()` except the hyper-parameter-search branch
(`integration_utils.py:830`). So `train.py` owns `run.finish()` at the end and
prints `run.url`, mirroring `eval.py:log_run`.

**Unsloth gotcha (real, found in source).** `unsloth/models/rl.py:395-408`
wraps `train()` and, *if `train()` has already completed once in this process*
(`self._unsloth_training_completed`, set at `:437`), calls `wandb.finish()` and
resets `WandbCallback._initialized = False` so HF re-inits a fresh run
(upstream issue unslothai/unsloth#3954). Consequence for `train.ipynb`: re-running
the train cell in the same kernel **abandons our configured run and starts an
unconfigured one** (project from `WANDB_PROJECT`, name from `args.run_name`, no
tags/job_type/config). The first `train()` in a process is unaffected. Mitigation:
re-run the `wandb.init` cell too, or restart the runtime — state it in the notebook.
Unsloth's other wandb code (`unsloth_zoo/mlx/trainer.py`) is the Apple-MLX path and
never runs on a T4. `unsloth/import_fixes.py:5026 disable_broken_wandb()` only
neuters wandb when it is installed but unimportable (protobuf clash).

### 2. How the eval callback logs onto `train/loss`'s step axis

**The mechanism.** `WandbCallback.setup()` installs a custom x-axis
(`integration_utils.py:768-771`):

```python
self._wandb.define_metric("train/global_step")
self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)
```

and `on_log` logs **without `step=`** (`:899`):

```python
self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step})
```

So TRL's `train/loss` does *not* live on wandb's internal step at all: it lives on
the `train/global_step` metric, and the glob `"*"` makes that the x-axis for
**every** metric in the run, ours included.

**Recommended call** — in the eval callback, on the run object `train.py` opened:

```python
flat = {f"dev/{k}": v for k, v in flatten(result).items()}   # src/eval.py:flatten
self.run.log({**flat, "train/global_step": state.global_step})
```

No `step=`, no `commit=`. This is byte-for-byte the shape of `:899`, and it is the
same shape TRL's own `LogCompletionsCallback` uses
(`trl/trainer/callbacks.py:539-540`: `if "wandb" in args.report_to:
wandb.log({"completions": table})`). `flatten()` already emits `micro_avg/f1`,
`PerpInd/p`, `diagnostics/relevance_acc`, so the `dev/` prefix yields
`dev/micro_avg/f1` — the map's Q4 naming, unchanged from M2.
(`step_sync=True` means omitting `train/global_step` would also work — wandb
back-fills the last value — but being explicit is free and removes the dependence
on which writer logged last.)

**Why not the alternatives:**

- **`trainer.log(...)` mangles the keys.** `Trainer.log` (`trainer.py:3844`) fans
  out to `on_log`, and `WandbCallback.on_log` runs the dict through
  `rewrite_logs` (`integration_utils.py:549`), which prefixes `eval_`→`eval/`,
  `test_`→`test/` and **everything else with `train/`**. `trainer.log({"dev/micro_avg/f1": x})`
  lands as `train/dev/micro_avg/f1`. `Trainer.evaluate(metric_key_prefix="dev")`
  does not help either: `dev_micro_avg/f1` is not a recognised prefix, so it also
  becomes `train/dev_micro_avg/f1`. Only the literal prefix `eval` survives clean.
  `trainer.log` also injects `epoch` and appends to `state.log_history`
  (`trainer.py:3856-3864`), which is harmless but not what we want for 55 scalars.
- **Returning metrics from `on_evaluate` does nothing — and is actively
  dangerous.** `CallbackHandler.call_event` (`trainer_callback.py`) does
  `result = getattr(callback, event)(...)` then `if result is not None: control = result`.
  A returned metrics dict is silently assigned to `control`, and the next
  `control.should_log` access raises `AttributeError`. `TrainerCallback` hooks may
  only return a `TrainerControl`.
- **`on_evaluate` may never fire at all.** It is reached from
  `_maybe_log_save_evaluate` only when `control.should_evaluate`, i.e. when
  `eval_strategy` is set and `Trainer.evaluate()` runs a forward-pass loop we do
  not want (our eval is generate + `parse_target` + `score`). Hook **`on_step_end`**
  with our own cadence instead — TRL's `LogCompletionsCallback` does exactly this
  (`trl/trainer/callbacks.py:504-511`, incl. the `_last_logged_step` guard, because
  the hook can fire more than once per step). Note `on_step_end` fires *before*
  `_maybe_log_save_evaluate` for that step (`trainer.py:1774` then `:1775`), which
  is fine on the custom axis and matters only in the raw-Step view.

**What breaks when two writers share a step (W&B's monotonic-step rule).**
W&B's history is append-only per step: "It is not possible to write to a specific
history step. W&B only writes to the 'current' and 'next' step."
(https://docs.wandb.ai/models/track/log/, fetched 2026-09-22). `Run.log`'s own
docstring says the same (`wandb/sdk/wandb_run.py:1731-1733`): "By default, each
call to `log` creates a new 'step'. The step must always increase, and it is not
possible to log to a previous step." Client side, `Run._log`
(`wandb/sdk/wandb_run.py:1629-1662`) only ever advances its counter
(`if step > self._local_step: self._local_step = step`) but still forwards the
requested `step` to the backend (`interface.py:709-710`); the enforcement itself
lives in wandb-core, which is a compiled Go binary — so the *drop* behaviour is
documented, not source-verified here (see caveats).

Concretely, three cases:

- **Both writers omit `step=` (recommended).** Each `log()` call commits its own
  internal step. Nothing is lost. The two series sit on different internal steps
  but the same `train/global_step`, so every chart on the custom axis lines up.
  Only the raw "Step" x-axis shows them interleaved — cosmetic.
- **We pass `step=state.global_step` (the tempting mistake).** Unsloth's default
  is `logging_steps: 1` (`unsloth/models/rl.py:2606`), so the internal step tracks
  `global_step` — until the first extra `log()` call. `Trainer.evaluate` itself
  calls `self.log(output.metrics)` (`trainer.py:2596`), and so does every one of
  our own eval points; each adds one internal step. From then on the internal step
  **runs ahead** of `global_step`, our `step=` is a write into the past, and the
  history row is dropped (the run summary keeps the last value, so the number
  appears in the sidebar but the chart has a hole). Silent data loss.
- **`commit=False` accumulation across two writers.** Fragile: neither writer knows
  when the other commits, so our metrics can be flushed into whatever step TRL
  commits next. Do not use.

### 3. Naming, tagging, `job_type`

Because we own `wandb.init` (item 1), all of this comes from the YAML through the
init call, exactly as in `eval.py:log_run` — only `job_type` changes:

```python
run = wandb.init(
    project=config["wandb_project"],     # muc4-event-extraction
    name=config["name"],                 # qwen3-0.6b-smoke
    tags=config.get("tags", []),         # the smoke YAML has no tags: key yet
    job_type="train",                    # M2 hardcoded "eval"; map Q5 says train
    config=payload.config,               # YAML verbatim + provenance stamps
)
```

- **`run_name` vs `WANDB_NAME`.** Both are unnecessary once we init first, but the
  precedence is worth knowing: explicit `wandb.init(name=...)` wins over
  `WANDB_NAME`, because env vars are applied to the *global* settings
  (`wandb/sdk/wandb_setup.py:232 update_from_env_vars`) and the init kwargs are
  layered on top (`wandb/sdk/wandb_init.py:301 settings.update_from_settings(init_settings)`,
  with the kwargs folded into `init_settings` at `:1419-1428`). `WANDB_NAME` is a
  real wandb env var (`wandb/env.py:30`), as are `WANDB_TAGS` (`:59`),
  `WANDB_JOB_TYPE` (`:54`), `WANDB_RUN_GROUP` (`:46`).
  `TrainingArguments.run_name` defaults to `None` in transformers 5.5.0
  (`training_args.py:1034`; the v4-era "defaults to output_dir" post-init is gone)
  and only reaches wandb as `init_args["name"]` *if the run does not exist yet*
  (`integration_utils.py:751-752`). **Still set `run_name=config["name"]` in
  `SFTConfig`**: it costs nothing, it is what `WANDB_LOG_MODEL` artifact names key
  off (`:868`), and it is the fallback identity if the Unsloth double-`train()`
  path from item 1 ever fires.
- **`job_type` has no `TrainingArguments` equivalent** — `WandbCallback` never sets
  it. It can only come from our `wandb.init(job_type="train")` (or `WANDB_JOB_TYPE`).
  Same for `tags`. This is the single strongest argument for owning the init.
- **`WANDB_PROJECT` is irrelevant once we init first** — it is read only inside the
  `if self._wandb.run is None` branch (`integration_utils.py:762`). M2's rule
  ("`wandb_project` from YAML, key from env only") carries over unchanged;
  `WANDB_API_KEY` stays env-only.
- **The smoke YAML has no `tags:` key** (`configs/qwen3-0.6b-smoke.yaml`, read
  2026-09-22). Ticket 10 should add one, e.g. `tags: [smoke, qlora, qwen3-0.6b]`,
  so the M2 tag convention holds.

### 4. `wandb.Table` and an adapter artifact in a TRL-owned run

**Both are allowed, with no special ceremony.** There is one run object; anything
`eval.py` does to a run it opened, the callback can do to the run `train.py`
opened. TRL ships the precedent itself: `LogCompletionsCallback` logs a table into
the trainer's run from a plain `TrainerCallback`
(`trl/trainer/callbacks.py:539-540`).

- **Table.** Log it from `on_train_end` with the M2 `TABLE_COLUMNS`
  (`src/eval.py:TABLE_COLUMNS`):
  `self.run.log({"predictions": wandb.Table(columns=list(TABLE_COLUMNS), data=rows), "train/global_step": state.global_step})`.
  Our `on_train_end` runs *after* `WandbCallback.on_train_end` (we are appended
  later, `trainer.py:559`) and after the final `train_runtime` log
  (`trainer.py:1838` precedes `:1854`), and the run is still open (item 1). Build a
  **fresh `wandb.Table` per log call** — `Media.bind_to_run`
  (`wandb/sdk/data_types/base_types/media.py:109-145`) mutates `self._path` and
  clears `_is_tmp`, so re-logging one Table object is a known footgun; TRL likewise
  rebuilds its DataFrame each time (`callbacks.py:537`). If the callback logs a
  per-eval-point table, log it under a per-step key or accept one table version per
  internal step.
- **Artifact.** `self.run.log_artifact(wandb.Artifact(f"{name}-adapter", type="model"))`
  with `artifact.add_dir(adapter_dir)` — same call `eval.py:log_run` already makes
  for `type="predictions"`. A 0.6B r16 LoRA is ~20 MB, so this is cheap
  (*estimate*: r=16 over 7 projections of Qwen3-0.6B, fp16 ≈ 20–40 MB). Use a name
  distinct from `model-{run.name}`, which is what `WANDB_LOG_MODEL` would use
  (`integration_utils.py:866-869`), so the two never version-collide. The adapter is
  also pushed to the Hub (map Q6), so the artifact is belt-and-braces; recording the
  Hub repo id in `run.config`/`run.summary` may be enough — ticket 10's call.

### 5. What `WandbCallback` logs on its own (what we would duplicate)

All from `integration_utils.py`, transformers 5.5.0:

| What | Where | Default | Overlap with us |
|---|---|---|---|
| `run.config` ← `args.to_dict()` (109 `TrainingArguments` fields) merged over `model.config.to_dict()`, plus `peft_config` | `:738-745`, `:766` | always | **`config.update(..., allow_val_change=True)` silently overwrites colliding keys of the config we passed to `wandb.init`.** Checked the smoke YAML against all 109 fields: the only collision is **`warmup_ratio`** (same value, so harmless today — but a YAML key that ever diverges from its `SFTConfig` counterpart will show the `SFTConfig` value). |
| `model/num_parameters` | `:781` | always | new, useful, keep |
| custom x-axis `train/global_step` + `define_metric("*", ...)` | `:768-771` | always | this is what makes item 2 work — **do not** call `define_metric` ourselves and do not fight it |
| `run._label(code="transformers_trainer")` | `:777` | always | telemetry only |
| `train/loss`, `train/grad_norm`, `train/learning_rate`, `train/epoch` (+ `train/num_input_tokens_seen` if enabled) | `:899` via `trainer.py:2057-2066` | always | the loss curve we want; we add `dev/*` alongside |
| summary-only scalars `train_runtime`, `train_samples_per_second`, `train_steps_per_second`, `train_loss`, `total_flos` | `:881-896` | always | summary, not history |
| `eval/*` (`eval/loss`, `eval/runtime`, …) | `rewrite_logs` `:549` | only if `eval_strategy` set | we are **not** using `Trainer.evaluate` (item 2), so this stays empty — our metric is `dev/*`, which cannot be confused with a teacher-forced `eval/loss` |
| **gradients / parameters** via `wandb.watch` | `:773-776` | **OFF** — `os.getenv("WANDB_WATCH", "false")` | nothing to turn off; just never set `WANDB_WATCH`. On a T4 leave it off (it uploads per-tensor histograms every `max(100, logging_steps)` steps). |
| **model as an artifact** (initial architecture at setup `:792-815`; final model at `on_train_end` `:839-878`; full checkpoint dir every save `:901-922`) | | **OFF** — `WandbLogModel(os.getenv("WANDB_LOG_MODEL", "false"))` at `:708`, `:672-690` | nothing to turn off. **Keep it off**: with map Q6 ("checkpoint at every eval point"), `WANDB_LOG_MODEL=checkpoint` would upload the whole checkpoint dir (optimizer state included) on every save. To be explicit and disconnect-proof in the notebook, set `os.environ["WANDB_LOG_MODEL"] = "false"` before importing; an unrecognised value degrades to `FALSE` with a warning (`:686-690`). Our own adapter artifact (item 4) is the controlled replacement. |

Note there is **no `WANDB_DISABLED`** in transformers 5.x (grep of
`integration_utils.py`, 0 hits, 2026-09-22) — the off switch is `report_to="none"`
or `wandb.init(mode="disabled")` / `WANDB_MODE=disabled`. Mirror M2's
`eval.py --wandb`: W&B stays off unless asked for, so CPU tests never touch it.

### Open caveats

- **Nothing was executed.** No Unsloth/TRL install exists locally (this repo's venv
  has transformers 5.17.0 and wandb 0.30.0 but no trl/unsloth/torch-cuda), and the
  brief forbids installing. Every claim is static source reading + official docs.
  The first smoke run on Colab is the real test; the cheap check is that
  `dev/micro_avg/f1` and `train/loss` share an x-axis in the same chart.
- **The step-drop behaviour is documented, not source-verified.** wandb's Python
  client only clamps its own counter upward (`wandb_run.py:1656-1657`); the actual
  rejection of a past step happens in wandb-core (compiled Go). The claim rests on
  the `Run.log` docstring and https://docs.wandb.ai/models/track/log/. The
  *recommendation* (never pass `step=`) does not depend on which way it fails.
- **Adapter size is an estimate** (~20–40 MB), not measured.
- **wandb 0.30.0 is the version in this repo's venv**, not necessarily the version
  Colab resolves in 2026-09. The APIs used (`init`, `log`, `Table`, `Artifact`,
  `define_metric`, `finish`) are long-stable; `WandbCallback` calls
  `define_metric` behind a `getattr` guard (`:769`), so a newer wandb cannot break
  the axis. Newer wandb docs push `run.log` over module-level `wandb.log` —
  `wandb.log` is still a live alias in 0.30.0 (`wandb/__init__.py:89`, not
  deprecated) and HF/TRL still use it, but our code should hold the run object.
- **transformers ≥5.6 not checked**, only 5.5.0 and 5.17.0 (identical
  `WandbCallback`). Outside the Unsloth pin anyway.
- **The Unsloth double-`train()` behaviour** (`rl.py:395-408`) was read in the
  2026.9.9 wheel; 2026.9.x patch releases could change the guard. It is only
  reachable on a second `train()` in one process.
- **Not answered here** (belongs to tickets 06/10): the eval cadence arithmetic
  (`eval_every` in epochs → step interval), whether the 50-doc dev score is meant
  to be comparable to the 200-doc baselines, and whether the callback also logs a
  per-eval-point predictions table or only a final one.
