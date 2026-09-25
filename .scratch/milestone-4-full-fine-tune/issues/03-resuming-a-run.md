# 03 Resuming a run from the Hub

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

The smoke adapter repo already holds a resumable `last-checkpoint/`
(`optimizer.pt`, `scheduler.pt`, `rng_state.pth`, `trainer_state.json`). What
does resuming from it actually take under our pinned stack (transformers
5.5.0, trl 0.24.0, unsloth 2026.9.11 / zoo 2026.9.7, peft 0.20.0, wandb)?
Read the installed sources; cite file:line.

1. **Trainer**: `resume_from_checkpoint` on a local
   `snapshot_download(allow_patterns="last-checkpoint/*")` path — what it
   restores (adapter weights via PEFT? optimizer, scheduler, RNG, scaler,
   `global_step`, data order / skipped batches) and what it needs present
   (e.g. `training_args.bin` compatibility). Any Unsloth override of
   `_load_from_checkpoint` or of the resume path for a 4-bit PEFT model.
2. **Our callback**: `ScoringCallback` in `src/train.py` hooks
   `on_step_end` with its own cadence — does it fire correctly after resume
   (no duplicate eval at the resumed step, no skipped one)? Does
   `TrainerState` carry anything it depends on?
3. **W&B**: resuming the same run id (`wandb.init(id=…, resume="must")`),
   how the HF `WandbCallback` behaves when `wandb.run` already exists on
   resume, step monotonicity with `train/global_step` as the x-axis, and
   whether the config update on train begin collides with our nested config.
4. **Unsloth's `wandb.finish()` on a second `train()`** (milestone-3 ticket
   04): does it bite a resume in a fresh process, or only a second call in
   the same process?
5. **Hub push on resume**: does `hub_strategy="checkpoint"` keep pushing to
   the same repo cleanly after a resume (existing `last-checkpoint/`,
   `save_total_limit`)?

End with the minimal sequence of calls a `--resume` path must make.

## Answer

Resolved 2026-09-25 by a research subagent reading the pinned wheels (no GPU
run). Full findings, every claim cited file:line: branch
`research/resuming-a-run` (commit `478271a`),
`.scratch/milestone-4-full-fine-tune/research/03-resuming-a-run.md`.

**Minimal resume sequence** — a fresh process,
`python -m train --config <same yaml> [--limit <same N>] --resume <wandb_run_id>`:

1. `snapshot_download(adapter, allow_patterns="last-checkpoint/*")` into the
   HF cache — **not** into `output_dir`.
2. Read `global_step` from the checkpoint's `trainer_state.json`.
3. Rebuild model, LoRA, dataset and `SFTConfig` exactly as a fresh run does;
   peft loads the saved weights into our own `get_peft_model` adapter and
   ignores the checkpoint's `adapter_config.json`.
4. `wandb.init(id=…, resume="must", …same name/tags/job_type/config)` before
   building the trainer.
5. `trainer.train(resume_from_checkpoint=<path string>)` — `True` searches
   the empty `output_dir` for `checkpoint-N` and fails.
6. The tail is unchanged.

Restored by the trainer: adapter, optimizer, scheduler, fp16 scaler, RNG,
`global_step`, `log_history`, and the data position (same seeded shuffle,
already-seen batches skipped). `training_args.bin` is never read. Unsloth
overrides none of this.

**Traps:**

- **Resuming a finished checkpoint** trains nothing, so `ScoringCallback`
  never fires and the predictions table is empty. The smoke repo's
  `last-checkpoint` is in exactly that state (step 21 of 21). `--resume`
  must detect it and refuse.
- **Dataset revision:** `train()` fetches the latest revision, so a re-pushed
  dataset silently changes the data. Resume must take the revision from the
  resumed run's config.
- **Config drift:** save and logging intervals come from
  `trainer_state.json`, while `max_steps` and our eval interval are
  recomputed. Any change to the YAML or `--limit` desynchronises eval from
  save.
- **W&B:** the config merges without collisions (the value passed to `init`
  wins) and no rows are dropped. Steps between the checkpoint and the crash
  are logged twice on `train/global_step`, which is cosmetic.
- **Run id:** nothing in the checkpoint stores it. Pass it explicitly, or
  parse it from the Hub `README.md`'s "Training run:" line.
- **Unsloth's `wandb.finish()`** fires only on a second `.train()` of the
  *same trainer object*, so it is harmless to a resume in a fresh process.
- **Hub:** pushes continue into the same repo and overwrite `last-checkpoint/`
  file by file. A *fresh* run with the same repo id clobbers a pending
  checkpoint, and downloading into `output_dir` would re-upload a stale copy
  on every save.
- **Size:** `optimizer.pt` is 21 MB, not ticket 02's ~81 MB, because Unsloth
  defaults to 8-bit AdamW.
