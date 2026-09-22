# 04 Who owns the W&B run during training: TRL or us?

Type: research
Status: claimed
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
