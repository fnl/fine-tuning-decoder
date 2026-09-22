# 10 The shape of a training run in W&B

Type: grilling
Status: resolved
Blocked by: 04, 06
HITL: yes

## Question

Milestone 2 fixed the shape of an *eval* run (config = YAML + stamps, 55
flat scorer metrics, a predictions artifact and table, `job_type=eval`).
Decide the training run's shape so the two are siblings, not strangers:

- Namespaces: `train/*` from TRL versus `dev/*` from our callback — and
  whether the 55 scorer keys keep their milestone-2 spelling under the `dev/`
  prefix so a W&B report can put a smoke run and a baseline run on one axis.
- Which of the diagnostics ticket 06 named must be visible per eval point,
  and which only at the end.
- Config: the YAML verbatim plus which stamps (git SHA, dataset revision,
  GPU, resolved library versions — milestone 2 logged `engine_version`; the
  training equivalent is a list).
- End-of-run artifacts: a predictions table from the last eval point? the
  adapter as a W&B artifact, or is the Hub repo the only home (DESIGN §9
  names the Hub)? A link from the run to the Hub repo.
- Naming, tags, `job_type=train`, and what the notebook prints at the end
  (milestone 2 prints the run URL last — keep that).
- The cross-run question: how a reader compares this smoke run to the
  baselines when the smoke run scores 50 dev documents and the baselines
  scored 200.

Fixes the spec's W&B section.

## Answer

Settled 2026-09-22 (delegated). Mechanics are ticket 04's; this fixes the
shape.

### One run, ours, `job_type=train`

`train.py` calls `wandb.init(project, name, tags, job_type="train", config=…)`
exactly as `eval.py:log_run` does, *then* passes `report_to="wandb"`; it owns
`run.finish()`. Ticket 04 verified `WandbCallback` only inits when
`wandb.run is None` and that nothing in TRL calls `finish`.

### Namespaces

- `train/*` — TRL's own (loss, learning rate, epoch, grad norm).
- `dev/*` — ours: the 55 keys of `flatten(score())` verbatim under the prefix,
  so `dev/micro_avg/f1`, `dev/diagnostics/parse_failure_rate`, and so on.
- Logged as `run.log({**{f"dev/{k}": v}, "train/global_step": step})` with
  **no** `step=` argument (ticket 04: an explicit `step=` silently drops rows).

**Why the prefix stays, given it blocks a naive overlay with milestone 2's
runs:** because those numbers are *not* comparable. Ticket 06 ruled the
callback's 50-document score a training signal, never a headline; milestone 4
produces the comparable figure over all 200 dev documents, and its eval run
will use `eval.py`'s unprefixed keys. Identical spelling under a distinct
prefix means a report can put them side by side deliberately, while nothing
averages them by accident.

### Cadence and volume

All 55 keys at every eval point (6 for the smoke run — cheap, and
`diagnostics/n_docs` records the subset size at each). At `on_train_end`,
additionally: the `predictions` table with milestone 2's `TABLE_COLUMNS`
(docid, gold, output, parse_ok, cut_off, n_gold_events, n_pred_events) built
from the **last** eval point, and the proof-criteria summary of ticket 06
(first- and last-quarter mean `train/loss`, unmasked-token count of the first
batch, dropped-example count).

### Config

The YAML verbatim, plus stamps. Milestone 2's single `engine_version` becomes
a `versions` mapping — unsloth, unsloth_zoo, trl, transformers, torch, peft,
bitsandbytes — since ticket 01 showed the pin chain is what determines
behaviour. Reused from `CONFIG_STAMPS`: `git_sha`, `git_dirty`,
`dataset_revision`, `gpu`. Added: resolved `total_steps`,
`gradient_accumulation_steps`, `eval_steps`, `n_dropped` (ticket 08) and
`n_train_examples` — every derived number, so a run is reproducible from its
own config.

### Artifacts: the Hub is the adapter's home

The adapter is **not** logged as a W&B artifact. DESIGN §9 names the Hub, and
ticket 02 measured 38.55 MiB per push at every eval point; duplicating that
into W&B buys nothing. Instead the run config carries `adapter` (the repo id)
and the summary carries `adapter_revision` (the final commit sha) plus the
repo URL, so the run points at the artifact rather than copying it.
`WANDB_LOG_MODEL` and `WANDB_WATCH` stay off (ticket 04 — the checkpoint mode
would upload optimizer state at every save).

### Ergonomics

`name` and `tags` from the YAML; the notebook prints `run.url` last, as the
baseline cells do. Ticket 04's Unsloth gotcha — a second `train()` in one
process calls `wandb.finish()` and abandons our configured run — is a note for
ticket 11, not something the run shape can defend against.
