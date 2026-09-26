# 07 The resume contract

Type: grilling
Status: resolved
Blocked by: 03, 06
HITL: yes

## Question

Given what resuming takes (03) and the cadence (06): how is a resume
invoked (a `--resume` flag, a YAML key, auto-detect an existing
`last-checkpoint/` on the Hub?), what does it refuse to do (a config that
differs from the checkpoint's), how does it find and continue the W&B run,
and what CPU tests prove it without a GPU? Name the new `CONTEXT.md` terms,
if any.

## Answer

Resolved 2026-09-26 by grilling (all recommendations accepted). The mechanics
come from ticket 03; this is the contract.

1. **Invocation:** `python -m train --config <same yaml> [--limit N] --resume
   <wandb_run_id>`, a CLI flag, not a YAML key (the YAML stays values-only) and
   not auto-detection. The run id is printed at the start of every run and is in the
   adapter card's "Training run:" link.
2. **What `--resume` takes and refuses** (a custom `ResumeError`, naming the reason):
   - takes the **dataset revision** from the resumed run's config, never the
     Hub's latest;
   - refuses a **finished checkpoint** (`global_step` = total steps; it would train
     nothing and log an empty table);
   - refuses any difference between the YAML (after `--limit`) and the run's
     `experiment` config, **naming the keys**;
   - refuses a missing `last-checkpoint/`;
   - **allows a different git sha**: the original `git_sha` stays in the config,
     and each resume appends `{step, git_sha}` to a `resumes` list in the run summary.
3. **Fresh-run guard:** a run without `--resume` refuses if the adapter repo
   already has any `last-checkpoint/`, naming the repo. A rerun needs a new
   adapter name or a manual repo deletion, a deliberate human act.
4. **`--limit` runs never push to the Hub.** They are pipeline checks and save
   locally only. Today's notebook `--limit 8` cell overwrites the milestone-3
   smoke adapter; this stops that, and it keeps `--limit` clear of the guard.
5. **CPU tests:** pure functions for every refusal in 2 and 3, the `resumes`
   record, and `--resume` parsing. No Hub or W&B mocks; the glue is proven by 6.
6. **Resume drill** (a build step before the real run):
   `configs/qwen3-0.6b-resume-drill.yaml` is the smoke YAML with `train_docs: 32`,
   `dev_docs: 8`, adapter `fnl-es/qwen3-0.6b-muc4-lora-drill`. That gives 6 steps, with an eval point and
   checkpoint at every step. Start it, interrupt the cell after the first pushed checkpoint, then
   resume with the printed id. It passes if the run finishes as **one** W&B run with 6 eval
   points and one `resumes` entry, a second `--resume` refuses (finished), and a
   fresh run on the drill YAML refuses (guard). Afterwards the build deletes the drill YAML
   and cells; deleting the drill's Hub repo is the user's call.
7. **`train.ipynb` becomes milestone 4's notebook:** cells 1–5 unchanged; a
   4B pipeline check (`qwen3-4b-r16.yaml --limit 8`, no push); a resume-drill
   section (removed after the drill); the run; a resume cell with
   `RUN_ID = ""  # @param`, which **skips itself in Python when `RUN_ID` is
   empty**, so "Run all" never resumes a just-finished run. Milestone 3's
   smoke-run commands stay **commented out** under a note that they were
   milestone 3's smoke check, kept for the record and superseded by the 4B
   check. The load-back cell goes; the final evaluation supersedes it. The title
   and intro are widened, and `tests/test_notebook.py` follows.
8. **Vocabulary:** `CONTEXT.md` gains **Resume** and **Rerun**.

## Comments

### 2026-09-26: amendment from ticket 12

Item 2 gains one refusal: `--resume` refuses if the running torch version
differs from the resumed run's `versions.torch`, naming both. Every other
training package is pinned in the install cell, so torch alone can drift
between sessions.

### 2026-09-26: amendment from ticket 10

Item 1's invocation drops `[--limit N]`. `--resume` and `--limit` are mutually
exclusive (argparse error), because item 4 makes a `--limit` run push nothing,
so it has no `last-checkpoint/` to resume.

### 2026-09-26: the resume drill passed

W&B run `z31g72jy` (`configs/qwen3-0.6b-resume-drill.yaml`, commit `6d38d10`,
Tesla T4) ran item 6's drill:

- **The interrupted process got further than its output showed.** It trained all
  6 steps in 52 s (20:15:52–20:16:44 UTC), about 10 s per step with the eval.
  Its checkpoints reached the Hub trailing the training, and the last one to arrive
  was step 5 (20:17:07). The step-6 checkpoint was lost with the deleted runtime.
  A disconnect loses the work since the last checkpoint that *reached the Hub*,
  not the last one saved locally.
- **The resume** in a new runtime (`resume: run z31g72jy from step 5`) retrained
  step 6 and finished the same W&B run. The retrained step 6 reproduced the
  original's loss (0.653) and dev F1 (0.145) exactly: optimizer, scheduler, RNG
  and data position were restored. The run shows eval points 1–6, with step 6
  logged twice (cosmetic, as ticket 03 said), `resumes: [{"step": 5, "git_sha":
  "6d38d10…"}]` and `adapter_revision` `640bcd7`.
- **The second `--resume` was refused:** `ResumeError: the run finished: its
  checkpoint is at step 6 of 6`.
- **The fresh run was refused.** No second W&B run exists and no Hub commit
  follows 20:21:40. Rerunning `check_fresh_run` against the repo's live files
  raises `ResumeError: fnl-es/qwen3-0.6b-muc4-lora-drill holds a
  last-checkpoint/: …`. The cell's `2>&1 | tail -1` showed an Unsloth banner
  instead of the error: a pipe block-buffers stdout until exit, while the
  traceback goes to stderr at once.

Only one step was left to resume, but every part of the glue ran: W&B lookup,
checkpoint download, `resume="must"`, trainer restore, pushing onward, and the
tail. The drill section is removed from `train.ipynb`. The drill YAML is kept by
the user's choice, as with the probe eval. Deleting the Hub repo is the user's
call.
