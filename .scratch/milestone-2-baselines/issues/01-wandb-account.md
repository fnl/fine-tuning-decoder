# 01 Set up the W&B account and API key

Type: task
Status: resolved
Blocked by: —
HITL: yes

## Question

The user has no W&B account. Before the logging schema can be judged in the
real UI and before any run can be logged, the account must exist and a key
must be verified from this machine.

Checklist for the user:

1. Create an account at https://wandb.ai (personal entity; the username
   becomes the default entity, nothing in code depends on it).
2. Copy the API key from https://wandb.ai/authorize.
3. Export it locally: `export WANDB_API_KEY=...` (shell profile or
   `direnv`; never in the repo).
4. Verify from the repo: `uv run wandb login --verify` then
   `uv run python -c "import wandb; r=wandb.init(project='muc4-event-extraction', name='hello'); r.log({'ok':1}); r.finish()"`.
5. Delete the `hello` run in the UI (or keep it; harmless).

Resolved when step 4 succeeds. Record in the answer: entity name, the
project URL.

## Answer

Resolved by the user 2026-09-20. Account created; `WANDB_API_KEY` lives in
`.envrc` (git-ignored, loaded by direnv — not visible to shells that skip
direnv, so the agent verifies with `direnv exec . uv run wandb login --verify`
when needed). Project `muc4-event-extraction` under the personal entity.
Entity name / project URL not recorded by the user; nothing in code depends on
the entity (W&B picks the default entity from the key).
