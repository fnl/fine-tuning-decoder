# 10 The shape of a training run in W&B

Type: grilling
Status: open
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
