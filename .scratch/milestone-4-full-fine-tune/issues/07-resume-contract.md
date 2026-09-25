# 07 The resume contract

Type: grilling
Status: open
Blocked by: 03, 06
HITL: yes

## Question

Given what resuming takes (03) and the cadence (06): how is a resume
invoked (a `--resume` flag, a YAML key, auto-detect an existing
`last-checkpoint/` on the Hub?), what does it refuse to do (a config that
differs from the checkpoint's), how does it find and continue the W&B run,
and what CPU tests prove it without a GPU? Name the new `CONTEXT.md` terms,
if any.
