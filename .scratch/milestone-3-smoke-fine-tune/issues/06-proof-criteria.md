# 06 What counts as proof that the loop works?

Type: grilling
Status: open
Blocked by: —
HITL: yes

## Question

DESIGN §11 says milestone 3 "proves loop, masking, checkpoint push, eval
callback" — four words that have to become observable checks the spec can
state and `/implement` can verify. Pin each one:

- **Loss goes down** is the weakest possible claim. What is the actual
  check: training loss below some value, a monotone-ish trend, or simply
  "a loss curve exists in W&B"?
- **Masking** — is the proof the CPU test from ticket 05, a first-step loss
  in the expected range for completion-only loss, or an inspection of one
  batch's labels on the GPU?
- **Checkpoint push** — the adapter repo exists on the Hub and
  `PeftModel.from_pretrained` loads it back? Does the proof require loading
  it back in the *same* notebook?
- **Eval callback** — `dev/*` metrics appear in the run at the expected
  steps, or does the smoke run have to *improve* on some diagnostic?

And the sharp question behind all four: **may the smoke run be allowed to
produce a bad score?** A 0.6B model on 100 documents for 1 epoch may well
score below the 4B zero-shot baseline (micro-F1 18.6) or near zero, and that
would be a *successful* smoke run. Which numbers, if any, are allowed to
fail the milestone — and which diagnostics (parse-failure rate, relevance
accuracy, event-count accuracy) are the ones that would actually indicate a
broken pipeline rather than a small model?

This fixes the spec's "Done criterion" section and feeds ticket 10 (which
metrics must be visible in the run).
