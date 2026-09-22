# 12 Write the milestone-3 spec

Type: task
Status: resolved
Blocked by: 06, 08, 09, 10, 11
HITL: yes

## Question

With every decision on this map resolved, write
`.scratch/milestone-3-smoke-fine-tune/spec.md` using the `to-spec` skill and
`.scratch/milestone-2-baselines/spec.md` as the model: goal, decisions
(pointing at tickets rather than restating them), the code contract verbatim,
testing decisions, out of scope, done criterion, and an order of work that
de-risks earliest.

Before writing, update `CONTEXT.md` with whatever vocabulary tickets 05 and
09 fixed (the masking terms are not in the glossary yet), and check DESIGN
§7 against what this map decided — §7 was written before TRL 0.24 was
pinned, and anything it now contradicts should be corrected in place with a
note, not silently.

The map closes with this ticket. What follows is
`/implement @.scratch/milestone-3-smoke-fine-tune/spec.md`.

## Answer

Written 2026-09-22: [`spec.md`](../spec.md), `ready-for-agent`.

Prerequisites done first, as the question required:

- `CONTEXT.md` gained a **Training** section — prompt, completion, masked
  token, completion-only loss, sequence budget, subset, eval point,
  checkpoint — and *dropped document* was widened to cover the training-time
  case as well as data preparation.
- `docs/DESIGN.md` corrected in place with dated notes rather than silently:
  §9's smoke experiment is **3 epochs**, not 1 (ticket 06's floor); §7 records
  that the completion-only mechanism is ours and why the two library
  alternatives were rejected, that `padding_free` is off explicitly, and the
  dropped-example policy with the measured 2082-token document.

Correction made while checking: the "1 epoch" figure is in DESIGN **§9**, not
§11 — §11's milestone 3 states no epoch count. Tickets 06 and the map said §11
and were fixed.

**Map closed.** Next: `/implement @.scratch/milestone-3-smoke-fine-tune/spec.md`,
whose first step is the T4 smoke run that ticket 07 would have been.
