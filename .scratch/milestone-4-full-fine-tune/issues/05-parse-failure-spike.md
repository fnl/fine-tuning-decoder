# 05 The step-12 parse-failure spike

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

Smoke run `63pxlpvn` (W&B `flowing/muc4-event-extraction`) logged
parse-failure rates of 6 / 8 / 4 / **38** / 20 / 10 / 8 % across its seven
eval points. Only the final `predictions` table was logged. What happened at
step 12, and does it threaten the 4B run?

1. From what W&B holds (history of every `dev/*` key, including cut-off
   counts, event-count and relevance diagnostics; the final predictions
   table): were the step-12 failures **cut-off outputs** (length limit),
   malformed JSON, repetition loops, or something else? Read `src/eval.py` /
   `src/generate.py` for how each is counted.
2. Is it plausibly a learning-rate / schedule artefact (mid-epoch 2, near
   peak of cosine after warmup) that settles by the end, or a pipeline smell
   per milestone-3 ticket 06?
3. What would have let us see the outputs: logging a small predictions table
   at every eval point — what does that cost in W&B terms?

End with a recommendation for the 4B run: accept, change a knob, or add
per-eval-point logging. The call is ticket 06's.

## Answer

Resolved 2026-09-25 by a research subagent. Full findings: branch
`research/parse-failure-spike` (commit `e669d94`), `.scratch/milestone-4-full-fine-tune/research/05-parse-failure-spike.md`.

**Verdict: accept — the spike does not threaten the 4B run. Change no
training or generation knob; add per-eval-point logging.** The step-12
failures were almost certainly **repetition loops running to the 512-token
limit** before the first event closed. The evidence is indirect, because the
training callback logs **no cut-off count** (the question assumed it did):

- Parsed events at step 12 carry **4.1 entities each**, against 2.1–2.9 at
  every other eval point and 3.05 in gold — duplicated entities.
- Step 12 was the slowest eval point (235 s), consistent with every
  generation batch running to 512 tokens.
- In the final predictions table, 3 of the 4 failures are the same
  repeated-role loops.

**Not an lr artefact:** warmup is 1 step, peak at step 2; step 12 runs at
1e-4 (half the peak), with 85 % of the run's cumulative lr already applied.
Failure counts 3/4/2/19/10/5/4 of 50 are far beyond noise but fade as the lr
anneals — a transient phase of the small model. **Not a pipeline smell:** the
pipeline is identical at every eval point; the final 8 % (4/50) is within
noise of zero-shot's 7.5 % (15/200); no prose or code fences. Zero-shot 4B
failures were malformed brackets or prose, never loops.

**For ticket 06:** log a predictions table at every eval point (≤ ~0.4 MB and
milliseconds per 4B run) and add `dev/diagnostics/n_cut_off`. Keep
`max_new_tokens` at 512 — dev gold reaches 500 tokens. Budget each eval
point's time for the worst case, every batch running to 512 tokens.
