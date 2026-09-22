# 06 What counts as proof that the loop works?

Type: grilling
Status: resolved
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

## Answer

Settled 2026-09-22. The loss criterion is the user's; items 2–5 were
delegated ("write down how you will attack them and move on") and are the
agent's decisions, recorded here so they can be reversed knowingly rather
than rediscovered.

**The milestone passes on mechanical criteria only. No score gates it.**

### 1. Loss — user's answer

A loss curve exists in W&B and trends downward monotone-ish. No threshold on
the absolute value, no per-step monotonicity: the check is that the mean of
the last quarter of logged `train/loss` points is below the mean of the first
quarter.

**Consequence for ticket 08 — the skeleton config cannot satisfy this.**
100 documents at `effective_batch_size: 16` for 1 epoch is **6 optimizer
steps**, and `eval_every: 0.5` gives **2 eval points**. A trend over 6 noisy
points is not a trend, and 2 eval points barely demonstrate a cadence.
Measured alternatives:

| effective batch | epochs | steps | eval points |
|---|---|---|---|
| 16 | 1 | 6 | 2 |
| 16 | 3 | 18 | 6 |
| 8 | 3 | 36 | 6 |
| 4 | 2 | 50 | 4 |

Requirement this ticket imposes: **≥ ~18 optimizer steps and ≥ 4 eval
points**. Recommendation to ticket 08: keep `effective_batch_size: 16` (it is
the 4B run's value, so the smoke exercises the knob milestone 4 uses) and
raise the smoke to **3 epochs** — 18 steps, 6 eval points, still minutes on a
T4 at 0.6B. DESIGN **§9** — not §11 — names the smoke config as "(100 docs, 1
epoch, 50 dev docs)" and is superseded; §11's milestone 3 states no epoch
count. Ticket 12 corrects §9 rather than editing it silently.

### 2. Masking — three layers, cheapest first

1. **CPU test** (ticket 05's function, both tokenizers): every prompt token is
   `-100`, every completion token is not, the boundary sits exactly at
   `len(prompt_ids)`, the sequence ends at `<|im_end|>`, and the
   empty-document case yields exactly **2** unmasked tokens (ticket 05 settled
   this: the template's trailing newline is truncated away). This is the real
   proof and it costs no GPU.
2. **One-batch assertion at train time**: `train.py` logs once at startup the
   number of unmasked label tokens in the first batch and the *decoded*
   unmasked span of its first example. One line of notebook output showing
   the JSON target and nothing else is what a human can actually check.
3. **First-step loss as a corroborating signal**: completion-only loss over
   JSON targets should start well below full-sequence loss over a news
   document. Record the observed value in the first `/implement` smoke run
   (ticket 07 was closed unrun) so a later regression has a number to compare
   against. Not a gate — a silently broken mask
   (`labels == input_ids`) would show up as a markedly higher initial loss.

### 3. Checkpoint push — the adapter must survive the session

Proof: the Hub repo exists with at least one revision, and a **final cell in
`train.ipynb` loads the pushed adapter back onto a freshly-loaded base model
and generates one document**. Loading it back in the same notebook is
required, not optional: milestone 3 exists to prove the artifact survives a
Colab disconnect, and verifying the in-memory object proves nothing about
what landed on the Hub. It is cheap at 0.6B and it exercises exactly the path
milestones 4 and 5 use to evaluate an adapter. → decides the open question in
ticket 11.

### 4. Eval callback — cadence and shape, not quality

Proof: `dev/*` metrics appear at the expected steps (the count matches the
cadence arithmetic of ticket 09), and the key set equals `flatten(score())`'s
so a smoke run and a baseline run can be overlaid in one W&B panel. The smoke
run is **not** required to improve on any diagnostic.

### 5. May the smoke run produce a bad score? — yes, explicitly

A 0.6B model trained on 100 documents may score near-zero micro-F1, below the
4B zero-shot baseline (18.6). That is a **successful** smoke run and the spec
must say so, so that nobody later "fixes" a working pipeline.

The diagnostics that would indicate a *broken pipeline* rather than a small
model, to be read at the last eval point:

- **Parse-failure rate above the zero-shot baseline's 7.5 %.** After training
  on 100 well-formed JSON targets the model should emit parseable JSON more
  reliably than the un-tuned 4B, not less. A rise is the strongest single
  smell that masking or the chat template is wrong.
- **Collapse to all-`[]`** (relevance accuracy ≈ 46 %, the trivial baseline).
  Allowed — it is a real local optimum when 40 % of the subset is empty — but
  it must be *reported*, and combined with a flat loss curve it means the run
  taught the model nothing.
- Prose or markdown fences in the output: what fine-tuning should buy
  immediately, and `parse_target` tolerating them must not hide their return.

### 6. Dev-subset comparability — resolved here, clearing the map's fog

The callback's 50-document dev score is a **training signal, not a headline
number**. It is never compared to the 200-document baselines in `RESULTS.md`;
milestone 4 produces the comparable number over the full dev split. It keeps
the identical metric names so W&B *can* overlay the two, with the differing
subset stated in the run config. This closes the "Dev-subset comparability"
patch in the map's Not yet specified.
