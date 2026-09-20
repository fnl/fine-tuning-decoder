# 07 Which three train documents are the 3-shot exemplars?

Type: prototype
Status: resolved
Blocked by: —
HITL: yes

## Question

Decided: fixed exemplars pinned by docid in the YAML, multi-turn chat
format, ideally one empty doc, one single-event, one multi-event, all short.
Prototype: a script (scratch, not in `src/`) that lists candidate train docs
by (event count, `n_input_tokens`), renders the full 3-shot prompt with the
Qwen chat template for a sample dev doc, and prints total token count.
The user picks the three (or a selection rule) from the rendered output.

Also settle: how exemplar target JSON is produced (reuse
`render_target(canonical_events(doc))`), and whether exemplars are excluded
from anything later (they are train docs; dev is untouched, so probably no).

## Answer

Prototyped 2026-09-20 with `prototype_exemplars.py` (this directory; throwaway,
run `uv run python .scratch/milestone-2-baselines/prototype_exemplars.py [3 docids]`
to list candidates per event-count bucket, render the 3-shot conversation for
every dev doc and report token stats). User accepted:

**Exemplars, in this order** (train split of `fnl-es/muc4-chat`):

| docid | body tokens | events | teaches |
|---|---|---|---|
| `DEV-MUC3-0512` | 103 | 0 | men arrested *with* bombs and attack plans, no event occurred → planned ≠ happened |
| `DEV-MUC3-0126` | 104 | 1 `attack` | all five roles in one event; `PerpInd` with two coreferent mentions; `PerpOrg` ≠ `PerpInd` |
| `DEV-MUC3-0094` | 109 | 2 `bombing` | second device = second event; `PerpOrg` abbreviation coreference (`mrta`); roles shared across events |

Rejected: `DEV-MUC3-0509` (default shortest single-event pick; gold omits the
injured guard, and the set would show no `Victim`), `DEV-MUC3-0334` (Victim
only), `DEV-MUC3-1190` (one car bomb double-typed attack+bombing — confusing).
Selection rule used: shortest per bucket with body ≥ 100 tokens (system
prompt is 183 of `n_input_tokens`), then hand-checked for role coverage.

**Sizing** (official `Qwen/Qwen3-4B-Instruct-2507` tokenizer): shared prefix
system + 3 shots = 665 tokens; dev 3-shot input min 755, **max 1857**
(`TST2-MUC4-0006`) vs budget 3072 − 512 = 2560 → 0 docs over budget, ~700
tokens headroom. Confirms ticket 06's `max_model_len=3072`.

**Target JSON**: no re-rendering. `generate.py` loads the three train rows by
docid and splices their `user`/`assistant` messages in verbatim between the
system message and the target document's user turn; the assistant message
already *is* `render_target(canonical_events(doc))`.

**Exclusion**: none. Exemplars are ordinary train docs; dev/test untouched;
milestone 3 trains on them like any other document.
