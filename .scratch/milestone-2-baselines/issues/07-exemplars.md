# 07 Which three train documents are the 3-shot exemplars?

Type: prototype
Status: open
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
