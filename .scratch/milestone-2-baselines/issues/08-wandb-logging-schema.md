# 08 W&B run schema for baselines

Type: grilling
Status: open
Blocked by: 01
HITL: yes

## Question

With a real W&B project to look at, decide what one baseline run logs and
under which names, so milestones 3–5 reuse the shape:

- config: the YAML verbatim plus stamped fields (git SHA, dataset revision,
  engine + version, `n_truncated`) — graduates the "reproducibility fields"
  fog;
- metrics: flattening of `score()` (`micro_avg/f1`, `PerpInd/p`, …,
  `diagnostics/relevance_acc`);
- artifacts: raw outputs JSONL (name/type), and whether gold is attached;
- a predictions `wandb.Table` (docid, gold, output, parse_ok) — yes/no and
  columns;
- run naming/tags (`baseline`, `zero-shot`, `3-shot`, `always-empty`);
- how `eval.py --wandb` is invoked (run name from config, `WANDB_PROJECT`
  from YAML, key from env only).
