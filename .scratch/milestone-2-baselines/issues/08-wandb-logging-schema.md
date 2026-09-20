# 08 W&B run schema for baselines

Type: grilling
Status: resolved
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

## Answer

Grilled 2026-09-20; all recommendations accepted.

**Run ownership**: `eval.py --wandb` opens and owns the one run per baseline.
`generate.py` is W&B-free: it writes `outputs/<name>/<split>.jsonl` (rows) and
a `meta.json` sidecar: `model`, `engine`, `engine_version`
(`"vllm 0.29.0"` / `"constant"`), `git_sha`, `git_dirty` (bool),
`dataset_revision` (Hub commit sha of `fnl-es/muc4-chat`), `split`, `gpu`
(`torch.cuda.get_device_name()` or `"cpu"`), `n_docs`, `n_truncated`,
`n_cut_off`, `wall_seconds`. The always-`[]` baseline on CPU proves the
whole path; milestone 3's eval callback reuses the same log functions on an
already-open training run.

**`run.config`** (reproducibility fog graduated) = the YAML dict verbatim
(nested) + stamped from `meta.json`: `git_sha`, `git_dirty`,
`dataset_revision`, `engine_version`, `split`, `gpu`. Not stamped: system
prompt (code, covered by `git_sha`), torch/transformers versions.

**Metrics**: `flatten(result)` in `eval.py` joins nested keys with `/` —
all 7 slots × 7 fields (`micro_avg/f1`, `PerpInd/p_num`, `incident_type/r`,
…) + 6 `diagnostics/*` = 55 scalars, plus `diagnostics/n_truncated` and
`diagnostics/n_cut_off` from `meta.json`. Logged once with `run.log(flat)`
(summary + step-0 point that a training callback can extend).

**Artifact**: `outputs/<name>/` (`<split>.jsonl` + `meta.json`) logged as
`wandb.Artifact(name=f"{name}-{split}", type="predictions")`. Gold is not
attached — `dataset_revision` is the reference.

**Table**: `wandb.Table` logged as `predictions`, one row per doc:
`docid, gold, output, parse_ok, cut_off, n_gold_events, n_pred_events`
(gold = target string, output = raw text).

**Identity**: `name` = YAML `name`; `tags` = YAML `tags:` verbatim
(`[baseline, zero-shot]`, `[baseline, 3-shot]`, `[baseline, always-empty]`);
`job_type="eval"` hardcoded (milestone 3: `train`); no group. A re-run is a
new run with the same name; stale runs are deleted by hand in the UI.

**Invocation**: `uv run python -m src.eval --config configs/<name>.yaml
--pred outputs/<name>/dev.jsonl --gold <gold> --wandb`. `--wandb` off by
default (tests/local stay offline); `wandb_project` from YAML; entity = key
default; `WANDB_API_KEY` from env only, missing key = wandb's own failure.
Without `--wandb`, `--config` is optional and the text table prints as
today. With `--wandb`, the text table prints *and* `run.url` is printed
last so the Colab cell ends in a clickable link. `--gold` semantics on
Colab (prepared file vs Hub split) → ticket 09.
