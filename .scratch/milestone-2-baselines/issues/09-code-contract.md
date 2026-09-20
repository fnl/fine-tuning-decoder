# 09 Contract of generate.py, eval.py and the baseline YAMLs

Type: grilling
Status: open
Blocked by: 06, 07, 08
HITL: yes

## Question

Pin the interfaces the spec will state verbatim:

- `src/generate.py`: callable core signature (examples + model handle →
  `{docid, output}` rows), CLI (`--config`, `--split`, `--limit`, `--out`),
  where the few-shot turns are assembled (here vs. `data/prepare.py`), and
  how the always-`[]` baseline is expressed (an engine `constant` in the
  same CLI vs. a separate path);
- `src/eval.py`: new `--wandb` flag and what it needs from the config;
- YAML schema shared by the three baseline configs and compatible with
  `configs/qwen3-0.6b-smoke.yaml` (keys: `model`, `dataset`, `engine`,
  `shots`/`exemplars`, `generation`, `wandb_project`);
- GPU imports lazy, CPU tests for prompt assembly and the constant engine,
  `mypy` overrides for vllm/unsloth.
