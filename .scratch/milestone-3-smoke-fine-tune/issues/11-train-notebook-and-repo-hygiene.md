# 11 `train.ipynb` layout and the repo changes around it

Type: grilling
Status: resolved
Blocked by: 01, 07, 08
HITL: yes

## Question

Milestone 2's notebook is 8 cells and took three attempts to make work on a
T4; its lessons (auto-restart guard, `%cd` after every restart, chaining with
`&&`, secrets asserted loudly, a `--limit 5` smoke cell before the real run)
are the starting point, not a rediscovery. Decide:

- The cell list for `notebooks/train.ipynb`: runtime check → guarded Unsloth
  install + restart → clone@`REF` → `pip install -e .` → secrets → dev gold
  dump → a `--limit` smoke cell → the real training cell. Is there also a
  "load the pushed adapter back and generate one document" cell, given
  ticket 06's proof criteria?
- The rename `colab.ipynb` → `baselines.ipynb`: does it happen in this spec
  (and with it the README's Colab link, the `test_notebook.py` fixture and
  the DESIGN §3 tree), or is it left alone to keep the diff small?
- How `test_notebook.py` extends to two notebooks: same assertions (loads,
  no outputs, referenced config paths exist) parameterised over both?
- Whether `train.ipynb` needs anything from `data/prepared/` at all, given
  the dataset is on the Hub and the callback scores in-process.
- README: a "Fine-tuning" section mirroring "Running a baseline", and what
  the prerequisites table gains (a Hub token with write scope is already
  listed "from milestone 3").
- `.gitattributes` / nbstripout already cover a second notebook — confirm.

Fixes the spec's notebook and docs sections.

## Answer

Settled 2026-09-22 (delegated). Milestone 2's notebook lessons are inherited,
not rediscovered — but ticket 01 removed the worst of them: Colab's torch
2.11.0 satisfies every Unsloth bound, so **there is no restart**, and with it
no `%cd` dance, no "run all twice", no `torchaudio` uninstall.

### `notebooks/train.ipynb` — 7 cells

1. **Runtime check** — `!nvidia-smi`, `torch.__version__`, `sys.version`.
   Milestone 2's snapshot went stale; this prints what is actually there.
2. **Install** — guarded `if importlib.util.find_spec("unsloth") is None:
   !pip install unsloth`. Plain, *not* Unsloth's notebook recipe (ticket 01:
   its torch→xformers map has no entry for 2.11). **No `do_shutdown`.**
3. **Clone @ `REF`** (a `@param` form field, as in the baselines notebook) and
   `!pip install --quiet -e .`.
4. **Versions** — print the resolved unsloth / trl / transformers / torch /
   peft / bitsandbytes. They are what ticket 10 stamps into the run config,
   and ticket 01's pin chain is the thing most likely to drift.
5. **Secrets** — `HF_TOKEN`, `WANDB_API_KEY` from Colab Secrets, asserted
   loudly (milestone 2's pattern verbatim).
6. **Smoke** — `!python -m train --config configs/qwen3-0.6b-smoke.yaml
   --limit 8`, which is ticket 07's closing note made routine: a handful of
   steps that proves the labels reach the collator and the loss is finite
   before the real cell runs.
7. **Train** — `!python -m train --config configs/qwen3-0.6b-smoke.yaml`,
   ending on the W&B run URL.
8. **Load back** — the proof criterion of ticket 06: `PeftModel.from_pretrained`
   pulls the *pushed* adapter onto a freshly-loaded base model and generates
   one dev document. (Eight cells, then — the load-back earns its place.)

A markdown cell at the top carries the one warning ticket 04 surfaced:
**re-running the train cell in the same runtime abandons the configured W&B
run**, because Unsloth calls `wandb.finish()` on a second `train()` in one
process. Restart the runtime instead.

### Decisions

- **`colab.ipynb` → `baselines.ipynb`: yes, in this spec.** Two notebooks with
  one named after the platform and the other after its job is a trap. It costs
  a `git mv`, the README's Colab link, the DESIGN §3 tree and the notebook
  test's path list.
- **`test_notebook.py` is parameterised over both notebooks**: loads under
  `nbformat`, no cell carries outputs, every config path it mentions exists.
  One new assertion, cheap and general: no cell calls `do_shutdown` in
  `train.ipynb` (the restart is exactly what must not creep back).
- **No gold-dump cell.** The baselines notebook writes
  `data/prepared/dev.jsonl` because `eval.py --gold` needs a file;
  `train.py` scores in-process from the Hub dataset, so `train.ipynb` touches
  `data/` not at all.
- **README** gains a "Fine-tuning" section mirroring "Running a baseline",
  with its own Colab badge for `train.ipynb`. The prerequisites table already
  says the Hub write token is needed "from milestone 3" — that line becomes
  true and loses the qualifier.
- **`.gitattributes`** already matches `*.ipynb`, so nbstripout covers the new
  notebook with no change. Confirmed.
