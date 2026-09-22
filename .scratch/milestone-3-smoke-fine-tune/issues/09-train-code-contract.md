# 09 Contract of `train.py` and where the in-process engine lives

Type: grilling
Status: open
Blocked by: 01, 02, 03, 05, 07
HITL: yes

## Question

Pin the interfaces the spec will state verbatim, the way milestone 2's
ticket 09 did for `generate.py` / `eval.py`:

- **`tokenize_example`** (ticket 05's shape): signature, return type, failure
  mode, and where it lives — `train.py`, `generate.py` beside `render_inputs`
  (they render the same prompt), or a new module.
- **The in-process engine**: is it `hf_engine(model, tokenizer, ...) -> Engine`
  in `generate.py` next to `vllm_engine` and `constant_engine`, or a private
  helper in `train.py`? It must satisfy the existing
  `Engine = Callable[[list[str]], list[GenerationResult]]` so the callback is
  just `generate()` + `parse_target` + `score()`.
- **The eval callback**: a `TrainerCallback` subclass or a
  `compute_metrics`-style function; what it receives, what it logs, how it
  restores training state, and how it is tested with fakes.
- **Step arithmetic**: `eval_every` in epochs → steps, given subset size,
  batch size and accumulation — a pure function with its own test, including
  the awkward cases (fewer steps than one eval interval, non-integer).
- **`train.py`'s own shape**: a callable core plus a thin CLI
  (`python -m train --config configs/<name>.yaml [--limit N]`), lazy Unsloth
  and TRL imports so the module is importable on CPU, and the `mypy`
  overrides `unsloth.*` / `trl.*` / `peft.*` need.
- **Packaging**: `src/train.py` added to hatch `only-include`; `wandb` is
  already a core dependency, Unsloth and TRL stay notebook-only.
- **Tests** (Q9): the masking test against both tokenizers, the step
  arithmetic, the callback with a fake engine and a fake trainer state, the
  config loader — and what is deliberately left untested because it needs a
  GPU.
- **Docs**: which README and `docs/DESIGN.md` sections change (§3's tree,
  §7's knobs, §11).

If ticket 07 found that Unsloth re-tokenises, this ticket also chooses the
prompt-completion fallback shape instead.
