# 09 Contract of `train.py` and where the in-process engine lives

Type: grilling
Status: resolved
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

## Comments

**2026-09-22, from ticket 01 (pre-tokenised datasets):** the fallback branch of
this ticket is **not needed** — Unsloth accepts `input_ids` + `labels` verbatim
(`unsloth_zoo/dataset_utils.py:2582-2588`). But two consequences replace it:
(a) `max_length` is **not enforced** on a pre-tokenised split, because Unsloth's
prep replaces the one that calls `truncate_dataset` — `train.py` must truncate
or assert the budget itself, and that check needs its own CPU test; (b) Unsloth
silently auto-enables `padding_free` — masking survives it, but set
`padding_free=False` explicitly for the smoke run rather than inherit a rewrite.

**2026-09-22, from ticket 05 (mask prototype):** the signature is settled —
`tokenize_example(example, tokenizer, *, stop_after_eos=True) -> {input_ids,
labels}`, raising `MaskingError(ValueError)` naming the docid. Reference
implementation: `../prototype_mask.py`.

**2026-09-22, from ticket 03 (generation):** the engine is
`unsloth_engine(model, tokenizer, *, max_new_tokens, batch_size)` beside
`vllm_engine`; generation batch size must be a YAML knob (16 at 0.6B, 4-8 at 4B).

## Answer

Settled 2026-09-22 (delegated), on ticket 01's source-verified finding that a
pre-tokenised dataset survives Unsloth. The fallback branch of this ticket is
not needed.

### `src/generate.py` gains one engine

```python
def unsloth_engine(model, tokenizer, *, max_new_tokens: int, batch_size: int) -> Engine
```

Beside `vllm_engine` and `constant_engine`, satisfying the existing
`Engine = Callable[[list[str]], list[GenerationResult]]`. It sets
`tokenizer.padding_side = "left"`, length-sorts the inputs, generates with
`do_sample=False, temperature=None, top_p=None, top_k=None` and an explicit
`pad_token_id`, slices completions by prompt width, derives `finish_reason`
from eos presence, restores the original order. It does **not** call
`for_inference` (ticket 03: `from_pretrained` already wraps `generate`, and
calling it poisons the restore). Lives here, not in `train.py`, so the
callback is nothing but `generate()` + `parse_target` + `score()`.

### `src/train.py`

```python
IGNORE_INDEX = -100

class MaskingError(ValueError): ...

def load_config(path: Path) -> dict            # validates; rejects unknown keys
def tokenize_example(example, tokenizer, *, stop_after_eos=True) -> dict
def select_subset(split, n: int | None) -> list[Example]        # first N rows
def build_dataset(examples, tokenizer, *, max_seq_len) -> tuple[Dataset, int]
def training_steps(n, *, effective_batch_size, epochs) -> int
def eval_interval(total_steps, *, eval_every, epochs) -> int
class ScoringCallback(TrainerCallback): ...    # hooks on_step_end
def train(config: dict) -> str                 # returns the W&B run url
def main(argv: Sequence[str] | None = None) -> None
```

- `build_dataset` tokenises, **drops** over-budget examples and returns the
  count (ticket 08): Unsloth does not enforce `max_length` on a pre-tokenised
  split, so nothing truncates silently.
- `ScoringCallback` hooks **`on_step_end`** with its own cadence, not
  `on_evaluate` — ticket 04 found the latter never fires without
  `eval_strategy`. It receives `model` and `processing_class` from the
  callback kwargs, calls `generate(dev_examples, engine, tokenizer,
  max_input_tokens=…)`, parses with `parse_target`, scores with `score()`,
  logs per ticket 10, and keeps the last result for `on_train_end`.
- GPU imports (`unsloth`, `trl`, `peft`) are function-local, so the module
  imports on CPU and every pure function above is testable without a GPU.
- CLI: `uv run python -m train --config configs/<name>.yaml [--limit N]`.
  `--limit` overrides `train_docs`/`dev_docs` for a fast notebook smoke.

### Packaging, typing

`src/train.py` added to hatch `only-include`. `mypy` `ignore_missing_imports`
for `unsloth.*`, `trl.*`, `peft.*`, `bitsandbytes.*` (`vllm.*`, `torch.*`
already present). No new core dependency: `wandb` is already one; Unsloth and
TRL stay notebook-installed.

### Tests (CPU, pytest) — `tests/test_train.py`

1. **Masking**, both tokenizers, `tokenizer` marker: every prompt token is
   `-100`, every completion token is not, the seam is at `len(prompt_ids)`,
   the sequence ends at `<|im_end|>`, an empty document yields exactly 2
   unmasked tokens, and both tokenizers agree on the unmasked count.
2. **`MaskingError`** names the docid when the prefix breaks (construct a
   mismatching example rather than monkey-patching the template).
3. **Budget**: an over-length example is dropped and counted; a dev example is
   untouched.
4. **Step arithmetic**: `training_steps` and `eval_interval` over the real
   smoke numbers (100 docs, batch 16, 3 epochs → 18 steps, interval 3) plus
   the awkward cases — fewer steps than one interval, non-integer
   `eval_every`, `epochs=1`.
5. **`select_subset`**: first N, N absent, N larger than the split.
6. **`load_config`**: unknown key rejected by name; indivisible
   `effective_batch_size` rejected; every key of the smoke YAML accepted.
7. **`ScoringCallback`** with a fake engine and a fake `TrainerState`: fires
   at the expected steps, logs exactly the `dev/`-prefixed keys of
   `flatten(score())` plus `train/global_step`, never passes `step=`.
8. **`unsloth_engine`** is *not* tested (needs a GPU); `constant_engine`
   already covers the `Engine` contract.

### Docs

README gains a "Fine-tuning" section mirroring "Running a baseline". DESIGN
§3's tree gains `train.py`; §7 gains the masking mechanism, `padding_free=False`,
the dropped-example policy and the 3-epoch smoke; §11's milestone 3 line is
annotated where it said 1 epoch. `CONTEXT.md` gains the terms ticket 12 fixes.
