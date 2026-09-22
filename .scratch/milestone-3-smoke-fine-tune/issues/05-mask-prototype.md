# 05 See the mask: prototype `tokenize_example` on real documents

Type: prototype
Status: resolved
Blocked by: —
HITL: yes

## Question

Completion-only loss is "non-negotiable" (DESIGN §7) and the mechanism is
ours (Q3), so the mask must be *seen* before any GPU time is spent. Build a
throwaway script in this directory (not in `src/`, cf.
`.scratch/milestone-2-baselines/prototype_exemplars.py`) that, for both
`Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B-Instruct-2507` and a handful of real
prepared examples (one empty document, one single-event, one multi-event):

1. Renders prompt (`messages[:2]`, `add_generation_prompt=True`,
   `enable_thinking=False`) and full (`messages`), asserts the token-level
   prefix, and builds `input_ids` / `labels`.
2. Prints the boundary region token by token — decoded token, id, label —
   for ~10 tokens either side of the prompt/completion seam, so the seam is
   visually checkable, including the `<think>\n\n</think>\n\n` block on the
   0.6B (it must be *masked*, because our inference prompt supplies it).
3. Prints, per example, prompt length, completion length, and the share of
   unmasked tokens; and over the whole 100-document smoke subset the total
   and mean unmasked tokens (40 of those documents have a 3-token
   completion — the number the run learns from is worth knowing up front).

Decisions this prototype is built to settle:

- Does the trailing `\n` after `<|im_end|>` stay in the labels or get
  dropped? (The model can never emit it: generation stops at eos.)
- Is `assert` the right failure mode for a broken prefix, or a named
  exception like `generate.py`'s over-budget `ValueError`?
- Does the function take a whole example, or prompt/target strings?
- Do the two tokenizers agree closely enough that one code path serves both
  (the answer decides whether the smoke run really proves the 4B's path)?

Link the script from this file and record the decisions under `## Answer`.

## Answer

Prototyped 2026-09-22: [`prototype_mask.py`](../prototype_mask.py) (throwaway;
run with `uv run python .scratch/milestone-3-smoke-fine-tune/prototype_mask.py`).
All four decisions settled; the user had delegated them ("figure them out and
move on"), so they are the agent's, recorded for knowing reversal.

### What the seam looks like

For `Qwen3-0.6B`, the masked prompt ends:

```
  761  '<think>'     label=-100
  762  '\n\n'        label=-100
  763  '</think>'    label=-100
  764  '\n\n'        label=-100
  765  '[]'          label=1294   <<< seam
  766  '<|im_end|>'  label=151645
```

**This is the visual proof that the Q3 decision was right.** The empty
`<think>` block sits on the *masked* side, because our inference prompt
supplies it (`enable_thinking=False`, `add_generation_prompt=True`). TRL's
patched Qwen3 template and Unsloth's `train_on_responses_only` both put it on
the *unmasked* side — they would have trained the smoke model to emit a block
the prompt already contains.

For `Qwen3-4B-Instruct-2507` the same seam sits directly after
`<|im_start|>assistant\n`, with no think block at all.

### 1. The trailing newline: dropped

The template emits `<|im_end|>\n`; everything after the final eos is always
exactly `['\n']` across all 100 documents and both tokenizers (verified). The
model can never emit it — generation stops at eos — so it is a label for a
token that is never produced. It is not negligible: for the 40 empty
documents it is **1 of 3 unmasked tokens, a third of everything they teach**
(120 → 80 unmasked tokens over those 40 documents).

Decision: **truncate the sequence after the final `<|im_end|>`**, in
`input_ids` as well as `labels` — not a blind "drop one token", so a template
change cannot silently cut real content. An empty document then contributes
exactly 2 unmasked tokens, `[]` and `<|im_end|>`, and the model is still
trained to stop.

### 2. Failure mode: a named exception, not `assert`

`MaskingError(ValueError)` naming the docid and both lengths, mirroring
`generate.py`'s over-budget `ValueError`. `assert` is wrong here: it vanishes
under `python -O`, and this check is the one guarding the project's single
non-negotiable (DESIGN §7).

### 3. Signature: the whole example in, `{input_ids, labels}` out

`tokenize_example(example, tokenizer, *, stop_after_eos=True) -> dict`. Taking
the example (not prompt/target strings) lets the error name the docid and
makes it a drop-in for `Dataset.map`; returning the two columns TRL consumes
keeps the function honest about its purpose.

### 4. One code path serves both models: **yes, proven**

Both tokenizers produce **identical unmasked token counts** over the smoke
subset (3,459) and identical completion statistics (min 2, mean 34.6, max
152); only the prompt side differs, by the four think-block tokens. The
targets tokenise the same because the family shares a tokenizer. So the smoke
run genuinely exercises the 4B's path, which is the whole justification for
milestone 3.

### Numbers for the spec

| | 0.6B | 4B-Instruct-2507 |
|---|---|---|
| tokens over the 100-document subset | 58,595 | 58,195 |
| unmasked (loss-bearing) | 3,459 (5.9 %) | 3,459 (5.9 %) |
| completion length min / mean / max | 2 / 34.6 / 152 | 2 / 34.6 / 152 |

**5.9 % of tokens carry loss.** Worth stating in the spec: it is the number
that makes completion-only loss matter, and the first-step loss check in
ticket 06 should be read against it.
