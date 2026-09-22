# 05 See the mask: prototype `tokenize_example` on real documents

Type: prototype
Status: open
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
