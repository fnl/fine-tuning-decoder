# 11 The 4B generates empty outputs

Type: task
Status: claimed
Blocked by: —
HITL: yes

## Question

Both 4B probes (ticket 01: `4lkcxlgt`, `wk6ypti0`) scored 50 dev documents
with the in-process Unsloth engine and got `""` for every output: 49 ended
"stop", 1 "length". The same code scored 16.3 at 0.6B, and vLLM served the
same base model in fp16 fine for the baselines (18.6). Separately, the 4B's
step-1 training loss is 3.14 against the 0.6B's 1.37 on the same leading
documents. Find out why, on a GPU, before any cadence decision (06) relies on
callback times.

Leading hypothesis: fp16 numerics in Unsloth's 4B path on sm_75 (activation
overflow in the NF4-dequantised forward). It would degrade training (high loss) and
generation (degenerate special-token output) alike, while vLLM, a different
kernel path, is unaffected. Alternatives: the model Unsloth resolves for
`Qwen/Qwen3-4B-Instruct-2507` (its `-unsloth-bnb-4bit` repo) or its tokenizer
(pad `<|vision_pad|>`, versus `<|endoftext|>` officially); Unsloth's fast
generation path at 4B.

Discriminating checks (throwaway cells, one GPU session):

1. **Raw token ids** of a few outputs from the untrained 4B via the same
   `unsloth_engine` (`skip_special_tokens=False`): EOS at once, pad, NaN-driven
   token 0, …?
2. **Base-model loss** of a handful of train examples, no adapter, with
   Unsloth NF4 fp16 vs plain transformers fp16 (no Unsloth) vs Unsloth NF4 with
   fp32 logits/upcast if exposed: does 3.14 reproduce outside Unsloth?
3. **Serve the probe adapter** (`fnl-es/qwen3-4b-muc4-lora-probe` @ `9354e17`)
   through vLLM fp16 + LoRA: if it outputs JSON, the adapter is sound and
   Unsloth generation is the culprit. This doubles as ticket 08's T4 vLLM-LoRA
   de-risk at 4B.
4. If a fix is found: re-measure one callback's wall time and cut-offs at
   generation batch 4 and 8 (ticket 01's missing numbers).

Resolve with the cause, the fix or workaround (a knob, a pin, an engine
change), and the callback numbers.

## Comments

### 2026-09-26: round 1, `probes/diag_4b.py` (commit `e766536`)

W&B runs `bfbvxxhk` (unsloth-fp16), `lz1zddiy` (unsloth-fp32), `nagkh17l`
(hf-nf4), `ryqasv13` (hf-fp16), Tesla T4, base model with zero-init LoRA,
first 8 train examples, first 8 dev documents.

| mode | mean loss | per-example loss | hidden max | outputs |
|---|---|---|---|---|
| unsloth-fp16 | **8.14** | 2.5, **11.9, 13.2**, 4.4, **12.9**, 5.9, **12.8**, 1.5 | 7,552 | valid JSON |
| unsloth-fp32 | 14.85 | ≈ 15 everywhere | 8,294 | `!!!!` / `celcel…` garbage |
| hf-nf4 | 1.17 | 1.69, 0.18, 0.00, 2.80, 0.00, 3.67, 0.00, 1.05 | 7,524 | valid JSON |
| hf-fp16 | 1.53 | 2.01, 2.35, 0.00, 3.08, 0.00, 3.62, 0.00, 1.16 | 7,400 | valid JSON |

- **The fp16-overflow hypothesis is refuted.** Hidden states peak at ≈ 7,500 in
  every fp16 mode, including plain transformers, far below 65,504. No non-finite logits anywhere.
- **The untrained Unsloth 4B generates correctly.** Its outputs are token-for-token
  identical to hf-nf4 on the first dev document, at generation batch 1 and 4. So the
  generation path is sound, and the probes' empty outputs come from what 12 steps of
  training did to the model.
- **Unsloth's loss is wrong.** The examples that plain transformers finds nearly free
  (≈ 0, presumably irrelevant documents whose target is `[]`) cost ≈ 12–13 nats under Unsloth. This fits
  labels shifted one position too far: the model is scored on `<|im_end|>`
  where `[]` belongs. Training on such targets teaches it to end the turn at
  once, which matches the empty outputs; the 3.14 step-1 loss fits too.
- **unsloth-fp32 is broken for this model.** Not a workaround.
- **`git_dirty`**: `huggingface_tokenizers_cache/` and `unsloth_compiled_cache/`
  are untracked in the clone. They should go in `.gitignore`.

Round 2 (`probes/diag_loss.py`, commit `bb7d22a`) recomputes cross-entropy from
Unsloth's own logits at label shifts 0/1/2 against its reported loss, for 4B
and 0.6B.

### 2026-09-26: round 2, `probes/diag_loss.py` (commit `bb7d22a`); root cause

W&B runs `09lghiwk` (4B), `re31jlou` (0.6B). Unsloth's reported loss equals
cross-entropy recomputed from its own logits at the normal shift of 1, on every
example and for both models. **The loss computation is correct; the shift hypothesis is refuted.**
The difference is in the **labels**: completion lengths are 6 vs 2 tokens on
`[]` documents and 76 vs 72 on the first, so the 4B's completions carry 4 extra
tokens.

Confirmed locally (CPU, `tokenize_example` on train[1]):

- `Qwen/Qwen3-4B-Instruct-2507`: 2 completion tokens, `'[]<|im_end|>'`
- `unsloth/Qwen3-4B-Instruct-2507` (the tokenizer `FastLanguageModel` returns):
  6 tokens, `'<think>\n\n</think>\n\n[]<|im_end|>'`
- `Qwen/Qwen3-0.6B`: 2 tokens, `'[]<|im_end|>'`

**Cause:** Unsloth's chat template for Qwen3-4B-Instruct-2507 renders an empty
think block into the assistant turn of a full conversation. It does not add one to the
generation prompt. The prompt is still a token prefix, so `MaskingError` passes, but every target
starts with `<think>\n\n</think>\n\n`, which this instruct model never emits.
That explains the ≈ 12–13 nat losses on `[]` documents, the 3.14 step-1 loss and,
after 12 steps, outputs that are only the (special, decoded-away) think block.
Inference and the vLLM baselines use the official template, so training and
evaluation disagree. The 0.6B's template has no such block, so milestone 3 never saw it.
