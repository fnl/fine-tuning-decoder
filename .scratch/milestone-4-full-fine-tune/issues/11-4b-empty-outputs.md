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
