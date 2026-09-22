# 03 Generating with the model under training, for the eval callback

Type: research
Status: claimed
Blocked by: —
HITL: no

## Question

The eval callback (Q4) must generate on 50 dev documents every ½ epoch from
*inside* the training loop, with the model already on the GPU in 4-bit. vLLM
is not available in this environment. Establish:

1. How to generate from an Unsloth 4-bit model mid-training: whether
   `FastLanguageModel.for_inference` / `for_training` still exist in
   `unsloth 2026.9.x` (milestone 2 ticket 05 found the official Qwen3
   notebook no longer calls `for_inference`), and what the current
   recommended call is.
2. Batched `model.generate` on a 4-bit Qwen3-0.6B: left padding, the pad
   token (Qwen has `<|endoftext|>` as pad, `<|im_end|>` as eos),
   `do_sample=False` (the generation config ships `do_sample=true`), the
   `use_cache` / gradient-checkpointing interaction, and whether the KV cache
   must be re-enabled and then re-disabled around the callback.
3. Memory at 0.6B on a T4 while the optimizer state is resident: a safe
   batch size for 50 documents of ~700 input + 512 new tokens, and rough
   wall time per callback (this runs 2–6 times per smoke run).
4. Whether the model must be switched to `eval()` / `torch.no_grad()`
   explicitly, and what the known failure mode is when it is not.
5. Whether the same code path works unchanged for a **merged-nothing**
   LoRA model at 4B (milestone 4 reuses this callback).
6. Any TRL-native alternative we are ignoring (`SFTTrainer`'s own
   `compute_metrics` / `predict` path) and why generation-based scoring does
   not fit it.

Answer in this file with sources and dates. Feeds ticket 09, which decides
whether the in-process engine lives in `generate.py` beside `vllm_engine` or
in `train.py`.
