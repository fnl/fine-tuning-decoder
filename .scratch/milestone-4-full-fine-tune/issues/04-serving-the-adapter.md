# 04 Serving the finished adapter for the final evaluation

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

The final evaluation scores the adapter over all 200 dev documents. The
milestone-2 baselines ran on vLLM (in `baselines.ipynb`, its own stack); the
training callback runs the Unsloth in-process engine. The adapter was trained
against an **NF4-quantised** base. Establish, with sources and dates:

1. **vLLM + LoRA on a T4** (the version `baselines.ipynb` installs): is LoRA
   supported on Turing / fp16, with what flags (`enable_lora`, `max_lora_rank`,
   `LoRARequest`), and can it combine with a bitsandbytes-quantised base
   (`quantization="bitsandbytes"`)? Memory for a 4B fp16 base + r16 adapter
   on 15 GB.
2. **The numerics question**: an adapter trained on an NF4 base, served over
   an fp16 base — what does the literature / Unsloth / PEFT documentation say
   about the mismatch (QLoRA paper, Unsloth docs, known issues)? Is serving
   over the same NF4 base the "correct" comparison, and does that change what
   the number means relative to the fp16 baselines?
3. **The in-process alternative**: cost of the Unsloth engine
   (`unsloth_engine` in `src/generate.py`) over 200 documents at 4B, from
   the 0.6B callback timings (95–245 s per 50 docs) and whatever scaling
   evidence exists.
4. **Adapter provenance gotcha**: milestone-3 ticket 02 found
   `adapter_config.json` records `base_model_name_or_path:
   unsloth/…-bnb-4bit`. Does that matter for loading into vLLM or plain PEFT
   over the official `Qwen/Qwen3-4B-Instruct-2507`?

End with the options and their trade-offs; the choice is ticket 08's.

## Answer

Resolved 2026-09-25 by a research subagent reading vLLM 0.29.0,
`vllm-bnb-plugin` 0.0.3, peft 0.20.0 and unsloth 2026.9.11 / zoo 2026.9.7
sources, the QLoRA paper, and W&B runs `63pxlpvn`, `meduja8a`, `t9nr1gsf`.
Nothing was run on a GPU. Full findings, cited file:line: branch
`research/serving-the-adapter` (commit `3ef9469`),
`.scratch/milestone-4-full-fine-tune/research/04-serving-the-adapter.md`.

**Three options; the choice is ticket 08's.**

| | A. vLLM, fp16 base + LoRA | B. vLLM, NF4 dynamic base + LoRA | C. In-process Unsloth, NF4 base |
|---|---|---|---|
| Stack | `baselines.ipynb` (vLLM 0.29.0) | A + `vllm-bnb-plugin==0.0.3` | `train.ipynb` |
| Numerics vs training | base differs by quantisation error `q` (small: dynamic quant keeps the worst layers 16-bit); fp16 compute | same NF4 weights, but a **bf16** 4-bit matmul emulated on sm_75 | identical to training and the callback |
| Comparable to M2 baselines | **best**: same weights, engine, rendering, sampling; the only difference is the adapter | confounded with base quantisation | confounded with quantisation *and* engine |
| Est. wall, 200 docs | ≈ 5–8 min | ≈ 10+ min (unknown) | ≈ 15–35 min standalone |
| Risk | LoRA Triton on sm_75 unverified for 0.29 (a T4 bug was fixed in triton 3.4) | young plugin, bf16 on T4, bnb slow in vLLM | lowest: already ran in M3 |
| The number means | the adapter as deployed on the official base | the trained function, served by vLLM | the trained function, exactly |

Key facts:

1. **vLLM LoRA** needs only `enable_lora=True` plus a `LoRARequest` per call.
   The defaults (`max_lora_rank=16`, `max_loras=1`, fp16 LoRA dtype) fit our
   r16 adapter exactly. Kernels are Triton-only with no sm_75 gate. Pass a
   local `snapshot_download(repo, revision=<sha>, allow_patterns=["adapter_*"])`
   path: a Hub id would pull `last-checkpoint/` too, and `LoRARequest` takes
   no revision. Memory: +≈ 63 MiB, so it fits the M2 budget. **bitsandbytes
   left vLLM core in 0.28**; it now needs the out-of-tree plugin.
2. **Numerics:** QLoRA trains ΔW on `dequant(W_NF4)`; serving over fp16
   removes `q`. The literature calls the effect small and unpredictable in
   sign; the one first-hand report (Qwen3-8B, T4) saw −0.004. **Unsloth's own
   default** export folds NF4-trained adapters onto the 16-bit base and says
   in-code that this beats the dequantised base. Neither choice is wrong: the
   comparison note must state which base the number was served on. **Free
   cross-check:** the training run's final eval point already scores the
   final adapter on NF4 over `dev[:50]`, so comparing it with the served rows
   for the same docids sizes the mismatch (±3 F1 noise).
3. **In-process cost (bears on ticket 06):** at 0.6B a callback averaged
   **162 s per 50 docs** (≈ 75–80 ms per decode step; most batches run to the
   512 cap). The **4B estimate is ≈ 150 ms/step**, i.e. roughly **2× the 0.6B
   callback** at the same batch size, and more at training-time batch 4–8.
   This is extrapolated — ticket 01 must measure it.
4. **Provenance:** `adapter_config.json` will name
   `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`. vLLM and
   `PeftModel.from_pretrained(base, …)` ignore it; `AutoPeftModel` and
   Unsloth follow it to the NF4 base. **Leave it** — it is true. The eval
   records the served base, the adapter revision sha, and
   `served_base_quantization`.

If B or C is chosen, an **NF4 zero-shot baseline** is the one extra run that
separates fine-tuning gain from quantisation cost — also what milestone 5
needs.
