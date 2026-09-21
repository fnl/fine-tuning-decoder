# 05 Does vLLM run Qwen3-4B on a free Colab T4 today?

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

DESIGN §8 prefers vLLM for inference. A free Colab T4 is compute 7.5: fp16
only, no bf16, no FlashAttention-2, 15 GB VRAM. Determine, with sources and
dates:

1. The current pip install line for vLLM on Colab (CUDA/torch version it
   pins vs. what Colab ships), install time, and known breakages with the
   Colab preinstalled torch.
2. Whether vLLM supports Qwen3-4B (arch `Qwen3ForCausalLM`) in fp16 on
   sm_75, which attention backend it falls back to, and a working
   `max_model_len` for ~2300 tokens (input ≤1800 + 512 new) on 15 GB
   including KV cache for batched generation of 200 docs.
3. How to pass a chat-templated multi-turn prompt (system + few-shot turns)
   through vLLM (`LLM.chat` vs. pre-rendered text) and set greedy decoding.
4. The fallback: batched `transformers.generate` with left padding on
   Qwen3-4B fp16 on a T4 — memory and rough throughput for 200 docs.
5. Whether Unsloth's `FastLanguageModel.for_inference` is a sensible third
   option on T4 (and whether it drags in the training install).

The answer feeds ticket 06 (engine decision); it does not decide.

## Answer

Researched 2026-09-20. Summary: a free T4 is still a supported vLLM target
(docs floor is now compute capability 7.5: V100 dropped, T4 kept) and users
report vLLM 0.21–0.28 running on T4 with `--dtype half`. Colab's current
runtime (2026.07) ships Python 3.12.13 + torch 2.11.0; vLLM 0.29.0
(2026-09-09) pins torch 2.13.0 and transformers ≥5.10.4, so it replaces
Colab's torch; the CUDA-12.9 wheel on vLLM's own index is the safe choice. On
sm_75 vLLM auto-selects the Triton attention backend; Qwen3-4B fits in fp16
with room for ~12–15 concurrent 2.3k-token docs (estimate). Unsloth cannot
share an env with vLLM ≥0.27 and always brings the training stack.

### 1. Install line, versions, breakages
- Colab 2026.07 (latest pinnable): Python 3.12.13, torch 2.11.0 ([runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html));
  bump requests to torch 2.13/2.14 still open ([colabtools #6093](https://github.com/googlecolab/colabtools/issues/6093), 2026-09-03).
  On runtime 2026.04 (2026-07-31): nvcc 12.8, driver reports CUDA 13.0;
  `vllm==0.19.0` (torch 2.10/cu128) worked, other versions gave
  `ImportError: libcudart.so.13` ([datawookie](https://datawookie.dev/blog/2026-07-31-running-llms-google-colab/)).
  Check `!nvidia-smi` (driver ≥580 ⇒ CUDA 13 OK) and `torch.__version__` first.
- vLLM 0.29.0 PyPI: `torch==2.13.0`, `transformers>=5.10.4`, Python ≥3.10,<3.15.
  CUDA 13.0 default wheel since v0.20.0 (2026-04-27), torch 2.13 since v0.27.0
  (2026-08-10) ([releases](https://github.com/vllm-project/vllm/releases));
  the install page's "CUDA (12.9)" is stale ([#44335](https://github.com/vllm-project/vllm/issues/44335)).
- Safe line (CUDA-12.x build; index verified 2026-09-20; recipe [#43435](https://github.com/vllm-project/vllm/issues/43435)):
  `pip install vllm==0.29.0 --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129`
  Plain `pip install vllm==0.29.0` only if driver ≥580. `vllm==0.26.0`
  (2026-07-27) pins `torch==2.11.0` and would keep Colab's torch.
- Breakages: torch swap (~550 MB wheel + ~3 GB libs; est. 3–8 min, unverified):
  install before importing torch, then restart ([#52300](https://github.com/vllm-project/vllm/issues/52300)).

### 2. Qwen3-4B on sm_75, backend, max_model_len
- Docs: "compute capability 7.5 or higher (e.g., T4 …)" ([GPU install v0.29.0](https://docs.vllm.ai/en/v0.29.0/getting_started/installation/gpu.html));
  `Qwen3ForCausalLM` is core. On <8.0 `supported_dtypes=[float16, float32]`,
  `dtype="auto"` warns and falls back to fp16 (`vllm/platforms/cuda.py` L236–243,
  `config/model.py` L2302); pass `dtype="half"` (bf16 also crashes a kernel, PR [#52224](https://github.com/vllm-project/vllm/pull/52224)).
- Backend order FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX; FA needs ≥8.0
  (`flash_attn.py` L198), FlashInfer floor raised to 8.0 as "broken on SM75"
  (`flashinfer.py` L504–514), Triton takes all (`triton_attn.py` L373) ⇒
  **TRITON_ATTN**, kv-cache fp16 only. Runs in practice: 0.28.0 `--dtype half`
  on sm_75 ([#54950](https://github.com/vllm-project/vllm/issues/54950), 2026-09-02), 0.21.0 on Kaggle T4 ([#43576](https://github.com/vllm-project/vllm/issues/43576)).
- Memory: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 4.02 B params → 8.04 GB fp16; 36 layers × 8 KV
  heads × 128 dim → 144 KiB/token, 316 MiB per 2 300-token doc. At
  `gpu_memory_utilization=0.9`: 13.5 GiB − 7.5 weights − ~1 activations ⇒
  ~4.5–5 GiB KV ⇒ ~12–15 docs in flight (estimate). Suggested `LLM(model,
  dtype="half", max_model_len=2560, gpu_memory_utilization=0.9, max_num_seqs=16)`;
  measure the longest rendered 3-shot prompt first. `enforce_eager=True` skips
  compile/graph warm-up (est. 2–4 min). 200 docs: est. 6–12 min.

### 3. Chat template and greedy decoding
- `LLM.chat(conversations, sampling_params, chat_template_kwargs=...)`
  (`entrypoints/llm.py` L612–627), or render with `tokenizer.apply_chat_template(
  msgs, tokenize=False, add_generation_prompt=True)` and `llm.generate(texts, sp)`
  (identical tokens; Qwen has no BOS). Instruct-2507 is non-thinking only;
  `enable_thinking=False` "is no longer required" (model card).
- Greedy must be explicit: `SamplingParams(temperature=0.0, max_tokens=512)`;
  vLLM otherwise applies `generation_config.json` (temp 0.7, top_p 0.8, top_k 20).

### 4. Fallback: batched `transformers.generate` fp16
- `from_pretrained(..., dtype=torch.float16, device_map="cuda")`, `padding_side="left"`,
  `generate(do_sample=False, max_new_tokens=512)` (config has `do_sample=true`).
- Estimates: batch 8 ≈ 8.04 GB + 8 × 316 MiB ≈ 11 GB (batch 16 ≈ 13.5 GB, tight);
  ~200 tok/s at batch 8 ⇒ 200 docs ≈ 10–25 min (static batching pads to the
  slowest doc). No published T4 numbers; fp16 overflow unverified — watch parse failures.

### 5. Unsloth `for_inference` on T4
- `unsloth==2026.9.7` (2026-09-18) requires `torch<2.13.0`, `transformers<=5.5.0`,
  xformers, bitsandbytes, peft, trl: full training stack, no inference-only
  extra. Official `nb/Qwen3_(4B)-Instruct.ipynb` (2026-08-18) installs the same
  plus `transformers==4.56.2`, loads `unsloth/Qwen3-4B-Instruct-2507` with
  `load_in_4bit=True`, and no longer calls `for_inference` (`models/llama.py` L3849).
- Compatible with Colab's torch 2.11, incompatible with vLLM ≥0.27 (torch) and
  ≥0.24 (transformers); `fast_inference=True` needs vLLM and only resolves with
  0.20–0.23. Sensible only inside the training notebook; NF4 would change the
  baseline vs. fp16 ([install docs](https://unsloth.ai/docs/get-started/install/pip-install)).

### Open caveats
- Colab's live default may be newer than the 2026.07 snapshot; verify torch/driver in-notebook.
- No first-hand vLLM 0.29.0 + Colab T4 run found; nearest: 0.28.0 on sm_75, 0.19.0/0.21.0 on Colab/Kaggle T4.
- Install time, KV budget and all throughput figures are estimates.
- `transformers>=5.10.4` may clash with other preinstalled Colab packages (unverified).

## Comments

**2026-09-21 (first-hand, free Colab T4):** the install line
`pip install vllm==0.29.0 --extra-index-url https://wheels.vllm.ai/0.29.0/cu129 --extra-index-url https://download.pytorch.org/whl/cu129`
from `notebooks/colab.ipynb` installs cleanly, followed by `pip install -e .`
and the runtime restart. This closes the "no first-hand 0.29.0 + Colab T4
report" caveat for the *install*; the `vllm==0.26.0` fallback is not needed
for it. Still unverified: generation on sm_75 (TRITON_ATTN, `dtype="half"`),
install time and throughput. The notebook did not run through on that
attempt, but not because of vLLM: the smoke cell failed with
`FileNotFoundError: configs/qwen3-4b-zero-shot.yaml` — the runtime restart
resets the working directory to `/content`, and only the clone cell changed
it. Fixed in the notebook by prefixing every post-restart cell with
`%cd /content/fine-tuning-decoder` and chaining the smoke cell's commands
with `&&`. Generation on the T4 remains the next thing to verify.

**2026-09-21, second attempt:** the "`transformers>=5.10.4` may clash with
other preinstalled Colab packages" caveat is confirmed: `from vllm import LLM`
→ transformers `audio_utils` → `import torchaudio` →
`RuntimeError: PyTorch has CUDA version 12.9 whereas TorchAudio has CUDA
version 12.8` (Colab's torchaudio predates the torch swap). Fix in the
notebook: `pip uninstall --yes torchaudio` right after the vLLM install;
transformers imports it only when present and nothing here needs audio.

**2026-09-21, third attempt (run by the agent through Claude in Chrome) —
generation verified, ticket fully closed.** Free Colab T4, driver 580.82.07
(CUDA 13.0), Python 3.13.15, torch 2.13.0+cu129 after the swap. With
`torchaudio` removed, `generate --limit 5` ran end to end: architecture
`Qwen3ForCausalLM`, bf16 cast to fp16, attention backend `TRITON_ATTN`
(FA2 refused on compute capability 7.5, FlashInfer sampler falls back too),
weights download 103 s, model load 7.64 GiB in 145 s total, engine init
27 s, KV cache 4.8 GiB = 34,928 tokens = 11.4 concurrent 3072-token
requests. 5 documents generated in 12 s (input 186 tok/s, output 46 tok/s),
0 cut off. The estimates in the answer (12–15 concurrent docs, ~8 GB
weights) held; the throughput and install figures are no longer estimates.
Caveat found on the way: a runtime where vLLM was installed *before* the
torchaudio fix keeps the broken torchaudio, so the notebook now uninstalls
it unconditionally, outside the install guard.
Full runs the same evening: zero-shot 200 dev documents ≈ 4.5 min wall
(W&B `meduja8a`), 3-shot ≈ 5 min (`t9nr1gsf`), each including ~1 min model
load from the cached weights; 3 and 1 cut-off outputs respectively.
