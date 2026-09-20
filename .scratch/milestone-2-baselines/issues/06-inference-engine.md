# 06 Inference engine for baselines

Type: grilling
Status: resolved
Blocked by: 04, 05
HITL: yes

## Question

Given the research in 04 and 05: vLLM, batched `transformers.generate`, or
Unsloth fast inference for the two GPU baselines on a free T4? Decide:

- the engine and its install line for the notebook;
- fp16 dtype, greedy, `max_new_tokens=512`, batching strategy;
- `max_model_len` / the truncated-document policy with few-shot context
  (graduates the "few-shot input budget" fog);
- how `generate.py` isolates the engine (one function per engine? one
  engine only, fallback deferred?) so milestone 3's eval callback can reuse
  the callable core on an in-process model.

## Answer

Grilled 2026-09-20; all recommendations accepted.

**Measured facts** (official `Qwen/Qwen3-4B-Instruct-2507` tokenizer,
`data/prepared`): train `n_input_tokens` deciles 345…984, max 1767 — the
count *includes* the 183-token system prompt; targets median 18 tokens, p90
83, max 993; shortest input+target per event-count bucket: 0 events 229,
1 event 254, 2 events 299, 3+ events 326. Dev max `n_input_tokens` 1375,
0 truncated documents. Three short exemplars add ≈700–1000 tokens, so the
longest 3-shot input is ≈2100–2400 → ticket 05's `max_model_len=2560` is
too tight for 3-shot.

**Engine**: vLLM only, **`vllm==0.29.0`** pinned, notebook-only dependency:
`pip install vllm==0.29.0 --extra-index-url https://wheels.vllm.ai/0.29.0/cu129
--extra-index-url https://download.pytorch.org/whl/cu129`, preceded by a
cell printing `nvidia-smi` and `torch.__version__`, followed by a runtime
restart. Fallback order if the install or first generation breaks:
`vllm==0.26.0` (keeps Colab's torch 2.11), then open a ticket for a
`transformers.generate` engine — not coded speculatively. Milestone 3 writes
the in-process engine against its real need.

**Knobs**: `LLM(model, dtype="half", max_model_len=3072,
gpu_memory_utilization=0.9, max_num_seqs=16, enforce_eager=True, seed=0)`,
`SamplingParams(temperature=0.0, max_tokens=512)`. `max_model_len` and
`max_new_tokens` define the experiment and live in the YAML
(`generation:` — exact keys in ticket 09); the T4-specific rest are constants
inside `vllm_engine()`. Baseline `model:` is the official `Qwen/` id, not
the Unsloth mirror. No 0.6B baseline config; the notebook smoke is
`--limit 5` on the 4B.

**Few-shot input budget** (fog graduated): one input budget only — the
dataset's `truncated` flag from `prepare.py`. `generate.py` counts the
rendered input tokens and **fails loudly with the docid** if any exceeds
`max_model_len − max_new_tokens`; no second truncation layer, no per-doc
exemplar dropping. Ticket 07's prototype prints the max rendered 3-shot
length to confirm 3072 holds on dev. `n_truncated` reported from the
dataset flag.

**Rendering**: pre-render every conversation with
`tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
enable_thinking=False)` using the official tokenizer (identical to
`prepare.py._n_input_tokens`; `enable_thinking=False` always passed — no-op
on 2507, required on hybrid templates), then `llm.generate(texts, sp)`.
Engine-agnostic and immune to the mirror-template trap (ticket 04).

**Seam**: `Engine = Callable[[list[str]], list[GenerationResult]]` (rendered
texts in; output text + finish reason out). `vllm_engine(config) -> Engine`
(lazy `import vllm`), `constant_engine(text) -> Engine` for always-`[]` —
the same pipeline rows → JSONL → eval → W&B, not a separate path. Core:
`generate(examples, engine, exemplars) -> rows` (signature pinned in 09).
All 200 inputs go to the engine in one call; vLLM's progress bar is the
progress; `--limit N` is the smoke knob. Prefix caching (on by default in
vLLM V1) makes the shared system + exemplar prefix nearly free.

**Cut-off outputs**: the engine records `finish_reason == "length"` per row
so eval/W&B can count cut-off outputs (CONTEXT.md term); field name in 09.
