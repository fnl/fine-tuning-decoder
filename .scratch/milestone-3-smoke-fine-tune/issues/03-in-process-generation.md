# 03 Generating with the model under training, for the eval callback

Type: research
Status: resolved
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

## Answer

Researched 2026-09-22 against `unsloth 2026.9.9` (wheel, 2026-09-19), `unsloth_zoo
2026.9.7`, `trl 0.24.0`, `transformers 5.5.0`, `peft 0.21.0` sources downloaded and
read locally. **Headline: do not write a mode-switching dance at all.** Unsloth
replaces `model.generate` with `unsloth_fast_generate`, which snapshots the training
mode and the gradient-checkpointing mode, calls `for_inference`, generates under
`torch.inference_mode()` + `autocast`, and calls `for_training` back with the *same*
GC mode. A callback that calls `model.generate(...)` on the trainer's model is
correct as written. The two things the engine must still do itself are **left
padding** and **explicit greedy decoding**.

### 1. `for_inference` / `for_training` in `unsloth 2026.9.x`

- **Both still exist and are not deprecated.** `FastLlamaModel.for_inference`
  (`unsloth/models/llama.py:3907`) and `for_training` (`:3953`); the generic
  `FastLanguageModel.for_inference` / `for_training` forward to
  `FastBaseModel` / `FastDiffusionModel` (`unsloth/models/loader.py:1214-1223`).
  Both are also bound onto the model and every wrapped submodule as
  `model.for_inference` / `model.for_training` (`llama.py:3049-3054`, and again
  after `get_peft_model` at `llama.py:3895-3900`).
- **The current recommended call is plain `model.generate(...)`.** `from_pretrained`
  swaps in the patched generate: `model._old_generate = model.generate;
  model.generate = types.MethodType(unsloth_fast_generate, model)`
  (`llama.py:3058-3061`). That wrapper does (`llama.py:2171-2249`):
  - `restore_training_mode = self.training` (`:2172`);
  - snapshots the real GC mode off the modules, keeping the *string* `"unsloth"`
    rather than collapsing it to a bool (`:2174-2178`);
  - `FastLlamaModel.for_inference(self)` (`:2180`);
  - forces `kwargs["cache_implementation"] = "dynamic"` (`:2209`) — so no static
    cache, no `torch.compile` recompile per batch shape on this path;
  - defaults `pad_token_id` to `config.eos_token_id` (`:2234`);
  - `with _get_inference_mode_context_manager(self), torch.autocast(...):
    output = self._old_generate(...)` (`:2236-2241`) — `torch.inference_mode()`
    for us (`_utils.py:4584-4591`; `no_grad` only for torchao);
  - `if restore_training_mode: FastLlamaModel.for_training(self,
    use_gradient_checkpointing = <the snapshot>)` (`:2243-2247`).
- **Qwen3 takes this path.** `model_type == "qwen3"` dispatches to `FastQwen3Model`
  (`loader.py:943-951`), which is `class FastQwen3Model(FastLlamaModel)`
  (`qwen3.py:354`) and delegates to `FastLlamaModel.from_pretrained` (`qwen3.py:399`).
  Not the `vision.py` / `FastBaseModel` path (whose `unsloth_base_fast_generate`
  uses `inference_mode` but does **not** call `for_inference`/`for_training`).
- **Through PEFT too.** `PeftModelForCausalLM.generate` →
  `self.base_model.generate(...)` (`peft/peft_model.py:2229-2250`, and
  `PeftModel.generate` → `self.get_base_model().generate(...)` at `:1009-1012`),
  and `LoraModel.__getattr__` forwards to the inner model, so the patched
  `unsloth_fast_generate` is what actually runs.
- The official Qwen3 Colab notebooks agree: `nb/Qwen3_(4B)-Instruct.ipynb`
  (unslothai/notebooks, last commit **2026-08-18**) runs its inference cell (cell 33)
  *after* `trainer.train()` (cell 30) and calls plain
  `model.generate(**tokenizer(text, return_tensors="pt").to("cuda"), ...)`; the string
  `for_inference` does not appear anywhere in the notebook. Same for
  `nb/Qwen3_(0.6B)-Reasoning-Conversational-ExecuTorch.ipynb`.
- The docs page [Unsloth Inference](https://unsloth.ai/docs/basics/inference-and-deployment/unsloth-inference)
  still shows `FastLanguageModel.for_inference(model)`, but it says *"Last updated 8
  months ago"* (≈ 2026-01) — **stale, not wrong**: the call is harmless, just redundant.
- **Recommendation:** call `model.generate(...)`. Do not call `for_inference` /
  `for_training` manually — calling `for_inference` yourself before `generate` makes
  the wrapper's `restore_training_mode` read `False`, so it will **not** call
  `for_training` afterwards and training resumes with GC off and `use_cache` on.

### 2. Batched `generate` on 4-bit Qwen3-0.6B

- **Left padding: you must set it, the wrapper sets it too late.** `for_inference`
  sets `_saved_temp_tokenizer.padding_side = "left"` (`llama.py:3919`) — but that
  happens *inside* `generate`, after your engine has already tokenized. And the last
  thing that touched the tokenizer before your callback ran was
  `patch_peft_model`, which sets `padding_side = "right"` (`llama.py:3884-3888`),
  or `for_training` (`:3969`). So the engine must do
  `tokenizer.padding_side = "left"` before `tokenizer(batch, padding=True)`.
  This is exactly what TRL's own in-training generation callback does
  (`trl/trainer/callbacks.py:515`, `tokenizer.padding_side = "left"`).
  **Safe for training:** TRL 0.24's SFT collator hard-codes
  `padding_side="right"` on every `pad()` call (`trl/trainer/sft_trainer.py:193-215`)
  and never reads `tokenizer.padding_side`, so leaving it flipped cannot corrupt the
  training batches. `for_training` flips it back to `"right"` anyway on exit.
- **Pad token.** `Qwen/Qwen3-0.6B` `tokenizer_config.json`: `pad_token` =
  `<|endoftext|>` (151643), `eos_token` = `<|im_end|>` (151645) — as the ticket says.
  Note `generation_config.json` lists **both** as EOS:
  `"eos_token_id": [151645, 151643]`. `unsloth/Qwen3-0.6B` deliberately differs:
  `pad_token` = `<|vision_pad|>` (151654), `padding_side: "left"` baked in. Either
  works; pass `pad_token_id=tokenizer.pad_token_id` **as a kwarg**, because
  `unsloth_fast_generate` otherwise substitutes `config.eos_token_id` (`:2234`) and
  a kwarg beats a `generation_config` field.
- **`do_sample=False` must be explicit and the sampling knobs must be unset.**
  `Qwen/Qwen3-0.6B/generation_config.json` (and `unsloth/`'s) ship
  `do_sample: true, temperature: 0.6, top_k: 20, top_p: 0.95`. Passing only
  `do_sample=False` leaves the three knobs set and transformers 5.5.0 emits three
  "this flag is only used in sample-based generation modes" warnings
  (`generation/configuration_utils.py:639-659`); the guard is
  `self.temperature is not None and self.temperature != 1.0`, so
  `temperature=None, top_p=None, top_k=None` silences them cleanly.
- **`use_cache` / gradient checkpointing: handled, but only because `for_inference`
  runs.** The chain is:
  - `prepare_model_for_training` calls `disable_use_cache(model)` whenever
    `use_gradient_checkpointing in (True, "unsloth")`
    (`unsloth_zoo/training_utils.py:496-498`), which sets `use_cache = False` on
    every config and records the originals on `model._unsloth_use_cache_originals`
    (`:275-304`);
  - `for_inference` calls `restore_use_cache(model)` (`llama.py:3944-3948` →
    `training_utils.py:307-319`) **and** sets `module.gradient_checkpointing = False`
    on every module plus `model.eval()` (`llama.py:3914-3932`);
  - `for_training` re-disables it: `disable_use_cache(model)` when the record exists
    (`llama.py:4000-4007`) and restores `module.gradient_checkpointing = <mode>`
    (`:3984-3987`).
  So **yes, the KV cache must be re-enabled and re-disabled around the callback —
  and Unsloth already does both**, provided you let its `generate` wrapper do it.
- **Suggested call** (per batch, after `tokenizer.padding_side = "left"`):
  `model.generate(**batch, max_new_tokens=512, do_sample=False, temperature=None,
  top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id,
  eos_token_id=[151645, 151643])`. Tokenize with `add_special_tokens=False` to match
  `generate.py`'s existing length check (`src/generate.py:68`); Qwen adds no BOS, so
  this is cosmetic but keeps the two paths identical.
- **`finish_reason`.** `generate` returns only ids — there is no vLLM-style
  `finish_reason`. Derive it from the generated span:
  `"stop" if any(i in EOS_IDS for i in generated_ids) else "length"`. Slice the span
  as `outputs[:, batch["input_ids"].shape[1]:]`, which is only valid **because** of
  left padding (all rows share one prompt length) — the same trick TRL uses
  (`callbacks.py:97-99`).

### 3. Memory and wall time at 0.6B on a T4 with optimizer state resident

Arithmetic from `Qwen/Qwen3-0.6B/config.json` (28 layers, 8 KV heads, head_dim 128,
hidden 1024, vocab 151936, `tie_word_embeddings: true`):

- KV cache = 2 × 28 × 8 × 128 × 2 B = **112 KiB / token**. At 768 prompt + 512 new =
  1280 tokens → **140 MiB / sequence**. Batch 8 ≈ 1.1 GiB, batch 16 ≈ 2.2 GiB,
  batch 32 ≈ 4.4 GiB.
- Weights: embedding 151,936 × 1,024 = 155.6 M params stay fp16 (bnb does not
  quantize embeddings; lm_head is tied) ≈ **311 MB**; the remaining ≈ 440 M params in
  NF4 with double quant ≈ **250 MB**. Total ≈ **0.56 GB**.
- LoRA r=16 on q/k/v/o/gate/up/down × 28 layers ≈ 10.1 M params → ≈ 40 MB fp32 +
  ≈ 20 MB `adamw_8bit` state + grads ≈ **0.1 GB**. Optimizer state being resident is
  a rounding error at this size.
- **Recommendation: `batch_size = 16`** (≈ 2.9 GB total, on a 15 GB T4 with ~13.5 GB
  usable). 32 fits on paper; 16 leaves room for fragmentation and for the 4B reuse to
  keep the same code. Sort the 50 inputs by token length before batching to cut
  padding waste (prompts run 444–765 tokens per the map), and restore the original
  order before returning.
- **Wall time (estimate, not measured).** 50 docs / 16 = 4 batches; each decode step
  is a full forward of a 0.6 GB NF4 model at batch 16, est. 20–40 ms on a T4 (bnb
  dequant is the bottleneck, and NF4 decode is *slower* than fp16 —
  [bitsandbytes #611](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/611)).
  `generate` stops only when **every** row in the batch has finished, so mixed
  batches (25 of the first 50 dev docs are empty → 3-token completions) run close to
  the 512-step cap. → **≈ 15–25 s per batch, ≈ 1.5–3 min per callback**; call it
  **2–5 min** with prefill and the mode switches. At 2–6 callbacks that is **5–30 min**
  added to the smoke run. Length-sorted batching should pull the low end down
  materially, because the empty-document batches then finish in ~10 steps.

### 4. `eval()` / `torch.no_grad()` — and the failure mode without them

- **Not needed if you call `model.generate`**: `for_inference` does `model.eval()`
  (`llama.py:3927`) and the wrapper enters `torch.inference_mode()`
  (`llama.py:2236-2241`). Adding your own `with torch.no_grad():` around the call is
  harmless but redundant.
- **The failure mode if the model is left in train mode with GC on is not a crash —
  it is a silent quadratic slowdown.** transformers 5.5.0
  `modeling_layers.py:59-91`: `GradientCheckpointingLayer.__call__` does, verbatim,
  `if self.gradient_checkpointing and self.training:` → force `use_cache = False`,
  `past_key_values = None`, warn once *"Caching is incompatible with gradient
  checkpointing in Qwen3DecoderLayer"*, then run the layer through
  `self._gradient_checkpointing_func`. So every one of 512 decode steps re-runs the
  entire ~700-token prefix through checkpointed (i.e. doubly-executed) layers with no
  KV reuse. A 2-minute callback becomes hours.
- Second failure mode: without `inference_mode`/`no_grad`, 512 decode steps build and
  retain an autograd graph → OOM on a T4, and the retained activations survive into
  the next optimizer step.
- Third, subtler: tensors produced under `inference_mode` cannot be used in autograd
  later. Unsloth's own RL path clones them for exactly this reason (`models/rl.py:199-207`,
  *"`.clone` is required because inference_mode is forced here"*). Our engine decodes
  to `str` immediately, so this does not bite — but do not stash the raw output tensor.
- **Do not wrap the call in TRL's `unwrap_model_for_generation`** (`trl/models/utils.py:308-351`).
  It calls `gradient_checkpointing_disable()` on entry and `gradient_checkpointing_enable()`
  on exit — Unsloth documents that this *"overwrit[es] Unsloth's `unsloth` wrapper,
  which for Gemma-4 corrupts forward numerics and blows GRPO KL divergence up to
  ~10^12 at step 1"* (`unsloth/models/rl.py:3118`, and again `:3474`), which is why
  Unsloth monkey-patches it to a no-op — **but only for the RL trainers**
  (`rl.py:304-312`), not for `SFTTrainer`. On a single T4 `accelerator.unwrap_model`
  is a no-op anyway, so the context manager buys nothing and costs the GC mode.
- **Known regression, already fixed in our version:** generating inside a
  `TrainerCallback` raised `ValueError: Invalid target device: None` in
  `unsloth 2025.10.3`
  ([unsloth #3538](https://github.com/unslothai/unsloth/issues/3538), opened
  2025-10-31; fix in unsloth-zoo #1229). `unsloth_zoo 2026.9.7` carries the
  regression test — `tests/test_per_layer_device.py:101` asserts *"a None index is
  exactly the #3538 crash"*. Not a risk on this stack.
- Not a risk either: `_unsloth_install_pretrain_detector` (`models/_utils.py:262-288`)
  only flags **grad-enabled** forwards **before** `trainer.train()`, so it can drop a
  poisoned `torch.compile` cache. Our callback runs mid-training under
  `inference_mode`, which the detector explicitly treats as clean.

### 5. Does the same path work for the 4B LoRA (nothing merged)?

**Yes, unchanged — except the batch size, which must be a config knob, not a constant.**

- `Qwen3-4B-Instruct-2507/config.json` is also `model_type: "qwen3"` /
  `Qwen3ForCausalLM`, so it takes the same `FastQwen3Model` → `FastLlamaModel`
  dispatch and the same patched `generate`. Milestone 4 changes no engine code.
- Sizes: 36 layers, 8 KV heads, head_dim 128 → KV = **144 KiB/token**; at 1800 + 512 =
  2312 tokens that is **325 MiB/sequence**. Weights: embedding 151,936 × 2,560 =
  389 M params fp16 ≈ 778 MB + ≈ 3.6 B params NF4 ≈ 2.0 GB → **≈ 2.8 GB**.
  With training state resident, budget ≈ 8–9 GB for KV → **batch 4–8 at 4B**
  (estimate; batch 16 = 5.2 GB is arithmetically possible but leaves no slack).
- `max_position_embeddings` is 262,144 at 4B (40,960 at 0.6B), so the wrapper's
  `input_len + max_new_tokens > max_position_embeddings` guard
  (`llama.py:2186-2194`) cannot trip at either size with our ~2.3k budget.
- `generation_config.json` differs only in the sampling defaults (temp 0.7 / top_p
  0.8) — irrelevant once we pass `do_sample=False` with the knobs unset.
- The LoRA is never merged: `PeftModelForCausalLM.generate` reaches the patched
  generate through `base_model`, with the adapters live. That is the same code path
  at both sizes.

### 6. The TRL-native alternatives, and why they do not fit

- **`SFTTrainer.compute_metrics` / `Trainer.predict` — wrong kind of output and
  absurd memory.** `SFTTrainer` is `class SFTTrainer(BaseTrainer)` and
  `class BaseTrainer(Trainer)` (`trl/trainer/base_trainer.py:27`); `SFTConfig` is
  `class SFTConfig(TrainingArguments)` (`sft_config.py:22`). Neither derives from
  `Seq2SeqTrainer` / `Seq2SeqTrainingArguments`, which is the only place
  `predict_with_generate` exists (transformers 5.5.0
  `training_args_seq2seq.py:56`). So `compute_metrics` receives an `EvalPrediction`
  of **teacher-forced logits**, not generations. Two independent killers:
  1. *Semantics.* Our score is `parse_target(free-running output)` → `score()`. Under
     teacher forcing every position is conditioned on the gold prefix, so the metric
     measures something the model never has to produce at inference. It also cannot
     produce a `cut_off` / `finish_reason`, and cannot produce a parse failure — the
     very signal milestone 2 reports (7.5 % / 5 %).
  2. *Memory.* Without `preprocess_logits_for_metrics` the eval loop accumulates
     `50 × ~1200 × 151,936 × 4 B ≈ 36 GB` of logits. That parameter
     (`sft_trainer.py:559-562, 588, 856`) exists precisely because this is
     unworkable.
- **`LogCompletionsCallback` — right mechanism, wrong job.** TRL 0.24 *does* generate
  inside training (`trl/trainer/callbacks.py:484-548`, via `_generate_completions`
  at `:67-102`), but it only builds a `wandb` table of `(step, prompt, completion)`;
  it computes no metric, requires `trainer.eval_dataset` with a literal `"prompt"`
  column, and routes through `unwrap_model_for_generation` (see §4). **Copy its
  shape, not its wrapper**: set `padding_side="left"`, batch at
  `per_device_eval_batch_size`, slice `generation[len(prompt):]`, decode.
- **Conclusion:** nothing TRL-native is being ignored. A plain `TrainerCallback` that
  calls our own `Engine` is the right seam, and it keeps `score()` / `flatten()` and
  the W&B payload shared with the milestone-2 eval path.

### Recommended code shape (fits `Engine = Callable[[list[str]], list[GenerationResult]]`)

Lives beside `vllm_engine` in `src/generate.py` (same seam, same `GenerationResult`,
same `constant_engine` neighbourhood); `train.py` builds it from the trainer's model
and passes it to the existing `generate()`.

```python
def unsloth_engine(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_new_tokens: int,
    batch_size: int = 16,
) -> Engine:
    """The model under training, greedy and batched; Unsloth's patched `generate`
    owns the eval/train, use_cache and gradient-checkpointing switches."""
    import torch

    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]

    def engine(inputs: list[str]) -> list[GenerationResult]:
        tokenizer.padding_side = "left"
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        results: dict[int, GenerationResult] = {}
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch = tokenizer(
                [inputs[i] for i in index],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(model.device)
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
            for i, row in zip(index, outputs[:, batch["input_ids"].shape[1] :], strict=True):
                ids = row.tolist()
                reason = "stop" if any(t in eos_ids for t in ids) else "length"
                results[i] = GenerationResult(
                    tokenizer.decode(ids, skip_special_tokens=True), reason
                )
            del outputs, batch
        torch.cuda.empty_cache()
        return [results[i] for i in range(len(inputs))]

    return engine
```

Used from a `TrainerCallback`, whose `on_step_end(args, state, control, **kwargs)`
is handed `model=trainer.model` and `processing_class=tokenizer`
(transformers 5.5.0 `trainer_callback.py:543-556`) — so the callback needs no
reference to the trainer beyond those kwargs. No `for_inference`, no
`for_training`, no `unwrap_model_for_generation`, no manual `use_cache` juggling.

### Open caveats

- **Nothing here was executed.** Every claim is read from source or from the official
  notebooks; no generation was run on a T4 with this stack. The wall-time and
  batch-size figures in §3 and §5 are arithmetic plus estimate, in the same spirit as
  milestone 2 ticket 05 before its first-hand comments.
- **The 20–40 ms/decode-step figure is a guess.** No published Qwen3-0.6B-NF4-on-T4
  throughput number was found; the only grounded neighbour is milestone 2's vLLM
  Qwen3-4B fp16 run (46 output tok/s at ~5 concurrent). Measure it on the first
  callback and put the number in the ticket.
- **`max_new_tokens` early-stop behaviour is assumed, not verified**: batches should
  terminate as soon as every row emits an EOS, so length-sorted batching should be a
  large win. Unverified that Unsloth's forced `cache_implementation="dynamic"` does
  not defeat transformers' early `StoppingCriteria` short-circuit.
- **Version skew with the official notebook.** `nb/Qwen3_(4B)-Instruct.ipynb` pins
  `transformers==4.56.2` + `trl==0.22.2`; our locked stack is transformers ≤5.5.0 +
  TRL 0.24.0. The `unsloth_fast_generate` reading above is from the 2026.9.9 wheel,
  which is the version that will actually install, but the notebook's exact pin is
  not what we run.
- **`unsloth 2026.9.9` ships no sdist** (only a wheel); the source above was read
  from the unpacked `unsloth-2026.9.9-py3-none-any.whl`. `unsloth_zoo` is 2026.9.7,
  matching what was already extracted; a newer zoo may install alongside.
- **`tokenizer.padding_side` is mutated globally.** Safe against TRL 0.24's collator
  (which hard-codes right padding), but any *other* consumer of the same tokenizer
  object during training would see "left" until the next `for_training`. If the smoke
  run ever adds a second consumer, save and restore it explicitly.
- **`model.device` on a PeftModel** is assumed to resolve through `__getattr__` to the
  base model's device (TRL relies on the same at `callbacks.py:93`). If it ever
  returns `meta` or `None`, use `next(model.parameters()).device`.
- **`fp16_full_eval`** ([Unsloth troubleshooting FAQ](https://unsloth.ai/docs/basics/troubleshooting-and-faqs))
  is irrelevant here: we never enter `Trainer`'s evaluation loop, so `eval_dataset`
  stays `None` and `eval_strategy` stays `"no"`.
