# 01 The Unsloth + TRL stack on a free Colab T4, and the pre-tokenised dataset

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

The whole map rests on one assumption: that we can hand TRL's `SFTTrainer` a
dataset we tokenised and masked ourselves (`input_ids` + `labels`) and have
Unsloth leave it alone. Establish, with sources and dates:

1. **Install.** The pip line for Unsloth on a free Colab T4 today, and the
   versions it actually resolves (`unsloth`, `unsloth_zoo`, `trl`,
   `transformers`, `torch`, `peft`, `bitsandbytes`, `accelerate`,
   `datasets`). Whether a runtime restart is needed, install time, and any
   clash with Colab's preinstalled packages of the kind that bit milestone 2
   (`torchaudio` built against a different CUDA). Charting established that
   `unsloth 2026.9.9` pins `trl<=0.24.0`, `transformers<=5.5.0`,
   `torch<2.13` — confirm this is still what resolves.
2. **Pre-tokenised datasets under Unsloth.** Does Unsloth's `SFTTrainer`
   wrapper (it patches TRL, see `unsloth_zoo`) accept a dataset that already
   has `input_ids` and `labels`, or does it re-tokenise / require
   `dataset_text_field` / drop the `labels` column? Read the installed
   source, not the docs. **This is the decisive item**: if the answer is no,
   the fallback is TRL's prompt-completion form and ticket 09 changes shape.
3. **TRL 0.24 API names** the spec must state verbatim: `SFTConfig` field
   names for sequence length, batch size, accumulation, epochs, lr,
   scheduler, warmup ratio, eval/save cadence, `report_to`, seed — and which
   of them moved or were renamed between 0.18 and 0.24.
4. **`FastLanguageModel.from_pretrained`** for `Qwen3-0.6B` on a T4:
   arguments for 4-bit NF4, fp16 compute (no bf16 on sm_75), `max_seq_length`,
   and `get_peft_model` for the LoRA config in DESIGN §7 (r=16, α=16,
   dropout 0, all seven linear targets). Whether Unsloth insists on its own
   `unsloth/Qwen3-0.6B` mirror and, if so, whether that mirror's chat template
   differs from the official one (milestone 2 found the 4B mirror injects an
   empty `<think>` block — the same trap would corrupt our masking).
5. Gradient checkpointing and any T4-specific knob Unsloth recommends
   (`use_gradient_checkpointing="unsloth"`, `packing`, fp16 vs
   `fp16_full_eval`), plus what it does to memory at 0.6B / 2048 tokens.

Answer in this file with sources and dates. It does not decide anything:
it feeds tickets 07, 08, 09 and 11.

## Answer

Researched 2026-09-22. **Item 2 verdict: yes.** Unsloth *replaces* TRL's
`SFTTrainer._prepare_dataset` with `unsloth_zoo.dataset_utils.sft_prepare_dataset`,
whose very first branch is `if "labels" in column_names: … do_tokenize = False`
(`unsloth_zoo/dataset_utils.py:2582-2588`). A dataset carrying `input_ids` +
`labels` is passed through untouched: no re-tokenisation, no `dataset_text_field`,
no `formatting_func`, the `labels` column kept and used verbatim by the collator.
The one real consequence is the mirror image: because Unsloth's prep replaces
TRL's, **nothing truncates a pre-tokenised split on TRL 0.24** — `max_length` is
inert and we must cap lengths ourselves. Two further live findings: the plain
`pip install unsloth` resolution (trl 0.24.0 / transformers 5.5.0) differs from
Unsloth's own Colab notebooks (which force `trl==0.22.2`/`0.25.1` and
`transformers==4.56.2`/`4.57.3` with `--no-deps`); and Unsloth silently
**auto-enables `padding_free`** on SFT because it rewrites the config default to
`None`.

Sources below are the actual downloaded artefacts: `unsloth 2026.9.9`
(wheel, uploaded 2026-09-22 17:46 UTC), `unsloth_zoo 2026.9.7` (sdist,
2026-09-22 14:58 UTC), `trl 0.24.0`, `trl 0.18.2`, `transformers 5.5.0`. Paths
are package-relative.

### 1. Install line, resolved versions, restart, clashes

**Pins confirmed** (verbatim from `unsloth-2026.9.9.dist-info/METADATA`,
downloaded 2026-09-22):
`trl!=0.19.0,<=0.24.0,>=0.18.2` · `transformers!=4.52.0-3,!=4.53.0,!=4.54.0,
!=4.55.0,!=4.55.1,!=4.57.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0,<=5.5.0,>=4.51.3` ·
`torch<2.13.0,>=2.4.0` · `peft!=0.11.0,>=0.18.0` ·
`bitsandbytes!=0.46.0,!=0.48.0,>=0.45.5` · `accelerate>=0.34.1` ·
`datasets!=4.0.*,!=4.1.0,<4.4.0,>=3.4.1` · `unsloth_zoo>=2026.9.7` ·
`xformers>=0.0.27.post2` (linux x86_64) · `triton>=3.0.0` · `torchvision`,
`diffusers`, `tyro`, `hf_transfer`, `structlog`, `typer`, `pydantic`.
`unsloth_zoo 2026.9.7` allows `trl<=1.13.0` and `transformers<=5.17.0`, so the
**intersection is TRL 0.24.0 / transformers 5.5.0** — the charting fact holds,
including the `torch<2.13` bound.

**What actually resolves.** `pip install --dry-run --ignore-installed unsloth`
on Python 3.12/linux-x86_64, 2026-09-22 (101 packages):

| package | version | | package | version |
|---|---|---|---|---|
| unsloth | 2026.9.9 | | transformers | 5.5.0 |
| unsloth_zoo | 2026.9.7 | | torch | 2.12.1 |
| trl | 0.24.0 | | triton | 3.7.1 |
| peft | 0.21.0 | | xformers | 0.0.35 |
| accelerate | 1.15.0 | | torchvision | 0.27.1 |
| bitsandbytes | 0.50.2 | | torchao | 0.18.0 |
| datasets | 4.3.0 | | cut-cross-entropy | 25.1.1 |
| huggingface_hub | 1.32.0 | | numpy | 2.5.3 |

**Colab specifics.** Latest runtime listed on the
[runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html)
(fetched 2026-09-22) is **2026.07: Python 3.12.13, torch 2.11.0**. torch 2.11.0
satisfies both `torch<2.13.0,>=2.4.0` (unsloth) and `torch>=2.10` (xformers
0.0.35), so **pip should keep Colab's torch** — no 3 GB torch swap, and therefore
none of the milestone-2 `torchaudio`-vs-CUDA breakage, which was a *consequence*
of replacing torch. Total new download with torch kept ≈ **100 MB** (measured by
HEAD on the resolved wheel URLs: bitsandbytes 43 MB, unsloth 24 MB, transformers
10 MB, diffusers 5.9 MB, hf_transfer 3.6 MB, torchao 3.4 MB, xformers 3.3 MB,
rest <2 MB each). Install time therefore ≈ **1–3 min** (estimate) versus vLLM's
3–8 min. Colab's own notebooks contain **no runtime-restart cell** (checked
`nb/Qwen3_(4B)-Instruct.ipynb` and `nb/Qwen3_(0_6B)-Phone_Deployment.ipynb`,
2026-09-22), consistent with torch being untouched.

**Recommended line:** `!pip install unsloth` (plain), *not* the notebooks'
`--no-deps` recipe. Reason, verified on PyPI 2026-09-22: the official notebooks
map torch→xformers as `{'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}`
with default `0.0.34`; Colab's torch **2.11** is not in that map, so it installs
`xformers==0.0.34`, a 110 MB `cp39-abi3` compiled wheel that declares
`torch==2.10.0` — the classic ABI mismatch, hidden by `--no-deps`. The resolver
instead picks **xformers 0.0.35** (2026-02-20), a 3.3 MB `py39-none-manylinux`
wheel declaring only `torch>=2.10`, i.e. no ABI pin. The notebooks also force
`transformers==4.56.2` + `trl==0.22.2` (4B notebook, last commit 2026-08-18) or
`transformers==4.57.3` + `trl==0.25.1` (0.6B notebook) via `--no-deps`, both
outside unsloth's own declared range. Item 2's answer is unaffected either way
(see below), so prefer the resolver's own answer.

Clash watch-list for the notebook (unverified, check in-cell): Colab's
preinstalled `transformers` will be moved to 5.5.0 and `datasets` to <4.4.0;
`torchvision` may be bumped. Print `torch.__version__`, `transformers.__version__`
and `nvidia-smi` in the first cell as `baselines.ipynb` already does.

### 2. Pre-tokenised datasets under Unsloth — **YES, accepted verbatim**

Read from source, not docs. Chain, in order:

1. `import unsloth` → `unsloth/models/llama.py:4027` calls `PatchFastRL(...)` at
   module scope → `patch_trl_rl_trainers()` (`unsloth/models/rl.py:3458`) which
   regenerates each TRL trainer's source and rebinds `trl.SFTTrainer`,
   `trl.trainer.SFTTrainer`, `trl.trainer.sft_trainer.SFTTrainer` **and** the
   three `SFTConfig` spellings to the generated `UnslothSFTTrainer` /
   `UnslothSFTConfig` (`rl.py:3086-3117`). So `from trl import SFTTrainer,
   SFTConfig` after `import unsloth` yields the patched classes.
2. In that regeneration, `rl_replacements.sft_trainer_prepare_dataset`
   (`unsloth/models/rl_replacements.py:581-763`) takes the *source* of
   `unsloth_zoo.dataset_utils.sft_prepare_dataset`, renames it
   `_prepare_dataset` (`:708`) and substitutes it for TRL's.
3. `unsloth_zoo/dataset_utils.py:2573-2593` then decides on a **probed row**
   (`column_names = set(next(iter(dataset)).keys())`):
   ```python
   if "labels" in column_names:
       self.data_collator = DataCollatorForSeq2Seq(tokenizer)
       used_column_names.append("labels")
       do_tokenize = False
   elif "input_ids" in column_names:
       self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm = False)
       do_tokenize = False
   elif "prompt" in column_names and "completion" in column_names: …
   elif dataset_text_field not in column_names:
       do_formatting_func = True   # ← the "you must specify a formatting_func" path
   ```
   With `labels` present the function falls straight through to `if packing:`
   (`:2748`) and, with `packing=False`, `return dataset` — **unchanged, every
   column kept** (`used_column_names` is only consulted inside the packing
   branch's `select_columns`).

Corollaries, all read off source:

- **`dataset_text_field` is never required** for us; the `RuntimeError("Unsloth:
  You must specify a `formatting_func`")` is unreachable once `labels` is present.
- **The `labels` column is not dropped.** TRL 0.24's
  `_set_signature_columns_if_needed` already lists it
  (`trl/trainer/sft_trainer.py:1078`, confirms the charting fact), and Unsloth
  additionally rewrites that list for older TRLs (`rl.py:2999-3035`).
- **Which collator actually runs.** The `self.data_collator =
  DataCollatorForSeq2Seq(...)` assignment inside prep is *dead on TRL 0.24*:
  TRL builds its collator at `sft_trainer.py:748-764`, calls `_prepare_dataset`
  at `:805`, then `super().__init__(data_collator=data_collator)` at `:847`,
  and `Trainer.__init__` overwrites `self.data_collator`. Unsloth's own
  collator-swap block (`rl.py:2476-2497`) is spliced *before* `super().__init__`
  (template `rl.py:523` then `:532`) and both of its branches test
  `isinstance(data_collator, …)`, so with `data_collator=None` it is a no-op.
  Net: the live collator is TRL 0.24's `DataCollatorForLanguageModeling`, which
  does `if "labels" in examples[0]: labels = [torch.tensor(e["labels"]) …]`
  (`sft_trainer.py:160`) and pads with `-100`. Our `-100` prompt span survives
  end to end. **Do not pass a custom collator**: it would (a) trip TRL's
  `ValueError("Passing a custom data collator is not supported when using
  padding-free")` at `:709` and (b) set `blocked=True` in Unsloth's wrapper.
- **`completion_only_loss` resolves to `False`** for us — `args.completion_only_loss`
  defaults to `None` and TRL then sets it from the dataset shape
  (`"prompt" in dataset_sample and "completion" in dataset_sample`,
  `sft_trainer.py:737`). So the collator's completion-mask branch never fires and
  nothing overwrites our labels.
- **`max_length` is NOT enforced on a pre-tokenised split.** TRL's own
  `truncate_dataset(dataset, args.max_length, …)` (`sft_trainer.py:1060`) lives
  in the function Unsloth replaced, and Zoo's replacement does not truncate. The
  elaborate pre-tokenised truncation block Unsloth generates
  (`rl.py:2185-2460`) is gated on the string `"`max_length` is not enforced"`
  appearing in TRL's source (`rl.py:2184`), which is a **TRL ≥1.0 guard absent
  from 0.24**. Consequence for ticket 09: `tokenize_example` must enforce the cap
  itself (and assert it), because neither TRL nor Unsloth will on this stack.
- **Unsloth knows this path exists deliberately** — `rl.py:167-198` and
  `pretokenized_within_cap` (`rl.py:806`) are written entirely around "rows that
  already carry `input_ids`". This is a supported shape, not an accident.
- Version-robustness: because the accepting code is Zoo's, not TRL's, the answer
  is the same under `trl==0.22.2` / `0.25.1` as under `0.24.0`. Historic
  counter-evidence ([unsloth#1843](https://github.com/unslothai/unsloth/issues/1843),
  Feb 2025, "UnslothTrainer applies ChatML template although the dataset is
  pre-tokenized") predates this branch and concerns `UnslothTrainer`, not
  `SFTTrainer`.

**Surprise to plan for: Unsloth auto-enables padding-free.** `rl.py:2620`
rewrites the generated `UnslothSFTConfig` default to `"padding_free": None,
# None = user didn't set it, allows auto-enable detection` (TRL 0.24's own
default is `False`, `trl/trainer/sft_config.py:204`). `unsloth/trainer.py:1284-1290`
then calls `configure_padding_free(config)` — which sets `padding_free=True` and
`remove_unused_columns=False` (`unsloth/utils/packing.py:170-174`) — unless
`blocked`, and `blocked` is False for a plain PEFT-wrapped Qwen3 with no custom
collator. After `__init__` it wraps the collator to add `packed_seq_lengths`,
inferring lengths from `len(input_ids)` when the split has no `seq_lengths`
(`packing.py:235-277`). This is *safe on a T4 despite no FA2*: Unsloth builds
block-diagonal SDPA / xFormers masks itself (`packing.py:27-118`,
`build_sdpa_packed_attention_mask`, `build_xformers_block_causal_mask`) and
filters TRL's "attention implementation is not …" warning (`packing.py:122-128`).
Masking still survives: the collator concatenates the `labels` lists verbatim
(`sft_trainer.py:183-185`) and only adds `labels[position_ids == 0] = -100`, i.e.
each sequence's first token, which is already a prompt token for us.
**Recommendation for the spec: set `padding_free = False` explicitly** for the
smoke run, so the smoke run tests the ordinary padded path and the loss is
comparable to a plain-PEFT rerun; flip it on later as a speed experiment.

### 3. TRL 0.24 API names the spec must state verbatim

`SFTConfig` in TRL 0.24 subclasses `transformers.TrainingArguments`, so most
names come from transformers 5.5.0 (`transformers/training_args.py`).

| purpose | name on TRL 0.24 / transformers 5.5.0 | default | note |
|---|---|---|---|
| sequence length | **`max_length`** | `1024` | `sft_config.py:182` |
| batch size | `per_device_train_batch_size` | 8 (Unsloth: 4) | `training_args.py:768` |
| eval batch size | `per_device_eval_batch_size` | 8 (Unsloth: 4) | |
| accumulation | `gradient_accumulation_steps` | 1 (Unsloth: 2) | `:840` |
| epochs | `num_train_epochs` (float) | 3.0 | `:769` |
| step cap | `max_steps` | -1 | `:770` |
| lr | `learning_rate` | 2e-5 on SFTConfig (Unsloth: 5e-5) | `sft_config.py` |
| scheduler | `lr_scheduler_type` (+`lr_scheduler_kwargs`) | `"linear"` | `:779` |
| warm-up | **`warmup_steps`** (int = steps, float in [0,1) = ratio) | 0 | `:789` |
| warm-up (old) | `warmup_ratio` | `None`, **deprecated** | `:1428`, help says "removed in v5.2" yet still present in 5.5.0 |
| eval cadence | `eval_strategy` ("no"/"steps"/"epoch") + `eval_steps` | `"no"` / `None` | `:1059`, `:1063` |
| save cadence | `save_strategy` + `save_steps` + `save_total_limit` | `"steps"` / 500 | `:1130`, `:1136`, `:1154` |
| logging | `logging_strategy`, `logging_steps` | SFTConfig overrides `logging_steps=10`; Unsloth sets 1 | |
| tracking | `report_to` | **`"none"`** in transformers 5.x | `:1028` — W&B needs `report_to="wandb"` explicitly |
| run name | `run_name` | `None` | used by wandb |
| seed | `seed` | 42 (Unsloth: 3407) | `:1246`; also `data_seed` |
| precision | `fp16`, `bf16`, `fp16_full_eval`, `bf16_full_eval` | see §5 | `:865-883` |
| other | `optim` (Unsloth: `"adamw_8bit"`), `weight_decay` (Unsloth: 0.001), `max_grad_norm`, `output_dir`, `remove_unused_columns`, `push_to_hub`, `hub_model_id`, `hub_strategy`, `hub_private_repo`, `load_best_model_at_end`, `gradient_checkpointing(_kwargs)` | | |

**What moved between TRL 0.18.2 and 0.24.0** (diffed field-by-field on the two
sdists):
- **Removed: `max_seq_length`.** In 0.18.2 it still existed as a deprecated
  alias whose `__post_init__` copied it into `max_length`
  (`trl-0.18.2/trl/trainer/sft_config.py:183-198`, "removed in version 0.20.0").
  In 0.24.0 it is **gone**. Unsloth re-adds a `max_seq_length` field to its
  generated `UnslothSFTConfig` (`rl.py:2676-2682`) as a convenience, so both
  spellings work *under Unsloth* — but the spec should say `max_length`.
- **Added in 0.24 vs 0.18:** `packing_strategy`, `assistant_only_loss`,
  `loss_type`, `chat_template_path`, plus SFT-specific re-declarations of
  `logging_steps` (10), `gradient_checkpointing` (**True**) and `bf16` (`None`).
- **`bf16` trap:** TRL 0.24's `SFTConfig.__post_init__` does
  `self.bf16 = not self.fp16 if self.bf16 is None else self.bf16`
  (`sft_config.py:261`) — i.e. a bare `SFTConfig()` turns **bf16 on**, which is
  fatal on sm_75. Unsloth's generated config forces `bf16=False, fp16=False`
  (`rl.py:2613-2614`) and then auto-selects; still, **state `fp16=True,
  bf16=False` explicitly** in the YAML→config mapping.
- Transformers-side rename to state: **`eval_strategy`**, not
  `evaluation_strategy` (zero occurrences of the latter in 5.5.0), and
  `processing_class=`, not `tokenizer=`, on the trainer.
- Unsloth's other `UnslothSFTConfig` default rewrites (`rl.py:2595-2626`):
  `torch_empty_cache_steps=250`, `eval_accumulation_steps=2`,
  `dataloader_pin_memory=True`, `auto_find_batch_size=False`,
  `include_num_input_tokens_seen=False`, `loss_type="nll"`, and
  `warmup_steps=0.1` when transformers ≥5.0 else `warmup_ratio=0.1`.
- **`max_length` gotcha:** Unsloth overwrites `args.max_length` at trainer
  construction (`rl.py:2160-2180`) from `model.max_seq_length`, treating the
  caller's value as "explicit" **only if it differs from TRL's dataclass default**
  (`rl.py:2121-2136`). So `SFTConfig(max_length=1024)` is indistinguishable from
  "unset" and gets widened to the model's `max_seq_length`. Pick 2048 (or
  whatever `from_pretrained(max_seq_length=…)` got) and keep the two equal.

### 4. `FastLanguageModel.from_pretrained` / `get_peft_model` for Qwen3-0.6B on a T4

Signature (`unsloth/models/loader.py:444-478`), relevant defaults:
`max_seq_length=2048`, `dtype=None`, `load_in_4bit=True`, `full_finetuning=False`,
`use_gradient_checkpointing="unsloth"`, `use_exact_model_name=False`,
`random_state=3407`, `fast_inference=False`.

- **NF4 is what `load_in_4bit=True` builds** — no extra argument needed:
  `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)`
  (`unsloth/models/llama.py:2680-2686`). Double-quant is on.
- **fp16 on sm_75 is automatic.** `SUPPORTS_BFLOAT16` is set from
  `torch.cuda.get_device_capability()[0] >= 8` (`unsloth/models/_utils.py:2440-2444`),
  so False on a T4. `dtype=None` → `torch.float16`
  (`llama.py:2440-2441`); passing `torch.bfloat16` is *downgraded* with
  `"Device does not support bfloat16. Will change to float16."` (`llama.py:2442-2444`).
  `is_bfloat16_supported()` (`_utils.py:3275`) is the public helper for the YAML's
  `dtype` knob.
- **Suggested call** (everything else default):
  ```python
  model, tokenizer = FastLanguageModel.from_pretrained(
      model_name      = "unsloth/Qwen3-0.6B",
      max_seq_length  = 2048,
      dtype           = None,            # → float16 on a T4
      load_in_4bit    = True,            # → NF4 + double quant
      use_gradient_checkpointing = "unsloth",
      random_state    = 3407,
  )
  model = FastLanguageModel.get_peft_model(
      model, r = 16, lora_alpha = 16, lora_dropout = 0.0, bias = "none",
      target_modules = ["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
      use_gradient_checkpointing = "unsloth", random_state = 3407,
  )
  ```
  DESIGN §7's LoRA config **is** `get_peft_model`'s default set, verbatim:
  `r=16, target_modules=[the same seven], lora_alpha=16, lora_dropout=0.0,
  bias="none", use_gradient_checkpointing="unsloth", random_state=3407`
  (`unsloth/models/llama.py:3117-3146`). Stating them explicitly is still worth it.
- **Yes, Unsloth redirects to its own mirror.** Unless `use_exact_model_name=True`,
  `get_model_name()` (`unsloth/models/loader_utils.py:992-1052`) resolves through
  `FLOAT_TO_INT_MAPPER`; `mapper.py:813-824` maps both `Qwen/Qwen3-0.6B` and
  `unsloth/Qwen3-0.6B` to **`unsloth/Qwen3-0.6B-unsloth-bnb-4bit`** under
  `load_in_4bit=True`. A name it does not know but a *newer* unsloth does raises
  `NotImplementedError("… not supported in your current Unsloth version")`
  (`loader_utils.py:1043`).
- **The mirror's chat template does NOT differ in effect.** Fetched
  `tokenizer_config.json` for `Qwen/Qwen3-0.6B`, `unsloth/Qwen3-0.6B`,
  `unsloth/Qwen3-0.6B-bnb-4bit` and `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`
  (2026-09-22). All three Unsloth repos share one template (sha256 `5da44855…`,
  4905 chars) that differs textually from Qwen's (`a55ee1b1…`, 4168 chars), but
  the diff is purely defensive Jinja: `message.content is string` →
  `is defined and is not none`, the reverse-iteration loop rewritten as a forward
  loop with an index, `startswith/endswith` replaced by slice comparisons, and
  `if reasoning_content` → `if not reasoning_content.strip() == ''`. Rendered with
  Jinja2 on `[system, user, assistant]` with plain string contents, the two
  templates produce **byte-identical output** in both thinking modes, and the
  `messages[:2] + add_generation_prompt=True` rendering is an exact **string**
  prefix of the full rendering in both:
  - `enable_thinking=False`: prompt ends `…<|im_start|>assistant\n<think>\n\n</think>\n\n`,
    completion `'[]<|im_end|>\n'` — i.e. **exactly the 3 tokens** the charting note
    records for an empty document. Confirms that fact and pins down that the
    masking must pass `enable_thinking=False` on **both** renderings.
  - `enable_thinking=True` (or unset): prompt ends `…<|im_start|>assistant\n`,
    completion `'<think>\n\n</think>\n\n[]<|im_end|>\n'`. Still a prefix, but the
    empty `<think>` block lands **inside the trained span** — the milestone-2
    trap, and here it comes from Qwen's *official* template too, not from the
    mirror. The mirror is exonerated; the thinking flag is the real hazard.
  - The one genuine tokenizer difference: **`pad_token` is `<|vision_pad|>` on the
    Unsloth mirrors vs `<|endoftext|>` on Qwen's repo** (`eos_token` is
    `<|im_end|>` in both). Harmless for loss (TRL pads `labels` with `-100`
    independently, `sft_trainer.py:203-205`) but it changes the pad id our
    pre-tokenisation should *not* emit, and it is what `SFTConfig.pad_token`
    would otherwise have to override.

### 5. Gradient checkpointing and T4 knobs

- **`use_gradient_checkpointing="unsloth"`** selects
  `Unsloth_Offloaded_Gradient_Checkpointer`
  (`unsloth_zoo/gradient_checkpointing.py:330-336`): each layer's input hidden
  states are copied to pinned CPU RAM with `non_blocking=True` in the forward and
  streamed back in the backward, so activations cost host RAM rather than VRAM.
  The docstring's own claim is "Saves VRAM by smartly offloading to RAM. Tiny hit
  to performance". It is the default in both `from_pretrained` and
  `get_peft_model`, so the spec only needs to name it, not enable it. It is also
  re-applied at trainer construction: `model.for_training(use_gradient_checkpointing=…)`
  (`rl.py:2465-2467`).
- **Precision.** Leave `fp16`/`bf16` unset and Unsloth resolves them per device
  (`rl.py:2024-2030`: `use_bf16_amp = (not float16) and _bf16_supported()`,
  then `args.fp16 = not use_bf16_amp`, plus `ACCELERATE_MIXED_PRECISION`). On a T4
  that is `fp16=True, bf16=False`. `fp16_full_eval` / `bf16_full_eval` are
  mirrored from them when the user set neither (`rl.py:2074-2088`), so
  `fp16_full_eval=True` comes for free — relevant because our eval callback runs
  generation, not `compute_loss`. Setting `bf16=True` on a fp16 model raises
  `TypeError("Unsloth: Model is in float16 precision but you want to use
  bfloat16 …")` (`rl.py:2008`).
- **`packing`.** Default stays `False` (Unsloth does not rewrite it) and
  `_should_pack` (`unsloth/trainer.py:108-112`) only fires if the user asks.
  Leave it off: with 100 documents, packing would concatenate distinct documents
  into one sequence and muddy the completion-only loss for no throughput win.
- **`padding_free`** — see §2. Auto-enabled unless set; **set it `False`**.
- **Memory at 0.6B / 2048 tokens (estimates, not measured).** Qwen3-0.6B ≈ 0.75 B
  params → ~0.5 GB as NF4 weights + ~0.15 GB for the fp16 embedding/lm_head that
  bnb skips; LoRA r=16 over seven projections ≈ 10 M trainable params → ~20 MB fp16
  + ~80 MB of 8-bit Adam state (`optim="adamw_8bit"`); with
  `use_gradient_checkpointing="unsloth"` activations are per-layer, not per-model,
  so at `per_device_train_batch_size=2`–`4` × 2048 tokens the peak should sit in
  the **2–4 GB** band on a 15 GB T4 — i.e. batch size is not the binding
  constraint at this size, and the smoke run's real cost is the eval callback's
  generation. Treat these as order-of-magnitude figures to be replaced by the
  first `torch.cuda.max_memory_allocated()` reading.

### Open caveats

- **Nothing here was run on a GPU.** Every claim in §2 is from source reading of
  `unsloth 2026.9.9` + `unsloth_zoo 2026.9.7` + `trl 0.24.0`; the first Colab run
  is still the acceptance test. In particular the *auto padding-free* path and
  the block-diagonal SDPA mask on sm_75 are unexercised here.
- **The resolution in §1 was computed in a clean Python 3.12 env, not on Colab.**
  That torch 2.11.0 survives `pip install unsloth` is an inference from the
  version constraints (`torch<2.13`, `xformers>=0.0.27.post2` → 0.0.35's
  `torch>=2.10`), not an observation. If pip does swap torch, the milestone-2
  `torchaudio` CUDA clash returns and a restart is needed.
- **Colab's live default runtime may be newer than 2026.07**; the FAQ page does not
  label which version is current. Milestone 2's own first-hand note reports
  Python 3.13.15 in a live runtime, which does not match the 2026.07 row —
  verify `sys.version` and `torch.__version__` in cell 1 before trusting §1.
- **Install time (1–3 min)** and **all §5 memory figures** are estimates.
- **`xformers 0.0.35` being a non-ABI-pinned wheel is inferred from its filename
  and `Requires-Dist`**, not from importing it against torch 2.11.
- **transformers version skew:** the local masking work was verified on
  transformers 5.17; the Colab training env will be pinned to **5.5.0**. Chat
  template rendering is Jinja and should be identical, but re-assert the
  token-level prefix inside the notebook rather than trusting the local check.
- The chat-template comparison used **Jinja2 directly**, not
  `tokenizer.apply_chat_template`; transformers adds its own defaults (e.g.
  `enable_thinking` handling in 5.x) that could in principle differ.
- `unsloth 2026.9.9` was published **2026-09-22 17:46 UTC**, i.e. today; the
  behaviour above may be days old. `unsloth_zoo` releases roughly daily.
- **Not investigated here:** `resume_from_checkpoint` and checkpoint contents
  under Unsloth (ticket 02), whether `push_to_hub` on every save works from a
  Colab session, and TRL's `pack_dataset` behaviour on a `labels`-carrying split
  (only relevant if packing is ever turned on).
