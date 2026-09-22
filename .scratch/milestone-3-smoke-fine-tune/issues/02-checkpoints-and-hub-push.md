# 02 Adapter checkpointing and the Hub push during training

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

Decided: checkpoint at every eval point and push every save, so a Colab
disconnect never loses the adapter (map Notes, Q6). Establish how that is
actually expressed under TRL 0.24 + Unsloth + `transformers`:

1. The `TrainingArguments`/`SFTConfig` fields: `output_dir`, `save_strategy`,
   `save_steps`, `save_total_limit`, `push_to_hub`, `hub_model_id`,
   `hub_strategy` (does `"every_save"` still exist, and what does it push —
   the checkpoint directory, or only the final model?), `hub_private_repo`,
   `hub_token`. Whether the push is asynchronous and what happens when the
   runtime dies mid-push.
2. What a **LoRA checkpoint actually contains** when the model is a 4-bit
   Unsloth model: which files, how large at 0.6B/r=16, and whether the base
   model's weights leak into it. Confirm the pushed artifact is an adapter,
   never a merged model (DESIGN §9).
3. Whether `Trainer.push_to_hub` / `model.push_to_hub` / Unsloth's
   `save_pretrained` disagree about what lands in the repo, and which one
   produces a repo that `PeftModel.from_pretrained` (and, later, vLLM's LoRA
   loading) can read back.
4. The repo card: does the push write one automatically, and can we supply
   ours (milestone 1 wrote a dataset card by hand in `prepare.py`)?
5. `resume_from_checkpoint` after a disconnect: what must be present locally
   or pullable from the Hub for it to work, and whether the optimizer state
   is part of the pushed checkpoint at all.

Answer in this file with sources and dates. Feeds tickets 08 and 09; the
resume question also feeds the "disconnect recovery" fog on the map.

## Answer

Researched 2026-09-22 against the pinned stack, read from sdists/wheels
downloaded that day: `transformers 5.5.0` (PyPI 2026-04-02, the ceiling of
unsloth's `transformers<=5.5.0`), `trl 0.24.0` (2025-10-16), `peft 0.21.0`
(2026-09-15), `unsloth 2026.9.9` + `unsloth_zoo 2026.9.7` (both 2026-09-22),
`huggingface_hub 1.32.0` (2026-09-17). Dates from the PyPI JSON API.
`unsloth 2026.9.9`'s wheel METADATA re-confirms the map's pins:
`trl!=0.19.0,<=0.24.0,>=0.18.2`, `transformers<=5.5.0`, `peft!=0.11.0,>=0.18.0`.

**Headline.** Everything works through stock `transformers.Trainer`: Unsloth
rewrites only the banner and the TPU branch of `_inner_training_loop`, and TRL
0.24 does not touch `_save_checkpoint` except to regenerate the model card
first. `hub_strategy="every_save"` still exists and is still the default — but
it pushes only the **root mirror** of `output_dir` (adapter + tokenizer +
`training_args.bin` + `README.md`), never the checkpoint directory, so it
carries **no optimizer state and nothing to resume from**. The decision
"checkpoint at every eval point and push every save, so a disconnect never
loses the adapter" is spelled `hub_strategy="checkpoint"`: that is `every_save`
plus a second upload of the whole checkpoint folder into `last-checkpoint/`.
Recommendation: `save_strategy="steps"`, `save_steps == eval_steps`,
`save_total_limit=2`, `push_to_hub=True`, `hub_strategy="checkpoint"`,
`hub_always_push=True`, and an explicit `trainer.push_to_hub()` after `train()`.

### 1. The `TrainingArguments` / `SFTConfig` fields

`SFTConfig(TrainingArguments)` (`trl/trainer/sft_config.py:22`) adds **no**
overrides for any of these fields — they are plain `transformers 5.5.0`
`TrainingArguments`. Field definitions, `src/transformers/training_args.py`:

| field | line | default | note |
|---|---|---|---|
| `output_dir` | 762 | `None` → `"trainer_output"` (post-init, `:1449`) | also the name the repo defaults to |
| `save_strategy` | 1130 | `"steps"` | `no` / `steps` / `epoch` / `best` (`trainer_utils.py:394`) |
| `save_steps` | 1136 | `500` | int, or a float in `[0,1)` read as a ratio of total steps |
| `save_total_limit` | 1154 | `None` | `None` or `<=0` ⇒ never rotate (`trainer_utils.py:366`) |
| `save_only_model` | 1124 | `False` | `True` drops optimizer/scheduler/RNG — "Prevents resuming training from checkpoint" |
| `push_to_hub` | 1168 | `False` | |
| `hub_model_id` | 1186 | `None` → `user/<basename(output_dir)>` | `"org/name"` accepted |
| `hub_strategy` | 1189 | `"every_save"` | **yes, it still exists** |
| `hub_private_repo` | 1177 | `None` | `None` = Hub default (public unless the org forces private); **ignored if the repo already exists** |
| `hub_token` | 1171 | `None` | falls back to `huggingface_hub.get_token()` |
| `hub_always_push` | 1195 | `False` | skip a push while the previous one is unfinished |
| `hub_revision` | 1199 | `None` | branch/tag/sha to commit onto (5.x only) |

**What each strategy pushes** — `Trainer._push_from_checkpoint`
(`src/transformers/trainer.py:4074`), the only place `hub_strategy` is read:

- `"end"` → returns immediately (`:4076`). Nothing is pushed during training,
  **and nothing is pushed at the end either**: `args.push_to_hub` appears in
  `trainer.py` only at `:567` (`init_hf_repo`), `:3088` (this call), `:3800`
  (`save_model(_internal_call=False)`) and `:4029`. `_finalize_training` only
  calls `_finish_current_push()` (`:1857`). You must call
  `trainer.push_to_hub()` yourself.
- `"every_save"` → **copies** `adapter_config.json`, `adapter_model.bin`,
  `adapter_model.safetensors` (added to the list because `is_peft_available()`,
  `:4095`) out of `checkpoint-N/` into `output_dir/`, re-saves the tokenizer and
  `training_args.bin` there, then `upload_folder(folder_path=output_dir,
  ignore_patterns=["_*", "checkpoint-*"])` (`:4114`). The checkpoint directory
  itself is explicitly excluded. Commit message
  `"Training in progress, step N"`.
- `"checkpoint"` → the above **plus** a second `upload_folder` of the whole
  `checkpoint-N/` folder to `path_in_repo="last-checkpoint"` (`:4123`).
- `"all_checkpoints"` → same second upload but to `path_in_repo="checkpoint-N"`,
  one subfolder per save, never deleted.

So: `every_save` pushes the *model*, not the *checkpoint*. This is the
long-standing documentation complaint ([transformers#27728](https://github.com/huggingface/transformers/issues/27728),
[#30141](https://github.com/huggingface/transformers/issues/30141)); the
5.5.0 docstring (`training_args.py:497-501`) now states it correctly.

**Asynchrony.** Both uploads pass `run_as_future=True`. `huggingface_hub`
dispatches those onto `HfApi.run_as_future`, a **single** shared
`ThreadPoolExecutor(max_workers=1)` (`hf_api.py:2309`) — "Background jobs are
queued to preserve order but are not ran in parallel" (`:2278`). The futures
are wrapped in `PushInProgress` (`transformers/utils/hub.py:929`).
`_push_from_checkpoint` **skips the whole push** if the previous one is not
done and `hub_always_push` is False (`trainer.py:4079`) — with a 40 MB adapter
on Colab that will rarely bite, but it is a silent skip, so set
`hub_always_push=True` if "push every save" is meant literally. It does not
block training: it only appends to the same single-worker queue
(`:4137-4140`).

**Dying mid-push.** The upload is two-phase: `preupload_lfs_files` for the LFS
blobs (`hf_api.py:5271`), then one `_send_commit` request (`:5346`). A Colab
VM kill between them leaves the repo at its previous commit with an orphaned,
unreferenced blob — never a half-written revision. If instead the *cell* is
interrupted and the interpreter exits normally, `concurrent.futures.thread`
registers `_python_exit` via `threading._register_atexit`, so Python joins the
worker and the queued push completes before exit ([CPython
`concurrent/futures/thread.py`](https://github.com/python/cpython/blob/3.12/Lib/concurrent/futures/thread.py)).
`train()` itself ends with `_finish_current_push()` (`trainer.py:1857`), which
waits and logs "Waiting for the current checkpoint push to be finished".

**Token on Colab.** Leave `hub_token=None`: `init_hf_repo` (`:3900`) calls
`create_repo(token=None)`, and `huggingface_hub.get_token()` tries `HF_TOKEN`,
then the token file, then `_get_token_from_google_colab()` — the Colab secrets
vault (`utils/_auth.py:83,87`). `init_hf_repo` runs in `Trainer.__init__`
(`:566-568`), so a bad token fails before the first step, not at step N.

### 2. What a LoRA checkpoint actually contains

`_save_checkpoint` (`trainer.py:3031`) writes `output_dir/checkpoint-N/` with:

- `save_model()` → `_save()` → `self.model.save_pretrained(dir)` (`:3828`);
  `self.model` is a `PeftModel`, which is in `supported_classes` when peft is
  installed (`:3811`). So: **`adapter_config.json` + `adapter_model.safetensors`
  only**.
- `self.processing_class.save_pretrained(dir)` (`:3830`) → `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`, `added_tokens.json`,
  `special_tokens_map.json`, `chat_template.jinja`.
- `torch.save(self.args, "training_args.bin")` (`:3840`).
- unless `save_only_model`: `optimizer.pt`, `scheduler.pt`, `scaler.pt`,
  `rng_state.pth` (`:3065-3070`; names at `trainer.py:240-245`).
- `trainer_state.json` (`:3083`).
- TRL additionally regenerates `output_dir/README.md` **before** the parent
  save (`trl/trainer/sft_trainer.py:1200-1207`), so the card is in the
  checkpoint's parent, and peft's own card ends up inside the checkpoint.

**No base weights.** `get_peft_model_state_dict` "only includes the PEFT
parameters, not the parameters of the base model"
(`peft/utils/save_and_load.py:153`). Embeddings are added only if
`save_embedding_layers` resolves True, which happens when an embedding layer is
in `target_modules` (`:282`) or when the live `config.vocab_size` differs from
the base repo's (`:306-316`). Unsloth's `get_peft_model` defaults to
`["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
(`unsloth/models/llama.py:3120-3128`) and its pad-token fix reuses an existing
reserved token (`<|vision_pad|>`, `unsloth_zoo/pad_token.py:64-70`) rather than
resizing, so neither trigger fires. The NF4 base is never written: Unsloth's
`patch_saving_functions` wraps `model.save_pretrained` **only** to rewrite
`safe_serialization=None` → `True` and forwards everything else untouched
(`unsloth/save.py:6925-6957`); merging lives behind the separate
`save_pretrained_merged` / `push_to_hub_merged` names (`:6996-7000`).
DESIGN §9 is satisfied by default; you have to ask for a merge to get one.

**Size at 0.6B / r=16.** `Qwen3-0.6B` `config.json` (cached locally at
`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de.../config.json`):
28 layers, `hidden_size` 1024, 16 heads × `head_dim` 128 (q out 2048), 8 KV
heads (k/v out 1024), `intermediate_size` 3072. LoRA params
= `r × (fan_in + fan_out)` summed over the 7 projections
= 16 × (3072 + 2048 + 2048 + 3072 + 4096 + 4096 + 4096) = 360,448 per layer
× 28 = **10,092,544 params**. peft upcasts adapter weights to fp32
(`autocast_adapter_dtype=True` by default, `peft/mapping_func.py:110`;
`_cast_adapter_dtype` "only upcasts float16 and bfloat16 to float32",
`tuners_utils.py:628-632`) ⇒ 40,370,176 B + header.

Verified against real published adapters (HF API, 2026-09-22):

- [`kxm1k4m1/Qwen3-0.6B-unsloth-bnb-4bit-Phishing`](https://huggingface.co/kxm1k4m1/Qwen3-0.6B-unsloth-bnb-4bit-Phishing)
  (Unsloth, r=16, same 7 targets, `modules_to_save: null`):
  `adapter_model.safetensors` = **40,422,168 B = 38.55 MiB**. No base weights,
  no embeddings.
- [`bendavidsteel/Qwen3-0.6B-claim-extraction-ezstance`](https://huggingface.co/bendavidsteel/Qwen3-0.6B-claim-extraction-ezstance)
  r=8 → 20,236,472 B: exactly half. Arithmetic confirmed.
- [`CarlOwOs/distilled-qwen3-0.6b-qlora-mmlu`](https://huggingface.co/CarlOwOs/distilled-qwen3-0.6b-qlora-mmlu)
  is an `all_checkpoints` repo and shows a whole checkpoint:
  `optimizer.pt` 81,069,610 B (= 2 × adapter, AdamW `exp_avg` + `exp_avg_sq` in
  fp32), `rng_state.pth` 14,244, `scheduler.pt` 1,064, `scaler.pt` 988,
  `trainer_state.json` ~1.6 KB, `training_args.bin` 5,304, plus the 15.1 MiB
  tokenizer set.

⇒ **one local checkpoint ≈ 131 MiB** (38.5 adapter + 77.3 optimizer + 15.1
tokenizer + ~0.03 misc). **Per push:** `every_save` moves only the adapter
after the first commit — `create_commit` drops no-op operations whose remote
oid equals the local one (`hf_api.py:5296-5305`), so the unchanged tokenizer is
not re-uploaded; `"checkpoint"` adds the full ~131 MiB of `last-checkpoint/`,
of which ~116 MiB (adapter + optimizer) changes every time.
**Cheap leak assertion for the notebook:** `adapter_model.safetensors` must be
≈ 40.4 MB. If it is ~640 MB, `embed_tokens`/`lm_head` got saved (Qwen3-0.6B's
tied embedding is 151936 × 1024 = 155.6 M params).

### 3. `Trainer.push_to_hub` vs `model.push_to_hub` vs Unsloth `save_pretrained`

They disagree about the *repo contents*, not about the *adapter*.

- **`Trainer.push_to_hub(commit_message=...)`** (`trainer.py:3992`) —
  `save_model(_internal_call=True)` into `output_dir`, then
  `create_model_card(...)`, then `_finish_current_push()`, then
  `upload_folder(output_dir, ignore_patterns=["_*","checkpoint-*"])`, blocking
  by default. Repo root gets adapter + tokenizer + `training_args.bin` +
  `README.md`. **This is the one to use**, and it must be called explicitly
  after `train()` for the final state.
- **`model.push_to_hub(repo_id)`** — Unsloth replaces it with
  `unsloth_push_to_hub` (`unsloth/save.py:6825-6893`), a generated wrapper over
  peft's own `push_to_hub`: it appends the `unsloth` tag, suffixes the commit
  message with "(Trained with Unsloth)", and calls `upload_to_huggingface(...)`
  (`:2582`), which does `create_repo(exist_ok=False)` and pushes Unsloth's
  `MODEL_CARD` **only when it actually created the repo** — i.e. it is a no-op
  once `init_hf_repo` has run. It uploads the adapter only: **no tokenizer**,
  so you would also need `tokenizer.push_to_hub(...)`.
- **`model.save_pretrained(dir)`** under Unsloth — plain
  `PeftModel.save_pretrained` (see §2); adapter only, not merged.
- **`model.save_pretrained_merged(...)` / `push_to_hub_merged(...)`** — the
  merged 16-bit export. Explicitly out of scope (DESIGN §9); nothing calls it
  implicitly.

**Read-back.** All three produce `adapter_config.json` +
`adapter_model.safetensors` at the repo root, which is what
`PeftModel.from_pretrained(base, repo_id)` and vLLM's PEFT LoRA loader both
want ([vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora/); vLLM's
default `max_lora_rank` is 16, so r=16 needs no flag). Only the
Trainer path also gives a tokenizer in the same repo, which is what `eval.py`
will want in milestone 4.

**One real gotcha for read-back.** `FastLanguageModel.from_pretrained(
"Qwen/Qwen3-0.6B", load_in_4bit=True)` rewrites the name to
`unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (`unsloth/models/loader.py:618-640`,
mapping table `unsloth/models/mapper.py:813-822`), and the `-bnb-4bit`-stripping
fixup in `get_peft_model` is dead code (`unsloth/models/llama.py:3746-3752`,
guarded by `if False:`). So `adapter_config.json`'s `base_model_name_or_path`
will read `unsloth/Qwen3-0.6B-unsloth-bnb-4bit`. Consequences:
`PeftModel.from_pretrained(base, repo)` ignores it (the base you pass wins),
but `FastLanguageModel.from_pretrained("fnl-es/<repo>")` **follows** it
(`loader.py:826`) and will pull the 4-bit base. Set it deliberately
(`model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen3-0.6B"`
before the first save) if the adapter is meant to be advertised against the
fp16 base for milestone 4's vLLM run.

### 4. The repo card

**Yes, a card is written automatically, and it is rewritten at every save.**
TRL 0.24's `SFTTrainer._save_checkpoint` calls `self.create_model_card(
model_name=...)` *before* delegating to the parent
(`trl/trainer/sft_trainer.py:1200-1207`), and `BaseTrainer.create_model_card`
(`trl/trainer/base_trainer.py:33-86`) unconditionally does
`model_card.save(os.path.join(self.args.output_dir, "README.md"))`. It
overrides `transformers.Trainer.create_model_card` entirely (`SFTTrainer(BaseTrainer)`,
`BaseTrainer(Trainer)`), so the transformers branch that would have called
peft's `create_or_update_model_card` when the existing card says
`library_name: peft` (`trainer.py:3985-3986`) never runs.

The generated card (`trl/templates/lm_model_card.md`,
`trl/trainer/utils.py:1329-1336`) carries
`library_name: transformers`, `base_model: <model.config._name_or_path>`,
`tags: [generated_from_trainer, trl, sft, unsloth]` (the `unsloth` tag is added
because the config has `unsloth_version`, `base_trainer.py:65-66`), a W&B run
link, framework versions, and a `pipeline("text-generation", model=<repo>)`
snippet that does not work for an adapter-only repo.

**Two things to fix, both of them ours to do.** First, `library_name` should be
`peft` (what `PeftModel.create_or_update_model_card` sets,
`peft/peft_model.py:1715`) — with `transformers` the Hub will not render the
repo as an adapter. Second, the hand-written card must survive: a `README.md`
dropped into `output_dir` is clobbered at the next save. So mirror what
milestone 1's `prepare.py` did for the dataset card, but from **inside the
trainer**: subclass `SFTTrainer` and override `create_model_card()` to write
our own `README.md` into `args.output_dir` (signature
`(model_name=None, dataset_name=None, tags=None)` — `Trainer.push_to_hub`
forwards `model_name` and possibly `tags`, so accept both and ignore the rest).
That single override covers both the per-save pushes and the final
`trainer.push_to_hub()`. The lazier alternative — `HfApi().upload_file` of a
card after training — works but leaves every intermediate commit carrying the
TRL card.

### 5. `resume_from_checkpoint`

**It is a local path. There is no Hub download anywhere in the resume path.**
`train(resume_from_checkpoint=True)` resolves to
`get_last_checkpoint(args.output_dir)` (`trainer.py:1402-1405`) and raises if
nothing is there; a string is used as a directory verbatim.
`_load_from_checkpoint` (`:3279`) only ever calls `os.path.isfile` /
`os.listdir` on it, and for a peft model ends at
`model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)`
(`:3388`). `_load_optimizer_and_scheduler` / `_load_scaler` /
`_load_rng_state` read `optimizer.pt`, `scheduler.pt`, `scaler.pt`,
`rng_state.pth` from the same folder (`:1643-1644`, `:1683`), and
`_init_training_state` reads `trainer_state.json` to recover `global_step`
(`:1540-1543`).

**Minimum to resume**, in one directory: `adapter_config.json` +
`adapter_model.safetensors` (or the loop raises "Can't find a valid checkpoint
at ..."), `trainer_state.json`, `optimizer.pt`, `scheduler.pt`, `scaler.pt`,
`rng_state.pth`.

**Is the optimizer state pushed?** Only under `hub_strategy="checkpoint"` or
`"all_checkpoints"`. Under `every_save` — the default — it is not, and neither
is `trainer_state.json`: `_push_from_checkpoint`'s root upload copies only
config/weights/adapter files (`trainer.py:4085-4098`). **So `every_save` alone
cannot satisfy "a Colab disconnect never loses the run"; it only satisfies
"never loses the adapter".**

**Recovery recipe after a disconnect** (`hub_strategy="checkpoint"`):

```python
from huggingface_hub import snapshot_download
local = snapshot_download(repo_id=HUB_ID, allow_patterns="last-checkpoint/*")
trainer.train(resume_from_checkpoint=f"{local}/last-checkpoint")
```

The base model and the adapter shapes must match bit-for-bit (same Unsloth
model name, same `r`/`target_modules`), and a `transformers` version mismatch
against `config.transformers_version` only warns (`:3343`). Note the
checkpoint's own `trainer_state.json` drives the step counter, so
`num_train_epochs` / `max_steps` must be unchanged too.

**Also worth knowing.** `save_only_model=True` shrinks a checkpoint from
~131 MiB to ~54 MiB but its own help string says it "Prevents resuming training
from checkpoint" — leave it `False`. `save_total_limit` protects the most
recent checkpoint and the best one and deletes the oldest otherwise
(`trainer_utils.py:342-385`); `None` never deletes. Rotation is **local only** —
`upload_folder` is called without `delete_patterns`, so nothing is ever removed
from the Hub, which is what we want (`all_checkpoints` accumulates forever;
`checkpoint` overwrites `last-checkpoint/` and keeps the rest in git history).

**Unsloth does not interfere.** `unsloth_zoo/compiler.py:6262-6347` takes the
source of `Trainer._inner_training_loop` via `inspect.getsource`, replaces the
`logger.info("***** Running training *****")` banner with the Unsloth banner
and stubs out `is_torch_xla_available()`, then `exec`s it as
`_fast_inner_training_loop`. `_maybe_log_save_evaluate` → `_save_checkpoint` →
`_push_from_checkpoint` and the whole resume path are untouched. (This
source-rewriting is exactly why `transformers<=5.5.0` is pinned; 5.5.0 already
refactored the loop into `_run_epoch`/`_finalize_training` and the banner
anchor still matches.) `_push_from_checkpoint` is byte-identical between
5.5.0 and 5.17.0 apart from `upload_folder(...)` → `hf_api().upload_folder(...)`,
and `HubStrategy` has the same four members, so the current
[transformers docs](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
apply to our pin unchanged — no "newer API" caveat is needed here.

### Suggested config block

```yaml
adapter:
  output_dir: outputs/qwen3-0.6b-r16      # repo name falls out of the basename
  save_strategy: steps
  save_steps: <same arithmetic as eval_every>
  save_total_limit: 2                     # not 1: see caveats
  save_only_model: false
  push_to_hub: true
  hub_model_id: fnl-es/qwen3-0.6b-muc4-r16
  hub_strategy: checkpoint                # not every_save: optimizer state
  hub_always_push: true
  hub_private_repo: false                 # public, per map Q6
  # hub_token: omitted — Colab secrets vault supplies HF_TOKEN
```

If the eval callback fires on its own cadence rather than through
`eval_strategy`, `save_steps` is independent of it: derive both from the same
`eval_every` number, or have the callback set `control.should_save = True` in
`on_step_end` (respected at `trainer.py:2087`) so they cannot drift.

### Open caveats

- **Not run.** Nothing here was executed against a Colab T4; every claim is from
  source read on 2026-09-22 plus three published Hub repos. The first smoke run
  should print the pushed file list and assert `adapter_model.safetensors`
  ≈ 40.4 MB.
- **`save_total_limit=1` + `hub_strategy="checkpoint"` is a real race**
  (unverified, reasoned from source): `upload_folder` is decorated
  `@future_compatible`, so it enumerates `folder_path` *inside* the worker
  thread. If the next save's `rotate_checkpoints` deletes a folder whose upload
  is still queued, that upload fails. `save_total_limit=2` (or `None`) avoids it.
  Estimate: unlikely at 0.6B with minutes between saves.
- **Upload time is an estimate.** ~116 MiB of changed bytes per
  `"checkpoint"` push; at a plausible Colab uplink of 20–50 Mbit/s that is
  20–50 s, comfortably inside a half-epoch, but unmeasured. If it is not, the
  `hub_always_push=True` queue grows and the final `_finish_current_push()`
  stalls.
- **`_get_token_from_google_colab` needs notebook access enabled on the secret**;
  if it is off, `create_repo` raises inside `Trainer.__init__`. Untested here.
- **`base_model_name_or_path` fix is untested.** Mutating
  `model.peft_config["default"].base_model_name_or_path` before the first save
  is the obvious lever but was not run; peft may re-derive it.
- **`Trainer.push_to_hub(**kwargs)` forwards to TRL's three-parameter
  `create_model_card`**, so passing `language=`/`license=`/`tasks=` (valid for
  stock transformers) would raise `TypeError`. Only `model_name`, `dataset_name`
  and `tags` are safe.
- **`peft 0.21.0` declares a bare `Requires-Dist: transformers`** with no upper
  bound; it was released 2026-09-15, five months after transformers 5.5.0, and
  was not tested against it here. If it misbehaves, pin down to a peft released
  nearer 5.5.0 (unsloth only requires `>=0.18.0`).
- **[transformers#30724](https://github.com/huggingface/transformers/issues/30724)**
  ("`every_save` won't push the model to the Hub if large") concerns sharded
  index files; a 38 MB adapter never shards, so it does not apply — but it would
  at 4B if anyone ever merged.
