# Spec: Milestone 4 — full QLoRA fine-tune of Qwen3-4B on a T4, compared to the baselines

Status: ready-for-agent
Created: 2026-09-26
Source: `map.md` and `issues/01`–`12` in this directory; `docs/DESIGN.md` §7–§9, §11; `CONTEXT.md`

## Problem Statement

Milestone 3 proved the training loop on a 0.6B model and 100 documents.
Milestone 4 is the run that defines "done": `Qwen/Qwen3-4B-Instruct-2507`
fine-tuned with QLoRA on the whole train split on a Colab T4, the finished
adapter scored over all 200 dev documents, and a written comparison against
the milestone-2 baselines (zero-shot 18.6, 3-shot 20.0 micro-F1).

The map found what separates that from "write one YAML and run":

- The run takes ≈ 2.25 h, and a free-tier session guarantees nothing, so a
  disconnect is the normal case. Today a disconnect loses the run: nothing
  resumes, and a fresh run with the same adapter repo clobbers the pending
  checkpoint.
- The training stack is installed unpinned. A release between a run and its
  resume would change the run halfway through, and ticket 11 showed that
  Unsloth's bundled files change behaviour.
- There is no way to score an adapter over the full dev split. The training
  callback scores 50 documents in-process on NF4, which is a training signal
  and not a result comparable to the baselines.
- The comparison has no home, no shape and no rule for what it may claim.

## Solution

- **Training runs the new experiment `configs/qwen3-4b-r16.yaml`**, which
  changes values only. The run gains:
  - a `--resume <wandb_run_id>` flag that continues an interrupted run from
    the Hub's `last-checkpoint/` as the same W&B run, and refuses every unsafe
    case by name;
  - a guard that stops a fresh run from clobbering a resumable repo;
  - a pinned install cell;
  - richer per-eval-point logging.
- **`generate.py` learns to serve an adapter through vLLM** over the fp16
  official base. That is the baselines' own engine, so the adapter is the only
  variable. It is scored in its own `job_type=eval` run whose YAML pins the
  adapter revision and names the training run.
- **A hand-written comparison note** in `docs/` sets the fine-tune beside
  the baselines.
- **The GPU risks go first:**
  - an eval of the existing 4B probe adapter proves vLLM LoRA on a T4;
  - a 6-step 0.6B drill proves resume end to end.

Everything testable without a GPU is pure and tested on CPU.

## User Stories

1. AS A learner, I WANT the 4B run defined by one YAML that differs from the smoke YAML in values only, SO THAT milestone 3's promise holds and the experiment is diffable.
2. AS A learner, I WANT an interrupted run to continue from its last pushed checkpoint as the same W&B run, SO THAT a Colab disconnect costs minutes, not the run.
3. AS A learner, I WANT to resume by passing the run id the run printed, SO THAT nothing is auto-detected behind my back.
4. AS A learner, I WANT a resume to take the dataset revision from the run it resumes, SO THAT a re-pushed dataset cannot change the data mid-run.
5. AS A learner, I WANT a resume of a finished run refused, SO THAT I never log an empty eval into a completed run.
6. AS A learner, I WANT a resume refused when the YAML differs from the run's, with the differing keys named, SO THAT eval and save points cannot silently desynchronise.
7. AS A learner, I WANT a resume refused when Colab's torch differs from the run's, SO THAT the one unpinned package cannot change the run halfway.
8. AS A learner, I WANT a resume allowed on newer code but recorded, SO THAT a bug fix can rescue a run and the record still shows which steps ran on which commit.
9. AS A learner, I WANT a fresh run refused when its adapter repo already holds a checkpoint, SO THAT "Run all" can never destroy a pending resume.
10. AS A learner, I WANT `--limit` runs never to push, SO THAT a quick pipeline check never overwrites a real adapter.
11. AS A learner, I WANT `--resume` and `--limit` rejected together, SO THAT I cannot ask to resume a run that never pushed a checkpoint.
12. AS A learner, I WANT the training stack pinned to the versions every GPU run so far used, SO THAT a run and its resume execute the same code.
13. AS A learner, I WANT a predictions table and the cut-off count at every eval point, SO THAT a parse-failure spike can be diagnosed from the run itself.
14. AS A learner, I WANT the finished adapter scored over all 200 dev documents with the baselines' engine, sampling and metric keys, SO THAT its numbers line up with theirs in W&B.
15. AS A learner, I WANT the eval to score the exact adapter commit the training run recorded, SO THAT a later push to the repo cannot change what was scored.
16. AS A learner, I WANT the eval run to name its training run, SO THAT every result traces back to the run that produced it.
17. AS A learner, I WANT vLLM LoRA on a T4 proven on the probe adapter before the real run, SO THAT the engine question is settled while a fallback is still cheap.
18. AS A learner, I WANT resume proven on a 6-step drill before the real run, SO THAT a disconnect during the 2.25 h run is a known procedure.
19. AS A learner, I WANT a comparison note that sets the fine-tune beside every baseline on the same 200 documents, citing each run, SO THAT "done" is a document I can read and check.
20. AS A learner, I WANT the note to say the published GTT figure is a test-split number, SO THAT I do not compare dev against test.
21. AS A learner, I WANT the NF4-in-training versus fp16-served difference measured on 50 documents, SO THAT the quantisation gap is a number, not a guess.
22. AS A learner, I WANT the README to explain how to run, resume and evaluate today, SO THAT the procedures survive this conversation.
23. AS A maintainer, I WANT the notebook test to keep the install pins equal to the versions the run stamps, SO THAT the two lists cannot drift.
24. AS A maintainer, I WANT probe and drill YAMLs deleted once they have served, SO THAT `configs/` holds only experiments.

## Implementation Decisions

Every decision below is detailed in the ticket it cites. This section fixes the
contract and does not re-argue it.

### The experiment ([Callback cadence and the 4B YAML](issues/06-callback-cadence-and-4b-yaml.md))

`configs/qwen3-4b-r16.yaml`, verbatim:

```yaml
# Milestone 4: QLoRA fine-tune of the 4B on the whole train split (DESIGN §7, §9, §11).
name: qwen3-4b-r16
tags: [fine-tuned, qlora]
model: Qwen/Qwen3-4B-Instruct-2507
dataset: fnl-es/muc4-chat
adapter: fnl-es/qwen3-4b-muc4-lora-r16
dev_docs: 50                 # first N dev documents per eval point; train_docs absent = the whole split
max_seq_len: 2048
epochs: 3
lr: 2.0e-4
scheduler: cosine
warmup_ratio: 0.03
effective_batch_size: 16
per_device_batch_size: 2     # 4 is no lighter and slower on a T4 (ticket 01)
seed: 3407
quantization: nf4
lora:
  r: 16
  alpha: 16
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
eval_every: 0.5              # ≈ 41 steps, 6 eval points, each a pushed checkpoint
generation:
  max_new_tokens: 512
  batch_size: 8              # 243 s per 50 docs at 4B; 16 unmeasured (ticket 11)
wandb_project: muc4-event-extraction
```

Expected derived numbers: 1,298 kept examples (1 dropped), 246 steps, eval
interval 41. The smoke YAML's header comment ("omits the subsets") is corrected:
milestone 4 omits `train_docs` only.

### The callback

- `ScoringCallback` logs, at every eval point and in the same `log` call as
  the `dev/*` keys:
  - `dev/diagnostics/n_cut_off` (the count of rows with `cut_off`);
  - a `predictions` W&B table with the baselines' `TABLE_COLUMNS`.
- The end-of-run table in `train()` goes, because it duplicated the last eval
  point.
- No new config key.

### Resume ([Resuming a run from the Hub](issues/03-resuming-a-run.md), [The resume contract](issues/07-resume-contract.md))

```
python -m train --config <same yaml> --resume <wandb_run_id>
```

**Glue sequence** (ticket 03). Unsloth overrides none of these steps:

1. Fetch the resumed run's config and summary via the W&B API.
2. `snapshot_download(adapter, allow_patterns="last-checkpoint/*")` into the HF
   cache, never into `output_dir`.
3. Read `global_step` from `trainer_state.json`.
4. Rebuild model, LoRA, dataset (at the run's `dataset_revision`) and
   `SFTConfig` exactly as a fresh run does.
5. `wandb.init(id=<id>, resume="must", …same name, tags, job_type, config)`.
6. `trainer.train(resume_from_checkpoint=<path string>)`. Passing `True`
   searches the empty `output_dir` and fails.
7. The tail is unchanged.

**Refusals.** One `ResumeError(ValueError)`, and each message names its reason:

| case | refusal |
|---|---|
| no `last-checkpoint/` in the adapter repo | names the repo |
| checkpoint `global_step` ≥ the run's `derived.total_steps` | "finished" |
| the loaded YAML differs from the run's `experiment` config | names every differing key |
| running `torch.__version__` ≠ the run's `versions.torch` | names both ([Pin the training stack](issues/12-pin-the-training-stack.md)) |
| `--resume` together with `--limit` | argparse error (decided in this ticket: a `--limit` run never pushes, so it has no checkpoint to resume) |

**Allowed:** a different git sha. The run config keeps the original
`git_sha`. Each resume appends `{"step": <global_step>, "git_sha": <sha>}` to a
`resumes` list in the run summary.

**Fresh-run guard.** Without `--resume`, `train` refuses (a `ResumeError`,
naming the repo) if the adapter repo exists and has any file under
`last-checkpoint/`. A **rerun** needs a new adapter name, or a person deleting
the repo.

**`--limit N` never pushes.** It sets `push_to_hub=False` and saves locally
only. It stays clear of the guard.

The refusal checks, the guard's file-list check, the `resumes` record and the
argument parsing are **pure functions** taking plain values (dicts, lists of
repo paths, version strings). The glue calls them.

### The pinned stack ([Pin the training stack](issues/12-pin-the-training-stack.md))

`train.ipynb`'s install cell drops its `find_spec` guard and runs:

```
!pip install --quiet unsloth==2026.9.11 unsloth_zoo==2026.9.7 trl==0.24.0 \
    transformers==5.5.0 peft==0.20.0 bitsandbytes==0.50.2
```

A comment in the cell says that torch stays Colab's (and `--resume` checks it),
and that the pins stay until a deliberate, explained bump. Milestone 5 decides
its own stack.

### Serving an adapter ([Serving the finished adapter](issues/04-serving-the-adapter.md), [The final evaluation run](issues/08-final-evaluation-run.md))

- `generate.py` validates three new **optional** YAML keys: `adapter`,
  `adapter_revision` and `training_run`. The last two are required when
  `adapter` is set.
- With `adapter` present, `vllm_engine`:
  - snapshots **only** `adapter_*` files at `adapter_revision` to a local path
    (`allow_patterns=["adapter_*"]`, never `last-checkpoint/`);
  - builds the `LLM` with `enable_lora=True` (vLLM's default `max_lora_rank`
    of 16 covers r=16);
  - passes `LoRARequest(<adapter name>, 1, <local path>)` to `generate`.
- Everything else matches the baselines: fp16 official base, greedy,
  `max_new_tokens` 512, `max_model_len` 3072, `max_num_seqs` 16.
- `eval.py` is unchanged. It logs the YAML as run config, so the three keys
  become the eval run's link back to the adapter and its training run.
- `meta.json` records nothing new.

The eval YAML shape (the real one is written after training):

```yaml
# Final evaluation of the milestone-4 adapter over the whole dev split (DESIGN §8).
name: qwen3-4b-r16-eval
tags: [fine-tuned, qlora]
model: Qwen/Qwen3-4B-Instruct-2507
dataset: fnl-es/muc4-chat
engine: vllm
exemplars: []
adapter: fnl-es/qwen3-4b-muc4-lora-r16
adapter_revision: <the training run's summary.adapter_revision>
training_run: <the training run's W&B id>
generation:
  max_new_tokens: 512
  max_model_len: 3072
wandb_project: muc4-event-extraction
```

**Fallback**, only if the probe eval shows that vLLM LoRA fails on a T4:

- An `engine: unsloth` branch in `generate.py` loads the NF4 base plus the
  pinned adapter and generates with the existing `unsloth_engine`.
- The final eval then also runs an **NF4 zero-shot** (`qwen3-4b-zero-shot-nf4.yaml`,
  `engine: unsloth`, no adapter, all 200 dev docs). The note sets the fine-tune
  beside it as well as beside the fp16 baselines, which keeps the comparison to
  one variable. This settles the map's last fog patch.
- Not option (B), the bnb plugin.

### Notebooks

**`train.ipynb` becomes milestone 4's notebook**
([The resume contract](issues/07-resume-contract.md) item 7):

- Title and intro widen to cover the 4B run.
- Cells 1 and 3–5 are unchanged; the install cell is pinned as above.
- A 4B pipeline check: `qwen3-4b-r16.yaml --limit 8` (no push).
- A resume-drill section, removed after the drill.
- The run.
- A resume cell, `RUN_ID = ""  # @param {type:"string"}`, which **skips itself in
  Python when `RUN_ID` is empty**, so "Run all" never resumes a finished run.
- Milestone 3's smoke commands stay **commented out**, under a note saying they
  were milestone 3's check and are superseded by the 4B check.
- The load-back cell goes; the final eval supersedes it.

**`baselines.ipynb`:**

- Its title widens to "baselines and adapter evaluation".
- It gains generate → eval cells, first for the probe eval (later deleted),
  then, after training, for `qwen3-4b-r16-eval.yaml`.
- The final-eval cell and its YAML land in **one commit**, because the
  notebook test requires every named config to exist.

### The comparison note ([The comparison note](issues/09-comparison-note.md))

`docs/milestone-4-comparison.md` is hand-written, and every number cites its
W&B run id. Its headings are fixed:

1. **Setup**:
   - base model, adapter repo at its revision, served base (fp16 via vLLM, or NF4 on the fallback);
   - dev split, 200 documents, links to the four runs.
2. **Results**:
   - *headline table*: rows always-empty, zero-shot, 3-shot, QLoRA fine-tune
     (+ NF4 zero-shot on the fallback); columns micro P / R / F1, parse-failure rate;
   - *detail table*: per-role F1 (event type + 5 roles), relevance, event-count
     and event-type accuracy, cut-off outputs, truncated documents;
   - one sentence: published GTT ≈ 50–55 is a **test**-split figure, context only;
   - one sentence on how many documents the numbers rest on. If the gap to 3-shot
     is under ~5 points, list a paired bootstrap as a milestone-6 item.
3. **Training**:
   - steps, wall time, sessions and resumes, dropped documents, peak memory;
   - links to the W&B loss and eval-point F1 panels (no committed images).
4. **NF4 vs fp16**: one line with both micro-F1s on the first 50 dev documents
   and their difference, with no significance claim ("does not apply" on the
   fallback path).
5. **What surprised us**: short, e.g. the chat-template think block (ticket 11)
   and the parse-failure spike (ticket 05).

The note is frozen once complete. `RESULTS.md` (milestone 6) links to it and
does not absorb it.

### README and docs

The README is restructured around what a reader does now (decided in this ticket):

- **Fine-tuning**:
  - the 4B experiment and the pinned stack;
  - the adoption of the official chat template and the completion guard;
  - `--limit` checks that never push, and the fresh-run guard;
  - the predictions table and `n_cut_off` at every eval point.
  - The smoke run shrinks to one line: it proved the loop in milestone 3 and
    its YAML stays as the cheap end-to-end check.
- **Resuming a run** (new): the `RUN_ID` cell, `--resume <id>`, what it refuses
  and why, the `resumes` record, and the expectation of a GPU lockout after heavy use.
- **Evaluating an adapter** (new): the eval YAML's three keys, the pinned
  revision, the `baselines.ipynb` cells.
- **Results** (new, **post-run commit only**): a pointer to the comparison note.
- **Layout**: `docs/` gains the note.

DESIGN §3's tree follows the notebooks (`train.ipynb`: "4B check, run, resume";
`baselines.ipynb`: "baselines and adapter evals"). This ticket has already
added dated corrections for the map's other findings to DESIGN §7–§9 and §11.
`CONTEXT.md` already holds **Resume**, **Rerun** and **Comparison note**.

## Testing Decisions

The rule and prior art are milestone 3's. Tests drive public seams with real
data shapes and assert observable behaviour: the errors raised and the keys and
values produced. Prior art: `tests/test_train.py` (fake engine and fake trainer
state for the callback), `tests/test_generate.py`, `tests/test_notebook.py`.

1. **Resume refusals**, pure: finished checkpoint, config drift naming every
   differing key (and none when equal), torch mismatch naming both versions,
   missing checkpoint, allowed git-sha change.
2. **The `resumes` record**: appends `{step, git_sha}` to an absent or existing list.
3. **Fresh-run guard**: refuses on any `last-checkpoint/…` path, passes on an
   adapter-only repo and on a missing repo.
4. **CLI**: `--resume` parses; `--resume` with `--limit` is rejected; `--limit`
   turns pushing off.
5. **Callback** (extends the existing test): every eval point logs
   `dev/diagnostics/n_cut_off` and a `predictions` table in the same call,
   still without an explicit step.
6. **`generate` config**: `adapter` without `adapter_revision` or
   `training_run` is rejected by name; an adapter-free baseline YAML still loads.
7. **Adapter snapshot patterns** fetch `adapter_*` and never `last-checkpoint/`.
8. **Notebooks** (extends `test_notebook.py`):
   - `train.ipynb`'s install cell pins every package in `train.VERSIONED`
     except torch, at the exact versions stated in the pin;
   - the resume cell contains its empty-`RUN_ID` skip;
   - existing checks unchanged (no outputs, named configs exist, no restart in
     `train.ipynb`).

No Hub or W&B mocks. The glue that calls the Hub, W&B and the trainer is proven
by the resume drill and the probe eval on a T4.

## Out of Scope

- The rented-GPU / bf16 LoRA run and the quantisation-cost comparison
  (milestone 5). The NF4 zero-shot runs here only on the fallback path.
- `RESULTS.md`, the test split, the adapter's model card beyond the trainer's
  auto card, and a paired bootstrap (milestone 6).
- Choosing the adapter by best eval point, rank and lr sweeps, the plain-peft
  rewrite, the base-model variant, DocEE, merged or GGUF export (round two).
- Auto-detecting a resumable run; W&B artifact lineage between training and eval.
- Deleting Hub repos (probe, drill), which is the user's call.

## Further Notes

### Order of work that de-risks earliest

1. **Hygiene commit.** Delete `configs/qwen3-4b-probe-b{2,4}.yaml`, pin the
   install cell, add the pin test.
2. **Adapter serving.** Add the `generate.py` keys and snapshot plus tests, then
   `configs/qwen3-4b-probe-eval.yaml` serving `fnl-es/qwen3-4b-muc4-lora-probe`
   at the `adapter_revision` of run `6svccyzx`, first 50 dev docs, from
   `baselines.ipynb`.
   - **GPU.** Passes if it runs and scores within ±3 of that run's in-process
     15.8, allowing for the NF4-vs-fp16 difference.
   - **On failure**, build the fallback engine (and later the NF4 zero-shot)
     before going on.
   - On success, delete the probe-eval YAML and cell. Its W&B run stays as the
     evidence.
3. **Callback**: `n_cut_off` and the per-eval-point table; drop the end-of-run table.
4. **Resume**: the pure checks, the guard, the `--limit` no-push rule and their
   tests, then the glue.
5. **Resume drill.** `configs/qwen3-0.6b-resume-drill.yaml` is the smoke YAML
   with `train_docs: 32`, `dev_docs: 8`, adapter
   `fnl-es/qwen3-0.6b-muc4-lora-drill` (6 steps, an eval point at each).
   - **GPU.** Start it, interrupt after the first pushed checkpoint, then resume
     with the printed id.
   - Passes if the result is **one** W&B run with 6 eval points and one
     `resumes` entry, a second `--resume` refuses ("finished"), and a fresh run
     on the drill YAML refuses (the guard).
   - Then delete the drill YAML and cells.
6. `configs/qwen3-4b-r16.yaml`, the milestone-4 `train.ipynb`, the notebook
   test, the smoke YAML's comment.
7. The note's skeleton with the three baseline rows filled in (from W&B, each
   citing its run id); the README restructure; the DESIGN §3 tree.

### After the build: the run book

1. **Pipeline check.** "Run all" in `train.ipynb` runs the 4B `--limit 8`
   check, then the run.
   - The check should print a first completion that is the target JSON
     followed by `<|im_end|>`, with no `<think>`.
   - Step-1 loss should be ≈ 1.5.
2. **The run.** ≈ 2.25 h, one session if lucky. On a disconnect, open a new
   runtime, run the setup cells, set `RUN_ID` to the printed id and run the
   resume cell. Repeat until the run finishes.
3. **The eval.** Write `configs/qwen3-4b-r16-eval.yaml` with the run's
   `summary.adapter_revision` and id, add its cell to `baselines.ipynb`, and
   commit both together. Run the cell (≈ 5–8 min).
4. **The cross-check**, locally on CPU:
   ```bash
   uv run python -m data.prepare                     # data/prepared/dev.jsonl
   uv run wandb artifact get flowing/muc4-event-extraction/qwen3-4b-r16-eval-dev:latest \
       --root outputs/qwen3-4b-r16-eval
   head -50 data/prepared/dev.jsonl > outputs/dev-first50.jsonl
   uv run python -c "
   import json
   ids = {json.loads(l)['docid'] for l in open('outputs/dev-first50.jsonl')}
   for l in open('outputs/qwen3-4b-r16-eval/dev.jsonl'):
       if json.loads(l)['docid'] in ids: print(l, end='')" > outputs/eval-first50.jsonl
   uv run python -m eval --pred outputs/eval-first50.jsonl --gold outputs/dev-first50.jsonl
   ```
   Set its micro-F1 against the training run's `dev/micro_avg/f1` at its last
   eval point.
5. **The note.** Fill in the fine-tune row, training facts, the cross-check line
   and the surprises. Add the README's *Results* pointer. That commit closes
   milestone 4.

### Done criterion

Milestone 4 is done when all three hold:

- one `job_type=train` W&B run of `qwen3-4b-r16` that reached its last step
  with 6 eval points (resumes allowed, all inside that one run);
- one `job_type=eval` run over all 200 dev documents scoring the pinned
  `adapter_revision`;
- `docs/milestone-4-comparison.md` with no placeholder left.

**No score gates this.** A fine-tune below 3-shot's 20.0 passes, and the note
says why. What would mean a broken pipeline:

- a final-eval parse-failure rate above zero-shot's 7.5 %;
- all-`[]` predictions;
- a step-1 loss far from ≈ 1.5.

### Numbers worth knowing

- **Training:** ≈ 22 s/step at per-device 2, so ≈ 1.5 h of pure training;
  5.8 GiB training plateau; 10.2 GiB whole-device peak in the callback at
  generation batch 8.
- **Callback:** 243 s per 50 documents, ≈ 1,050 s in the worst case (every
  batch to 512 tokens). The worst-case run is 3.7 h, 2 sessions, with a 35-min
  checkpoint gap (accepted).
- **Checkpoints:** `optimizer.pt` is 21 MB (8-bit AdamW).
- **Probe reference:** run `6svccyzx`, 12 steps, dev micro-F1 15.8 on 50 docs
  (noise).
