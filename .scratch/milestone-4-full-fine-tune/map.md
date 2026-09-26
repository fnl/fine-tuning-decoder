# Map: Milestone 4 — full QLoRA fine-tune of Qwen3-4B on a T4, compared to the baselines

Label: wayfinder:map
Status: closed (2026-09-26)
Created: 2026-09-25
Source: `docs/DESIGN.md` §1, §7, §8, §9, §11 (milestone 4); the milestone-3 map's out-of-scope list; charting interview 2026-09-25

## Destination

A spec ready for `/implement` that reaches DESIGN §11 milestone 4: `Qwen3-4B-Instruct-2507`
fine-tuned with QLoRA on the whole train split on a free Colab T4, resumable
after a disconnect; the finished adapter evaluated over all 200 dev
documents in its own `job_type=eval` run; and a written comparison against
the milestone-2 baselines. Every setup task and every fact the spec must
state verbatim is settled inside this map; what follows the map is pure
build plus the run itself.

## Notes

- Vocabulary: `CONTEXT.md` (base model / adapter / run / eval point /
  checkpoint / baseline / engine / diagnostics). Consult before every ticket;
  the resume and comparison vocabulary is new and will need entries.
- Skills: `grill-me` + `domain-modeling` for grilling tickets; `to-spec` (or
  the milestone-3 spec as the model) for the spec ticket. There is **no**
  `research` skill in this install — research tickets are resolved by a
  subagent brief, findings committed on a `research/<name>` branch, answered
  in the ticket with sources and dates and a pointer to the branch.
- Prior art to copy, not re-derive: `.scratch/milestone-3-smoke-fine-tune/`
  (map, tickets, `spec.md`) and `.scratch/milestone-2-baselines/`.
  `src/train.py` was built so milestone 4 is "write one YAML and run"
  (M3 Q10, ticket 08) — challenge that only with evidence.
- **Colab credits (2026-09-26):** the free tier refused a GPU for usage
  limits, so the user bought credits to keep going. Runs may now be on paid
  T4s (same hardware, measurements valid). The free-tier constraint below
  still stands: the plan must be doable without credits.
- Decisions locked in the charting interview (do not re-open):
  - **Destination is a spec** (not an executed run).
  - **The spec covers three things**: the 4B training run, a separate final
    evaluation of the adapter over all 200 dev documents (`job_type=eval`),
    and a **written comparison note** that `RESULTS.md` (milestone 6) later
    absorbs (Q2 = c).
  - **The test split is not touched**; it is held for one final evaluation of
    baselines and fine-tunes together in milestone 6 (Q3).
  - **Resume after disconnect is in scope** (Q4): from the Hub's
    `last-checkpoint/`, continuing the same W&B run.
  - **One run, DESIGN §7 knobs, no tuning** (Q5); rerun only if the pipeline
    breaks. Rank/lr sweeps stay in round two.
  - **Free-tier T4 is a hard constraint** (Q6); the levers are batch sizes,
    generation length and callback cadence, not hardware.
  - **The reported adapter is the final step's** (Q7), not the best eval
    point — choosing by a noisy 50-document dev score would be tuning on dev.
- Facts established while charting (2026-09-25, from W&B `flowing/muc4-event-extraction`
  and the Hub; do not re-derive):
  - Milestone 3 smoke run `63pxlpvn` (Tesla T4, fp16, NF4): 21 steps, 0
    dropped, loss first-quarter mean 1.00 → last-quarter 0.38; 7 eval points
    at steps 3…21; final dev micro-F1 **16.3** (50 docs); parse failures
    6 / 8 / 4 / **38** / 20 / 10 / **8 %** across the eval points.
  - **Evaluation dominates wall time**: training steps ≈ 4 s at 0.6B, each
    50-document callback **95–245 s**; ≈ 90 % of the 1,238 s run. Peak GPU
    memory 9.3 GB of 15.
  - `fnl-es/qwen3-0.6b-muc4-lora-smoke` holds the adapter and a resumable
    `last-checkpoint/` (`optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
    `trainer_state.json`), pushed at every save.
  - Versions actually resolved on Colab: unsloth **2026.9.11**, unsloth_zoo
    2026.9.7, trl 0.24.0, transformers 5.5.0, peft **0.20.0**, bitsandbytes
    0.50.2, torch 2.11.0+cu128.
  - The config-nesting fix (`a4a0a5e`) postdates the last GPU run — untested
    on a GPU.
  - Whether the notebook's load-back cell generated is **unconfirmed** (only
    in the notebook's output).
  - Full train split at effective batch 16 ≈ 82 steps/epoch, ≈ 246 steps for
    3 epochs, 7 eval points at `eval_every: 0.5`.
  - Milestone 2 results to beat (200 dev docs): zero-shot micro-F1 **18.6**
    (7.5 % parse failures), 3-shot **20.0** (5 %), always-empty 0. Published
    GTT ≈ 50–55.
- User preferences: answers immediately with a recommendation; wants defaults
  justified; only commit when asked.

## Decisions so far

<!-- one line per resolved ticket: gist + link -->

- [Free Colab T4 session limits](issues/02-colab-session-limits.md): design
  every session for **≤ 3 h wall time**, push a checkpoint **≥ every 30 min**,
  expect a resume per 2–3 h and a possible 24 h GPU lockout after heavy use —
  resume is the normal path. The 12 h FAQ figure is a ceiling with no floor;
  no free-tier background execution; the idle timeout is unpublished.

- [The step-12 parse-failure spike](issues/05-parse-failure-spike.md):
  **accept, no knob changes.** The spike was repetition loops hitting the
  512-token limit, a transient of the 0.6B mid-training (it fades as the
  learning rate anneals), not a learning-rate peak or a pipeline fault. The
  callback logs no cut-off count, so ticket 06 should add
  `dev/diagnostics/n_cut_off` and a predictions table at every eval point.
  Keep `max_new_tokens` 512 (dev gold reaches 500 tokens) and size each eval
  point's time for the worst case, every batch at 512.

- [Resuming a run from the Hub](issues/03-resuming-a-run.md): **it works with
  stock calls.** Download `last-checkpoint/` into the cache, rebuild the run
  exactly as before, `wandb.init(id=…, resume="must")`, then
  `train(resume_from_checkpoint=<path>)`. The trainer restores optimizer,
  scheduler, scaler, RNG, step and data position, and Unsloth overrides none
  of it. Traps for ticket 07: a finished checkpoint trains nothing (refuse
  it); pin the dataset revision from the resumed run; any YAML or `--limit`
  change desynchronises eval from save; the run id is not stored in the
  checkpoint; a fresh run with the same repo id clobbers a pending
  checkpoint. `optimizer.pt` is 21 MB (Unsloth uses 8-bit AdamW).

- [Serving the finished adapter](issues/04-serving-the-adapter.md): **three
  options for ticket 08.** (A) vLLM with the fp16 base and the LoRA: most
  comparable to the baselines, ≈ 5–8 min, but LoRA on a T4 in vLLM 0.29 is
  unverified. (B) vLLM with the NF4 base: needs the young `vllm-bnb-plugin`,
  since bitsandbytes left vLLM core in 0.28, and computes in bf16 on sm_75.
  (C) In-process Unsloth on NF4: exactly the trained function, ≈ 15–35 min.
  The NF4-versus-fp16 base mismatch is expected to be small and of uncertain
  sign; the final training eval point gives a free 50-document cross-check.
  Leave the adapter's `base_model_name_or_path` alone. **For ticket 06:** the
  4B in-process callback is estimated at ≈ 2× the 0.6B's cost per step
  (unmeasured).

- [The final evaluation run](issues/08-final-evaluation-run.md): **vLLM,
  fp16 official base + LoRA**, the baselines' own path. `generate.py` gains
  optional `adapter` / `adapter_revision` / `training_run` keys; `eval.py`
  is unchanged. `qwen3-4b-r16-eval.yaml` is committed after training with the
  pinned sha and run from `baselines.ipynb`. The first build step is a 0.6B smoke eval
  to prove vLLM LoRA on a T4; the fallback is an in-process Unsloth engine on NF4.

- [Measure the 4B on a T4](issues/01-measure-4b-on-t4.md): **training
  fits and is ≈ 22–25 s/step** (5.8 GiB, both per-device batch sizes; 4 is not
  faster), so ≈ 1.5–1.7 h for 246 steps. Stack unchanged from milestone 3, and
  the config-nesting fix works on a GPU. **Callback not measured:** every 4B output was
  empty, and step-1 loss is 3.14 against the 0.6B's 1.37. Both are split off to
  [The 4B generates empty outputs](issues/11-4b-empty-outputs.md), which now
  blocks the cadence ticket.
- [The comparison note](issues/09-comparison-note.md): **hand-written
  `docs/milestone-4-comparison.md`, frozen once written.** A headline table and
  a per-role/diagnostics table of always-empty, zero-shot, 3-shot and the
  fine-tune (200 dev docs), each row citing its W&B run. Published GTT goes in
  a sentence, not a row (it is a test-split figure). Also: training facts, links
  to the W&B curves, the local 50-doc NF4-vs-fp16 line and the surprises. The
  build commits the skeleton with the baseline rows filled in; the post-run
  commit fills the rest. `RESULTS.md` links to the note; it does not absorb it.

- [The 4B generates empty outputs](issues/11-4b-empty-outputs.md): **Unsloth's
  4B chat template put an empty think block in every training target.**
  Training now adopts the official template, and `tokenize_example` refuses any
  completion that isn't exactly the target (`439cce1`). The rerun probes start at
  loss 1.49 and write real JSON. A callback on 50 docs takes **243 s at generation
  batch 8** (292 s at 4), ≈ 1,050 s in the worst case. The cadence ticket is unblocked.

- [Callback cadence and the 4B YAML](issues/06-callback-cadence-and-4b-yaml.md):
  **`eval_every: 0.5` × 50 dev docs** (6 eval points; ≈ 2.25 h, one session,
  checkpoint gap ≤ 22 min, and a 35-min gap accepted in the worst case), per-device batch 2,
  generation batch 8, `max_new_tokens` 512. Predictions table and `n_cut_off`
  at every eval point. `configs/qwen3-4b-r16.yaml` →
  `fnl-es/qwen3-4b-muc4-lora-r16`, and the promise of "values only" holds. The final eval's de-risk
  step now serves the 4B probe adapter.

- [The resume contract](issues/07-resume-contract.md): **`--resume
  <wandb_run_id>`** takes the dataset revision from the run. It refuses a finished
  checkpoint, any experiment-config drift (naming the keys) and a missing checkpoint,
  and allows new code, recorded in a `resumes` list. A fresh run refuses an adapter repo that
  already has `last-checkpoint/`, and `--limit` runs never push. CPU tests cover the pure
  checks; a 0.6B resume drill proves the glue before the real run. `train.ipynb`
  becomes milestone 4's, with a resume cell that skips itself when `RUN_ID` is empty.
  New terms: Resume, Rerun.

- [Pin the training stack](issues/12-pin-the-training-stack.md): **pin the
  resolved stack exactly** in `train.ipynb`'s install cell (unsloth 2026.9.11,
  unsloth_zoo 2026.9.7, trl 0.24.0, transformers 5.5.0, peft 0.20.0,
  bitsandbytes 0.50.2). torch stays Colab's, and `--resume` refuses a torch mismatch.
  A CPU test keeps the pins equal to `VERSIONED`, and the pins stay until a
  deliberate bump.

- [Write the milestone-4 spec](issues/10-write-spec.md): **[`spec.md`](spec.md)
  is ready for `/implement`.** Also settled: `--resume` excludes `--limit`; an
  NF4 zero-shot runs only on the vLLM-LoRA fallback; probe-eval and drill files
  are deleted after use; no score gate on done; the README gets restructured
  (Resuming, Evaluating an adapter, a post-run Results pointer). DESIGN §7–§9 and §11
  are corrected with dated notes. **The map is closed.**

## Not yet specified

None: the last patch (an NF4 zero-shot baseline) graduated into the spec's
fallback path via [Write the milestone-4 spec](issues/10-write-spec.md).

## Out of scope

- The rented-GPU / bf16 LoRA run and the quantization-cost comparison —
  milestone 5.
- `RESULTS.md` itself and the final test-split evaluation — milestone 6.
- The adapter's Hub model card, and a paired bootstrap on the dev gap if it
  is small — milestone 6 ([The comparison note](issues/09-comparison-note.md)).
- Choosing the adapter by best dev eval point; rank and lr sweeps; the
  plain-peft rewrite, base-model variant, DocEE, merged/GGUF export — round
  two (`docs/DESIGN.md` §11).
