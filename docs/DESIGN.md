# Design: Fine-tuning a small decoder for document-level event extraction

Outcome of a grill-me interview on 2026-09-14. Every decision below was
discussed and agreed; alternatives that were rejected are noted so we do not
re-litigate them.

## 1. Goal

**Learning exercise.** The deliverable is understanding the full pipeline
(data → chat template → LoRA/QLoRA → eval → adapter on the Hub) plus a
baseline-vs-fine-tuned comparison. SOTA on the task is explicitly *not* the
goal; ambition goes to `TODO.md` (round two).

Done = milestone 4 below with a written comparison. Milestones 5–6 are the
payoff.

## 2. Compute

- **Develop on free Colab (T4, 15 GB)** with QLoRA (bitsandbytes NF4) and a
  small data subset.
- **Full runs on rented hardware**: first choice Colab Pay-As-You-Go compute
  units (A100/L4, same notebook, zero setup change); fallback RunPod/Lambda/
  Vast Jupyter template.
- Consequence: everything must be portable. Data and adapters live on the HF
  Hub; scripts read secrets only from env vars; no Colab imports in `src/`.
- The local machine has a tiny GPU; it is **ignored by design**. Local = CPU
  only (data prep, scorer, tests).

## 3. Repo shape

Repo + thin notebook (rejected: notebook-only, for portability and
diff-ability).

```
pyproject.toml          # uv; py >=3.11,<3.14 (Colab is 3.13); core deps CPU-installable
configs/*.yaml          # one file per experiment
src/
  data/prepare.py       # download GTT JSON → chat JSONL → push to Hub
  generate.py           # render inputs (+ exemplars) → engine (vLLM | in-process | constant) → outputs JSONL
  train.py              # Unsloth + TRL SFTTrainer, config-driven
  eval.py               # GTT scorer + diagnostics → W&B
tests/                  # pytest, CPU only
notebooks/baselines.ipynb  # thin: clone@ref, pip install, secrets, gold, then one cell per GPU run
notebooks/train.ipynb      # thin: clone@ref, pip install, secrets, smoke, train, load the adapter back
docs/DESIGN.md          # this file
RESULTS.md, TODO.md     # written in milestone 6 / as ideas arise
```

GPU-only deps (`unsloth`, `trl`, `bitsandbytes`, `vllm`) are installed by the
notebook, not `pyproject`, because Unsloth's Colab install line changes often.

## 4. Dataset: MUC-4 (GTT preprocessed), DocEE-ready interface

**Chosen: MUC-4** via `xinyadu/gtt` `data/muc/processed/{train,dev,test}.json`
(1300/200/200 docs). Public domain data, MIT preprocessing → redistributed
publicly on the Hub.

Why: tiny schema (6 incident types × 5 roles: PerpInd, PerpOrg, Target,
Victim, Weapon), short docs (mean 320 words, p95 ~700), genuinely
document-level and **multi-event** (train: 600 docs with 0 templates, 466
with 1, 146 with 2, 88 with 3+), published baselines (~50–55 F1) to compare
against.

Known limitation: no published IAA; annotation is inconsistent, so the F1
ceiling is unknown. Accepted for round one.

**Runner-up: DocEE** (27K docs, 59 types, 356 roles, κ=0.94/0.81, human ≈85
F1 vs BERT ≈41). Cleaner signal but one event per doc, fat schema, long
free-text arguments. `prepare.py` and `eval.py` are written behind a small
dataset interface (`load(split) -> list[Doc]`, `score(pred, gold)`) so DocEE
is a drop-in second dataset.

Rejected: WikiEvents (246 docs, coref-heavy output), RAMS (trigger given,
5-sentence windows), ACE 2005 (LDC license), MAVEN/GENEVA/PHEE (sentence-level).

## 5. Base model

- **Target: `Qwen/Qwen3-4B-Instruct-2507`** (Apache-2.0, strong JSON,
  Unsloth-supported) — the official repo, not the Unsloth mirror, whose hybrid
  chat template injects `<think>` blocks into assistant turns.
- **Smoke: Qwen3-0.6B** (same family, full epoch in minutes on T4).
- **Instruct, not base**, so the zero-shot baseline on the identical checkpoint
  is meaningful and the chat template is native. Base-model variant = round two.
- Qwen3 thinking mode **off** (`enable_thinking=False`) in training and
  inference; the 2507 template ignores the flag, it is passed anyway so every
  rendering follows one convention.

## 6. Task framing

| Decision | Choice |
|---|---|
| Output format | JSON list of templates; empty doc → `[]` |
| Coreferent mentions | emit all mentions per entity (list of strings) |
| Character offsets | dropped |
| Empty roles | omitted from the target; parser fills them in |
| Prompt | one ~150-token system prompt with types, role definitions, the "military clashes are not incidents" rule and the output format — **identical** for zero-shot and fine-tuned runs |
| Template order | sorted by char offset of earliest mention; mentions within an entity by offset (deterministic target) |
| Long docs (>~1800 input tokens, ~2%) | dropped from train; kept in dev/test with input truncation, count reported |

## 7. Training

**Unsloth + TRL `SFTTrainer`** (plain peft/transformers rewrite = round two).
LoRA config and collator kept explicit in `train.py`.

Starting knobs:
- Completion-only loss (assistant tokens only) — non-negotiable.
  *Mechanism fixed 2026-09-22 (milestone-3 map, tickets 03/05): we pre-tokenise
  and mask the prompt span ourselves after asserting it is a token-level prefix
  of the full rendering. TRL's `assistant_only_loss` and Unsloth's
  `train_on_responses_only` are rejected — both would train the empty `<think>`
  block that our inference prompt already supplies.*
  *Added 2026-09-26 (milestone-4 map, ticket 11): training adopts the base
  model's official chat template, because Unsloth's copy for
  Qwen3-4B-Instruct-2507 renders an empty think block into every assistant turn.
  The masking step refuses any completion that is not exactly the target plus
  `<|im_end|>`.*
- `max_seq_len` 2048, packing off; `padding_free` off explicitly (Unsloth
  auto-enables it). Train examples over the sequence budget are dropped and
  counted — the 1800-token input budget bounds the prompt alone, so 1 of 1299
  train documents totals 2082 tokens. *Added 2026-09-22 (ticket 08).*
- LoRA r=16, α=16, dropout 0, targets all linear (q,k,v,o,gate,up,down).
- lr 2e-4, cosine, 3 % warmup, 3 epochs, effective batch 16 (bs 2–4 × accum).
  *Fixed 2026-09-26 (milestone-4 map, ticket 06): per-device batch 2 × accum 8
  for the 4B on a T4 (memory is equal at 4, and 2 is faster); generation batch 8.*
- fp16 on T4, bf16 on L4/A100; QLoRA on T4, bf16 LoRA on rented GPU.
- Eval callback on dev every ~½ epoch running the real scorer, not just loss.
  *Fixed 2026-09-26 (ticket 06): the first 50 dev documents, 6 eval points for
  the 4B, a checkpoint at each, and a predictions table plus the cut-off count
  at every one.*
- *Added 2026-09-26 (milestone-4 map, tickets 03, 07, 12): a run resumes after
  a disconnect with `--resume <wandb_run_id>` from the Hub's `last-checkpoint/`,
  continuing the same W&B run. It refuses a finished checkpoint, any config drift,
  or a torch version other than the run's. The training stack is pinned in
  `train.ipynb`.*

Estimate: 1300 docs × 3 epochs ≈ 250 steps ≈ 40–60 min for 4B QLoRA on T4.
*Corrected 2026-09-26 (milestone-4 map, tickets 01, 06, 11): measured ≈ 22 s
per step, so 246 steps take ≈ 1.5 h of pure training. With six 50-document
callbacks at ≈ 243 s each, the run takes ≈ 2.25 h, one free-tier session.*

## 8. Evaluation

- **Headline: GTT scorer** ported into `eval.py` (template alignment per doc,
  per-role P/R/F1 with any-mention matching, micro-F1) → comparable to
  published numbers.
- **Diagnostics:** relevance accuracy (`[]` vs ≥1 template), template-count
  exact match, incident-type accuracy on aligned templates, JSON parse-failure
  rate.
- **Baselines run before any training:** zero-shot and 3-shot
  Qwen3-4B-Instruct with the same system prompt; trivial always-`[]` for
  relevance (~46 %).
- Inference: vLLM (or Unsloth fast generate), greedy, `max_new_tokens` ≈ 512.
  *Fixed 2026-09-26 (milestone-4 map, ticket 08): a finished adapter is scored
  over the whole dev split by vLLM with the fp16 official base plus the LoRA,
  at the revision the training run recorded, in its own `job_type=eval` run. In-process
  generation is for the training callback only (and for the fallback if vLLM
  LoRA fails on a T4).*

## 9. Artifacts, tracking, secrets, config

- HF namespace: **`fnl-es`** (from `huggingface-cli whoami`). All public.
  - dataset `fnl-es/muc4-chat`
  - adapters `fnl-es/qwen3-4b-muc4-lora` (LoRA adapter only, not merged)
    *Corrected 2026-09-26 (milestone-4 map, ticket 06): the repo is named after
    the experiment, `fnl-es/qwen3-4b-muc4-lora-r16`.*
- **W&B** project `muc4-event-extraction` (chosen over TensorBoard-to-Drive
  for hosted, session-surviving, cross-machine run comparison; `report_to` is
  a one-word switch if we change our mind).
- Secrets: `HF_TOKEN`, `WANDB_API_KEY` via Colab Secrets in the notebook, env
  vars elsewhere.
- One YAML per experiment, logged to W&B: `qwen3-0.6b-smoke.yaml` (100 docs,
  3 epochs, 50 dev docs), `qwen3-4b-r16.yaml`, later `qwen3-4b-r16-bf16.yaml`.
  *Corrected 2026-09-22 (milestone-3 map, ticket 06): 1 epoch is 7 optimizer
  steps and 2 eval points — too few for the loss trend to carry information.
  Corrected again 2026-09-23 (implementation): the trainer counts the last,
  partial accumulation as a step, so 3 epochs are 21 steps, 7 eval points.*
  *Added 2026-09-26 (milestone-4 map, ticket 08): a scored adapter gets its
  own eval YAML, `qwen3-4b-r16-eval.yaml`, naming the adapter, its pinned
  revision and the training run.*
- Merged/GGUF export: out of scope.

## 10. Tests (CPU, pytest)

- Scorer on hand-built gold/pred pairs incl. the multi-template alignment case.
- `prepare.py` round-trip: templates → JSON target → parsed == original minus
  offsets.
- Prompt rendering with Qwen's tokenizer/chat template (tokenizer only).
- JSON repair on truncated output.

## 11. Milestones

1. Repo skeleton + data prep + scorer, tests green locally, dataset on Hub.
2. Zero-shot and 3-shot baselines of Qwen3-4B-Instruct on dev (T4, vLLM) → W&B.
3. Smoke fine-tune of Qwen3-0.6B on 100 docs — proves loop, masking,
   checkpoint push, eval callback.
4. Full QLoRA fine-tune of Qwen3-4B on T4; compare to 2. **← "done"**
   *Refined 2026-09-26 (milestone-4 map): resumable across sessions; the
   adapter is scored over all 200 dev documents in its own eval run; the
   comparison is `docs/milestone-4-comparison.md`, dev split only.*
5. One rented-GPU run, bf16 LoRA, same config — quantization cost + portability.
6. `RESULTS.md` (baselines vs fine-tunes vs published GTT/GRIT) + lessons.
   *Refined 2026-09-26 (milestone-4 map, ticket 09): on the test split; it
   links to the milestone comparison notes rather than absorbing them.*

Round two (`TODO.md`): DocEE, plain-peft rewrite, base-model variant,
rank/lr sweeps, sliding windows for long docs.
