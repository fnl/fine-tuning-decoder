# Results

**Draft, 2026-09-27.** The test-split evaluation (milestone 6) has not run
yet; every cell marked *pending* is filled from its W&B run once it has.

What a 4B decoder learns from 1,300 MUC-4 documents: `Qwen/Qwen3-4B-Instruct-2507`
fine-tuned with QLoRA on a free Colab T4, set beside its own zero-shot and
3-shot baselines and the published MUC-4 systems. The goal was to learn the
pipeline, not to reach the state of the art (DESIGN §1).

## Setup

- **Task:** document-level event extraction. Each document yields zero or more
  events, each an event type and five roles (PerpInd, PerpOrg, Target, Victim,
  Weapon); the model writes them as JSON.
- **Data:** `fnl-es/muc4-chat`, the GTT preprocessing of MUC-4
  (1,300 / 200 / 200 train / dev / test documents).
- **Metric:** the GTT scorer ported into `src/eval.py`: micro-F1 over the event
  type and the five roles, with per-document event alignment and any-mention
  matching. The published figures below use the same metric.
- **Systems:**
  - always-empty: answers `[]` for every document;
  - zero-shot and 3-shot: the base model with the fine-tune's own system prompt;
  - QLoRA fine-tune: adapter `fnl-es/qwen3-4b-muc4-lora-r16`, rank 16, 3 epochs
    on an NF4 base (`configs/qwen3-4b-r16.yaml`), served on the fp16 base by vLLM.
- **No document is truncated.** The longest dev and test inputs are 1,375 and
  1,352 tokens, under the 1,800-token input budget, so every system reads
  every document whole. Training dropped 2 of 1,300 train documents: one over
  the input budget, one over the 2,048-token sequence budget.
- **Milestone 5 was skipped** (`docs/milestone-5-skipped.md`), so there is no
  bf16 fine-tune row.

## Test split

All 200 test documents, greedy decoding, `max_new_tokens` 512. Micro-averaged,
in percent.

| run | P | R | F1 | parse failures |
|---|---:|---:|---:|---:|
| always-empty | *pending* | *pending* | *pending* | *pending* |
| zero-shot | *pending* | *pending* | *pending* | *pending* |
| 3-shot | *pending* | *pending* | *pending* | *pending* |
| QLoRA fine-tune | *pending* | *pending* | *pending* | *pending* |

Published systems on the same split and metric (see
`docs/milestone-4-comparison.md`, *Published results*):

| system | model | P | R | F1 |
|---|---|---:|---:|---:|
| TempGen | BART-large | 63.7 | 37.4 | 47.2 |
| GTT | BERT-base | 61.7 | 42.4 | 50.2 |
| IterX | T5-large encoder | 60.9 | 46.9 | 53.0 |

## Dev split

The development record is `docs/milestone-4-comparison.md`. In short, on the
200 dev documents the fine-tune scored **44.5** micro-F1 (P 50.6 / R 39.7)
against 20.0 for 3-shot and 18.6 for zero-shot, with parse failures down from
5–7.5 % to 0.5 %.

## Lessons

- **Check what the chat template renders into the training target.** Unsloth's
  copy of the 4B template put an empty `<think>` block into every assistant
  turn, and the first 4B probes generated nothing. Training now uses the base
  model's official template, and a guard refuses any completion that is not
  exactly the target.
- **Evaluation, not training, dominates wall time.** Each eval point on 50 dev
  documents took about four minutes, a quarter of the 4B run.
- **A 50-document eval point tracks the trend, not the level.** The same
  predictions scored 37.5 on the first 50 dev documents and 44.5 on all 200.
- **Fine-tuning bought precision more than volume.** The fine-tune predicted
  only slightly more items than 3-shot, but more than twice as many of them were
  right. Recall stays the limit, as it does for GTT.
- **One run against one measures nothing small.** Without the seed-to-seed
  spread, a gap of a few points between two runs has no meaning; that is why
  milestone 5 was skipped.

## Where to go next

`docs/TODO.md` lists round two, ordered by expected gain: stopping repetition
loops, training longer, and auditing how the prompt decides which documents
hold events.
