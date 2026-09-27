# Results

**Draft, 2026-09-27.** The test-split runs (milestone 6) and the checks on
them are in; the lessons come from the dev work (milestone 4).

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

All 200 test documents (812 gold items), greedy decoding, `max_new_tokens` 512,
commit `a82e74f`, on a Colab T4 (always-empty on CPU). Micro-averaged, in percent, using `Qwen/Qwen3-4B-Instruct-2507` (except always-empty).

| run | P | R | F1 | parse failures |
|---|---:|---:|---:|---:|
| always-empty ([`kyngxn5r`](https://wandb.ai/flowing/muc4-event-extraction/runs/kyngxn5r)) | 0.0 | 0.0 | 0.0 | 0.0 |
| zero-shot ([`1zg4m43j`](https://wandb.ai/flowing/muc4-event-extraction/runs/1zg4m43j)) | 18.3 | 25.9 | 21.4 | 4.5 |
| 3-shot ([`dnnt66g6`](https://wandb.ai/flowing/muc4-event-extraction/runs/dnnt66g6)) | 23.5 | 17.5 | 20.0 | 10.5 |
| QLoRA fine-tune ([`pqizcifd`](https://wandb.ai/flowing/muc4-event-extraction/runs/pqizcifd)) | 60.3 | 48.8 | **53.9** | 0.5 |

Published systems on the same split and metric (see
`docs/milestone-4-comparison.md`, *Published results*):

| system | model | P | R | F1 |
|---|---|---:|---:|---:|
| TempGen | BART-large | 63.7 | 37.4 | 47.2 |
| GTT | BERT-base | 61.7 | 42.4 | 50.2 |
| IterX | T5-large encoder | 60.9 | 46.9 | 53.0 |

**The fine-tune is on par with the best published systems under this metric.** A QLoRA-tuned 4B decoder, trained for 3 epochs on a free
T4, reaches 53.9 test micro-F1, beside IterX's 53.0 and GTT's 50.2. Its
precision (60.3) matches theirs and its recall (48.8) is the highest of the
four. One run cannot separate 53.9 from 53.0:

- **Document sampling alone spans ten points.** A percentile bootstrap over
  the 200 test documents (10,000 resamples, `python -m bootstrap`) gives a 95 %
  interval of **48.8 to 58.9**. 36 % of the resamples fall below IterX's 53.0
  and 7 % below GTT's 50.2. These shares are not significance tests: a paired
  comparison needs the published systems' per-document predictions, and their
  figures carry sampling noise of their own.
- **The interval is a lower bound on the uncertainty.** It covers which
  documents are in the split, not training randomness. The spread between
  seeds is unmeasured, so this is one draw from it.
- **Most of the gain over dev is the split.** Dev and test are different
  documents, and test is the denser split (812 gold items against 731). The
  same adapter scored 44.5 on dev, and the baselines moved only a point or two
  between the splits (zero-shot 18.6 to 21.4, 3-shot unchanged at 20.0).

**We have no figures for how the state of the art is reported today.** This metric
tops out at IterX (2023). Newer papers on the standard task, the fine-tuned
LLMs among them, report only CEAF-RME, where the best result we found is 42.5
(ThinkTwice, Qwen3-32B with a trained reranker;
`docs/milestone-4-comparison.md`). We have not
computed CEAF-RME, so we make no claim against those systems. Fine-tuned
Qwen3 at 0.6B to 14B, the 4B among them, has been evaluated on MUC-4 by Olsen
et al. (LREC 2026), but on an expanded task: Attack and Bombing only, up to 20
fields per event, and field-level exact-match scoring. Their figures are not
comparable with ours and stay out of the tables
(`docs/milestone-4-comparison.md`).

**Checks on the 53.9.**

- **Reproducible.** Rescoring the logged prediction artifact of `pqizcifd`
  on CPU gives 53.91 again.
- **No test documents in our fine-tuning data.** No test docid is in train or
  dev. Only one test document, `TST3-MUC4-0087`, shares most of its text with
  a train document, and it has no gold events; leaving it out keeps 53.91.
- **Pretraining contamination is unknown.** The split check says nothing about
  what Qwen3 saw in pretraining. MUC-4 is a public benchmark from 1992, and
  the GTT preprocessing, texts and annotations together, is on GitHub, so the
  test documents and their answers may be in the pretraining data. The zero-shot baseline's 21.4 shows the base
  model does not reproduce the test annotations unprompted. That is weak
  evidence: it does not rule out exposure that fine-tuning then draws on. The
  published encoders (BERT, BART, T5) were pretrained on web text as well, but
  on less of it and from earlier.
- **Our scorer understates the score slightly.** The original GTT `eval.py`
  cannot score documents with many events, so the check ran on the 197 test
  documents with at most four gold and four predicted events. There, the
  original counts 383 correct items where our port counts 380 (F1 55.3
  against 54.8). The gap is the event type. A few gold events carry a
  composite type, `attack / bombing` or `bombing / attack` (five events in
  test, one in dev). GTT accepts a predicted type that is a substring of the
  gold type, so `attack` matches `attack / bombing`; `src/eval.py` requires the
  types to be equal, on the mistaken belief (stated in its docstring) that the
  two rules agree on the six-type vocabulary. The three items are all in
  `TST3-MUC4-0056` and `TST3-MUC4-0061`, where the fine-tune predicted
  `attack` for an `attack / bombing` event. Scored with GTT's rule on all 200 documents, the
  test figures are:

  | run | P | R | F1 |
  |---|---:|---:|---:|
  | zero-shot | 18.5 | 26.2 | 21.7 |
  | 3-shot | 24.3 | 18.1 | 20.8 |
  | QLoRA fine-tune | 60.7 | 49.1 | 54.3 |

  The tables above keep our scorer's figures, which are what the W&B runs
  log. The difference is at most 0.7 points (3-shot) and 0.4 for the
  fine-tune, which changes nothing in the comparison. The published systems were scored with GTT's rule,
  so the comparable figure for the fine-tune is 54.3. The parity test missed
  this: its one composite dev event is either kept unchanged or dropped by the
  perturbations, so no single predicted type is ever scored against it.

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

## Where to go next

`docs/TODO.md` lists future work, ordered by expected gain: stopping repetition
loops, training longer, and auditing how the prompt decides which documents
hold events.
