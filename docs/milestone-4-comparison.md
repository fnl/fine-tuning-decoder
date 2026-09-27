# Milestone 4: the QLoRA fine-tune beside the baselines

Milestone 4's dev-split record: `Qwen/Qwen3-4B-Instruct-2507` fine-tuned with
QLoRA on the whole train split on a Colab T4, set beside the milestone-2
baselines. Every number is copied from the W&B run it cites. The note is frozen
once complete; `RESULTS.md` (milestone 6) links to it.

## Setup

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Adapter:** `fnl-es/qwen3-4b-muc4-lora-r16` at revision
  `2d3ac04008a333d6d3c1db0ce17990efb00d558e`, trained by
  `configs/qwen3-4b-r16.yaml`.
- **Served base:** fp16, via vLLM 0.29.0 with the adapter as a LoRA, the
  baselines' own engine, rendering and greedy sampling.
- **Data:** the dev split of `fnl-es/muc4-chat`, all 200 documents, for every row.
- **Runs:**
  [always-empty `rl997k8g`](https://wandb.ai/flowing/muc4-event-extraction/runs/rl997k8g),
  [zero-shot `meduja8a`](https://wandb.ai/flowing/muc4-event-extraction/runs/meduja8a),
  [3-shot `t9nr1gsf`](https://wandb.ai/flowing/muc4-event-extraction/runs/t9nr1gsf),
  QLoRA fine-tune TBD (eval run), trained by
  [`wef2kzoo`](https://wandb.ai/flowing/muc4-event-extraction/runs/wef2kzoo) (training run).

## Results

Micro-averaged over the event type and the five roles, in percent.

| run | P | R | F1 | parse failures |
|---|---:|---:|---:|---:|
| always-empty (`rl997k8g`) | 0.0 | 0.0 | 0.0 | 0.0 |
| zero-shot (`meduja8a`) | 18.0 | 19.2 | 18.6 | 7.5 |
| 3-shot (`t9nr1gsf`) | 23.5 | 17.4 | 20.0 | 5.0 |
| QLoRA fine-tune (TBD) | TBD | TBD | TBD | TBD |

Per-role F1 (the event type and the five roles) and the diagnostics, in percent; the last
two columns are counts.

| run | event type | PerpInd | PerpOrg | Target | Victim | Weapon | relevance | event count | event type acc. | cut-off outputs | truncated documents |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always-empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 42.0 | 42.0 | 0.0 | 0 | 0 |
| zero-shot | 38.4 | 11.7 | 14.1 | 16.5 | 4.8 | 19.0 | 63.5 | 50.0 | 53.0 | 3 | 0 |
| 3-shot | 38.5 | 14.2 | 23.9 | 11.1 | 5.7 | 17.2 | 66.0 | 50.5 | 56.6 | 1 | 0 |
| QLoRA fine-tune | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Under this metric, the published test-split figures are 50.2 for GTT
(Du et al., 2021) and 53.0 for IterX (Chen et al., 2023), the best we found
(DESIGN §12). They are context only and not comparable to these dev-split
rows.

The numbers rest on 200 documents, 116 of which hold events: 731 gold items to
recall (181 event types and 550 role entities), so one point of recall is about
seven items. TBD: if the
fine-tune's gap to 3-shot is under ~5 points, a paired bootstrap is a milestone-6
item.

## Training

- **Steps:** 246 of 246 (1,298 kept examples, 3 epochs at effective batch 16).
- **Wall time:** 1 h 48 min (6,446 s in the trainer, 6 eval points included),
  against the 2.25 h estimate.
- **Sessions and resumes:** one session, no resumes. Two other
  `qwen3-4b-r16` train runs exist:
  - [`tockbarz`](https://wandb.ai/flowing/muc4-event-extraction/runs/tockbarz)
    is the `--limit 8` pipeline check, which never pushes;
  - [`pfjiai30`](https://wandb.ai/flowing/muc4-event-extraction/runs/pfjiai30)
    is a start killed at step 14, before its first checkpoint.
- **Dropped documents:** 1, over the 2,048-token budget, as expected.
- **Peak memory:** 8.6 GiB on the whole device (W&B system metrics), under the
  10.2 GiB the milestone-4 map measured (tickets 01, 11).
- **Loss:** 1.67 at step 1 (≈ 1.5 expected); average 0.47 over the first
  quarter and 0.15 over the last.
- **Eval points** (NF4, first 50 dev documents), micro-F1: 26.2, 31.1, 38.8,
  32.9, 39.5, 40.5; 0 parse failures and 0 cut-off outputs at the last.
- **Panels:** the training loss and the eval-point dev F1 are in
  [`wef2kzoo`'s charts](https://wandb.ai/flowing/muc4-event-extraction/runs/wef2kzoo).

## NF4 vs fp16

NF4, in process: 40.5 (`wef2kzoo`, last eval point, first 50 dev documents).
fp16, served by vLLM: TBD (the eval run, the same 50 documents). Difference: TBD,
with no significance claim.

## What surprised us

TBD.
