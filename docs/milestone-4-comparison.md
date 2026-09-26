# Milestone 4: the QLoRA fine-tune beside the baselines

Milestone 4's dev-split record: `Qwen/Qwen3-4B-Instruct-2507` fine-tuned with
QLoRA on the whole train split on a Colab T4, set beside the milestone-2
baselines. Every number is copied from the W&B run it cites. The note is frozen
once complete; `RESULTS.md` (milestone 6) links to it.

## Setup

- **Base model:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Adapter:** `fnl-es/qwen3-4b-muc4-lora-r16` at revision TBD, trained
  by `configs/qwen3-4b-r16.yaml`.
- **Served base:** fp16, via vLLM 0.29.0 with the adapter as a LoRA, the
  baselines' own engine, rendering and greedy sampling.
- **Data:** the dev split of `fnl-es/muc4-chat`, all 200 documents, for every row.
- **Runs:**
  [always-empty `rl997k8g`](https://wandb.ai/flowing/muc4-event-extraction/runs/rl997k8g),
  [zero-shot `meduja8a`](https://wandb.ai/flowing/muc4-event-extraction/runs/meduja8a),
  [3-shot `t9nr1gsf`](https://wandb.ai/flowing/muc4-event-extraction/runs/t9nr1gsf),
  QLoRA fine-tune TBD (eval run), trained by TBD (training run).

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

The published GTT figure of ≈ 50–55 F1 is a **test**-split number, so it is
context only and not comparable to these dev-split rows.

The numbers rest on 200 documents, 116 of which hold events: 731 gold items to
recall (181 event types and 550 role entities), so one point of recall is about
seven items. TBD: if the
fine-tune's gap to 3-shot is under ~5 points, a paired bootstrap is a milestone-6
item.

## Training

- **Steps:** TBD of 246 (1,298 kept examples, 3 epochs at effective batch 16).
- **Wall time:** TBD.
- **Sessions and resumes:** TBD.
- **Dropped documents:** TBD (expected 1, over the 2,048-token budget).
- **Peak memory:** TBD.
- **Panels:** the training loss TBD and the eval-point dev F1 TBD.

## NF4 vs fp16

TBD: the training run's last eval point (NF4, in process) against the served
adapter (fp16, vLLM) on the same first 50 dev documents, both micro-F1s and their
difference, with no significance claim.

## What surprised us

TBD.
