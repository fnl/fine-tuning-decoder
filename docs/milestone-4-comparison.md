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
  [QLoRA fine-tune `y72d51il`](https://wandb.ai/flowing/muc4-event-extraction/runs/y72d51il)
  (eval run, 217 s on a T4), trained by
  [`wef2kzoo`](https://wandb.ai/flowing/muc4-event-extraction/runs/wef2kzoo) (training run).

## Results

Micro-averaged over the event type and the five roles, in percent.

| run | P | R | F1 | parse failures |
|---|---:|---:|---:|---:|
| always-empty (`rl997k8g`) | 0.0 | 0.0 | 0.0 | 0.0 |
| zero-shot (`meduja8a`) | 18.0 | 19.2 | 18.6 | 7.5 |
| 3-shot (`t9nr1gsf`) | 23.5 | 17.4 | 20.0 | 5.0 |
| QLoRA fine-tune (`y72d51il`) | 50.6 | 39.7 | **44.5** | 0.5 |

Per-role F1 (the event type and the five roles) and the diagnostics, in percent; the last
two columns are counts.

| run | event type | PerpInd | PerpOrg | Target | Victim | Weapon | relevance | event count | event type acc. | cut-off outputs | truncated documents |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| always-empty | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 42.0 | 42.0 | 0.0 | 0 | 0 |
| zero-shot | 38.4 | 11.7 | 14.1 | 16.5 | 4.8 | 19.0 | 63.5 | 50.0 | 53.0 | 3 | 0 |
| 3-shot | 38.5 | 14.2 | 23.9 | 11.1 | 5.7 | 17.2 | 66.0 | 50.5 | 56.6 | 1 | 0 |
| QLoRA fine-tune | 63.6 | 27.3 | 35.1 | 43.5 | 55.2 | 34.7 | 83.5 | 64.5 | 89.3 | 1 | 0 |

Under this metric, the published test-split figures are 50.2 for GTT
(Du et al., 2021) and 53.0 for IterX (Chen et al., 2023), the best we found
(see *Published results*). They are context only and not comparable to these
dev-split rows.

The numbers rest on 200 documents, 116 of which hold events: 731 gold items to
recall (181 event types and 550 role entities), so one point of recall is about
seven items. The fine-tune's gap to 3-shot is 24.5 points: it recalls 290 gold items
where 3-shot recalls 127. That is far above the ~5-point threshold, so no
paired bootstrap is needed.

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

On the first 50 dev documents, NF4 in process scored 40.5 (`wef2kzoo`, last
eval point) and fp16 served by vLLM scored 37.5 (`y72d51il`, rescored locally
on the same 50). The difference is −3.0, with no significance claim.

## Where the points go

Computed from the eval run's predictions (artifact `qwen3-4b-r16-eval-dev` of
`y72d51il`) with `.scratch/milestone-4-full-fine-tune/probes/error_analysis.py`.
These numbers are not logged to W&B. Gold items are counted as the scorer
counts them (731 on dev, about 7 per point of recall). The role rows pool
entities per document, ignoring event alignment.

| cause | cost | detail |
|---|---|---|
| event documents answered `[]` | 93 gold items, ≈ 12.7 points of recall | 21 of the 116 documents with events. Most are political news that mention an incident in passing (median 345 words, against 255 where events were found). Some are annotation oddities: `TST2-MUC4-0031`, an army-against-army clash the prompt excludes, is gold "attack", and three gold "attacks" have no role fillers. The opposite error is half as common: 11 event-free documents were given events. |
| one repetition loop | 23 gold items, ≈ 3.1 points of recall | `TST1-MUC3-0073` (three bombings, 18 role entities) repeats `"bomb"` until it is cut off at 512 tokens: the run's only cut-off output and only parse failure. The targets do not teach it; 2 of 2,809 train entities repeat a mention. |
| event splitting | — | Of the 94 event documents answered with events, 56 get the event count right, 23 too few, 15 too many. |
| PerpInd recall | 85 of 149 PerpInd entities missed | The hardest role for GTT too (44.0). |
| span boundaries | 25 near misses | "orlando zepeda" for gold "colonel orlando zepeda", "bus" for "buses", "two suspects" for "suspects". |

Not causes: made-up text (6 of 548 predicted mentions are not in the
document), parse failures (the one loop above) and truncated documents (none).

## Published results

Context for milestone 6's `RESULTS.md`, not a target (DESIGN §1). Every
figure below is on the **test** split; ours are on dev until milestone 6.
*(Milestone 6, 2026-09-27: the test-split figures are in `docs/RESULTS.md`.)*

**Three metrics circulate, and only one of them is ours.** Figures under
different metrics never go in one table.

- **GTT micro-F1**, called *CEAF-REE_impl* by Chen et al. (2023): the GTT
  scorer that `eval.py` ports. Per-document template alignment and
  any-mention matching, micro-averaged over the event type and the five roles.
  GTT, TempGen and IterX all report it, so it is the one we compare against.
- **CEAF-RME** (Chen et al., 2023): a corrected CEAF variant that scores
  mentions against coreferent entities and leaves out the event type. It gives
  the same systems far lower numbers (GTT: 50.2 under ours, 32.3 under CEAF-RME),
  and newer papers, the LLM ones included, report only this one.
- **Template detection F1** (Gantt et al., 2023): whether the right number and
  types of templates are found. It is an error analysis.

**Under our metric** (test split, micro P / R / F1):

| system | encoder / model | P | R | F1 | source |
|---|---|---:|---:|---:|---|
| TempGen | BART-large | 63.7 | 37.4 | 47.2 | Chen et al. 2023, Table 3 |
| GTT | BERT-base | 61.7 | 42.4 | 50.2 | Du et al. 2021, Table 2 |
| IterX | BERT-base | 52.3 | 51.1 | 51.7 | Chen et al. 2023, Table 3 |
| **IterX** (best) | T5-large encoder | 60.9 | 46.9 | **53.0** | Chen et al. 2023, Table 3 |

GTT's per-role test F1 (Du et al. 2021, Table 1), for comparison with our
detail table: event type 67.4, PerpInd 44.0, PerpOrg 41.8, Target 32.4,
Victim 54.1, Weapon 59.7.

**Under CEAF-RME** (test split, F1): GTT 32.3, IterX
(T5-large) 35.2. The ThinkTwice systems are fine-tuned reasoning LLMs. With
greedy decoding, Qwen3-32B scores 36.0 and DeepSeek-R1-Distill-Llama-70B 28.5.
Sampling candidates and choosing one with a trained reward model lifts them to
**42.5** and 41.1, the best published result we found (Zubillaga et al. 2026,
Table 2).

**Small fine-tuned decoders, on a different task** (added 2026-09-27; this
note first said no such result had turned up). Olsen et al. (2026) fine-tune
Qwen3 at 0.6B, 4B, 8B and 14B (LoRA r16, thinking disabled) and OLMo-3-7B on
MUC-4 with the same 1,300 / 200 / 200 split. Their task is not ours. They keep
only Attack and Bombing events and predict up to 20 fields per event, adding
date, location, counts and categorical fields to the five span roles. They
align events with CEAF-RME and report field-level exact-match micro-F1 and
event-detection F1. Qwen3-4B scores 31.7 exact-match F1 (P 21.4 / R 61.1),
0.6B 20.6, 8B 49.9 and 14B 50.0 (Table 3). None of these figures align with
the tables above.

**Where we stand.** The fine-tune's 44.5 (P 50.6 / R 39.7) on the 200 *dev*
documents is 5.7 points below GTT and 8.5 below IterX, and like GTT it loses most
on recall. Dev and test are different documents, so only milestone 6's
test-split figure may stand beside it. *(Milestone 6, 2026-09-27: on test the
fine-tune scores 53.9, on par with IterX's 53.0 and GTT's 50.2; see
`docs/RESULTS.md`.)*


**Why the gap is not only the model.** Gantt et al. (2023) re-annotated 42
MUC-4 documents and found that experts disagree on how to split incidents into
templates. They also found that models barely split them at all. Merging each
system's predicted templates into one per type costs about a point, but doing
the same to the gold templates costs 12.7 (100 → 87.3). The noise ceiling is
unknown (DESIGN §4).

References:

- Du, Rush, Cardie. *Template Filling with Generative Transformers.* NAACL 2021.
  <https://aclanthology.org/2021.naacl-main.70>
- Chen, Gantt, Gu, Chen, White, Van Durme. *Iterative Document-level
  Information Extraction via Imitation Learning.* EACL 2023.
  <https://aclanthology.org/2023.eacl-main.136>
- Gantt, Kriz, Chen, Vashishtha, White. *On Event Individuation for
  Document-Level Information Extraction.* Findings of EMNLP 2023.
  <https://aclanthology.org/2023.findings-emnlp.862>
- Olsen, Velldal, Øvrelid. *MUC-4 Revisited: Document-level Event Analysis
  Beyond Span-based Arguments.* LREC 2026, pages 7766–7780.
  <https://aclanthology.org/2026.lrec-1.617.pdf>
- Zubillaga, Sainz, Lopez de Lacalle, Agirre. *Do not be greedy, Think Twice:
  Sampling and Selection for Document-level Information Extraction.*
  AACL-IJCNLP 2026. <https://arxiv.org/abs/2601.18395>

## What surprised us

- **The think block.** The 4B's first probes generated nothing but `""`.
  Unsloth's copy of the chat template renders an empty `<think>` block into
  every assistant turn, so every training target began with one. Training now
  adopts the base model's official template, and a guard refuses any completion
  that is not exactly the target plus `<|im_end|>` (ticket 11).
- **The parse-failure spike did not come back.** The 0.6B smoke run's 38 %
  spike at step 12 was repetition loops. The 4B run logged a predictions table
  and the cut-off count at every eval point, and neither ever showed more than
  1 of 50: parse failures were 2 % at the first eval point and 0 % after it
  (ticket 05).
- **The NF4-vs-fp16 gap changes sign.** The 12-step probe scored 6.1 points
  higher when served in fp16 (15.8 → 21.9, `ecgsh42q`), and the finished adapter
  3.0 lower. On 50 documents that is noise either way.
- **The first 50 dev documents are harder than the other 150.** The same
  predictions score 37.5 on the first 50 and 44.5 on all 200. Target looked
  like the weakest role in training (17.6 at the last eval point) but is one
  of the strongest on all 200 (43.5). The training callback tracks the trend,
  not the level.
- **Fine-tuning bought precision more than volume.** The fine-tune predicts
  573 items and 3-shot 541, but 290 of the fine-tune's are right against 3-shot's
  127. Both stay well short of the 731 gold items, so recall is still the limit,
  as it is for GTT. Our weakest roles are PerpInd (27.3), Weapon (34.7) and
  PerpOrg (35.1); GTT's test figures for them are 44.0, 59.7 and 41.8.
