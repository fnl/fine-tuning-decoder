# Map: Milestone 3 — smoke fine-tune of Qwen3-0.6B on 100 documents

Label: wayfinder:map
Created: 2026-09-22
Source: `docs/DESIGN.md` §7, §9, §10, §11 (milestone 3); charting interview 2026-09-22

## Destination

A spec ready for `/implement` that reaches DESIGN §11 milestone 3: `src/train.py`
fine-tunes `Qwen3-0.6B` on the first 100 train documents with completion-only
loss, an eval callback that runs the real scorer on 50 dev documents every ½
epoch into the same W&B run, and the LoRA adapter pushed to the Hub — driven
from a new `notebooks/train.ipynb` on a free-tier T4. Every setup task and
every fact the spec must state verbatim is settled inside this map; what
follows the map is pure build, and the smoke run itself is what `/implement`
verifies at the end.

## Notes

- Vocabulary: `CONTEXT.md` (smoke model / smoke run / adapter / experiment /
  run / engine / target / diagnostics). Consult before every ticket; the
  masking vocabulary is new and will need entries.
- Skills: `grill-me` + `domain-modeling` for grilling tickets; `prototype` for
  ticket 05; `to-spec` for ticket 12. There is **no** `research` skill in this
  install — research tickets are resolved by a subagent brief and answered in
  the ticket with sources and dates, as in the milestone-2 map.
- Prior art to copy, not re-derive: `.scratch/milestone-2-baselines/` (map,
  tickets, `spec.md`). `src/generate.py`'s `Engine` seam and `src/eval.py`'s
  `score()` / `flatten()` / W&B payload are designed to be reused here.
- Decisions locked in the charting interview (do not re-open):
  - **Destination is a spec**, not an executed run (Q1).
  - **Two notebooks** (Q2): new `notebooks/train.ipynb` for the Unsloth stack;
    `colab.ipynb` renamed `baselines.ipynb` for the vLLM stack. Unsloth and
    vLLM cannot share an environment (M2 ticket 05).
  - **Masking = our own pre-tokenisation** (Q3): `tokenize_example` renders
    prompt (`messages[:2]`, `add_generation_prompt=True`) and full
    (`messages`), asserts the token-level prefix, emits `input_ids` + `labels`
    with the prompt span `-100`; TRL consumes a processed dataset. Fallback if
    Unsloth re-tokenises: TRL's prompt-completion form (`completion_mask`),
    same arithmetic. Rejected: `assistant_only_loss` (TRL ≤0.24 cannot patch
    Qwen templates, and TRL 1.13's patched Qwen3 template trains the `<think>`
    block the inference prompt already supplies) and Unsloth's
    `train_on_responses_only` (same `<think>` skew, Unsloth-bound, untestable
    on CPU).
  - **Eval callback** (Q4) = `render_inputs` + an in-process engine + a
    `parse_target` + `score()` pass on the dev subset, logged to the same W&B
    run under a `dev/` prefix; cadence from a YAML `eval_every` in epochs.
  - **One W&B run per training run** (Q5), `job_type=train`; the M2-style
    separate eval run reappears in milestone 4.
  - **Checkpoint at every eval point and push every save** (Q6) — the Colab
    disconnect insurance milestone 3 exists to prove; explicit `adapter:` key
    in the YAML; adapter public, never merged.
  - **NF4 stays on for the smoke run** (Q7): it exercises the exact path the
    4B run will take. Knobs `quantization: nf4 | none` and a `dtype` chosen
    from the GPU.
  - **First N rows** of each split for the subsets (Q8) — the first 100 train
    documents are 40 empty / 40 one-event / 15 two / 5 three, the first 50 dev
    documents 25 empty / 25 relevant; same semantics as `generate --limit`.
  - **CPU-testable split** (Q9): pure functions (config, subset, masking, step
    arithmetic) tested with the real tokenizer; the callback tested with fakes;
    Unsloth/TRL glue imported lazily and untested locally.
  - `train.py` accepts the full-split config shape from the start so milestone
    4 is "write one YAML and run" (Q10).
- Facts established while charting (2026-09-22, do not re-derive):
  - `unsloth 2026.9.9` pins `trl!=0.19.0,>=0.18.2,<=0.24.0`,
    `transformers<=5.5.0`, `torch<2.13`; `unsloth_zoo 2026.9.7` allows
    `trl<=1.13.0`, so the intersection lands **TRL 0.24.0**.
  - TRL 0.24.0 accepts a pre-tokenised dataset: `is_processed = "input_ids" in
    column_names` skips tokenisation (`sft_trainer.py:887`), the collator uses
    a supplied `labels` column verbatim (`:160`), `labels` is in
    `_signature_columns` (`:1078`), and `truncate_dataset` slices every list
    column so `labels` stays aligned.
  - TRL 0.24.0 has no `chat_template_utils.py`; the auto-patched Qwen3
    templates exist only from TRL 1.13, out of reach under Unsloth's pin.
  - Neither `Qwen3-0.6B` nor `Qwen3-4B-Instruct-2507` carries `{% generation %}`.
  - The prompt is an exact **token-level** prefix of the full rendering for
    both models (verified on real examples, transformers 5.17): `DEV-MUC3-0001`
    632/705 tokens on 0.6B, 628/701 on 4B. An empty document's completion is 3
    tokens (`[]`, `<|im_end|>`, `\n`); prompts run 444–765 tokens.
  - Milestone 2 results to beat: zero-shot micro-F1 **18.6** (7.5 % parse
    failures), 3-shot **20.0** (5 %), always-empty 0. Published GTT ≈ 50–55.
- User preferences: answers immediately with a recommendation; wants defaults
  justified; only commit when asked.

## Decisions so far

<!-- one line per resolved ticket: gist + link -->

## Not yet specified

- **The fallback path if Unsloth re-tokenises.** If ticket 01/07 shows the
  pre-tokenised dataset does not survive Unsloth's wrapper, the reshape to
  TRL's prompt-completion form becomes its own decision (which column layout,
  where `chat_template_kwargs` goes, how the CPU test changes). Cannot be
  phrased sharply before the answer.
- **Disconnect recovery.** Whether `train.ipynb` gains a resume path (resume
  from the last pushed checkpoint) or simply restarts, and what that costs at
  0.6B. Hangs on what ticket 02 finds about checkpoint contents and
  `resume_from_checkpoint` under Unsloth.
- **Dev-subset comparability.** Whether the callback's 50-document dev score
  is meant to be comparable to the 200-document baselines at all, or is purely
  a training signal. Hangs on ticket 06's acceptance criteria.
- **New vocabulary.** Masking, completion, label span, checkpoint, eval point:
  the terms the spec will use are not in `CONTEXT.md` yet; which of them earn
  glossary entries graduates once tickets 05 and 09 fix the words.

## Out of scope

- The full QLoRA fine-tune of Qwen3-4B and `configs/qwen3-4b-r16.yaml` —
  milestone 4.
- Evaluating a finished adapter M2-style (vLLM + LoRA over all 200 dev
  documents) and the fp16-base / NF4-adapter numerics question it raises —
  milestone 4.
- Colab Pay-As-You-Go, rented GPUs, bf16 LoRA — milestone 5.
- `RESULTS.md` and the written comparison — milestone 6.
- Merged / GGUF export, rank and lr sweeps, the plain-peft rewrite, the
  base-model variant, DocEE — round two (`docs/DESIGN.md` §11).
