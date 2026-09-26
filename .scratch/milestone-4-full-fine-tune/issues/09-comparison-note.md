# 09 The comparison note

Type: grilling
Status: resolved
Blocked by: 08
HITL: yes

## Question

Milestone 4 ends with a written comparison. Where does it live
(`docs/`, the adapter's Hub model card, a W&B report?), what goes in it
(micro-F1, per-role F1, diagnostics vs zero-shot 18.6 / 3-shot 20.0 /
always-empty, published GTT ≈ 50–55; training curve; the NF4 caveat from 04),
who writes it (generated from W&B or hand-written), and how `RESULTS.md` in
milestone 6 absorbs it?

## Answer

Resolved 2026-09-26 by grilling (all recommendations accepted).

1. **Where:** a markdown file in the repo, `docs/milestone-4-comparison.md`.
   Not a W&B report (not in git) and not the model card (describes the
   adapter, not the experiment). It links out to the W&B runs as sources.
2. **Who:** hand-written; numbers copied from W&B run summaries, each citing
   its run id. No generator script. If milestone 6 needs one for `RESULTS.md`,
   it is built there.
3. **Tables** (all rows on 200 dev docs, same vLLM path): rows are
   always-empty, zero-shot, 3-shot and the QLoRA fine-tune.
   - *Headline table:* micro P/R/F1 and parse-failure rate.
   - *Detail table:* per-role F1 (event type + 5 roles), the diagnostics, and
     the counts of cut-off outputs and truncated documents.
   - **Published GTT ≈ 50–55 gets no row.** A sentence below the table says
     it is a test-split figure, so it is context, not a comparable result.
4. **Beyond the tables:**
   - training-run facts: steps, wall time, resumes, dropped docs, peak memory
   - loss curve and eval-point F1, as **links** to W&B panels (no committed
     images)
   - the NF4-vs-fp16 cross-check line
   - a short "what surprised us" section (e.g. ticket 11's empty outputs, the
     parse-failure spike), which is raw material for milestone 6's lessons
5. **NF4-vs-fp16 cross-check:** a local CPU step,
   `eval.py --pred <eval JSONL, first 50 rows> --gold <dev gold, first 50>`
   without `--wandb`. Its micro-F1 is set against the training run's final
   eval point (same 50 docids). The note reports both numbers and their
   difference in one line, with no significance claim. The spec states the
   exact commands. If the eval fell back to the NF4 engine, the note says the
   check does not apply.
6. **Uncertainty:** no bootstrap in milestone 4. The note gives the raw
   numbers and one sentence on the document count. If the gap to 3-shot is
   under ~5 points, the note lists a paired bootstrap as a milestone-6 item.
7. **`RESULTS.md` absorption:** the note stays frozen as milestone 4's
   dev-split record. `RESULTS.md` (test split) summarises it in a paragraph
   and links to it. Whether milestone 5's bf16 run gets its own note or a
   section in this one is decided in milestone 5.
8. **Model card:** not in milestone 4; the trainer's auto-pushed card stays.
   A proper card belongs with the final numbers in milestone 6.
9. **Timing and spec content:** the spec fixes the note's path, headings and
   table columns. The **build** commits the skeleton with the three baseline
   rows filled in, citing their W&B run ids. **After the run**, the fine-tune
   row, training facts, cross-check line and surprises are filled in, and
   that commit closes milestone 4. Acceptance: no placeholder left in the
   note.
10. **Glossary:** added **Comparison note** to `CONTEXT.md`
    (Models and experiments).
