# 10 Write the milestone-4 spec

Type: task
Status: resolved
Blocked by: 06, 07, 08, 09, 12
HITL: yes

## Question

With every decision on this map resolved, write
`.scratch/milestone-4-full-fine-tune/spec.md` using the `to-spec` skill and
`.scratch/milestone-3-smoke-fine-tune/spec.md` as the model: goal, decisions
(pointing at tickets rather than restating them), the code contract
verbatim, testing decisions, out of scope, done criterion, and an order of
work that de-risks earliest.

Before writing, update `CONTEXT.md` with whatever vocabulary tickets 07–09
fixed, and check DESIGN §7–§9 and §11 against the map — correct in place with
dated notes, not silently.

The map closes with this ticket.

## Answer

Resolved 2026-09-26. The spec is [`spec.md`](../spec.md) (`Status: ready-for-agent`).
Before writing it, the user settled six decisions (all recommendations accepted):

1. **`--resume` and `--limit` are mutually exclusive** (argparse error). A
   `--limit` run never pushes, so it has no checkpoint to resume. This amends
   [The resume contract](07-resume-contract.md) item 1.
2. **NF4 zero-shot baseline**: only on the fallback path. If vLLM LoRA fails on
   a T4, the final eval also runs `qwen3-4b-zero-shot-nf4.yaml` through the
   `unsloth` engine, so the comparison keeps one variable. Graduates the map's last
   fog patch into the spec.
3. **Probe-eval YAML and cell are deleted** once the de-risk passes, like the
   resume drill. The final-eval YAML and its notebook cell land in one post-training
   commit, because the notebook test requires every named config to exist.
4. **Done criterion**: one finished train run with 6 eval points (resumes
   inside it), one 200-doc eval run of the pinned revision, and a note with no
   placeholder. No score gate.
5. **Order of work**: hygiene → adapter serving + probe eval (first GPU step)
   → callback → resume → resume drill (GPU) → 4B YAML and notebook →
   note skeleton, README, DESIGN §3. After the build, the spec's run book covers
   the pipeline check, the run and its resumes, the eval, the local cross-check
   and the note.
6. **README restructure** (raised by the user): Fine-tuning rewritten for the
   4B; new *Resuming a run* and *Evaluating an adapter* sections; a *Results*
   pointer added only in the post-run commit; the smoke run cut to one line.

Also done here: dated corrections in DESIGN §7 (template adoption, batch
sizes, cadence, resume and pins, measured time), §8 (final eval via vLLM
fp16 + LoRA), §9 (adapter repo name, eval YAML) and §11 (milestone 4
refined; `RESULTS.md` links rather than absorbs). `CONTEXT.md` needed nothing
new: **Resume**, **Rerun** and **Comparison note** were already there.
