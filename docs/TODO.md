# Round two

Work deferred past the six milestones (DESIGN §1, §11). Milestone 6 is not
listed here; milestone 5 was skipped (`docs/milestone-5-skipped.md`). Each item says why it is here and what it costs on the free
T4.

## From the milestone-4 error analysis

Ordered by expected gain per GPU hour. The evidence is in
`docs/milestone-4-comparison.md`, *Where the points go*.

1. **Stop repetition loops at generation time.** One looping document
   (`TST1-MUC3-0073`) cost about 3 points of dev recall. Candidates:
   - a mild repetition penalty (it must not suppress legitimately repeated
     words across entities);
   - guided JSON decoding in vLLM;
   - a sampled retry of any cut-off output.

   Cost: eval runs only, about 5 min each. A generation change is a second
   variable, so the baselines are re-run with it before any comparison.
2. **Audit the relevance decision.** 21 event documents answered `[]` cost
   about 13 points of dev recall. Check whether the prompt's "military clashes
   are not events" rule is stricter than the gold annotations (`TST2-MUC4-0031`
   says it is), and how the train split labels incidents mentioned in passing.
   Cost: CPU reading first; a prompt change means a new run.
3. **Sample and select.** Generate several candidates per document and pick one
   by majority, by F1 voting against the others, or with a trained reward model,
   as ThinkTwice (Zubillaga et al. 2026) does to lift a greedy Qwen3-32B by
   6.5 CEAF-RME points. Cost: n times the eval time, and a selector.
4. **Rank and lr sweeps.** r16 at lr 2e-4 is the only point tried. Cost: one
   run per point; do it only after item 2 has settled the epoch count.
5. **Train longer.** GTT trained for 18 epochs; we trained for 3, and our last
   eval point was the best. Try 6 epochs: 12 eval points and about 4 h on a
   T4, so two free-tier sessions and a resume. Choose the adapter by its best
   eval point, not the last; the Hub already keeps a commit for every
   checkpoint. (Note: Unlikely to have significant impact.)

## Carried over from the design and the milestone specs

Possible next steps:
- **DocEE** as a second dataset, behind the existing dataset interface (DESIGN §4).
- **Plain peft/transformers rewrite** of `train.py`, without Unsloth (DESIGN §7).
- **Base-model variant**: fine-tune the base checkpoint instead of Instruct (DESIGN §5).

Possible improvements:
- **One shared config reader** for `generate.py`, `train.py` and `eval.py`
  (milestone-3 ticket 08).
- **The bf16 / `quantization: none` training path**, never run on a GPU
  since milestone 5 was skipped (`docs/milestone-5-skipped.md`). Exercise it
  the first time an experiment runs on an Ampere-or-newer GPU.
- **Merged or GGUF export** of the adapter (milestone-3 and -4 specs).
- **Match GTT on composite event types, or keep documenting it.** GTT accepts
  `attack` against a gold `attack / bombing`; `src/eval.py` requires equality,
  which costs the fine-tune 3 test items (`docs/RESULTS.md`, *Checks on the
  53.9*). The module docstring's claim that the two rules are equivalent is
  wrong, and the parity test does not cover the case.
