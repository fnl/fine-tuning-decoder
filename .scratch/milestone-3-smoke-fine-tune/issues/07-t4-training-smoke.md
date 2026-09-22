# 07 Prove the Unsloth stack and a masked 10-step run on a free T4

Type: task
Status: resolved
Blocked by: 01
HITL: yes

## Question

Milestone 2's lesson: the install and the GPU path diverge from the research
in ways no amount of reading finds (`torchaudio` CUDA mismatch, the working
directory reset after the restart, the vLLM wheel index). That was discovered
*during* implementation and cost three attempts. Here the same risk sits
under a load-bearing design decision, so it is paid down inside the map.

From a scratch Colab notebook on a free T4 (not yet `train.ipynb`; throwaway
cells are fine), using ticket 01's install line:

1. Install the Unsloth stack, restart, and record the resolved versions of
   `unsloth`, `unsloth_zoo`, `trl`, `transformers`, `torch`, `peft`,
   `bitsandbytes` — and every clash that had to be worked around.
2. Load `Qwen3-0.6B` 4-bit with a LoRA config per DESIGN §7 and report the
   memory footprint.
3. Build ~20 examples with **our** `tokenize_example` (ticket 05's function,
   pasted in), hand them to `SFTTrainer` as a pre-tokenised dataset, and run
   ~10 steps. **The decisive observation**: does the dataset survive — are
   `labels` still there at the collator, is the loss finite and consistent
   with completion-only loss, does Unsloth warn or re-tokenise? If it does
   re-tokenise, record exactly how it fails.
4. Generate from the model mid-run (ticket 03's recipe) on two documents and
   confirm output comes back and training continues afterwards.
5. Save the adapter and push it to a throwaway Hub repo; confirm the repo
   contents and delete it.

Record versions, memory, wall times, workarounds and failures under
`## Answer`. This is the ticket that turns Q3 from a decision into a fact,
and it unblocks the code contract (09) and the notebook layout (11).

## Answer

**Closed 2026-09-22 without being run — its premise changed.** Ruled out of
the map's scope, not resolved on the route.

This ticket existed to buy down one risk: that the pre-tokenised dataset would
not survive Unsloth's wrapper, discovered only after the spec was written
(milestone 2's install lesson). Ticket 01 settled that from source instead —
`unsloth_zoo/dataset_utils.py:2582-2588`, `if "labels" in column_names:
do_tokenize = False`, with the live collator being TRL 0.24's own. A file:line
in the code that will actually run is stronger evidence than one T4 session,
and it arrived without needing a GPU.

The second justification also evaporated: ticket 01 found Colab's torch 2.11.0
satisfies every Unsloth bound, so there is **no torch swap, no restart, no
`torchaudio` clash** — none of the failure modes that cost milestone 2 three
attempts apply here.

What this ticket would still have produced — peak memory, per-callback wall
time, end-to-end timings — are figures the spec does not state verbatim and
that the first `/implement` run produces as a by-product. They are estimates
in tickets 01, 02 and 03, each flagged as such.

**So it becomes step 1 of the implementation order of work**, not a map
ticket: the first thing `/implement` does is run the smoke config with
`--limit` on a T4 and confirm the loss is finite, the labels survive to the
collator, and generation works mid-run — before any of the remaining code is
written. Ticket 12 states this.

Tickets 09 and 11 proceed on ticket 01's source-verified answer. If it turns
out to be wrong on a real T4, the fallback is already documented: TRL's
prompt-completion form, same arithmetic (map Notes, Q3).
