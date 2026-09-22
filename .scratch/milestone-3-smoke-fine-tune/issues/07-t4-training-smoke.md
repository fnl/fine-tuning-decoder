# 07 Prove the Unsloth stack and a masked 10-step run on a free T4

Type: task
Status: open
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
