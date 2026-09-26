# 12 Pin the training stack

Type: grilling
Status: resolved
Blocked by: —
HITL: yes

## Question

`train.ipynb` installs whatever `unsloth` is newest (`pip install unsloth`
when absent), while `baselines.ipynb` pins `vllm==0.29.0`. Colab resolved
the same training stack in milestone 3, in both probes and in the diagnostics:
unsloth 2026.9.11, unsloth_zoo 2026.9.7, trl 0.24.0, transformers 5.5.0,
peft 0.20.0, bitsandbytes 0.50.2, torch 2.11.0+cu128. Two things make drift
matter now: ticket 11 showed that Unsloth's bundled files change behaviour, and
ticket 07 made resume a new session that reinstalls the stack, so a release
between a run and its resume would change the run mid-way. Should the install
cell pin the stack, which packages, how, and what happens to the pin after
milestone 4?

## Answer

Resolved 2026-09-26 by grilling (all recommendations accepted).

1. **Pin the resolved stack exactly** in `train.ipynb`'s install cell:
   `unsloth==2026.9.11 unsloth_zoo==2026.9.7 trl==0.24.0 transformers==5.5.0
   peft==0.20.0 bitsandbytes==0.50.2`. These are the versions every GPU run so far resolved to.
   The cell drops its `find_spec("unsloth") is None` guard: a pinned
   install is a quick no-op when satisfied, and it corrects a mismatched preinstall.
2. **torch is not pinned** (it belongs to Colab's image). Instead, **`--resume` refuses if
   the running torch differs from the resumed run's `versions.torch`**, naming both.
   This amends ticket 07's contract. With the pins, torch is the only package that can still drift.
3. **Kept honest by a CPU test:** `tests/test_notebook.py` asserts that the install
   cell pins every package in `train.VERSIONED` except torch, so the stamp list and
   the pin list cannot diverge. There is no new file: GPU dependencies stay in the notebook
   (AGENTS.md).
4. **After milestone 4** the pins stay until a deliberate, explained bump.
   Milestone 5's rented GPU decides its own stack. The install cell's comment says so.
