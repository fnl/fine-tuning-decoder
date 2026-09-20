# 02 Set up Colab free tier with secrets

Type: task
Status: resolved
Blocked by: 01
HITL: yes

## Question

The user has no Colab setup. The notebook (DESIGN §9) reads `HF_TOKEN` and
`WANDB_API_KEY` from Colab Secrets; both must exist and a T4 runtime must be
obtainable before the notebook's secrets cell can be designed.

Checklist for the user:

1. Open https://colab.research.google.com with the Google account you'll use;
   accept the free tier (no Pay-As-You-Go for this map).
2. New notebook → Runtime → Change runtime type → **T4 GPU**. Run
   `!nvidia-smi` and confirm a Tesla T4 with ~15 GB.
3. Left sidebar → 🔑 Secrets → add `HF_TOKEN` (a *write* token from
   https://huggingface.co/settings/tokens, needed later for adapter pushes)
   and `WANDB_API_KEY` (from ticket 01). Toggle "Notebook access" on.
4. In a cell: `from google.colab import userdata; print(bool(userdata.get('HF_TOKEN')), bool(userdata.get('WANDB_API_KEY')))` → `True True`.
5. Note the Python and CUDA versions (`!python -V; !nvcc --version`) —
   ticket 05/06 need them.

Resolved when steps 2 and 4 succeed. Record: GPU seen, Python/CUDA versions.

## Answer

Resolved by the user 2026-09-20. Colab free tier on the user's Google
account; `HF_TOKEN` and `WANDB_API_KEY` added as Colab Secrets with notebook
access. Runtime seen: Python 3.13.15, nvcc CUDA 12.8 (V12.8.93, built
2025-02-21). GPU not recorded (T4 assumed per free tier). Note: Python 3.13
is newer than the 2026.07 snapshot in ticket 05 (3.12.13) — vLLM 0.29.0
supports <3.15, so fine; check `torch.__version__` and `nvidia-smi` driver
in-notebook before choosing the wheel index (ticket 06).
