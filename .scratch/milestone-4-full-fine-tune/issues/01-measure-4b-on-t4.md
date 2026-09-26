# 01 Measure the 4B on a T4

Type: task
Status: resolved
Blocked by: —
HITL: yes

## Question

Every cadence and batch-size decision on this map waits on four numbers we
only have for the 0.6B: step time, callback time, peak memory, and whether
the stack still resolves the same. Produce them for `Qwen3-4B-Instruct-2507`
on a free Colab T4, NF4, fp16:

- **Training**: seconds per optimizer step and peak GPU memory at
  `per_device_batch_size` 2 and 4 (effective batch 16, `max_seq_len` 2048),
  over enough steps to include the longest examples — or state the worst
  case explicitly.
- **Callback**: wall time and peak memory of one in-process scoring pass over
  the 50-document dev subset at generation `batch_size` 4 and 8,
  `max_new_tokens` 512; how many outputs were cut off.
- **Stack**: the versions `train.py` stamps (`versions` in the W&B config),
  and whether the config-nesting fix (`a4a0a5e`) produces `experiment` /
  `derived` in the run config as intended.

The agent writes the probe (a throwaway YAML such as
`configs/qwen3-4b-probe.yaml` with a small `train_docs` and `dev_docs`, or a
notebook cell) and a checklist; the human runs it on Colab; the agent reads
the numbers back from W&B. Record the run ids. Also confirm whether the
milestone-3 load-back cell generated.

## Comments

### 2026-09-26: the probe and the Colab checklist

Probe: `configs/qwen3-4b-probe-b2.yaml` (per-device 2, generation batch 4) and
`configs/qwen3-4b-probe-b4.yaml` (per-device 4, generation batch 8). Both use
the first 192 train documents (12 steps, 1 epoch). That range holds the longest kept example
(1,807 tokens, train index 184; the only example over 2,048 is index 1148,
2,078 tokens, dropped). Every epoch therefore meets the worst-case batch. Each run
has one eval point and one checkpoint push at step 12 (50 dev docs,
`max_new_tokens` 512). Both push to the throwaway repo
`fnl-es/qwen3-4b-muc4-lora-probe`, and W&B tags them `probe`.

Checklist (human, free Colab T4):

1. Commit and push the two YAMLs to `origin/main` (the notebook clones from GitHub).
2. Open `notebooks/train.ipynb` on a T4 runtime and run cells 1–5 (GPU,
   install, clone at `REF = "main"`, versions, secrets). Note the printed
   versions.
3. Add a cell and run `!python -m train --config configs/qwen3-4b-probe-b4.yaml`.
   If it OOMs, note where (training step or the callback) and the traceback's last line.
4. Add a cell and run `!python -m train --config configs/qwen3-4b-probe-b2.yaml`.
5. Report both W&B run ids (or URLs) and the wall time of each cell. If a run
   died before W&B started, paste the output tail instead.
6. In the milestone-3 Colab notebook's saved output, check whether cell 8
   (load-back) printed a `gold:` / `output:` pair, and say whether it did.

The agent then reads back from W&B: step time from the `_timestamp` gaps
between `train/loss` rows, callback wall time from the gap at step 12, peak
memory from `system/gpu.0.memoryAllocatedBytes` split by phase, cut-offs
from the `cut_off` column of the `predictions` table, and `versions` /
`experiment` / `derived` from the run config.

### 2026-09-26: checklist step 1 done; GPU refused

Probe YAMLs pushed to `origin/main` as `0409d6d`. The agent then tried to
connect Colab (driving Chrome), but got "Cannot connect to GPU backend: You cannot currently
connect to a GPU due to usage limits in Colab". This is the free-tier lockout
from ticket 02, one day after the milestone-3 GPU work. To get past it, the
user bought Colab credits and ran steps 2–6 themselves on a paid T4.
It's the same hardware, so the measurements hold for the free tier. The design
constraint is unchanged: the milestone must be doable without credits.

## Answer

Resolved 2026-09-26 from W&B runs `4lkcxlgt` (`qwen3-4b-probe-b4`: per-device
4, generation batch 8) and `wk6ypti0` (`qwen3-4b-probe-b2`: 2 / 4), Tesla T4,
fp16, NF4, git `0409d6d`. **Training numbers are valid; callback numbers are
not.** Both callbacks produced **empty outputs for all 50 documents**, split off
as ticket 11.

**Training** (12 steps each, the worst-case 1,807-token example in range):

| | per-device 4 (accum 4) | per-device 2 (accum 8) |
|---|---|---|
| s / optimizer step (mean of steps 2–11) | ≈ 25 (21–31) | ≈ 22 (20–26) |
| GPU memory, training plateau | 5.8 GiB | 5.8 GiB |

Memory is flat and identical at both batch sizes (Unsloth's gradient
checkpointing offloads activations), with no OOM risk at 15 GiB. Per-device 4 is
**not** faster. At ≈ 22–25 s/step, 246 steps take **≈ 1.5–1.7 h of pure
training**.

**Callback: not measured.** Wall times were 164 s (generation batch 8, peak 6.8 GiB)
and 123 s (batch 4, peak 5.8 GiB), but every output was `""` (parse_ok 0/50,
1/50 cut off). With empty outputs these times don't reflect real work, so ticket 11 re-measures.

**Loss anomaly:** the 4B starts at loss **3.14** (step 1) and ends at 0.72
(step 12). The 0.6B smoke run `63pxlpvn` started at **1.37** on the same leading documents.
A 4B instruct should start lower, not 2.3× higher. The two runs are
bit-for-bit alike across batch sizes (3.137 / 3.138), so this is
deterministic, not noise.

**Stack:** `versions` = unsloth 2026.9.11, unsloth_zoo 2026.9.7, trl 0.24.0,
transformers 5.5.0, peft 0.20.0, bitsandbytes 0.50.2, torch 2.11.0+cu128,
identical to milestone 3's Colab resolution. The config-nesting fix works on
a GPU: `experiment` and `derived` arrive intact (`eval_steps` 12,
`warmup_steps` 1, not `None`). `git_dirty` is **True** on a clean checkout
of `0409d6d`, presumably an artefact of `pip install -e .`; unexplained.

**Milestone-3 load-back cell:** confirmed by the user, it printed its
`gold:` / `output:` pair.

Probe adapter commits: `fnl-es/qwen3-4b-muc4-lora-probe` @ `9354e17` (b4),
`6686fe8` (b2).
