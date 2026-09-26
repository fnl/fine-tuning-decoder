# 01 Measure the 4B on a T4

Type: task
Status: claimed
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
