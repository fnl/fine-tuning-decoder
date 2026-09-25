# 01 Measure the 4B on a T4

Type: task
Status: open
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
