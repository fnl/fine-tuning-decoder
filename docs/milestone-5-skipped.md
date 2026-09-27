# Milestone 5: skipped

**Decided 2026-09-27: milestone 5 is skipped, and milestone 6 follows
milestone 4 directly.** The bf16 LoRA run on a rented GPU would only show that
training also runs on a newer GPU in bf16. It would not improve the model, and
it could not measure the quantization cost it was meant to measure.

## What milestone 5 was

DESIGN §11: "One rented-GPU run, bf16 LoRA, same config — quantization cost +
portability." That means the milestone-4 config (`configs/qwen3-4b-r16.yaml`)
with `quantization: none`, trained in bf16 on an Ampere-or-newer GPU, set
beside the NF4 adapter.

## Why we skip it

**It cannot measure the quantization cost.** The comparison would be one bf16
run against one NF4 run on 200 dev documents (731 gold items, about seven per
point of recall). We have never measured how far the same config moves between
seeds, so a gap of a few points between two single runs has no meaning. To
measure the cost, we would need several seeds per configuration, which is a
sweep, not a milestone. The cost we would be looking for is also expected to be
small: QLoRA (Dettmers et al., 2023) matches 16-bit LoRA on its benchmarks.

**It does not touch the errors we have.** The config would be the same, so the
causes found in the milestone-4 error analysis would remain
(`docs/milestone-4-comparison.md`, *Where the points go*):

- about 12.7 points of recall from event documents answered with `[]`;
- about 3.1 points from one repetition loop.

The next gains come from those items, which are in `docs/TODO.md`, not from the
precision the run trains in.

**Portability is the only thing it would prove, and it is not worth a milestone.**
`src/train.py` already has the path: it picks bf16 from the device's compute
capability and accepts `quantization: none`. That path has never run on a GPU.
Proving it would cost about an hour of rented GPU time (an estimate) and would
teach nothing about the task.

## What changes

- Milestone 6 reports the baselines and the NF4 fine-tune on the test split.
- The bf16 / `quantization: none` path stays untested on a GPU. It is a known
  gap, and it gets exercised the first time a round-two experiment runs on
  newer hardware.

## References

- Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of
  Quantized LLMs.* NeurIPS 2023. <https://arxiv.org/abs/2305.14314>
