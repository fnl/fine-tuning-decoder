# 08 The final evaluation run

Type: grilling
Status: open
Blocked by: 04
HITL: yes

## Question

How is the finished adapter scored over all 200 dev documents? Choose the
engine (vLLM + LoRA over an fp16 or bnb base, or the Unsloth in-process
engine over NF4) from 04's options, and settle: which notebook runs it, its
experiment YAML (a `qwen3-4b-r16-eval`?), `job_type=eval` with the same
metric keys as the milestone-2 baseline runs so they line up in W&B, and how
it records which adapter revision it scored and links back to the training
run.
