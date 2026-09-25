# 06 Callback cadence and the 4B YAML

Type: grilling
Status: open
Blocked by: 01, 02, 05
HITL: yes

## Question

With the 4B's measured step and callback times (01) and a session budget
(02), write `configs/qwen3-4b-r16.yaml`'s values and settle the callback:

- How many eval points and how many dev documents per point fit the budget?
  Does checkpoint cadence stay tied to eval cadence (milestone-3 ticket 08),
  or does resume insurance want checkpoints more often than evaluation?
- `per_device_batch_size`, generation `batch_size`, `max_new_tokens`.
- Whether to add per-eval-point output logging (05).
- Confirm milestone-3's promise that milestone 4 "changes values only, no new
  keys" — or name the key that breaks it and why.
