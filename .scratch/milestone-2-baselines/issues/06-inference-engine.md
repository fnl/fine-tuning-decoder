# 06 Inference engine for baselines

Type: grilling
Status: open
Blocked by: 04, 05
HITL: yes

## Question

Given the research in 04 and 05: vLLM, batched `transformers.generate`, or
Unsloth fast inference for the two GPU baselines on a free T4? Decide:

- the engine and its install line for the notebook;
- fp16 dtype, greedy, `max_new_tokens=512`, batching strategy;
- `max_model_len` / the truncated-document policy with few-shot context
  (graduates the "few-shot input budget" fog);
- how `generate.py` isolates the engine (one function per engine? one
  engine only, fallback deferred?) so milestone 3's eval callback can reuse
  the callable core on an in-process model.
