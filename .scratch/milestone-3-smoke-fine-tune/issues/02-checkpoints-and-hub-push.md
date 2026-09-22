# 02 Adapter checkpointing and the Hub push during training

Type: research
Status: claimed
Blocked by: —
HITL: no

## Question

Decided: checkpoint at every eval point and push every save, so a Colab
disconnect never loses the adapter (map Notes, Q6). Establish how that is
actually expressed under TRL 0.24 + Unsloth + `transformers`:

1. The `TrainingArguments`/`SFTConfig` fields: `output_dir`, `save_strategy`,
   `save_steps`, `save_total_limit`, `push_to_hub`, `hub_model_id`,
   `hub_strategy` (does `"every_save"` still exist, and what does it push —
   the checkpoint directory, or only the final model?), `hub_private_repo`,
   `hub_token`. Whether the push is asynchronous and what happens when the
   runtime dies mid-push.
2. What a **LoRA checkpoint actually contains** when the model is a 4-bit
   Unsloth model: which files, how large at 0.6B/r=16, and whether the base
   model's weights leak into it. Confirm the pushed artifact is an adapter,
   never a merged model (DESIGN §9).
3. Whether `Trainer.push_to_hub` / `model.push_to_hub` / Unsloth's
   `save_pretrained` disagree about what lands in the repo, and which one
   produces a repo that `PeftModel.from_pretrained` (and, later, vLLM's LoRA
   loading) can read back.
4. The repo card: does the push write one automatically, and can we supply
   ours (milestone 1 wrote a dataset card by hand in `prepare.py`)?
5. `resume_from_checkpoint` after a disconnect: what must be present locally
   or pullable from the Hub for it to work, and whether the optimizer state
   is part of the pushed checkpoint at all.

Answer in this file with sources and dates. Feeds tickets 08 and 09; the
resume question also feeds the "disconnect recovery" fog on the map.
