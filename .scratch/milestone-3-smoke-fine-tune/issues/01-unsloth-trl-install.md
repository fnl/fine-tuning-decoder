# 01 The Unsloth + TRL stack on a free Colab T4, and the pre-tokenised dataset

Type: research
Status: claimed
Blocked by: —
HITL: no

## Question

The whole map rests on one assumption: that we can hand TRL's `SFTTrainer` a
dataset we tokenised and masked ourselves (`input_ids` + `labels`) and have
Unsloth leave it alone. Establish, with sources and dates:

1. **Install.** The pip line for Unsloth on a free Colab T4 today, and the
   versions it actually resolves (`unsloth`, `unsloth_zoo`, `trl`,
   `transformers`, `torch`, `peft`, `bitsandbytes`, `accelerate`,
   `datasets`). Whether a runtime restart is needed, install time, and any
   clash with Colab's preinstalled packages of the kind that bit milestone 2
   (`torchaudio` built against a different CUDA). Charting established that
   `unsloth 2026.9.9` pins `trl<=0.24.0`, `transformers<=5.5.0`,
   `torch<2.13` — confirm this is still what resolves.
2. **Pre-tokenised datasets under Unsloth.** Does Unsloth's `SFTTrainer`
   wrapper (it patches TRL, see `unsloth_zoo`) accept a dataset that already
   has `input_ids` and `labels`, or does it re-tokenise / require
   `dataset_text_field` / drop the `labels` column? Read the installed
   source, not the docs. **This is the decisive item**: if the answer is no,
   the fallback is TRL's prompt-completion form and ticket 09 changes shape.
3. **TRL 0.24 API names** the spec must state verbatim: `SFTConfig` field
   names for sequence length, batch size, accumulation, epochs, lr,
   scheduler, warmup ratio, eval/save cadence, `report_to`, seed — and which
   of them moved or were renamed between 0.18 and 0.24.
4. **`FastLanguageModel.from_pretrained`** for `Qwen3-0.6B` on a T4:
   arguments for 4-bit NF4, fp16 compute (no bf16 on sm_75), `max_seq_length`,
   and `get_peft_model` for the LoRA config in DESIGN §7 (r=16, α=16,
   dropout 0, all seven linear targets). Whether Unsloth insists on its own
   `unsloth/Qwen3-0.6B` mirror and, if so, whether that mirror's chat template
   differs from the official one (milestone 2 found the 4B mirror injects an
   empty `<think>` block — the same trap would corrupt our masking).
5. Gradient checkpointing and any T4-specific knob Unsloth recommends
   (`use_gradient_checkpointing="unsloth"`, `packing`, fp16 vs
   `fp16_full_eval`), plus what it does to memory at 0.6B / 2048 tokens.

Answer in this file with sources and dates. It does not decide anything:
it feeds tickets 07, 08, 09 and 11.
