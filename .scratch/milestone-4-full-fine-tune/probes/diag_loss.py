"""Throwaway diagnosis for ticket 11, round 2: is Unsloth's reported loss shifted wrongly?

    python .scratch/milestone-4-full-fine-tune/probes/diag_loss.py {4b|0.6b}

Loads the model exactly as train.py does (Unsloth, NF4, fp16, zero-init LoRA) and, for the first
train examples, compares the loss Unsloth reports with cross-entropy recomputed from its own
logits at label shifts 0, 1 (the causal-LM convention) and 2. Also scores with a plain PyTorch
forward of the same model object without labels. Prints a report and logs one W&B run.
"""

import os
import sys

MODELS = {"4b": "Qwen/Qwen3-4B-Instruct-2507", "0.6b": "Qwen/Qwen3-0.6B"}
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
if KEY not in MODELS:
    sys.exit(f"usage: diag_loss.py {{{'|'.join(MODELS)}}}")
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
import unsloth  # noqa: F401  (first: it patches transformers on import)

# isort: split
import torch
import torch.nn.functional as F
from datasets import load_dataset
from unsloth import FastLanguageModel

import wandb
from train import tokenize_example

N_EXAMPLES = 8
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def shifted_ce(logits: torch.Tensor, labels: torch.Tensor, shift: int) -> float:
    """Mean cross-entropy of ``logits[t]`` against ``labels[t + shift]`` over unmasked labels."""
    end = logits.shape[1] - shift
    return float(
        F.cross_entropy(logits[0, :end].float(), labels[0, shift : shift + end], ignore_index=-100)
    )


def main() -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODELS[KEY], max_seq_length=2048, dtype=torch.float16, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    train = load_dataset("fnl-es/muc4-chat")["train"]
    rows = []
    for i in range(N_EXAMPLES):
        example = tokenize_example(train[i], tokenizer)
        ids = torch.tensor([example["input_ids"]], device="cuda")
        labels = torch.tensor([example["labels"]], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, labels=labels)
            plain = model(input_ids=ids)
        n_completion = sum(label != -100 for label in example["labels"])
        rows.append(
            [
                train[i]["docid"],
                n_completion,
                float(out.loss),
                *(shifted_ce(out.logits, labels, s) for s in (0, 1, 2)),
                shifted_ce(plain.logits, labels, 1),
                bool(torch.equal(out.logits, plain.logits)),
            ]
        )
    columns = [
        "docid",
        "n_completion",
        "reported",
        "ce_shift0",
        "ce_shift1",
        "ce_shift2",
        "plain_ce_shift1",
        "same_logits",
    ]
    print(f"==== {KEY} {getattr(model.config, '_name_or_path', '?')}")
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row))
    run = wandb.init(
        project="muc4-event-extraction",
        name=f"diag-loss-{KEY}",
        tags=["probe", "diagnostic"],
        job_type="diagnostic",
        config={"model": MODELS[KEY], "gpu": torch.cuda.get_device_name()},
    )
    run.log({"losses": wandb.Table(columns=columns, data=rows)})
    run.finish()


if __name__ == "__main__":
    main()
