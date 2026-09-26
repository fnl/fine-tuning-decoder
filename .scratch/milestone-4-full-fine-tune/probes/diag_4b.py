"""Throwaway diagnosis for ticket 11: why the 4B's in-process outputs are empty.

One mode per process, because Unsloth patches transformers on import:

    python .scratch/milestone-4-full-fine-tune/probes/diag_4b.py MODE

    unsloth-fp16  the training stack exactly as train.py builds it (NF4, fp16, zero-init LoRA)
    unsloth-fp32  the same in fp32: does the fp16 overflow hypothesis hold?
    hf-nf4        plain transformers + bitsandbytes NF4, fp16 compute: no Unsloth
    hf-fp16       plain transformers, unquantised fp16: the reference

Each mode measures the base model's completion-only loss on the first train examples, the largest
hidden-state magnitude per decoder layer (fp16 overflows at 65504), and raw token ids of greedy
outputs at generation batch 1 and 4. It prints a report and logs it as one W&B run
(job_type=diagnostic, tag diagnostic).
"""

import os
import sys

MODES = ("unsloth-fp16", "unsloth-fp32", "hf-nf4", "hf-fp16")
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in MODES:
    sys.exit(f"usage: diag_4b.py {{{'|'.join(MODES)}}}")
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Unsloth returns empty logits in training otherwise
if MODE.startswith("unsloth"):
    import unsloth  # noqa: F401  (first: it patches transformers on import)

# isort: split
import subprocess
from contextlib import nullcontext

import torch
from datasets import load_dataset

import wandb
from generate import render_inputs
from train import tokenize_example

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
N_LOSS = 8
N_GEN = 8
MAX_NEW_TOKENS = 128
DTYPE = torch.float32 if MODE.endswith("fp32") else torch.float16


def load() -> tuple:
    if MODE.startswith("unsloth"):
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL, max_seq_length=2048, dtype=DTYPE, load_in_4bit=True
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
        return model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if MODE == "hf-nf4"
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, quantization_config=quantization, device_map="cuda"
    )
    return model, AutoTokenizer.from_pretrained(MODEL)


def layer_maxima(model, input_ids: torch.Tensor) -> list[float]:
    """The largest absolute hidden-state value each decoder layer outputs for one input."""
    maxima: list[float] = []
    layers = [m for m in model.modules() if type(m).__name__.endswith("DecoderLayer")]

    def hook(module, args, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        maxima.append(float(hidden.detach().float().abs().nan_to_num(float("inf")).max()))

    handles = [layer.register_forward_hook(hook) for layer in layers]
    try:
        with torch.no_grad(), autocast():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return maxima


def autocast():
    # the trainer runs fp16 under AMP autocast; mirror it
    return torch.autocast("cuda", dtype=torch.float16) if DTYPE == torch.float16 else nullcontext()


def losses(model, tokenizer, examples) -> tuple[list[float], int]:
    values, n_nonfinite_logits = [], 0
    for ex in examples:
        row = tokenize_example(ex, tokenizer)
        ids = torch.tensor([row["input_ids"]], device="cuda")
        labels = torch.tensor([row["labels"]], device="cuda")
        with torch.no_grad(), autocast():
            out = model(input_ids=ids, labels=labels)
        values.append(float(out.loss))
        if out.logits is not None and out.logits.numel():
            n_nonfinite_logits += int((~torch.isfinite(out.logits)).sum())
    return values, n_nonfinite_logits


def outputs(model, tokenizer, examples, batch_size: int) -> list[list]:
    """Greedy outputs as rows [batch_size, docid, n_tokens, first 24 ids, text with special tokens]."""
    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    inputs = render_inputs(examples, [], tokenizer)
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = tokenizer(
            inputs[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to("cuda")
        out = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
        for j, ids in enumerate(out[:, batch["input_ids"].shape[1] :].tolist()):
            text = tokenizer.decode(ids, skip_special_tokens=False)
            rows.append(
                [batch_size, examples[start + j]["docid"], len(ids), str(ids[:24]), text[:500]]
            )
    return rows


def main() -> None:
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout
    dataset = load_dataset("fnl-es/muc4-chat")
    train = [dataset["train"][i] for i in range(N_LOSS)]
    dev = [dataset["dev"][i] for i in range(N_GEN)]

    model, tokenizer = load()
    loss_values, n_nonfinite = losses(model, tokenizer, train)
    maxima = layer_maxima(
        model, torch.tensor([tokenize_example(train[0], tokenizer)["input_ids"]], device="cuda")
    )
    rows = outputs(model, tokenizer, dev[:2], 1) + outputs(model, tokenizer, dev, 4)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    summary = {
        "mode": MODE,
        "resolved_model": str(getattr(model.config, "_name_or_path", "?")),
        "pad_token": str(tokenizer.pad_token),
        "loss_mean": sum(loss_values) / len(loss_values),
        "n_nonfinite_logits": n_nonfinite,
        "hidden_max": max(maxima) if maxima else float("nan"),
        "hidden_max_layer": maxima.index(max(maxima)) if maxima else -1,
        "n_empty_outputs": sum(not row[4].replace("<|im_end|>", "").strip() for row in rows),
        "peak_gib": peak_gib,
        "git_status": git_status or "(clean)",
    }
    print(f"==== {MODE}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("losses:", [round(v, 3) for v in loss_values])
    print("layer maxima:", [round(v) for v in maxima])
    for row in rows:
        print(f"-- batch {row[0]} {row[1]} ({row[2]} tokens) ids {row[3]}\n{row[4]!r}")

    run = wandb.init(
        project="muc4-event-extraction",
        name=f"diag-4b-{MODE}",
        tags=["probe", "diagnostic"],
        job_type="diagnostic",
        config={"mode": MODE, "model": MODEL, "gpu": torch.cuda.get_device_name()},
    )
    run.summary.update({**summary, "losses": loss_values, "layer_maxima": maxima})
    run.log(
        {
            "outputs": wandb.Table(
                columns=["batch_size", "docid", "n_tokens", "first_ids", "text"], data=rows
            )
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
