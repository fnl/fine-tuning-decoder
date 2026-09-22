"""Prototype for ticket 05: see the completion-only mask.

Throwaway. Renders prompt and full example with the chat template, asserts the
token-level prefix, builds labels, and prints the seam token by token so the
mask can be eyeballed before any GPU time is spent.

    uv run python .scratch/milestone-3-smoke-fine-tune/prototype_mask.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B-Instruct-2507")
PREPARED = Path("data/prepared/train.jsonl")
IGNORE = -100


class MaskingError(ValueError):
    """The prompt is not a token-level prefix of the full rendering."""


def tokenize_example(
    example: dict, tokenizer: PreTrainedTokenizerBase, *, stop_after_eos: bool = True
) -> dict:
    """``{input_ids, labels}`` with every prompt token masked to -100.

    The sequence ends at the target's end-of-turn token: the chat template emits
    a trailing newline after it that the model can never generate, and for an
    empty document that newline would be a third of the loss signal.
    """
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:2], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True,
    )["input_ids"]
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, enable_thinking=False, return_dict=True,
    )["input_ids"]
    if full[: len(prompt)] != prompt:
        raise MaskingError(
            f"document {example['docid']}: the rendered prompt ({len(prompt)} tokens) is not a "
            f"prefix of the full rendering ({len(full)} tokens)"
        )
    if stop_after_eos:
        eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
        full = full[: max(i for i, t in enumerate(full) if t == eos) + 1]
    labels = [IGNORE] * len(prompt) + full[len(prompt) :]
    return {"input_ids": full, "labels": labels}


def show_seam(example: dict, tokenizer: PreTrainedTokenizerBase, width: int = 8) -> None:
    out = tokenize_example(example, tokenizer)
    ids, labels = out["input_ids"], out["labels"]
    seam = next(i for i, lab in enumerate(labels) if lab != IGNORE)
    lo, hi = max(0, seam - width), min(len(ids), seam + width)
    print(f"    seam at token {seam}; prompt {seam}, completion {len(ids) - seam}")
    for i in range(lo, hi):
        mark = "<<< seam" if i == seam else ""
        lab = "-100" if labels[i] == IGNORE else str(labels[i])
        print(f"      {i:5d}  {tokenizer.decode([ids[i]])!r:>26}  id={ids[i]:<8} label={lab:<8}{mark}")


def main() -> None:
    rows = [json.loads(line) for line in PREPARED.open()]

    def n_events(row: dict) -> int:
        return len(json.loads(row["messages"][2]["content"]))

    picks = {}
    for row in rows[:100]:
        key = min(n_events(row), 2)
        picks.setdefault(key, row)
    labels = {0: "empty document", 1: "one event", 2: "multi-event"}

    for model in MODELS:
        tok = AutoTokenizer.from_pretrained(model)
        print(f"\n{'=' * 78}\n{model}\n{'=' * 78}")
        for key, row in sorted(picks.items()):
            print(f"\n  {labels[key]} — {row['docid']}")
            show_seam(row, tok)

        total_tokens = total_unmasked = 0
        completions = []
        for row in rows[:100]:
            out = tokenize_example(row, tok)
            unmasked = sum(1 for lab in out["labels"] if lab != IGNORE)
            total_tokens += len(out["input_ids"])
            total_unmasked += unmasked
            completions.append(unmasked)
        print("\n  over the 100-document smoke subset:")
        print(f"    tokens {total_tokens:,} | unmasked {total_unmasked:,} "
              f"({100 * total_unmasked / total_tokens:.1f} %)")
        print(f"    completion length: min {min(completions)}, mean "
              f"{sum(completions) / len(completions):.1f}, max {max(completions)}")
        print(f"    documents whose completion is 3 tokens: "
              f"{sum(1 for c in completions if c == 3)}")

        tail = tokenize_example(rows[0], tok)["input_ids"][-2:]
        print(f"    last two tokens of a completion: {[tok.decode([t]) for t in tail]!r}")


if __name__ == "__main__":
    main()
