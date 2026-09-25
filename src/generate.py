"""Generation: render inputs, hand them to an engine, keep outputs as JSONL rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# One example as loaded from the dataset: docid, messages, n_input_tokens, truncated.
Example = Mapping[str, Any]


@dataclass
class GenerationResult:
    """One output and why the engine stopped: ``"stop"`` or ``"length"``."""

    text: str
    finish_reason: str


Engine = Callable[[list[str]], list[GenerationResult]]


def render_inputs(
    examples: Sequence[Example],
    exemplars: Sequence[Example],
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Render every example as one input: system prompt, exemplar turns, document, generation prompt."""
    exemplar_turns = [turn for exemplar in exemplars for turn in exemplar["messages"][1:]]
    inputs = []
    for ex in examples:
        messages = [ex["messages"][0], *exemplar_turns, ex["messages"][1]]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert isinstance(rendered, str)
        inputs.append(rendered)
    return inputs


def generate(
    examples: Sequence[Example],
    engine: Engine,
    tokenizer: PreTrainedTokenizerBase,
    exemplars: Sequence[Example] = (),
    *,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    """Outputs of ``engine`` for every example as rows ``{"docid", "output", "cut_off"}``.

    Raises ``ValueError`` naming the first example whose rendered input exceeds
    ``max_input_tokens``: the dataset's ``truncated`` flag is the only truncation.
    """
    inputs = render_inputs(examples, exemplars, tokenizer)
    for ex, text in zip(examples, inputs, strict=True):
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if n_tokens > max_input_tokens:
            raise ValueError(
                f"document {ex['docid']}: input has {n_tokens} tokens, over {max_input_tokens}"
            )
    results = engine(inputs)
    return [
        {"docid": ex["docid"], "output": r.text, "cut_off": r.finish_reason == "length"}
        for ex, r in zip(examples, results, strict=True)
    ]


def vllm_engine(model: str, *, max_model_len: int, max_new_tokens: int) -> Engine:
    """vLLM on one GPU, greedy decoding; the T4-specific knobs are constants here."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model,
        dtype="half",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
        enforce_eager=True,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def engine(inputs: list[str]) -> list[GenerationResult]:
        outputs = llm.generate(inputs, params)
        return [GenerationResult(o.outputs[0].text, o.outputs[0].finish_reason) for o in outputs]

    return engine


def unsloth_engine(
    model: Any, tokenizer: PreTrainedTokenizerBase, *, max_new_tokens: int, batch_size: int
) -> Engine:
    """The model under training, in process: batched, greedy, left-padded per batch.

    Unsloth's patched ``model.generate`` switches to inference and back to training
    itself (mode, KV cache, gradient checkpointing); calling ``for_inference``
    here would break that restore, so this engine never does.
    """
    import torch

    eos_ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]

    def engine(inputs: list[str]) -> list[GenerationResult]:
        order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
        results: dict[int, GenerationResult] = {}
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch = tokenizer(
                [inputs[i] for i in index],
                return_tensors="pt",
                padding=True,
                # per call, so completions start at one shared column: every generate
                # resets the tokenizer's own padding_side to "right" on its way out
                padding_side="left",
                add_special_tokens=False,
            ).to(model.device)
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
            for i, row in zip(index, outputs[:, batch["input_ids"].shape[1] :], strict=True):
                ids = row.tolist()
                reason = "stop" if any(t in eos_ids for t in ids) else "length"
                text = str(tokenizer.decode(ids, skip_special_tokens=True))
                results[i] = GenerationResult(text, reason)
            del outputs, batch
        torch.cuda.empty_cache()
        return [results[i] for i in range(len(inputs))]

    return engine


def constant_engine() -> Engine:
    """The always-empty baseline: ``[]`` for every input."""
    return lambda inputs: [GenerationResult("[]", "stop") for _ in inputs]


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def _gpu_name() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return str(torch.cuda.get_device_name()) if torch.cuda.is_available() else "cpu"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate outputs for one split of the dataset.")
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, help="only the first N documents")
    parser.add_argument("--out", type=Path, help="output directory (default outputs/<name>)")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = args.out or Path("outputs") / config["name"]

    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    revision = HfApi().dataset_info(config["dataset"]).sha
    dataset = load_dataset(config["dataset"], revision=revision)
    examples = dataset[args.split]
    if args.limit is not None:
        examples = examples.select(range(args.limit))
    train = dataset["train"]
    index = {docid: i for i, docid in enumerate(train["docid"])}
    exemplars = [train[index[docid]] for docid in config["exemplars"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model"])

    generation = config["generation"]
    if config["engine"] == "vllm":
        engine = vllm_engine(
            config["model"],
            max_model_len=generation["max_model_len"],
            max_new_tokens=generation["max_new_tokens"],
        )
        engine_version = f"vllm {version('vllm')}"
    elif config["engine"] == "constant":
        engine, engine_version = constant_engine(), "constant"
    else:
        raise ValueError(f"unknown engine {config['engine']!r}; expected vllm or constant")

    started = time.perf_counter()
    rows = generate(
        examples,
        engine,
        tokenizer,
        exemplars,
        max_input_tokens=generation["max_model_len"] - generation["max_new_tokens"],
    )
    wall_seconds = time.perf_counter() - started

    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{args.split}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "model": config["model"],
        "engine": config["engine"],
        "engine_version": engine_version,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "dataset_revision": revision,
        "split": args.split,
        "gpu": _gpu_name(),
        "n_docs": len(rows),
        "n_truncated": sum(examples["truncated"]),
        "n_cut_off": sum(row["cut_off"] for row in rows),
        "wall_seconds": round(wall_seconds, 3),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{args.split}: {len(rows)} documents, {meta['n_cut_off']} cut off -> {out}")


if __name__ == "__main__":
    main()
