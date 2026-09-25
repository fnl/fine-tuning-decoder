"""Training: completion-only LoRA fine-tune of one experiment, scoring the dev subset as it goes."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from data.prepare import parse_target
from eval import TABLE_COLUMNS, flatten, parse_rows, predictions_table, score
from generate import Engine, Example, _git, _gpu_name, generate, unsloth_engine

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        PreTrainedTokenizerBase,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

# The label of a masked token: excluded from the loss.
IGNORE_INDEX = -100
END_OF_TURN = "<|im_end|>"

# Keys of an experiment YAML: every one required unless optional; nested blocks by their own keys.
CONFIG_KEYS = frozenset(
    {
        "name", "tags", "model", "dataset", "adapter", "max_seq_len", "epochs", "lr",
        "scheduler", "warmup_ratio", "effective_batch_size", "per_device_batch_size", "seed",
        "quantization", "lora", "eval_every", "generation", "wandb_project",
    }
)  # fmt: skip
OPTIONAL_KEYS = frozenset({"train_docs", "dev_docs"})
BLOCK_KEYS = {
    "lora": frozenset({"r", "alpha", "dropout", "target_modules"}),
    "generation": frozenset({"max_new_tokens", "batch_size"}),
}
QUANTIZATIONS = ("nf4", "none")
# The pinned training stack, stamped into every run's config: it determines behaviour.
VERSIONED = ("unsloth", "unsloth_zoo", "trl", "transformers", "torch", "peft", "bitsandbytes")

MODEL_CARD = """\
---
library_name: peft
base_model: {model}
datasets: [{dataset}]
tags: [lora, event-extraction, muc-4]
---
# {adapter}

A LoRA adapter (r={r}, alpha={alpha}) of `{model}` for document-level event
extraction on MUC-4: given a news document and the dataset's system prompt, the
model answers with the document's events as a JSON array, `[]` for none.

Trained with completion-only loss on {n_train} documents of `{dataset}` for
{epochs} epochs, experiment `configs/{name}.yaml` of the fine-tuning-decoder
project. The adapter only, never merged; load it onto the base model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{model}"), "{adapter}")
```

Training run: {run_url}
"""


class MaskingError(ValueError):
    """The rendered prompt is not a token-level prefix of the full rendering."""


def load_config(path: Path) -> dict[str, Any]:
    """The experiment YAML at ``path``, validated: a typo is an error, not an ignored setting."""
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_keys(path.name, config, CONFIG_KEYS, OPTIONAL_KEYS)
    for block, keys in BLOCK_KEYS.items():
        _check_keys(f"{path.name}: {block}", config[block], keys)
    if config["effective_batch_size"] % config["per_device_batch_size"]:
        raise ValueError(
            f"{path.name}: effective_batch_size {config['effective_batch_size']} is not a "
            f"multiple of per_device_batch_size {config['per_device_batch_size']}"
        )
    if config["quantization"] not in QUANTIZATIONS:
        raise ValueError(
            f"{path.name}: quantization {config['quantization']!r} is not one of {QUANTIZATIONS}"
        )
    return config


def _check_keys(
    where: str,
    config: Mapping[str, Any],
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    if unknown := config.keys() - required - optional:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    if missing := required - config.keys():
        raise ValueError(f"{where}: missing keys {sorted(missing)}")


def tokenize_example(
    example: Example, tokenizer: PreTrainedTokenizerBase, *, stop_after_eos: bool = True
) -> dict[str, list[int]]:
    """``{input_ids, labels}`` of one example with every prompt token masked.

    The prompt is rendered exactly as at inference (generation prompt, no
    thinking), so the completion is what the model will be asked to produce.
    With ``stop_after_eos`` the sequence ends at the completion's end-of-turn
    token: the newline the chat template emits after it can never be generated.
    """
    messages = example["messages"]
    prompt = _render_ids(tokenizer, messages[:2], add_generation_prompt=True)
    full = _render_ids(tokenizer, messages, add_generation_prompt=False)
    if full[: len(prompt)] != prompt:
        raise MaskingError(
            f"document {example['docid']}: the rendered prompt ({len(prompt)} tokens) is not "
            f"a prefix of the full rendering ({len(full)} tokens)"
        )
    if stop_after_eos:
        eos = tokenizer.convert_tokens_to_ids(END_OF_TURN)
        assert isinstance(eos, int)
        full = full[: len(full) - full[::-1].index(eos)]
    return {"input_ids": full, "labels": [IGNORE_INDEX] * len(prompt) + full[len(prompt) :]}


def _render_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    assert isinstance(rendered, str)
    ids: list[int] = tokenizer.encode(rendered, add_special_tokens=False)
    return ids


def select_subset(split: Sequence[Example], n: int | None) -> list[Example]:
    """The first ``n`` examples of ``split``, or all of them when ``n`` is absent."""
    end = len(split) if n is None else min(n, len(split))
    return [split[i] for i in range(end)]


def build_dataset(
    examples: Sequence[Example], tokenizer: PreTrainedTokenizerBase, *, max_seq_len: int
) -> tuple[Dataset, int]:
    """The tokenised training examples and how many were dropped for exceeding ``max_seq_len``.

    Over-budget examples are dropped, never truncated: truncation would cut the
    target, and the trainer does not enforce the budget on a pre-tokenised dataset.
    """
    from datasets import Dataset

    rows = [tokenize_example(ex, tokenizer) for ex in examples]
    kept = [row for row in rows if len(row["input_ids"]) <= max_seq_len]
    return Dataset.from_list(kept), len(rows) - len(kept)


def training_steps(n: int, *, effective_batch_size: int, epochs: float) -> int:
    """Optimizer steps of a run: the trainer counts a last, partial batch as a step."""
    return math.ceil(epochs * math.ceil(n / effective_batch_size))


def eval_interval(total_steps: int, *, eval_every: float, epochs: float) -> int:
    """Steps between eval points: ``eval_every`` epochs rounded down, within 1..``total_steps``."""
    return min(total_steps, max(1, math.floor(total_steps / epochs * eval_every)))


class ScoringCallback:
    """At every eval point, score the dev subset by generating with the model under training.

    Fires every ``interval`` optimizer steps and at the last one, logging the
    scorer's flat result under ``dev/`` beside ``train/global_step``: never with
    an explicit ``step=``, which the W&B integration answers by dropping rows.
    The rows of the last eval point stay in ``rows`` for the end of the run.

    Hooks ``on_step_end`` because ``on_evaluate`` never fires without an
    evaluation strategy. Mixed into a ``TrainerCallback`` by ``train`` so that
    this module imports no ``transformers`` ahead of Unsloth.
    """

    def __init__(
        self,
        examples: Sequence[Example],
        engine: Engine,
        tokenizer: PreTrainedTokenizerBase,
        *,
        interval: int,
        max_input_tokens: int,
        log: Callable[[dict[str, Any]], None],
    ) -> None:
        self.examples = examples
        self.golds = {
            ex["docid"]: parse_target(ex["messages"][-1]["content"])[0] for ex in examples
        }
        self.engine = engine
        self.tokenizer = tokenizer
        self.interval = interval
        self.max_input_tokens = max_input_tokens
        self.log = log
        self.rows: list[dict[str, Any]] = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        step = state.global_step
        if step % self.interval and step != state.max_steps:
            return
        self.rows = generate(
            self.examples, self.engine, self.tokenizer, max_input_tokens=self.max_input_tokens
        )
        preds, parse_failures = parse_rows(self.rows)
        result = score(preds, self.golds, parse_failures)
        self.log({**{f"dev/{k}": v for k, v in flatten(result).items()}, "train/global_step": step})


def run_config(
    config: Mapping[str, Any],
    derived: Mapping[str, Any],
    versions: Mapping[str, str],
    stamps: Mapping[str, Any],
) -> dict[str, Any]:
    """The W&B config of a training run: the experiment and its derived numbers nested.

    Nested because the W&B integration writes every trainer setting and the model
    config into the same top level, overwriting any key of ours that shares a
    name (``warmup_ratio``, ``seed``, ``eval_steps``, ...).
    """
    return {
        "experiment": dict(config),
        "derived": dict(derived),
        "versions": dict(versions),
        **stamps,
    }


def train(config: dict[str, Any]) -> str:
    """Fine-tune the experiment's model, push the adapter to the Hub, return the W&B run URL."""
    import unsloth  # noqa: F401  (first: it patches transformers, trl and peft on import)

    # isort: split
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    import wandb

    revision = HfApi().dataset_info(config["dataset"]).sha
    dataset = load_dataset(config["dataset"], revision=revision)
    train_examples = select_subset(dataset["train"], config.get("train_docs"))
    dev_examples = select_subset(dataset["dev"], config.get("dev_docs"))

    bf16 = torch.cuda.get_device_capability()[0] >= 8  # Ampere and newer; the T4 is sm_75
    model, tokenizer = FastLanguageModel.from_pretrained(
        config["model"],
        max_seq_length=config["max_seq_len"],
        dtype=torch.bfloat16 if bf16 else torch.float16,
        load_in_4bit=config["quantization"] == "nf4",
    )
    lora = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=config["seed"],
    )

    train_dataset, n_dropped = build_dataset(
        train_examples, tokenizer, max_seq_len=config["max_seq_len"]
    )
    total_steps = training_steps(
        len(train_dataset),
        effective_batch_size=config["effective_batch_size"],
        epochs=config["epochs"],
    )
    derived = {
        "n_train_examples": len(train_dataset),
        "n_dropped": n_dropped,
        "gradient_accumulation_steps": config["effective_batch_size"]
        // config["per_device_batch_size"],
        "total_steps": total_steps,
        "eval_steps": eval_interval(
            total_steps, eval_every=config["eval_every"], epochs=config["epochs"]
        ),
        "warmup_steps": math.ceil(config["warmup_ratio"] * total_steps),
        "precision": "bf16" if bf16 else "fp16",
    }
    print(f"train: {len(train_dataset)} examples, {n_dropped} dropped over the sequence budget")

    run = wandb.init(
        project=config["wandb_project"],
        name=config["name"],
        tags=config["tags"],
        job_type="train",
        config=run_config(
            config,
            derived,
            {package: version(package) for package in VERSIONED},
            {
                "git_sha": _git("rev-parse", "HEAD"),
                "git_dirty": bool(_git("status", "--porcelain")),
                "dataset_revision": revision,
                "gpu": _gpu_name(),
            },
        ),
    )

    class ScoringTrainerCallback(ScoringCallback, TrainerCallback):
        pass

    class AdapterTrainer(SFTTrainer):  # type: ignore[misc]
        """Writes our adapter card: TRL rewrites ``README.md`` at every save otherwise."""

        def create_model_card(
            self,
            model_name: str | None = None,
            dataset_name: str | None = None,
            tags: str | list[str] | None = None,
        ) -> None:
            card = MODEL_CARD.format(**config, **lora, n_train=len(train_dataset), run_url=run.url)
            Path(self.args.output_dir, "README.md").write_text(card, encoding="utf-8")

    scoring = ScoringTrainerCallback(
        dev_examples,
        unsloth_engine(
            model,
            tokenizer,
            max_new_tokens=config["generation"]["max_new_tokens"],
            batch_size=config["generation"]["batch_size"],
        ),
        tokenizer,
        interval=derived["eval_steps"],
        max_input_tokens=config["max_seq_len"] - config["generation"]["max_new_tokens"],
        log=run.log,
    )
    trainer = AdapterTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        callbacks=[scoring],
        args=SFTConfig(
            output_dir=str(Path("outputs") / config["name"]),
            num_train_epochs=config["epochs"],
            learning_rate=config["lr"],
            lr_scheduler_type=config["scheduler"],
            warmup_steps=derived["warmup_steps"],
            per_device_train_batch_size=config["per_device_batch_size"],
            gradient_accumulation_steps=derived["gradient_accumulation_steps"],
            seed=config["seed"],
            fp16=not bf16,  # both stated: SFTConfig derives bf16 = not fp16 when bf16 is unset
            bf16=bf16,
            max_length=config["max_seq_len"],
            packing=False,
            padding_free=False,  # Unsloth enables it silently otherwise
            logging_steps=1,
            save_strategy="steps",
            save_steps=derived["eval_steps"],  # a checkpoint at every eval point
            save_total_limit=2,
            push_to_hub=True,
            hub_model_id=config["adapter"],
            hub_strategy="checkpoint",  # "every_save" pushes nothing resumable
            hub_always_push=True,  # otherwise a push is skipped while the previous one runs
            hub_private_repo=False,
            report_to="wandb",
            run_name=config["name"],
        ),
    )

    first_batch = next(iter(trainer.get_train_dataloader()))["labels"]
    n_kept = int((first_batch != IGNORE_INDEX).sum())
    completion = tokenizer.decode(first_batch[0][first_batch[0] != IGNORE_INDEX])
    print(f"first batch: {n_kept} kept tokens; first completion: {completion!r}")

    trainer.train()
    commit = trainer.push_to_hub(commit_message="End of training")

    table = wandb.Table(
        columns=list(TABLE_COLUMNS), data=predictions_table(scoring.rows, scoring.golds)
    )
    run.log({"predictions": table, "train/global_step": trainer.state.global_step})
    first_quarter, last_quarter = _loss_quarters(trainer.state.log_history)
    run.summary.update(
        {
            "loss_first_quarter": first_quarter,
            "loss_last_quarter": last_quarter,
            "first_batch_kept_tokens": n_kept,
            "n_dropped": n_dropped,
            "adapter_revision": commit.oid,
            "adapter_url": f"https://huggingface.co/{config['adapter']}",
        }
    )
    url = run.url or run.id  # offline runs have no URL
    run.finish()  # nothing in the training stack finishes the run it did not start
    return str(url)


def _loss_quarters(log_history: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Mean training loss over the first and the last quarter of the logged steps."""
    losses = [entry["loss"] for entry in log_history if "loss" in entry]
    quarter = max(1, len(losses) // 4)
    return sum(losses[:quarter]) / quarter, sum(losses[-quarter:]) / quarter


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune one experiment and push its adapter.")
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--limit", type=int, help="only the first N train and dev documents")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.limit is not None:
        config["train_docs"] = config["dev_docs"] = args.limit
    print(train(config))


if __name__ == "__main__":
    main()
