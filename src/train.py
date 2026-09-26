"""Training: completion-only LoRA fine-tune of one experiment, scoring the dev subset as it goes."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
# The directory of an adapter repo that ``hub_strategy="checkpoint"`` pushes a resumable state to.
CHECKPOINT = "last-checkpoint"

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
# Provenance stamped into a run's config at its start; a resume keeps the original ones.
STAMPS = ("git_sha", "git_dirty", "dataset_revision", "gpu")

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


class ResumeError(ValueError):
    """A run cannot be resumed from its adapter repo, or a fresh run would clobber one."""


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
    The completion must then be exactly the target and that token, or a
    ``MaskingError`` names the document: a template that adds to the assistant
    turn would otherwise be learnt silently.
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
        completion = tokenizer.decode(full[len(prompt) :])
        if completion != messages[-1]["content"] + END_OF_TURN:
            raise MaskingError(
                f"document {example['docid']}: the completion {completion[:80]!r} is not the "
                "target followed by the end-of-turn token"
            )
    return {"input_ids": full, "labels": [IGNORE_INDEX] * len(prompt) + full[len(prompt) :]}


def adopt_chat_template(
    tokenizer: PreTrainedTokenizerBase, source: PreTrainedTokenizerBase
) -> None:
    """Render with ``source``'s chat template: the base model's own, as every evaluation does.

    Unsloth ships its own copy of a model's tokenizer, whose template may differ
    from the official one; for Qwen3-4B-Instruct-2507 it renders an empty
    thinking block into every assistant turn, which the model would then learn.
    """
    tokenizer.chat_template = source.chat_template


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
    scorer's flat result and the count of cut-off outputs under ``dev/``, and
    the ``predictions`` table, beside ``train/global_step``: never with an
    explicit ``step=``, which the W&B integration answers by dropping rows.

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
        import wandb

        rows = generate(
            self.examples, self.engine, self.tokenizer, max_input_tokens=self.max_input_tokens
        )
        preds, parse_failures = parse_rows(rows)
        result = score(preds, self.golds, parse_failures)
        table = wandb.Table(columns=list(TABLE_COLUMNS), data=predictions_table(rows, self.golds))
        self.log(
            {
                **{f"dev/{k}": v for k, v in flatten(result).items()},
                "dev/diagnostics/n_cut_off": sum(row["cut_off"] for row in rows),
                "predictions": table,
                "train/global_step": step,
            }
        )


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


def check_resume(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    global_step: int | None,
    torch_version: str,
) -> None:
    """Refuse to resume ``run`` (its W&B config) from a checkpoint at ``global_step``.

    ``global_step`` is ``None`` when the adapter repo holds no ``last-checkpoint/``.
    A different git sha is allowed: a bug fix may rescue a run, and the resume is recorded.
    """
    if global_step is None:
        raise ResumeError(f"{config['adapter']} holds no {CHECKPOINT}/ to resume from")
    if global_step >= run["derived"]["total_steps"]:
        raise ResumeError(
            f"the run finished: its checkpoint is at step {global_step} of "
            f"{run['derived']['total_steps']}"
        )
    if drift := sorted(
        key
        for key in config.keys() | run["experiment"].keys()
        if config.get(key) != run["experiment"].get(key)
    ):
        raise ResumeError(f"the experiment differs from the resumed run's in keys {drift}")
    if torch_version != run["versions"]["torch"]:
        raise ResumeError(
            f"torch {torch_version} is running, but the resumed run used {run['versions']['torch']}"
        )


def has_checkpoint(repo_files: Sequence[str]) -> bool:
    """Whether an adapter repo's files hold a resumable ``last-checkpoint/``."""
    return any(path.startswith(CHECKPOINT + "/") for path in repo_files)


def check_fresh_run(adapter: str, repo_files: Sequence[str]) -> None:
    """Refuse a fresh run whose pushes would clobber the checkpoint a resume is waiting for."""
    if has_checkpoint(repo_files):
        raise ResumeError(
            f"{adapter} holds a {CHECKPOINT}/: resume its run with --resume <wandb_run_id>, "
            "or rerun under a new adapter name"
        )


def resumes(
    record: Sequence[Mapping[str, Any]] | None, *, step: int, git_sha: str
) -> list[dict[str, Any]]:
    """A run summary's ``resumes`` record (absent: ``None``) with this resume appended."""
    return [*map(dict, record or []), {"step": step, "git_sha": git_sha}]


def train(config: dict[str, Any], *, push: bool = True, resume: str | None = None) -> str:
    """Fine-tune the experiment's model, push the adapter to the Hub, return the W&B run URL.

    Without ``push`` the adapter and its checkpoints are saved locally only. With
    ``resume`` (a W&B run id) the run continues from the adapter repo's
    ``last-checkpoint/`` as the same W&B run, or a ``ResumeError`` says why not.
    """
    import unsloth  # noqa: F401  (first: it patches transformers, trl and peft on import)

    # isort: split
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    from transformers import AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    import wandb

    hub = HfApi()
    try:
        repo_files = hub.list_repo_files(config["adapter"]) if push or resume else []
    except RepositoryNotFoundError:
        repo_files = []
    checkpoint = None
    if resume is None:
        if push:
            check_fresh_run(config["adapter"], repo_files)
        versions = {package: version(package) for package in VERSIONED}
        stamps = {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "dataset_revision": hub.dataset_info(config["dataset"]).sha,
            "gpu": _gpu_name(),
        }
    else:
        resumed = wandb.Api().run(f"{config['wandb_project']}/{resume}")
        global_step = None
        if has_checkpoint(repo_files):
            # into the HF cache: a copy in output_dir would be re-uploaded at every save
            snapshot = snapshot_download(config["adapter"], allow_patterns=f"{CHECKPOINT}/*")
            checkpoint = str(Path(snapshot, CHECKPOINT))
            state = json.loads(Path(checkpoint, "trainer_state.json").read_text(encoding="utf-8"))
            global_step = state["global_step"]
        check_resume(
            config, resumed.config, global_step=global_step, torch_version=torch.__version__
        )
        versions = resumed.config["versions"]
        stamps = {key: resumed.config[key] for key in STAMPS}
        print(f"resume: run {resume} from step {global_step}")
    revision = stamps["dataset_revision"]
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
    adopt_chat_template(tokenizer, AutoTokenizer.from_pretrained(config["model"]))
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
        config=run_config(config, derived, versions, stamps),
        id=resume,
        resume="must" if resume else None,
    )
    if resume is None:
        print(f"W&B run id: {run.id} (after a disconnect: --resume {run.id})")
    else:
        assert global_step is not None  # check_resume refused a missing checkpoint
        run.summary["resumes"] = resumes(
            resumed.summary_metrics.get("resumes"),
            step=global_step,
            git_sha=_git("rev-parse", "HEAD"),
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
            push_to_hub=push,
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

    trainer.train(resume_from_checkpoint=checkpoint)  # a path: True searches output_dir
    if push:
        commit = trainer.push_to_hub(commit_message="End of training")
        run.summary.update(
            {
                "adapter_revision": commit.oid,
                "adapter_url": f"https://huggingface.co/{config['adapter']}",
            }
        )

    first_quarter, last_quarter = _loss_quarters(trainer.state.log_history)
    run.summary.update(
        {
            "loss_first_quarter": first_quarter,
            "loss_last_quarter": last_quarter,
            "first_batch_kept_tokens": n_kept,
            "n_dropped": n_dropped,
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


@dataclass
class Invocation:
    """What one ``python -m train`` asks for: the experiment, whether to push, what to resume."""

    config: dict[str, Any]
    push: bool
    resume: str | None


def parse_invocation(argv: Sequence[str] | None = None) -> Invocation:
    """The command line as an ``Invocation``: a ``--limit`` run is a check that never pushes."""
    parser = argparse.ArgumentParser(description="Fine-tune one experiment and push its adapter.")
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    # exclusive: a --limit run pushes nothing, so it leaves no checkpoint to resume
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--limit", type=int, help="only the first N train and dev documents; never pushes"
    )
    group.add_argument("--resume", metavar="WANDB_RUN_ID", help="continue this interrupted run")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.limit is not None:
        config["train_docs"] = config["dev_docs"] = args.limit
    return Invocation(config, push=args.limit is None, resume=args.resume)


def main(argv: Sequence[str] | None = None) -> None:
    invocation = parse_invocation(argv)
    print(train(invocation.config, push=invocation.push, resume=invocation.resume))


if __name__ == "__main__":
    main()
