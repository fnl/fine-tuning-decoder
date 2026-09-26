"""Training tests: masking, the sequence budget, subsets, step arithmetic, config and the callback."""

import copy
from dataclasses import asdict, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from data.prepare import TOKENIZER_ID, load, make_example
from eval import flatten, score
from generate import GenerationResult, constant_engine, render_inputs
from train import (
    IGNORE_INDEX,
    MaskingError,
    ResumeError,
    ScoringCallback,
    adopt_chat_template,
    build_dataset,
    check_fresh_run,
    check_resume,
    eval_interval,
    load_config,
    parse_invocation,
    resumes,
    run_config,
    select_subset,
    tokenize_example,
    training_steps,
)

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
SMOKE_CONFIG = Path(__file__).parent.parent / "configs" / "qwen3-0.6b-smoke.yaml"
SMOKE_MODEL = "Qwen/Qwen3-0.6B"
# Unsloth's copy of the base model's tokenizer: what FastLanguageModel returns in training
UNSLOTH_TOKENIZER_ID = "unsloth/Qwen3-4B-Instruct-2507"
MODELS = (SMOKE_MODEL, TOKENIZER_ID)


@pytest.fixture(scope="module")
def tokenizers() -> dict[str, Any]:
    from transformers import AutoTokenizer

    return {model: AutoTokenizer.from_pretrained(model) for model in MODELS}


@pytest.fixture(scope="module", params=MODELS)
def tokenizer(request: pytest.FixtureRequest, tokenizers: dict[str, Any]) -> Any:
    return tokenizers[request.param]


@pytest.fixture(scope="module")
def examples(tokenizers: dict[str, Any]) -> list[dict[str, Any]]:
    """The fixture corpus as examples in dataset form, in corpus order; the first is empty."""
    docs = load("train", root=CORPUS)
    examples = [make_example(doc, tokenizers[TOKENIZER_ID], truncate=False) for doc in docs]
    return [asdict(ex) for ex in examples if ex is not None]


def prompt_length(example: dict[str, Any], tokenizer: Any) -> int:
    """Tokens of the input the engine would receive for ``example`` at inference."""
    [text] = render_inputs([example], [], tokenizer)
    return len(tokenizer.encode(text, add_special_tokens=False))


def kept(labels: list[int]) -> list[int]:
    return [label for label in labels if label != IGNORE_INDEX]


@pytest.mark.tokenizer
def test_every_prompt_token_is_masked(examples: list[dict[str, Any]], tokenizer: Any) -> None:
    for ex in examples:
        labels = tokenize_example(ex, tokenizer)["labels"]
        assert set(labels[: prompt_length(ex, tokenizer)]) == {IGNORE_INDEX}, ex["docid"]


@pytest.mark.tokenizer
def test_every_completion_token_is_kept_from_the_prompt_length_on(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    for ex in examples:
        out = tokenize_example(ex, tokenizer)
        n = prompt_length(ex, tokenizer)
        assert out["labels"][n:] == out["input_ids"][n:], ex["docid"]


@pytest.mark.tokenizer
def test_completion_is_the_target_ending_at_the_end_of_turn_token(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    for ex in examples:
        completion = tokenizer.decode(kept(tokenize_example(ex, tokenizer)["labels"]))
        assert completion == ex["messages"][2]["content"] + "<|im_end|>", ex["docid"]


@pytest.mark.tokenizer
def test_empty_document_keeps_exactly_two_tokens(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    assert len(kept(tokenize_example(examples[0], tokenizer)["labels"])) == 2


@pytest.mark.tokenizer
def test_both_tokenizers_keep_the_same_number_of_tokens(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    counts = {
        model: [len(kept(tokenize_example(ex, tok)["labels"])) for ex in examples]
        for model, tok in tokenizers.items()
    }
    assert counts[SMOKE_MODEL] == counts[TOKENIZER_ID]


@pytest.mark.tokenizer
def test_prompt_that_is_not_a_token_prefix_fails_naming_the_document(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    ex = copy.deepcopy(examples[0])
    # the 0.6B template lifts a reasoning block out of the target, so the full
    # rendering no longer starts with the prompt's empty thinking block
    ex["messages"][2]["content"] = "<think>x</think>[]"
    with pytest.raises(MaskingError, match=ex["docid"]):
        tokenize_example(ex, tokenizers[SMOKE_MODEL])


@pytest.fixture(scope="module")
def unsloth_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(UNSLOTH_TOKENIZER_ID)


@pytest.mark.tokenizer
def test_completion_that_is_not_the_target_fails_naming_the_document(
    examples: list[dict[str, Any]], unsloth_tokenizer: Any
) -> None:
    # Unsloth's template renders an empty thinking block into the assistant turn alone
    with pytest.raises(MaskingError, match=examples[0]["docid"]):
        tokenize_example(examples[0], unsloth_tokenizer)


@pytest.mark.tokenizer
def test_adopting_the_official_template_makes_the_completion_the_target(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(UNSLOTH_TOKENIZER_ID)
    adopt_chat_template(tokenizer, tokenizers[TOKENIZER_ID])
    for ex in examples:
        completion = tokenizer.decode(kept(tokenize_example(ex, tokenizer)["labels"]))
        assert completion == ex["messages"][2]["content"] + "<|im_end|>", ex["docid"]


def lengths(examples: list[dict[str, Any]], tokenizer: Any) -> list[int]:
    return [len(tokenize_example(ex, tokenizer)["input_ids"]) for ex in examples]


@pytest.mark.tokenizer
def test_example_over_the_sequence_budget_is_dropped_and_counted(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    budget = max(lengths(examples, tokenizer)) - 1
    dataset, n_dropped = build_dataset(examples, tokenizer, max_seq_len=budget)
    assert (len(dataset), n_dropped) == (len(examples) - 1, 1)


@pytest.mark.tokenizer
def test_example_within_the_sequence_budget_is_kept_untruncated(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    budget = max(lengths(examples, tokenizer)) - 1
    dataset, _ = build_dataset(examples, tokenizer, max_seq_len=budget)
    assert dataset[0] == tokenize_example(examples[0], tokenizer)


SPLIT = [{"docid": f"D-{i}"} for i in range(5)]


def test_subset_is_the_first_n_examples() -> None:
    assert select_subset(SPLIT, 2) == SPLIT[:2]


def test_absent_subset_size_means_the_whole_split() -> None:
    assert select_subset(SPLIT, None) == SPLIT


def test_subset_larger_than_the_split_is_the_whole_split() -> None:
    assert select_subset(SPLIT, 10) == SPLIT


def test_smoke_run_takes_seven_optimizer_steps_per_epoch_for_three_epochs() -> None:
    # 100 documents at effective batch 16: six full steps plus a last partial one,
    # which the trainer counts (it rounds the final accumulation up)
    assert training_steps(100, effective_batch_size=16, epochs=3) == 21


def test_single_epoch_takes_one_epoch_of_steps() -> None:
    assert training_steps(100, effective_batch_size=16, epochs=1) == 7


def test_smoke_run_evaluates_every_three_steps() -> None:
    assert eval_interval(21, eval_every=0.5, epochs=3) == 3


def test_non_integer_interval_rounds_down_to_evaluate_at_least_as_often() -> None:
    assert eval_interval(21, eval_every=1 / 3, epochs=3) == 2


def test_interval_below_one_step_evaluates_every_step() -> None:
    # --limit 8: one step per epoch
    assert eval_interval(3, eval_every=0.5, epochs=3) == 1


def test_interval_beyond_the_run_evaluates_once_at_the_last_step() -> None:
    assert eval_interval(7, eval_every=2, epochs=1) == 7


def smoke_config() -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
    return config


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_smoke_experiment_is_accepted_with_every_key() -> None:
    assert load_config(SMOKE_CONFIG) == smoke_config()


def test_config_without_subset_sizes_is_accepted(tmp_path: Path) -> None:
    config = smoke_config()
    del config["train_docs"], config["dev_docs"]
    assert load_config(write_config(tmp_path, config)) == config


def test_unknown_key_is_rejected_by_name(tmp_path: Path) -> None:
    config = {**smoke_config(), "learning_rate": 1e-4}
    with pytest.raises(ValueError, match="learning_rate"):
        load_config(write_config(tmp_path, config))


def test_unknown_nested_key_is_rejected_by_name(tmp_path: Path) -> None:
    config = smoke_config()
    config["lora"]["lora_alpha"] = 32
    with pytest.raises(ValueError, match="lora_alpha"):
        load_config(write_config(tmp_path, config))


def test_missing_key_is_rejected_by_name(tmp_path: Path) -> None:
    config = smoke_config()
    del config["adapter"]
    with pytest.raises(ValueError, match="adapter"):
        load_config(write_config(tmp_path, config))


def test_indivisible_batch_sizes_are_rejected(tmp_path: Path) -> None:
    config = {**smoke_config(), "per_device_batch_size": 3}
    with pytest.raises(ValueError, match="effective_batch_size"):
        load_config(write_config(tmp_path, config))


def test_unknown_quantization_is_rejected(tmp_path: Path) -> None:
    config = {**smoke_config(), "quantization": "nf8"}
    with pytest.raises(ValueError, match="nf8"):
        load_config(write_config(tmp_path, config))


class RecordingLog:
    """Fake ``wandb.log``: remembers every call's positional and keyword arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


def run_steps(
    examples: list[dict[str, Any]], tokenizer: Any, *, max_steps: int, interval: int
) -> RecordingLog:
    """Drive a callback over every step of a fake run with the constant engine."""
    log = RecordingLog()
    callback = ScoringCallback(
        examples, constant_engine(), tokenizer, interval=interval, max_input_tokens=2560, log=log
    )
    for step in range(1, max_steps + 1):
        state = SimpleNamespace(global_step=step, max_steps=max_steps)
        callback.on_step_end(None, state, None)
    return log


def score_once(examples: list[dict[str, Any]], tokenizer: Any, engine: Any) -> dict[str, Any]:
    """What a callback logs at its one eval point, generating with ``engine``."""
    log = RecordingLog()
    callback = ScoringCallback(
        examples, engine, tokenizer, interval=1, max_input_tokens=2560, log=log
    )
    callback.on_step_end(None, SimpleNamespace(global_step=1, max_steps=1), None)
    logged: dict[str, Any] = log.calls[0][0][0]
    return logged


@pytest.mark.tokenizer
def test_callback_scores_at_every_interval_and_at_the_last_step(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    log = run_steps(examples, tokenizers[TOKENIZER_ID], max_steps=7, interval=3)
    assert [args[0]["train/global_step"] for args, _ in log.calls] == [3, 6, 7]


@pytest.mark.tokenizer
def test_callback_logs_the_scorer_keys_cut_offs_predictions_and_the_global_step(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    log = run_steps(examples, tokenizers[TOKENIZER_ID], max_steps=3, interval=3)
    expected = {f"dev/{key}" for key in flatten(score({}, {}))} | {
        "dev/diagnostics/n_cut_off",
        "predictions",
        "train/global_step",
    }
    assert set(log.calls[0][0][0]) == expected


@pytest.mark.tokenizer
def test_callback_never_passes_an_explicit_step(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    log = run_steps(examples, tokenizers[TOKENIZER_ID], max_steps=7, interval=3)
    assert all(kwargs == {} for _, kwargs in log.calls)


@pytest.mark.tokenizer
def test_callback_scores_the_generated_outputs_against_the_targets(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    targets = [ex["messages"][2]["content"] for ex in examples]
    engine = lambda inputs: [GenerationResult(t, "stop") for t in targets]
    assert score_once(examples, tokenizers[TOKENIZER_ID], engine)["dev/micro_avg/f1"] == 1.0


@pytest.mark.tokenizer
def test_callback_counts_the_outputs_cut_off_by_the_length_limit(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    def engine(inputs: list[str]) -> list[GenerationResult]:
        return [GenerationResult("[", "length")] + [GenerationResult("[]", "stop")] * (
            len(inputs) - 1
        )

    logged = score_once(examples, tokenizers[TOKENIZER_ID], engine)
    assert logged["dev/diagnostics/n_cut_off"] == 1


@pytest.mark.tokenizer
def test_callback_logs_one_predictions_row_per_dev_document(
    examples: list[dict[str, Any]], tokenizers: dict[str, Any]
) -> None:
    table = score_once(examples, tokenizers[TOKENIZER_ID], constant_engine())["predictions"]
    assert [row[0] for row in table.data] == [ex["docid"] for ex in examples]


def smoke_run_config() -> dict[str, Any]:
    derived = {"total_steps": 21, "eval_steps": 3, "warmup_steps": 1}
    stamps = {"git_sha": "0" * 40, "git_dirty": False, "dataset_revision": "0" * 40, "gpu": "T4"}
    return run_config(smoke_config(), derived, {"trl": "0.24.0"}, stamps)


def test_run_config_shares_no_top_level_key_with_the_trainer_settings() -> None:
    # the W&B integration overwrites every top-level key named like a trainer setting;
    # warmup_ratio is one in the transformers Colab runs, no longer in the local one
    from transformers import TrainingArguments

    trainer_settings = {field.name for field in fields(TrainingArguments)} | {"warmup_ratio"}
    assert set(smoke_run_config()) & trainer_settings == set()


def test_run_config_keeps_the_experiment_verbatim() -> None:
    assert smoke_run_config()["experiment"] == smoke_config()


TORCH = "2.11.0+cu128"


def resumed_run_config() -> dict[str, Any]:
    """The W&B config of an interrupted smoke run, as the resume reads it back."""
    derived = {"total_steps": 21, "eval_steps": 3, "warmup_steps": 1}
    stamps = {"git_sha": "0" * 40, "git_dirty": False, "dataset_revision": "0" * 40, "gpu": "T4"}
    return run_config(smoke_config(), derived, {"torch": TORCH, "trl": "0.24.0"}, stamps)


def resume(config: dict[str, Any], *, global_step: int | None = 9, torch: str = TORCH) -> None:
    check_resume(config, resumed_run_config(), global_step=global_step, torch_version=torch)


def test_resume_of_an_interrupted_run_with_the_same_config_is_allowed() -> None:
    resume(smoke_config())


def test_resume_on_newer_code_than_the_run_started_on_is_allowed() -> None:
    # the running code's sha is never compared: a bug fix may rescue a run
    run = {**resumed_run_config(), "git_sha": "1" * 40, "git_dirty": True}
    check_resume(smoke_config(), run, global_step=9, torch_version=TORCH)


def test_resume_without_a_checkpoint_is_refused_naming_the_repo() -> None:
    with pytest.raises(ResumeError, match="fnl-es/qwen3-0.6b-muc4-lora-smoke"):
        resume(smoke_config(), global_step=None)


def test_resume_of_a_finished_run_is_refused() -> None:
    with pytest.raises(ResumeError, match="finished"):
        resume(smoke_config(), global_step=21)


def test_resume_of_a_finished_run_is_refused_as_finished_whatever_else_changed() -> None:
    with pytest.raises(ResumeError, match="finished"):
        resume(smoke_config(), global_step=21, torch="2.12.0+cu128")


def test_resume_with_a_different_config_is_refused_naming_every_differing_key() -> None:
    config = {**smoke_config(), "lr": 1.0e-4, "eval_every": 1.0}
    with pytest.raises(ResumeError, match=r"\['eval_every', 'lr'\]"):
        resume(config)


def test_resume_on_a_different_torch_is_refused_naming_both_versions() -> None:
    with pytest.raises(ResumeError, match=r"2\.12\.0\+cu128.*2\.11\.0\+cu128"):
        resume(smoke_config(), torch="2.12.0+cu128")


SHA = "f" * 40


def test_first_resume_starts_the_resumes_record() -> None:
    assert resumes(None, step=9, git_sha=SHA) == [{"step": 9, "git_sha": SHA}]


def test_later_resume_appends_to_the_resumes_record() -> None:
    record = [{"step": 3, "git_sha": "0" * 40}]
    assert resumes(record, step=9, git_sha=SHA) == [
        {"step": 3, "git_sha": "0" * 40},
        {"step": 9, "git_sha": SHA},
    ]


ADAPTER = "fnl-es/qwen3-4b-muc4-lora-r16"
ADAPTER_FILES = [".gitattributes", "README.md", "adapter_config.json", "adapter_model.safetensors"]


def test_fresh_run_on_a_repo_with_a_checkpoint_is_refused_naming_the_repo() -> None:
    with pytest.raises(ResumeError, match=ADAPTER):
        check_fresh_run(ADAPTER, [*ADAPTER_FILES, "last-checkpoint/trainer_state.json"])


def test_fresh_run_on_an_adapter_only_repo_is_allowed() -> None:
    check_fresh_run(ADAPTER, ADAPTER_FILES)


def test_fresh_run_on_a_missing_repo_is_allowed() -> None:
    check_fresh_run(ADAPTER, [])


def test_resume_flag_names_the_run_to_resume() -> None:
    invocation = parse_invocation(["--config", str(SMOKE_CONFIG), "--resume", "abcd1234"])
    assert invocation.resume == "abcd1234"


def test_resume_together_with_limit_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_invocation(["--config", str(SMOKE_CONFIG), "--resume", "abcd1234", "--limit", "8"])


def test_full_run_pushes_to_the_hub() -> None:
    assert parse_invocation(["--config", str(SMOKE_CONFIG)]).push is True


def test_limited_run_never_pushes() -> None:
    assert parse_invocation(["--config", str(SMOKE_CONFIG), "--limit", "8"]).push is False


def test_limited_run_takes_the_first_n_train_and_dev_documents() -> None:
    config = parse_invocation(["--config", str(SMOKE_CONFIG), "--limit", "8"]).config
    assert (config["train_docs"], config["dev_docs"]) == (8, 8)
