"""Generation tests: rendering, the input-limit check, the constant engine and the CLI."""

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from data.prepare import SYSTEM_PROMPT, TOKENIZER_ID, load, make_example
from eval import read_rows
from generate import GenerationResult, constant_engine, generate
from generate import main as generate_main

CORPUS = Path(__file__).parent / "fixtures" / "corpus"
MAX_INPUT_TOKENS = 3072 - 512  # the baselines' max_model_len - max_new_tokens


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.fixture(scope="module")
def examples(tokenizer: Any) -> list[dict[str, Any]]:
    """The fixture corpus as examples in dataset form, in corpus order."""
    examples = [make_example(doc, tokenizer, truncate=False) for doc in load("train", root=CORPUS)]
    return [asdict(ex) for ex in examples if ex is not None]


class RecordingEngine:
    """Fake engine: remembers the inputs it received and answers ``[]`` for each."""

    def __init__(self, finish_reason: str = "stop") -> None:
        self.finish_reason = finish_reason
        self.inputs: list[str] = []

    def __call__(self, inputs: list[str]) -> list[GenerationResult]:
        self.inputs = list(inputs)
        return [GenerationResult("[]", self.finish_reason) for _ in inputs]


def turns(text: str) -> list[str]:
    """The roles of the rendered turns, in order."""
    return re.findall(r"<\|im_start\|>(\w+)", text)


def rendered(examples: list[dict[str, Any]], tokenizer: Any, exemplars: Any = ()) -> str:
    """The one input the engine receives for ``examples`` (a single example)."""
    engine = RecordingEngine()
    generate(examples, engine, tokenizer, exemplars, max_input_tokens=MAX_INPUT_TOKENS)
    return engine.inputs[0]


@pytest.mark.tokenizer
def test_zero_shot_input_is_system_then_document_then_generation_prompt(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    assert turns(rendered(examples[:1], tokenizer)) == ["system", "user", "assistant"]


@pytest.mark.tokenizer
def test_input_holds_system_prompt_and_document_verbatim(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    text = rendered(examples[:1], tokenizer)
    document = examples[0]["messages"][1]["content"]
    assert f"system\n{SYSTEM_PROMPT}<|im_end|>" in text
    assert f"user\n{document}<|im_end|>\n<|im_start|>assistant\n" in text


@pytest.mark.tokenizer
def test_input_ends_with_generation_prompt_and_has_no_thinking_block(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    text = rendered(examples[:1], tokenizer)
    assert text.endswith("<|im_start|>assistant\n") and "<think>" not in text


@pytest.mark.tokenizer
def test_three_shot_input_splices_exemplar_turns_between_system_prompt_and_document(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    text = rendered(examples[3:4], tokenizer, exemplars=examples[:3])
    assert turns(text) == ["system"] + ["user", "assistant"] * 3 + ["user", "assistant"]


@pytest.mark.tokenizer
def test_exemplar_documents_and_targets_appear_verbatim_in_order_before_the_document(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    text = rendered(examples[3:4], tokenizer, exemplars=examples[:3])
    expected = [m["content"] for ex in examples[:3] for m in ex["messages"][1:]]
    expected.append(examples[3]["messages"][1]["content"])
    position = len(SYSTEM_PROMPT)
    for content in expected:  # each piece must follow the previous one
        position = text.find(content, position)
        assert position != -1, content


@pytest.mark.tokenizer
def test_rows_carry_docid_output_and_cut_off_per_example(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    out = generate(examples[:2], RecordingEngine(), tokenizer, max_input_tokens=MAX_INPUT_TOKENS)
    assert out == [
        {"docid": "DEV-MUC3-0005", "output": "[]", "cut_off": False},
        {"docid": "DEV-MUC3-0018", "output": "[]", "cut_off": False},
    ]


@pytest.mark.tokenizer
def test_output_stopped_by_the_length_limit_is_cut_off(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    out = generate(
        examples[:1], RecordingEngine("length"), tokenizer, max_input_tokens=MAX_INPUT_TOKENS
    )
    assert out[0]["cut_off"] is True


@pytest.mark.tokenizer
def test_input_over_the_engine_limit_fails_naming_the_docid(
    examples: list[dict[str, Any]], tokenizer: Any
) -> None:
    with pytest.raises(ValueError, match="DEV-MUC3-0018"):
        limit = examples[0]["n_input_tokens"]
        generate(examples[:2], RecordingEngine(), tokenizer, max_input_tokens=limit)


def test_constant_engine_answers_an_empty_array_stopped_normally_for_every_input() -> None:
    assert constant_engine()(["one", "two"]) == [
        GenerationResult("[]", "stop"),
        GenerationResult("[]", "stop"),
    ]


ALWAYS_EMPTY_YAML = """\
name: always-empty
tags: [baseline, always-empty]
model: Qwen/Qwen3-4B-Instruct-2507
dataset: fnl-es/muc4-chat
engine: constant
exemplars: []
generation:
  max_new_tokens: 512
  max_model_len: 3072
wandb_project: muc4-event-extraction
"""


@pytest.fixture(scope="module")
def cli_out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One CLI run with the constant engine on the first three dev documents."""
    root = tmp_path_factory.mktemp("cli")
    config = root / "always-empty.yaml"
    config.write_text(ALWAYS_EMPTY_YAML, encoding="utf-8")
    out = root / "out"
    generate_main(["--config", str(config), "--limit", "3", "--out", str(out)])
    return out


@pytest.mark.tokenizer
def test_cli_writes_one_row_per_document_up_to_the_limit(cli_out: Path) -> None:
    assert read_rows(cli_out / "dev.jsonl") == [
        {"docid": "TST1-MUC3-0001", "output": "[]", "cut_off": False},
        {"docid": "TST1-MUC3-0002", "output": "[]", "cut_off": False},
        {"docid": "TST1-MUC3-0003", "output": "[]", "cut_off": False},
    ]


@pytest.mark.tokenizer
def test_cli_metadata_names_model_engine_split_device_and_counts(cli_out: Path) -> None:
    meta = json.loads((cli_out / "meta.json").read_text(encoding="utf-8"))
    for key in ("git_sha", "git_dirty", "dataset_revision", "wall_seconds"):
        del meta[key]  # provenance and timing: covered by the next test
    assert meta == {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "engine": "constant",
        "engine_version": "constant",
        "split": "dev",
        "gpu": "cpu",
        "n_docs": 3,
        "n_truncated": 0,
        "n_cut_off": 0,
    }


@pytest.mark.tokenizer
def test_cli_metadata_stamps_provenance_and_timing(cli_out: Path) -> None:
    meta = json.loads((cli_out / "meta.json").read_text(encoding="utf-8"))
    assert (
        bool(re.fullmatch(r"[0-9a-f]{40}", meta["git_sha"])),
        isinstance(meta["git_dirty"], bool),
        bool(re.fullmatch(r"[0-9a-f]{40}", meta["dataset_revision"])),
        meta["wall_seconds"] >= 0.0,
    ) == (True, True, True, True)
