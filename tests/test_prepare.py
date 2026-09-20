"""Codec tests: corpus loading, canonical events, targets, examples and parsing."""

import random
from pathlib import Path
from typing import Any

import pytest

from data.prepare import (
    INPUT_BUDGET,
    SYSTEM_PROMPT,
    TOKENIZER_ID,
    CorpusEvent,
    Doc,
    canonical_events,
    corpus_event,
    event,
    load,
    make_example,
    parse_target,
    render_target,
)

CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def test_load_returns_one_document_per_corpus_line() -> None:
    docs = load("train", root=CORPUS)
    assert [d.docid for d in docs] == [
        "DEV-MUC3-0005",
        "DEV-MUC3-0018",
        "DEV-MUC3-0042",
        "DEV-MUC3-0112",
        "DEV-MUC3-0231",
    ]


def test_load_rejects_unknown_split_name() -> None:
    with pytest.raises(ValueError, match="split"):
        load("validation", root=CORPUS)


def test_load_rejects_event_missing_a_role(tmp_path: Path) -> None:
    line = '{"docid": "X-1", "doctext": "t", "templates": [{"incident_type": "attack"}]}\n'
    (tmp_path / "dev.json").write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match="X-1"):
        load("dev", root=tmp_path)


def by_docid(docid: str) -> Doc:
    return next(d for d in load("train", root=CORPUS) if d.docid == docid)


def test_irrelevant_document_has_no_canonical_events() -> None:
    assert canonical_events(by_docid("DEV-MUC3-0005")) == []


def test_duplicate_events_are_collapsed_to_one() -> None:
    # six annotated events, three of them identical once offsets are removed
    assert len(canonical_events(by_docid("DEV-MUC3-0112"))) == 4


def test_events_are_ordered_by_earliest_mention_offset() -> None:
    later = corpus_event("attack", Target=[[("bridge", 100)]])
    earlier = corpus_event("bombing", Weapon=[[("bomb", 5)]])
    doc = Doc("d", "", [later, earlier])
    assert canonical_events(doc) == [
        event("bombing", Weapon=[["bomb"]]),
        event("attack", Target=[["bridge"]]),
    ]


def test_events_without_mentions_come_after_events_with_mentions() -> None:
    doc = Doc("d", "", [corpus_event("attack"), corpus_event("bombing", Weapon=[[("bomb", 500)]])])
    assert canonical_events(doc) == [event("bombing", Weapon=[["bomb"]]), event("attack")]


def test_events_sharing_the_earliest_offset_are_ordered_by_type_then_json() -> None:
    doc = Doc(
        "d",
        "",
        [
            corpus_event("bombing", Target=[[("school", 7)]]),
            corpus_event("attack", Target=[[("bridge", 7)]]),
            corpus_event("attack", PerpInd=[[("rebels", 7)]]),
        ],
    )
    assert [ev["incident_type"] for ev in canonical_events(doc)] == ["attack", "attack", "bombing"]


def test_canonical_events_do_not_depend_on_annotation_order() -> None:
    doc = by_docid("DEV-MUC3-0112")
    expected = canonical_events(doc)
    shuffled = Doc(doc.docid, doc.text, random.Random(3).sample(doc.events, len(doc.events)))
    assert canonical_events(shuffled) == expected


def test_mentions_within_an_entity_are_ordered_by_offset_then_string() -> None:
    doc = by_docid("DEV-MUC3-0042")  # "dynamite sticks" and "dynamite" share offset 22
    assert canonical_events(doc)[0]["Weapon"] == [["dynamite", "dynamite sticks"]]


def test_empty_entities_are_dropped() -> None:
    ev: CorpusEvent = corpus_event("attack", Target=[[], [("bridge", 1)]])
    assert canonical_events(Doc("d", "", [ev]))[0]["Target"] == [["bridge"]]


def test_irrelevant_document_target_is_an_empty_array() -> None:
    assert render_target([]) == "[]"


def test_target_is_compact_json_with_type_first_and_empty_roles_omitted() -> None:
    events = [event("attack", Weapon=[["bomb"]], PerpInd=[["rebels", "the rebels"]])]
    assert (
        render_target(events)
        == '[{"incident_type":"attack","PerpInd":[["rebels","the rebels"]],"Weapon":[["bomb"]]}]'
    )


def test_target_keeps_non_ascii_mentions_verbatim() -> None:
    assert (
        render_target([event("attack", Victim=[["niño"]])])
        == '[{"incident_type":"attack","Victim":[["niño"]]}]'
    )


@pytest.mark.parametrize("docid", [d.docid for d in load("train", root=CORPUS)])
def test_parsing_a_rendered_target_round_trips_canonical_events(docid: str) -> None:
    events = canonical_events(by_docid(docid))
    assert parse_target(render_target(events)) == (events, True)


CUT_OFF_EVENTS = canonical_events(by_docid("DEV-MUC3-0112"))
CUT_OFF_TARGET = render_target(CUT_OFF_EVENTS)


@pytest.mark.parametrize("cut", range(len(CUT_OFF_TARGET)))
def test_cut_off_output_salvages_every_complete_event(cut: int) -> None:
    # an event is complete once the output includes its closing brace
    ends = [
        CUT_OFF_TARGET.index(render_target([ev])[1:-1]) + len(render_target([ev])) - 2
        for ev in CUT_OFF_EVENTS
    ]
    n_complete = sum(cut >= end for end in ends)
    assert parse_target(CUT_OFF_TARGET[:cut])[0] == CUT_OFF_EVENTS[:n_complete]


def test_cut_off_output_with_one_complete_event_is_not_a_parse_failure() -> None:
    target = render_target(
        [event("attack", Target=[["bridge"]]), event("bombing", Weapon=[["bomb"]])]
    )
    assert parse_target(target[:-8])[1] is True


def test_output_with_no_complete_event_is_a_parse_failure() -> None:
    assert parse_target('[{"incident_type":"att') == ([], False)


def test_output_without_json_is_a_parse_failure() -> None:
    assert parse_target("There are no incidents in this document.") == ([], False)


def test_parser_tolerates_prose_and_markdown_fence() -> None:
    output = 'Here are the events:\n```json\n[{"incident_type":"attack","Target":[["bridge"]]}]\n```\nDone.'
    assert parse_target(output) == ([event("attack", Target=[["bridge"]])], True)


def test_parser_fills_missing_roles_and_drops_unknown_keys() -> None:
    assert parse_target('[{"incident_type":"arson","Date":"1989","Target":[["bank"]]}]')[0] == [
        event("arson", Target=[["bank"]])
    ]


def test_parser_coerces_missing_event_type_to_attack() -> None:
    assert parse_target('[{"Target":[["bank"]]}]')[0] == [event("attack", Target=[["bank"]])]


def test_parser_coerces_bare_string_entity_to_single_mention_entity() -> None:
    assert parse_target('[{"incident_type":"attack","Target":["bank"]}]')[0] == [
        event("attack", Target=[["bank"]])
    ]


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@pytest.mark.tokenizer
def test_example_has_system_user_and_assistant_messages(tokenizer: Any) -> None:
    doc = by_docid("DEV-MUC3-0042")
    example = make_example(doc, tokenizer, truncate=False)
    assert example is not None and example.messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": doc.text},
        {"role": "assistant", "content": render_target(canonical_events(doc))},
    ]


@pytest.mark.tokenizer
def test_short_document_fits_the_input_budget_untruncated(tokenizer: Any) -> None:
    example = make_example(by_docid("DEV-MUC3-0042"), tokenizer, truncate=False)
    assert example is not None and (example.truncated, example.n_input_tokens <= INPUT_BUDGET) == (
        False,
        True,
    )


@pytest.mark.tokenizer
def test_input_rendering_ends_with_generation_prompt_and_no_thinking_block(tokenizer: Any) -> None:
    example = make_example(by_docid("DEV-MUC3-0042"), tokenizer, truncate=False)
    assert example is not None
    rendered = tokenizer.apply_chat_template(
        example.messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert rendered.endswith("<|im_start|>assistant\n") and "<think>" not in rendered


@pytest.mark.tokenizer
def test_system_prompt_is_about_150_tokens(tokenizer: Any) -> None:
    assert 100 <= len(tokenizer.encode(SYSTEM_PROMPT)) <= 250


def long_document() -> Doc:
    doc = by_docid("DEV-MUC3-0042")
    return Doc(doc.docid, doc.text * 12, doc.events)  # ~2000+ tokens


@pytest.mark.tokenizer
def test_long_train_document_is_dropped(tokenizer: Any) -> None:
    assert make_example(long_document(), tokenizer, truncate=False) is None


@pytest.mark.tokenizer
def test_long_dev_document_is_truncated_to_the_input_budget(tokenizer: Any) -> None:
    example = make_example(long_document(), tokenizer, truncate=True)
    assert example is not None and (example.truncated, example.n_input_tokens <= INPUT_BUDGET) == (
        True,
        True,
    )


@pytest.mark.tokenizer
def test_truncated_document_keeps_a_prefix_of_the_text(tokenizer: Any) -> None:
    doc = long_document()
    example = make_example(doc, tokenizer, truncate=True)
    assert example is not None and doc.text.startswith(example.messages[1]["content"])


def test_cut_off_output_with_an_empty_role_still_salvages_the_complete_event() -> None:
    output = '[{"incident_type":"attack","Target":[["bridge"]]},{"incident_type":"bombing","PerpInd":[],"Weapon":[["bo'
    assert parse_target(output) == ([event("attack", Target=[["bridge"]])], True)


def test_parser_skips_an_empty_array_in_prose_before_the_events() -> None:
    output = 'Not [] this time: [{"incident_type":"attack","Target":[["bridge"]]}]'
    assert parse_target(output) == ([event("attack", Target=[["bridge"]])], True)
