"""Scorer tests: hand-built gold/prediction pairs driven through ``score``."""

from data.prepare import Event, event
from eval import score


def attack(**roles: list[list[str]]) -> Event:
    return event("attack", **roles)


def bombing(**roles: list[list[str]]) -> Event:
    return event("bombing", **roles)


def test_identical_prediction_scores_perfect_micro_f1() -> None:
    gold = {"d1": [attack(PerpInd=[["guerrillas"]], Target=[["bridge"]])]}
    result = score(gold, gold)
    assert result["micro_avg"]["f1"] == 1.0


def test_mention_normalisation_ignores_case_punctuation_and_articles() -> None:
    gold = {"d1": [attack(PerpInd=[["guerrillas"]])]}
    pred = {"d1": [attack(PerpInd=[["The Guerrillas!"]])]}
    assert score(pred, gold)["PerpInd"]["f1"] == 1.0


def test_subset_match_accepts_predicted_mentions_within_one_gold_entity() -> None:
    gold = {"d1": [attack(PerpInd=[["guerrillas", "rebels"]])]}
    pred = {"d1": [attack(PerpInd=[["rebels"]])]}
    assert score(pred, gold)["PerpInd"]["f1"] == 1.0


def test_subset_match_rejects_entity_with_an_extra_mention() -> None:
    gold = {"d1": [attack(PerpInd=[["guerrillas"]])]}
    pred = {"d1": [attack(PerpInd=[["guerrillas", "army"]])]}
    assert score(pred, gold)["PerpInd"]["p_num"] == 0


def test_wrong_event_type_scores_nothing_even_with_correct_roles() -> None:
    gold = {"d1": [attack(Target=[["bridge"]])]}
    pred = {"d1": [bombing(Target=[["bridge"]])]}
    assert score(pred, gold)["micro_avg"]["f1"] == 0.0


def test_spurious_event_counts_against_precision_in_every_slot() -> None:
    gold = {"d1": [attack(Target=[["bridge"]])]}
    pred = {"d1": [attack(Target=[["bridge"]]), bombing(Weapon=[["bomb"]])]}
    # 2 correct slots (type + Target) out of 4 predicted (2 types + Target + Weapon)
    assert score(pred, gold)["micro_avg"]["p"] == 0.5


def test_missing_event_counts_against_recall_in_every_slot() -> None:
    gold = {"d1": [attack(Target=[["bridge"]]), bombing(Weapon=[["bomb"]])]}
    pred = {"d1": [attack(Target=[["bridge"]])]}
    assert score(pred, gold)["micro_avg"]["r"] == 0.5


def test_empty_gold_and_empty_prediction_score_zero_over_zero() -> None:
    result = score({"d1": []}, {"d1": []})
    assert result["micro_avg"] == {
        "p_num": 0,
        "p_den": 0,
        "r_num": 0,
        "r_den": 0,
        "p": 0,
        "r": 0,
        "f1": 0,
    }


def test_document_absent_from_predictions_scores_as_empty_prediction() -> None:
    gold = {"d1": [attack(Target=[["bridge"]])]}
    assert score({}, gold) == score({"d1": []}, gold)


def test_alignment_prefers_best_mapping_over_order_preserving_one() -> None:
    gold = {"d1": [attack(Target=[["bridge"]]), attack(Target=[["school"]])]}
    pred = {"d1": [attack(Target=[["school"]]), attack(Target=[["bridge"]])]}
    assert score(pred, gold)["micro_avg"]["f1"] == 1.0


def test_scoring_leaves_inputs_unmodified() -> None:
    pred = {"d1": [attack(PerpInd=[["The Guerrillas!"]])]}
    score(pred, {"d1": [attack(PerpInd=[["guerrillas"]])]})
    assert pred == {"d1": [attack(PerpInd=[["The Guerrillas!"]])]}


def test_relevance_accuracy_is_share_of_documents_agreeing_on_relevance() -> None:
    gold = {"d1": [attack(Target=[["bridge"]])], "d2": []}
    pred = {"d1": [], "d2": []}
    assert score(pred, gold)["diagnostics"]["relevance_acc"] == 0.5


def test_event_count_accuracy_is_share_of_documents_with_equal_event_count() -> None:
    gold = {"d1": [attack(), attack()], "d2": [attack()]}
    pred = {"d1": [attack()], "d2": [bombing()]}
    assert score(pred, gold)["diagnostics"]["event_count_acc"] == 0.5


def test_event_type_accuracy_pairs_events_regardless_of_type() -> None:
    gold = {"d1": [attack(Target=[["bridge"]]), bombing(Target=[["school"]])]}
    pred = {"d1": [bombing(Target=[["bridge"]]), bombing(Target=[["school"]])]}
    assert score(pred, gold)["diagnostics"]["event_type_acc"] == 0.5


def test_parse_failure_rate_is_share_of_documents_flagged() -> None:
    gold = {"d1": [attack()], "d2": []}
    assert score({}, gold, parse_failures={"d1"})["diagnostics"]["parse_failure_rate"] == 0.5


def test_greedy_alignment_is_used_and_counted_above_the_mapping_threshold() -> None:
    gold = {"d1": [attack(Target=[[f"target {i}"]]) for i in range(9)]}
    pred = {"d1": gold["d1"][:7]}  # (9 + 1) ** 7 candidate alignments > 1_000_000
    assert score(pred, gold)["diagnostics"]["greedy_alignment_docs"] == 1


def test_greedy_alignment_still_finds_the_perfect_alignment() -> None:
    gold = {"d1": [attack(Target=[[f"target {i}"]]) for i in range(9)]}
    pred = {"d1": list(reversed(gold["d1"][:7]))}
    assert score(pred, gold)["micro_avg"]["p"] == 1.0
