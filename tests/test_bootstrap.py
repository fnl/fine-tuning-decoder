"""Bootstrap interval tests: hand-built gold/prediction pairs driven through ``bootstrap_f1``."""

from bootstrap import bootstrap_f1
from data.prepare import event

GOLD = {
    "d1": [event("attack", Target=[["bridge"]])],
    "d2": [event("attack", Target=[["bridge"]])],
}
# d1 right in both slots; d2 has the wrong type, so its event scores nothing: F1 = 2/4 = 0.5
MIXED = {
    "d1": [event("attack", Target=[["bridge"]])],
    "d2": [event("bombing", Target=[["bridge"]])],
}


def test_point_estimate_is_the_micro_f1_of_the_whole_split() -> None:
    assert bootstrap_f1(MIXED, GOLD).point == 0.5


def test_perfect_predictions_give_a_zero_width_interval() -> None:
    interval = bootstrap_f1(GOLD, GOLD)
    assert (interval.low, interval.high) == (1.0, 1.0)


def test_a_fixed_seed_reproduces_the_interval() -> None:
    assert bootstrap_f1(MIXED, GOLD, seed=7) == bootstrap_f1(MIXED, GOLD, seed=7)


def test_interval_brackets_the_point_estimate() -> None:
    interval = bootstrap_f1(MIXED, GOLD)
    assert interval.low <= interval.point <= interval.high
