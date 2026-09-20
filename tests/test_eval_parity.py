"""Parity of the ported scorer with the vendored original GTT ``eval.py`` on dev gold."""

import copy
import json
import random
from pathlib import Path

import pytest
from oracle import gtt_eval

from data.prepare import ROLES, Event
from eval import SLOTS, score

# Dev gold as the original's __main__ loads it, limited to documents with at most four
# events: the original enumerates (|gold|+1)**|pred| alignments and hangs on the four
# dev documents with more (5, 5, 5 and 9 events).
FIXTURE = Path(__file__).parent / "fixtures" / "dev_gold.json"
COUNTS = ("p_num", "p_den", "r_num", "r_den")


def dev_gold() -> dict[str, list[Event]]:
    with FIXTURE.open(encoding="utf-8") as f:
        golds: dict[str, list[Event]] = json.load(f)
    return golds


def perturbed(golds: dict[str, list[Event]], seed: int) -> dict[str, list[Event]]:
    """Gold with random event/entity drops, mention edits, type swaps and spurious events."""
    rng = random.Random(seed)
    pool = [ev for events in golds.values() for ev in events]
    preds = copy.deepcopy(golds)
    for events in preds.values():
        events[:] = [ev for ev in events if rng.random() > 0.2]
        for ev in events:
            if rng.random() < 0.1:
                ev["incident_type"] = rng.choice(["attack", "bombing", "kidnapping"])
            for role in ROLES:
                ev[role] = [e for e in ev[role] if rng.random() > 0.2]
                for entity in ev[role]:
                    for k, mention in enumerate(entity):
                        edit = rng.random()
                        if edit < 0.1:
                            entity[k] = mention.upper()
                        elif edit < 0.2:
                            entity[k] = f"the {mention}!"
                        elif edit < 0.3:
                            entity[k] = f"{mention} extra"
        if rng.random() < 0.15:
            events.append(copy.deepcopy(rng.choice(pool)))
        rng.shuffle(events)
    return preds


def counts(result: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {slot: {c: result[slot][c] for c in COUNTS} for slot in (*SLOTS, "micro_avg")}


@pytest.mark.parametrize("seed", [None, 0, 1])
def test_ported_scorer_matches_original_counts(seed: int | None) -> None:
    golds = dev_gold()
    preds = golds if seed is None else perturbed(golds, seed)
    ours = score(preds, golds)
    theirs = gtt_eval.eval_tf(copy.deepcopy(preds), copy.deepcopy(golds), docids=list(golds))
    assert counts(ours) == counts(theirs)
