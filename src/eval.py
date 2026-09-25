"""Port of the GTT scorer (xinyadu/gtt ``eval.py``, MIT) plus DESIGN §8 diagnostics.

The alignment, subset match, spurious/missing accounting and micro average follow
the original line for line so numbers stay comparable to published GTT/GRIT
results. Differences from the original are deliberate and listed here:

* inputs are deep-copied, never mutated;
* a document in ``golds`` but absent from ``preds`` scores as an empty prediction;
* event types are compared by equality (the original's substring check is
  equivalent on the closed six-type vocabulary);
* above ``MAX_ALIGNMENTS`` candidate alignments a greedy alignment replaces the
  exhaustive search (the original would hang);
* ``diagnostics`` are added to the result dict.
"""

import argparse
import copy
import itertools
import json
import re
import string
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from data.prepare import ROLES, Event, parse_target, render_target

Result = dict[str, Any]

# Slots scored per event: the event type plus the five roles, in GTT order.
SLOTS: tuple[str, ...] = ("incident_type", *ROLES)
# Documents with more candidate alignments than this fall back to a greedy alignment.
MAX_ALIGNMENTS = 1_000_000

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = set(string.punctuation)


def normalize_mention(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace (GTT)."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", s).split())


def f1(p_num: int, p_den: int, r_num: int, r_den: int) -> float:
    p = 0.0 if p_den == 0 else p_num / float(p_den)
    r = 0.0 if r_den == 0 else r_num / float(r_den)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def matching(gold: list[str], pred: list[str]) -> bool:
    """Subset match: the predicted entity is correct iff all its mentions are in the gold entity."""
    return all(m in gold for m in pred)


# Per-slot counts of one document (or one aligned pair): slot -> [p_num, p_den, r_num, r_den].
Counts = dict[str, list[int]]


def _empty_counts() -> Counts:
    return {slot: [0, 0, 0, 0] for slot in SLOTS}


def _pair_counts(pred: Event, gold: Event, *, with_type: bool) -> Counts:
    """Numerators earned by aligning ``pred`` with ``gold`` (denominators are per document)."""
    counts = _empty_counts()
    if with_type:
        counts["incident_type"] = [1, 0, 1, 0]
    for role in ROLES:
        c = counts[role]
        for entity_pred in pred[role]:
            if any(matching(entity_gold, entity_pred) for entity_gold in gold[role]):
                c[0] += 1
        for entity_gold in gold[role]:
            if any(matching(entity_gold, entity_pred) for entity_pred in pred[role]):
                c[2] += 1
    return counts


def _doc_denominators(pred: list[Event], gold: list[Event]) -> Counts:
    """Denominators, identical under every alignment: every pred slot and every gold slot."""
    counts = _empty_counts()
    for ev in pred:
        counts["incident_type"][1] += 1
        for role in ROLES:
            counts[role][1] += len(ev[role])
    for ev in gold:
        counts["incident_type"][3] += 1
        for role in ROLES:
            counts[role][3] += len(ev[role])
    return counts


def _add(total: Counts, part: Counts) -> None:
    for slot in SLOTS:
        for i in range(4):
            total[slot][i] += part[slot][i]


def _micro_f1(counts: Counts) -> float:
    sums = [sum(counts[slot][i] for slot in SLOTS) for i in range(4)]
    return f1(*sums)


class Alignment(NamedTuple):
    pairs: dict[int, int]  # pred index -> gold index, or -1 when unaligned
    counts: Counts
    greedy: bool


def _align(pred: list[Event], gold: list[Event], *, type_agnostic: bool = False) -> Alignment:
    """Best one-to-one alignment of predicted to gold events and its per-slot counts.

    Headline mode reproduces GTT: a pair of unequal type earns nothing (both events
    count as spurious/missing) and the first candidate in ``itertools.product``
    order reaching the best micro-F1 wins. ``type_agnostic`` pairs events by roles
    alone (the event-type slot is ignored) for the event-type accuracy diagnostic;
    a pair sharing no entity then ties with leaving both unaligned, and the
    enumeration order pairs them, so the diagnostic also counts such pairs.
    """
    base = _doc_denominators(pred, gold)
    gains: dict[tuple[int, int], Counts] = {}
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if type_agnostic:
                gains[i, j] = _pair_counts(p, g, with_type=False)
            elif p["incident_type"] == g["incident_type"]:
                gains[i, j] = _pair_counts(p, g, with_type=True)

    greedy = (len(gold) + 1) ** len(pred) > MAX_ALIGNMENTS
    if greedy:
        pairs = _greedy(gains, len(pred))
    else:
        pairs = _exhaustive(gains, base, len(pred), len(gold))

    counts = copy.deepcopy(base)
    for i, j in pairs.items():
        if (i, j) in gains:
            _add(counts, gains[i, j])
    return Alignment(pairs, counts, greedy)


def _exhaustive(
    gains: dict[tuple[int, int], Counts], base: Counts, n_pred: int, n_gold: int
) -> dict[int, int]:
    best_f1 = -1.0
    best: dict[int, int] = {}
    for assignment in itertools.product([*range(n_gold), -1], repeat=n_pred):
        used = [j for j in assignment if j != -1]
        if len(used) != len(set(used)):
            continue
        counts = copy.deepcopy(base)
        for i, j in enumerate(assignment):
            if (i, j) in gains:
                _add(counts, gains[i, j])
        this_f1 = _micro_f1(counts)
        if this_f1 > best_f1:
            best_f1, best = this_f1, dict(enumerate(assignment))
    return best


def _greedy(gains: dict[tuple[int, int], Counts], n_pred: int) -> dict[int, int]:
    """Best-scoring pair first; ties in ``(pred, gold)`` index order."""
    pairs = {i: -1 for i in range(n_pred)}
    used_gold: set[int] = set()
    for (i, j), _ in sorted(gains.items(), key=lambda kv: -_gain(kv[1])):
        if pairs[i] == -1 and j not in used_gold:
            pairs[i] = j
            used_gold.add(j)
    return pairs


def _gain(counts: Counts) -> int:
    return sum(counts[slot][0] + counts[slot][2] for slot in SLOTS)


def _normalized(docs: dict[str, list[Event]]) -> dict[str, list[Event]]:
    out = copy.deepcopy(docs)
    for events in out.values():
        for ev in events:
            for role in ROLES:
                ev[role] = [[normalize_mention(m) for m in entity] for entity in ev[role]]
    return out


def score(
    preds: dict[str, list[Event]],
    golds: dict[str, list[Event]],
    parse_failures: Iterable[str] = (),
) -> Result:
    """Score predictions against gold, both keyed by docid; see the module docstring."""
    preds = _normalized(preds)
    golds = _normalized(golds)
    totals = _empty_counts()
    relevance_hits = count_hits = greedy_docs = 0
    type_hits = type_pairs = 0
    for docid, gold in golds.items():
        pred = preds.get(docid, [])
        headline = _align(pred, gold)
        _add(totals, headline.counts)
        greedy_docs += headline.greedy
        relevance_hits += bool(pred) == bool(gold)
        count_hits += len(pred) == len(gold)
        for i, j in _align(pred, gold, type_agnostic=True).pairs.items():
            if j != -1:
                type_pairs += 1
                type_hits += pred[i]["incident_type"] == gold[j]["incident_type"]

    result: Result = {}
    for slot in SLOTS:
        result[slot] = _prf(*totals[slot])
    result["micro_avg"] = _prf(*(sum(totals[slot][i] for slot in SLOTS) for i in range(4)))
    n_docs = len(golds)
    result["diagnostics"] = {
        "relevance_acc": _ratio(relevance_hits, n_docs),
        "event_count_acc": _ratio(count_hits, n_docs),
        "event_type_acc": _ratio(type_hits, type_pairs),
        "parse_failure_rate": _ratio(len(set(parse_failures) & golds.keys()), n_docs),
        "greedy_alignment_docs": greedy_docs,
        "n_docs": n_docs,
    }
    return result


def _ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _prf(p_num: int, p_den: int, r_num: int, r_den: int) -> dict[str, float | int]:
    return {
        "p_num": p_num,
        "p_den": p_den,
        "r_num": r_num,
        "r_den": r_den,
        "p": 0.0 if p_num == 0 else p_num / float(p_den),
        "r": 0.0 if r_num == 0 else r_num / float(r_den),
        "f1": f1(p_num, p_den, r_num, r_den),
    }


def read_gold(path: Path) -> dict[str, list[Event]]:
    """Gold from prepared JSONL: the assistant message of each example, parsed."""
    golds = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            golds[example["docid"]] = parse_target(example["messages"][-1]["content"])[0]
    return golds


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The rows of an outputs JSONL file: ``{"docid": ..., "output": <raw model output>, ...}``."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_predictions(path: Path) -> tuple[dict[str, list[Event]], set[str]]:
    """Predictions from JSONL rows ``{"docid": ..., "output": <raw model output>}``."""
    return parse_rows(read_rows(path))


def parse_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, list[Event]], set[str]]:
    """Predictions and the docids of parse failures from rows ``{"docid", "output", ...}``."""
    preds = {}
    parse_failures = set()
    for row in rows:
        preds[row["docid"]], parse_ok = parse_target(row["output"])
        if not parse_ok:
            parse_failures.add(row["docid"])
    return preds, parse_failures


def format_result(result: Result) -> str:
    """The GTT-style table plus diagnostics."""
    lines = []
    for slot in (*SLOTS, "micro_avg"):
        r = result[slot]
        lines.append(f"{slot:<14} P: {r['p']:6.2%}  R: {r['r']:6.2%}  F1: {r['f1']:6.2%}")
    for name, value in result["diagnostics"].items():
        lines.append(
            f"{name:<22} {value:.4f}" if isinstance(value, float) else f"{name:<22} {value}"
        )
    return "\n".join(lines)


def flatten(result: Result) -> dict[str, float | int]:
    """The result as one flat mapping: ``micro_avg/f1``, ``PerpInd/p_num``, ``diagnostics/n_docs``, …"""
    return {
        f"{section}/{name}": value
        for section, values in result.items()
        for name, value in values.items()
    }


# Columns of the ``predictions`` table logged with a run.
TABLE_COLUMNS = ("docid", "gold", "output", "parse_ok", "cut_off", "n_gold_events", "n_pred_events")

# Fields of ``meta.json`` stamped into the run config beside the experiment YAML.
CONFIG_STAMPS = ("git_sha", "git_dirty", "dataset_revision", "engine_version", "split", "gpu")


@dataclass
class WandbPayload:
    """What one evaluation logs to W&B: the run config, its metrics and the predictions table."""

    config: dict[str, Any]
    metrics: dict[str, float | int]
    table: list[list[Any]]


def wandb_payload(
    config: Mapping[str, Any],
    meta: Mapping[str, Any],
    result: Result,
    rows: Sequence[Mapping[str, Any]],
    golds: Mapping[str, list[Event]],
) -> WandbPayload:
    """Build the run payload from the experiment YAML, the generation sidecar, the score and the rows."""
    metrics = flatten(result)
    metrics["diagnostics/n_truncated"] = meta["n_truncated"]
    metrics["diagnostics/n_cut_off"] = meta["n_cut_off"]
    return WandbPayload(
        config={**config, **{key: meta[key] for key in CONFIG_STAMPS}},
        metrics=metrics,
        table=predictions_table(rows, golds),
    )


def predictions_table(
    rows: Sequence[Mapping[str, Any]], golds: Mapping[str, list[Event]]
) -> list[list[Any]]:
    """One ``TABLE_COLUMNS`` row per output row."""
    table = []
    for row in rows:
        gold = golds[row["docid"]]
        pred, parse_ok = parse_target(row["output"])
        table.append(
            [
                row["docid"],
                render_target(gold),
                row["output"],
                parse_ok,
                row["cut_off"],
                len(gold),
                len(pred),
            ]
        )
    return table


def log_run(config_path: Path, pred: Path, result: Result, golds: Mapping[str, list[Event]]) -> str:
    """Log one evaluation as a W&B run and return its URL.

    The run carries the experiment YAML plus provenance stamps as config, the
    flat metrics, the ``predictions`` table and the output directory next to
    ``pred`` (``<split>.jsonl`` + ``meta.json``) as a ``predictions`` artifact.
    """
    import wandb

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out_dir = pred.parent
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    payload = wandb_payload(config, meta, result, read_rows(pred), golds)
    run = wandb.init(
        project=config["wandb_project"],
        name=config["name"],
        tags=config["tags"],
        job_type="eval",
        config=payload.config,
    )
    table = wandb.Table(columns=list(TABLE_COLUMNS), data=payload.table)
    run.log({**payload.metrics, "predictions": table})
    artifact = wandb.Artifact(f"{config['name']}-{meta['split']}", type="predictions")
    artifact.add_dir(str(out_dir))
    run.log_artifact(artifact)
    url = run.url or run.id  # offline runs have no URL
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Score predictions against prepared gold.")
    parser.add_argument("--pred", type=Path, required=True, help="JSONL of {docid, output}")
    parser.add_argument("--gold", type=Path, required=True, help="prepared JSONL of the split")
    parser.add_argument("--config", type=Path, help="experiment YAML (required with --wandb)")
    parser.add_argument("--wandb", action="store_true", help="log the run to W&B")
    args = parser.parse_args()
    if args.wandb and args.config is None:
        parser.error("--wandb requires --config")
    preds, parse_failures = read_predictions(args.pred)
    golds = read_gold(args.gold)
    result = score(preds, golds, parse_failures)
    url = log_run(args.config, args.pred, result, golds) if args.wandb else None
    print(format_result(result))
    if url:
        print(url)


if __name__ == "__main__":
    main()
