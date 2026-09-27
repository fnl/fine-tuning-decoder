"""Percentile bootstrap interval of micro-F1 over the documents of a split.

The GTT metric aligns events within each document, so a split's micro counts are the
sum of its documents' counts. Each resample draws the split's documents with replacement
and scores their summed counts. The interval covers which documents are in the split,
not training randomness (another seed or data order).

    python -m bootstrap --pred outputs/qwen3-4b-r16-eval/test.jsonl \\
        --gold data/prepared/test.jsonl --below 53.0 50.2
"""

import argparse
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from data.prepare import Event
from eval import f1, read_gold, read_predictions, score

Counts = tuple[int, int, int, int]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    samples: tuple[float, ...]  # the resampled micro-F1s, sorted


def bootstrap_f1(
    preds: dict[str, list[Event]],
    golds: dict[str, list[Event]],
    *,
    n: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """The micro-F1 of the split with its ``level`` percentile bootstrap interval."""
    per_doc = list(_doc_counts(preds, golds).values())
    rng = random.Random(seed)
    samples = sorted(_micro_f1(rng.choices(per_doc, k=len(per_doc))) for _ in range(n))
    tail = int(n * (1 - level) / 2)
    return Interval(
        point=_micro_f1(per_doc),
        low=samples[tail],
        high=samples[n - 1 - tail],
        samples=tuple(samples),
    )


def _doc_counts(preds: dict[str, list[Event]], golds: dict[str, list[Event]]) -> dict[str, Counts]:
    counts = {}
    for docid, gold in golds.items():
        micro = score({docid: preds.get(docid, [])}, {docid: gold})
        counts[docid] = (
            micro["micro_avg"]["p_num"],
            micro["micro_avg"]["p_den"],
            micro["micro_avg"]["r_num"],
            micro["micro_avg"]["r_den"],
        )
    return counts


def _micro_f1(docs: Iterable[Counts]) -> float:
    totals = [0, 0, 0, 0]
    for doc in docs:
        for i, n in enumerate(doc):
            totals[i] += n
    return f1(*totals)


def format_interval(interval: Interval, level: float, below: Iterable[float] = ()) -> str:
    """The point and interval in percent, and the share of samples under each ``below`` F1."""
    lines = [
        (
            f"micro F1 {interval.point:.2%}, {level:.0%} interval "
            f"{interval.low:.2%} to {interval.high:.2%} ({len(interval.samples)} resamples)"
        )
    ]
    for threshold in below:
        share = sum(s < threshold / 100 for s in interval.samples) / len(interval.samples)
        lines.append(f"resamples below {threshold:.1f}: {share:.1%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap interval of micro-F1 over documents.")
    parser.add_argument("--pred", type=Path, required=True, help="JSONL of {docid, output}")
    parser.add_argument("--gold", type=Path, required=True, help="prepared JSONL of the split")
    parser.add_argument("--n", type=int, default=10_000, help="number of resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95, help="coverage of the interval")
    parser.add_argument(
        "--below", type=float, nargs="*", default=[], help="F1s in percent to report shares under"
    )
    args = parser.parse_args()
    preds, _ = read_predictions(args.pred)
    interval = bootstrap_f1(preds, read_gold(args.gold), n=args.n, seed=args.seed, level=args.level)
    print(format_interval(interval, args.level, args.below))


if __name__ == "__main__":
    main()
