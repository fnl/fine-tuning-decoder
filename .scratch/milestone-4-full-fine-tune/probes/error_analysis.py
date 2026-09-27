"""Where the milestone-4 fine-tune loses its points, for the comparison note's error analysis.

    uv run python -m data.prepare
    uv run wandb artifact get flowing/muc4-event-extraction/qwen3-4b-r16-eval-dev:latest \
        --root outputs/qwen3-4b-r16-eval
    uv run python .scratch/milestone-4-full-fine-tune/probes/error_analysis.py outputs/qwen3-4b-r16-eval

Diagnostic only; the scorer is `eval.py`. Role entities are pooled per document and role, ignoring
event alignment: a predicted entity is a hit if it shares a normalised mention with a gold entity, a
near miss if one mention contains the other, and a gold entity is missed if nothing hits it. Gold
items per document and split are counted as the scorer counts them, per event.
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from eval import normalize_mention, parse_rows, read_gold, read_rows

ROLES = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]
PREPARED = Path("data/prepared")


def entities(events: list[dict], role: str) -> set[frozenset[str]]:
    return {frozenset(normalize_mention(m) for m in e) for ev in events for e in ev.get(role, [])}


def gold_items(events: list[dict]) -> int:
    """The scorer's recall denominator: one per event type, one per role entity of each event."""
    return len(events) + sum(len(ev.get(role, [])) for ev in events for role in ROLES)


def split_stats(split: str) -> str:
    gold = read_gold(PREPARED / f"{split}.jsonl")
    counts = [len(evs) for evs in gold.values()]
    n_ents = sum(gold_items(evs) - len(evs) for evs in gold.values())
    n = len(gold)
    return (
        f"{split}: {n} docs, {100 * counts.count(0) / n:.0f} % without events, "
        f"{sum(counts) / n:.2f} events and {n_ents / n:.2f} role entities per doc"
    )


def main(pred_dir: Path) -> None:
    rows = read_rows(pred_dir / "dev.jsonl")
    preds, failed = parse_rows(rows)
    gold = read_gold(PREPARED / "dev.jsonl")
    texts = {
        json.loads(line)["docid"]: json.loads(line)["messages"][1]["content"]
        for line in (PREPARED / "dev.jsonl").open(encoding="utf-8")
    }

    docs: Counter[str] = Counter()
    for docid, g in gold.items():
        p = preds.get(docid, [])
        if g and not p:
            docs["event docs answered []"] += 1
            docs["gold items in them"] += gold_items(g)
        elif p and not g:
            docs["event-free docs given events"] += 1
        elif g:
            docs[
                "too few events"
                if len(p) < len(g)
                else "too many"
                if len(p) > len(g)
                else "right count"
            ] += 1
    missed = [d for d, g in gold.items() if g and not preds.get(d)]
    hit = [d for d, g in gold.items() if g and preds.get(d)]
    print(dict(docs), "| parse failures", sorted(failed))
    print(
        "median words: missed event docs",
        statistics.median(len(texts[d].split()) for d in missed),
        "| found event docs",
        statistics.median(len(texts[d].split()) for d in hit),
    )
    print("cut off:", [r["docid"] for r in rows if r.get("cut_off")])

    mentions = [
        (docid, normalize_mention(m))
        for docid, evs in preds.items()
        for ev in evs
        for role in ROLES
        for e in ev.get(role, [])
        for m in e
    ]
    not_verbatim = sum(m not in normalize_mention(texts[d]) for d, m in mentions)
    print(f"predicted mentions not in the document: {not_verbatim} of {len(mentions)}")

    for role in ROLES:
        c: Counter[str] = Counter()
        for docid, g in gold.items():
            gold_ents, found = entities(g, role), set()
            for e in entities(preds.get(docid, []), role):
                exact = {ge for ge in gold_ents if ge & e}
                near = {
                    ge
                    for ge in gold_ents
                    if any(a in b or b in a for a in e for b in ge if a and b)
                }
                c["hit" if exact else "near miss" if near else "spurious"] += 1
                found |= exact or near
            c["gold"] += len(gold_ents)
            c["gold missed"] += len(gold_ents - found)
        print(role, dict(c))

    for split in ("train", "dev", "test"):
        print(split_stats(split))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
