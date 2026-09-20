"""Data codec: corpus documents -> canonical events -> chat examples, and back."""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

Role = Literal["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]
ROLES: tuple[Role, ...] = ("PerpInd", "PerpOrg", "Target", "Victim", "Weapon")
SPLITS = ("train", "dev", "test")

CORPUS_URL = "https://raw.githubusercontent.com/xinyadu/gtt/master/data/muc/processed/{split}.json"
RAW_DIR = Path("data/raw")

# Base model whose tokenizer renders the chat template and counts input tokens (DESIGN §5).
TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"
# Maximum input tokens (system + user, rendered) so input + target fit max_seq_len 2048.
INPUT_BUDGET = 1800

SYSTEM_PROMPT = """\
Extract every terrorist event described in the document.

Event types: attack, bombing, kidnapping, arson, robbery, forced work stoppage.
Roles: PerpInd (individuals who carried out the event), PerpOrg (organisations \
responsible), Target (physical objects attacked or damaged), Victim (people killed, \
injured, kidnapped or otherwise harmed), Weapon (weapons or explosives used).
Military clashes between armed forces are not events.

Answer only with a JSON array of events in order of appearance, or [] if there is \
none. An event is an object with "incident_type" and its non-empty roles; a role is \
a list of entities; an entity is a list of every string in the document referring \
to it, copied verbatim. Example:
[{"incident_type":"bombing","PerpInd":[["guerrillas","the rebels"]],"Target":[["bridge"]]}]"""

Entity = list[str]
CorpusEntity = list[tuple[str, int]]  # (mention, offset) pairs


class Event(TypedDict):
    """One event in the offset-free shape shared by targets, predictions and the scorer."""

    incident_type: str
    PerpInd: list[Entity]
    PerpOrg: list[Entity]
    Target: list[Entity]
    Victim: list[Entity]
    Weapon: list[Entity]


def event(incident_type: str, **roles: list[Entity]) -> Event:
    """Build an event; roles not given are empty."""
    ev = Event(incident_type=incident_type, PerpInd=[], PerpOrg=[], Target=[], Victim=[], Weapon=[])
    for role, entities in roles.items():
        if role not in ROLES:
            raise KeyError(role)
        ev[role] = entities
    return ev


class CorpusEvent(TypedDict):
    """One event as annotated in the corpus: every mention carries its offset."""

    incident_type: str
    PerpInd: list[CorpusEntity]
    PerpOrg: list[CorpusEntity]
    Target: list[CorpusEntity]
    Victim: list[CorpusEntity]
    Weapon: list[CorpusEntity]


@dataclass
class Doc:
    """One corpus document with its annotated events."""

    docid: str
    text: str
    events: list[CorpusEvent]


def load(split: str, root: Path = RAW_DIR) -> list[Doc]:
    """Read the cached corpus file of ``split`` (one JSON document per line)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    docs = []
    with (root / f"{split}.json").open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            events = [_corpus_event(raw["docid"], t) for t in raw["templates"]]
            docs.append(Doc(raw["docid"], raw["doctext"], events))
    return docs


def _corpus_event(docid: str, raw: dict[str, Any]) -> CorpusEvent:
    missing = [key for key in ("incident_type", *ROLES) if key not in raw]
    if missing:
        raise ValueError(f"document {docid}: event is missing {missing}")
    return corpus_event(
        raw["incident_type"],
        **{role: [[(m, o) for m, o in entity] for entity in raw[role]] for role in ROLES},
    )


def corpus_event(incident_type: str, **roles: list[CorpusEntity]) -> CorpusEvent:
    """Build a corpus event; roles not given are empty."""
    ev = CorpusEvent(
        incident_type=incident_type, PerpInd=[], PerpOrg=[], Target=[], Victim=[], Weapon=[]
    )
    for role, entities in roles.items():
        if role not in ROLES:
            raise KeyError(role)
        ev[role] = entities
    return ev


def canonical_events(doc: Doc) -> list[Event]:
    """Deduplicated, deterministically ordered, offset-free events of a document.

    Order: earliest mention offset (events without mentions last), then event
    type, then the compact JSON of the event. Mentions within an entity are
    ordered by offset, then string. Empty entities are dropped.
    """
    keyed: list[tuple[tuple[float, str, str], Event]] = []
    for ce in doc.events:
        ev = event(
            ce["incident_type"],
            **{
                role: [
                    [m for m, _ in sorted(entity, key=lambda p: (p[1], p[0]))]
                    for entity in ce[role]
                    if entity
                ]
                for role in ROLES
            },
        )
        if any(ev == seen for _, seen in keyed):
            continue
        offsets = [offset for role in ROLES for entity in ce[role] for _, offset in entity]
        earliest = min(offsets) if offsets else float("inf")
        keyed.append(((earliest, ev["incident_type"], _compact(ev)), ev))
    return [ev for _, ev in sorted(keyed, key=lambda pair: pair[0])]


def _compact(ev: Event) -> str:
    """Compact JSON of one event: type first, roles in canonical order, empty roles omitted."""
    slim: dict[str, Any] = {"incident_type": ev["incident_type"]}
    slim.update({role: ev[role] for role in ROLES if ev[role]})
    return json.dumps(slim, separators=(",", ":"), ensure_ascii=False)


def render_target(events: list[Event]) -> str:
    """The JSON target: a compact array of events; ``[]`` for an irrelevant document."""
    return "[" + ",".join(_compact(ev) for ev in events) + "]"


def parse_target(text: str) -> tuple[list[Event], bool]:
    """Parse an output into a prediction: ``(events, parse_ok)``.

    Tolerates prose or a markdown fence around the JSON array. A cut-off output
    yields its complete events and still counts as parsed; an output with no
    recoverable event yields ``([], False)``. An empty array counts as a parsed
    empty prediction only when no array of events can be found anywhere.
    """
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch == "["]
    saw_empty_array = False
    for start in starts:
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            if parsed:
                return _coerce_events(parsed), True
            saw_empty_array = True
    for start in starts:  # cut-off output: longest prefix closing at an event boundary
        for end in reversed([i for i, ch in enumerate(text) if ch == "}" and i > start]):
            try:
                parsed = json.loads(text[start : end + 1] + "]")
            except json.JSONDecodeError:
                continue
            events = _coerce_events(parsed)
            if events:
                return events, True
    return [], saw_empty_array


def _coerce_events(parsed: list[Any]) -> list[Event]:
    return [_coerce_event(item) for item in parsed if isinstance(item, dict)]


def _coerce_event(raw: dict[str, Any]) -> Event:
    incident_type = raw.get("incident_type")
    if not isinstance(incident_type, str):
        incident_type = "attack"  # GTT's own coercion
    ev = event(incident_type)
    for role in ROLES:
        value = raw.get(role, [])
        entities = value if isinstance(value, list) else [value]
        for entity in entities:
            mentions = entity if isinstance(entity, list) else [entity]
            mentions = [m for m in mentions if isinstance(m, str)]
            if mentions:
                ev[role].append(mentions)
    return ev


@dataclass
class Example:
    """One document in chat form: system prompt, document text, JSON target."""

    docid: str
    messages: list[dict[str, str]]
    n_input_tokens: int
    truncated: bool


def make_example(
    doc: Doc, tokenizer: PreTrainedTokenizerBase, *, truncate: bool
) -> Example | None:
    """Render a document as an example, or ``None`` if it is dropped for exceeding the budget.

    With ``truncate`` (dev/test) an over-budget document is cut from the end, by
    tokens, to fit; without it (train) the document is dropped.
    """
    text = doc.text
    n_tokens = _n_input_tokens(tokenizer, text)
    truncated = n_tokens > INPUT_BUDGET
    if truncated:
        if not truncate:
            return None
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        keep = len(text_ids)
        while n_tokens > INPUT_BUDGET:
            keep -= n_tokens - INPUT_BUDGET
            text = str(tokenizer.decode(text_ids[:keep]))
            n_tokens = _n_input_tokens(tokenizer, text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
        {"role": "assistant", "content": render_target(canonical_events(doc))},
    ]
    return Example(doc.docid, messages, n_tokens, truncated)


def _n_input_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(rendered, str)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def download(split: str, root: Path = RAW_DIR) -> Path:
    """Fetch one corpus split into the cache; skipped when the file already exists."""
    path = root / f"{split}.json"
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL.format(split=split), path)
    return path


DATASET_ID = "fnl-es/muc4-chat"
PREPARED_DIR = Path("data/prepared")

DATASET_CARD = f"""\
# muc4-chat

MUC-4 document-level event extraction as chat-formatted examples, one per
document, in the corpus's `train`/`dev`/`test` splits.

**Source.** The GTT-preprocessed MUC-4 corpus from
[xinyadu/gtt](https://github.com/xinyadu/gtt) (`data/muc/processed/`; MIT
licence for the preprocessing, the MUC-4 data itself is in the public domain).

**Example.** `messages` holds a system prompt (identical for every example),
a user message with the document text, and an assistant message with the
target. `n_input_tokens` counts the rendered system + user turn with the
`{TOKENIZER_ID}` tokenizer; `truncated` marks dev/test documents whose text
was cut to fit {INPUT_BUDGET} input tokens (train documents over budget are
dropped instead).

**Target.** A compact JSON array of the document's events, `[]` for an
irrelevant document. Each event has `incident_type` (one of attack, bombing,
kidnapping, arson, robbery, forced work stoppage) followed by its non-empty
roles among PerpInd, PerpOrg, Target, Victim, Weapon, in that order. A role is
a list of entities; an entity is the list of its coreferent mentions, verbatim
from the document, without character offsets.

**Canonical form.** Duplicate events (identical once offsets are removed) are
collapsed to one. Events are ordered by the offset of their earliest mention
(events without mentions last), then by event type, then by their JSON text.
Mentions within an entity are ordered by offset, then by string. Empty
entities are dropped.

Prepared by `src/data/prepare.py` in the fine-tuning-decoder project.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the MUC-4 chat dataset.")
    parser.add_argument("--push", action="store_true", help=f"push to {DATASET_ID} on the Hub")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[Example]] = {}
    for split in SPLITS:
        download(split)
        docs = load(split)
        examples = [make_example(doc, tokenizer, truncate=split != "train") for doc in docs]
        kept = [ex for ex in examples if ex is not None]
        splits[split] = kept
        truncated = sum(ex.truncated for ex in kept)
        print(f"{split}: {len(kept)} kept, {len(docs) - len(kept)} dropped, {truncated} truncated")
        with (PREPARED_DIR / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for ex in kept:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    if args.push:
        from datasets import Dataset, DatasetDict
        from huggingface_hub import DatasetCard

        dataset = DatasetDict(
            {split: Dataset.from_list([asdict(ex) for ex in kept]) for split, kept in splits.items()}
        )
        dataset.push_to_hub(DATASET_ID, private=False)
        card = DatasetCard.load(DATASET_ID)  # keep the split/feature metadata push_to_hub wrote
        card.text = DATASET_CARD
        card.push_to_hub(DATASET_ID)
        print(f"pushed https://huggingface.co/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
