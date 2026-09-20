"""PROTOTYPE (ticket 07, throwaway): list 3-shot exemplar candidates and size the prompt.

Run: uv run python .scratch/milestone-2-baselines/prototype_exemplars.py [DOCID DOCID DOCID]

Without args: prints the shortest candidates per event-count bucket (0 / 1 / 2+)
and picks the default (shortest with a non-stub body). With three docids: uses
those. Then renders the full 3-shot conversation for every dev doc, prints the
longest one to `rendered_3shot.txt` next to this file, and reports token stats.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_TOKENS = 183  # rendered system prompt, included in n_input_tokens
MIN_BODY = 100  # tokens; below this a doc is a stub, not a teaching example
MAX_MODEL_LEN, MAX_NEW = 3072, 512

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
train = [json.loads(l) for l in (ROOT / "data/prepared/train.jsonl").open()]
dev = [json.loads(l) for l in (ROOT / "data/prepared/dev.jsonl").open()]
by_id = {r["docid"]: r for r in train}


def n_events(r):
    return len(json.loads(r["messages"][-1]["content"]))


def types(r):
    return ",".join(e["incident_type"] for e in json.loads(r["messages"][-1]["content"]))


def bucket(r):
    return min(n_events(r), 2)


def target_tokens(r):
    return len(tok.encode(r["messages"][-1]["content"], add_special_tokens=False))


if len(sys.argv) == 4:
    picks = sys.argv[1:]
else:
    picks = []
    for b in (0, 1, 2):
        cands = sorted((r for r in train if bucket(r) == b), key=lambda r: r["n_input_tokens"])
        print(f"\n=== bucket {b}{'+' if b == 2 else ''} events: shortest 8 candidates ===")
        for r in cands[:8]:
            body = r["n_input_tokens"] - SYSTEM_TOKENS
            print(f"{r['docid']}  body={body:4d} tgt={target_tokens(r):3d} [{types(r)}]")
            print("    " + r["messages"][1]["content"][:160].replace("\n", " ") + " …")
        picks.append(next(r for r in cands if r["n_input_tokens"] - SYSTEM_TOKENS >= MIN_BODY)["docid"])

print(f"\n=== exemplars: {picks} ===")
shots = []
for d in picks:
    r = by_id[d]
    shots += r["messages"][1:]  # user + assistant turns
    print(f"{d}: {n_events(r)} events [{types(r)}], input {r['n_input_tokens']}, target {target_tokens(r)}")
    print("  " + r["messages"][-1]["content"][:300])


def render(dev_row):
    msgs = [dev_row["messages"][0], *shots, dev_row["messages"][1]]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


lens = {r["docid"]: len(tok.encode(render(r), add_special_tokens=False)) for r in dev}
longest = max(lens, key=lens.get)
prefix = len(tok.encode(tok.apply_chat_template([dev[0]["messages"][0], *shots], tokenize=False), add_special_tokens=False))
print(f"\nshared prefix (system + 3 shots): {prefix} tokens")
print(f"dev 3-shot input: min {min(lens.values())}, max {lens[longest]} ({longest}), budget {MAX_MODEL_LEN - MAX_NEW}")
print(f"docs over budget: {sum(v > MAX_MODEL_LEN - MAX_NEW for v in lens.values())}")
out = Path(__file__).with_name("rendered_3shot.txt")
out.write_text(render(next(r for r in dev if r["docid"] == longest)))
print(f"longest rendered prompt written to {out}")
