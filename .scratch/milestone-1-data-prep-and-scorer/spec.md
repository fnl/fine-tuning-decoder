# Spec: Milestone 1 — repo skeleton, data preparation, GTT scorer

Status: ready-for-agent
Created: 2026-09-20
Updated: 2026-09-20 — vocabulary aligned with `CONTEXT.md`; event-type accuracy redefined
Source: `docs/DESIGN.md` §3, §4, §6, §8, §9, §10, §11 (milestone 1); `docs/PROJECT_PLANNING.md`; `CONTEXT.md`

Vocabulary follows `CONTEXT.md`: the extracted unit is an **event** (called
a *template* in the corpus and the GTT scorer), its category an **event
type** (stored as `incident_type`), entity comparison is the **subset
match**, and the two truncations are **truncated document** (input side)
and **cut-off output** (generation side).

## Problem Statement

The project has a complete design (`docs/DESIGN.md`) but no code. Before any
GPU work can start, the developer needs the three CPU-only foundations that
every later milestone builds on:

1. A repo skeleton that installs with `uv` on a CPU-only machine and passes
   `mypy`, `ruff`, and `pytest` (the workflow in `AGENTS.md`).
2. A data preparation step that turns the corpus (the raw GTT-preprocessed
   MUC-4 splits) into the dataset of chat-formatted examples (system / user
   / assistant messages) and publishes it as the public Hub dataset
   `fnl-es/muc4-chat`, so the Colab notebook can load training data without
   touching the corpus.
3. A faithful port of the GTT scorer plus the DESIGN §8 diagnostics, so
   baselines and fine-tuned runs (milestones 2–4) produce numbers comparable
   to the published GTT/GRIT results.

Without tests for these pieces, bugs in target formatting, event ordering,
or alignment would only surface during expensive GPU sessions.

## Solution

Build the `src/` package described in DESIGN §3 with two independently
testable pure modules that share one in-memory `Event` shape:

- **Data codec** (`src/data/prepare.py`): load corpus documents, produce
  canonical events (deduplicated, deterministically ordered, offset-free),
  render each document into an example whose assistant turn is the JSON
  target, and parse (possibly cut-off) output back into a prediction. A
  thin CLI downloads the corpus and pushes the dataset to the Hub.
- **Scorer** (`src/eval.py`): port of GTT `eval.py` (mention normalisation,
  subset match, exhaustive per-document alignment maximising micro-F1,
  per-role and micro P/R/F1) plus the four diagnostics. Its return value is
  a plain dict that later milestones log to W&B verbatim.

Milestone 1 is done when: all tests pass locally, type-checker and linter
are clean, and `fnl-es/muc4-chat` exists on the Hub with `train`, `dev`, and
`test` splits.

## User Stories

### Repo skeleton

1. AS A developer, I WANT `uv sync` to install every dependency on a CPU-only machine, SO THAT I can run data prep, the scorer, and tests without a GPU.
2. AS A developer, I WANT `uv run mypy src`, `uv run ruff check .`, and `uv run pytest` to pass from a fresh clone, SO THAT the `AGENTS.md` workflow is enforceable from the first commit.
3. AS A developer, I WANT GPU-only libraries (unsloth, trl, bitsandbytes, vllm) kept out of `pyproject.toml`, SO THAT local installs stay small and the notebook owns their frequently-changing install line.
4. AS A developer, I WANT the corpus cache and local artifacts ignored by git, SO THAT the repo contains only code, config, docs, and small test fixtures.
5. AS A developer, I WANT a `configs/` directory with the smoke experiment YAML skeleton, SO THAT milestone 3 has a config file to fill in rather than invent.
6. AS A developer, I WANT `src/` modules to be importable in tests and the notebook without `sys.path` hacks, SO THAT the same code runs locally and on Colab.

### Loading the corpus

7. AS A developer, I WANT a single command to download the three corpus split files into a local cache, SO THAT I never depend on the `xinyadu/gtt` repo at import time.
8. AS A developer, I WANT the download skipped when the cached file already exists, SO THAT repeated runs are fast and offline-friendly.
9. AS A developer, I WANT `load(split)` to return a list of `Doc` objects (docid, document text, events with mentions and offsets), SO THAT prepare and eval share one document representation.
10. AS A developer, I WANT the loader to reject a split name other than `train`, `dev`, `test`, SO THAT typos fail loudly.
11. AS A developer, I WANT the loader to validate that every event carries the event type and all five role keys, SO THAT a malformed corpus is caught before it reaches the target renderer.

### Canonical events

12. AS A developer, I WANT duplicate events within a document removed, SO THAT the target matches what the GTT scorer treats as gold (which dedups silently).
13. AS A developer, I WANT events ordered by the offset of their earliest mention, SO THAT the target is deterministic and reflects document order.
14. AS A developer, I WANT a fully specified tie-breaker for events sharing the same earliest offset, SO THAT two runs of prepare produce byte-identical targets.
15. AS A developer, I WANT events with no mentions at all placed after all events that have mentions, SO THAT the ordering rule covers every event in the corpus.
16. AS A developer, I WANT mentions within an entity ordered by offset, SO THAT entity mention lists are deterministic.
17. AS A developer, I WANT offsets dropped from the target, SO THAT the model only has to emit strings it can copy from the text.
18. AS A developer, I WANT empty roles omitted from the target, SO THAT targets are short and the model is not trained to emit boilerplate.
19. AS A developer, I WANT an irrelevant document to have the target `[]`, SO THAT relevance is learnt as part of the same output format.
20. AS A developer, I WANT the target's keys emitted in a fixed order (event type first, then roles in the canonical role order), SO THAT targets are deterministic and easy to read.

### Examples

21. AS A developer, I WANT every example to contain exactly one system message, one user message, and one assistant message, SO THAT it is directly consumable by TRL's conversational format with completion-only loss.
22. AS A developer, I WANT one ~150-token system prompt naming the six event types, defining the five roles, stating the exclusion rule and specifying the output format, SO THAT the task is fully specified to the model.
23. AS A developer, I WANT the system prompt defined in exactly one place and exported, SO THAT zero-shot, few-shot, and fine-tuned runs use an identical prompt.
24. AS A developer, I WANT the user message to contain only the document text, SO THAT the input stays minimal and consistent.
25. AS A developer, I WANT the assistant message to be the compact JSON target, SO THAT the model learns to emit parseable output without whitespace waste.
26. AS A developer, I WANT the input (system + user, via Qwen's chat template with thinking disabled) token-counted with the Qwen tokenizer, SO THAT length decisions use the real tokenization.
27. AS A developer, I WANT train documents whose input exceeds the input budget (~1800 tokens) dropped, SO THAT no example exceeds `max_seq_len` 2048 with its target.
28. AS A developer, I WANT dev and test documents over the input budget kept as truncated documents, SO THAT evaluation covers every document in the split.
29. AS A developer, I WANT each example to record whether its document was truncated and its input token count, SO THAT evaluation can report the truncated-document count.
30. AS A developer, I WANT prepare to print per-split counts (documents kept, dropped, truncated), SO THAT I can sanity-check against the known statistics (1300/200/200 docs, ~2 % long).

### Parsing output

31. AS A developer, I WANT `parse_target(text)` to turn a well-formed JSON target back into events with all five roles present (empty roles filled in), SO THAT the parsed shape matches the loader's shape.
32. AS A developer, I WANT the parser to salvage every complete event from a cut-off output, SO THAT a hit on `max_new_tokens` costs only the cut-off event, not the whole document.
33. AS A developer, I WANT the parser to tolerate surrounding prose or a markdown code fence around the JSON, SO THAT zero-shot baselines are not penalised for formatting alone.
34. AS A developer, I WANT the parser to report a parse failure rather than raise, SO THAT one bad output cannot abort an evaluation.
35. AS A developer, I WANT a parse failure to score as an empty prediction, SO THAT scoring remains defined for every document.
36. AS A developer, I WANT events with a missing or non-string event type coerced exactly as the original GTT scorer does, SO THAT the port stays comparable to published numbers.
37. AS A developer, I WANT entity values that are not lists of strings (e.g. a bare string) coerced to a single-mention entity, SO THAT minor format drift is not counted as total failure.

### Scoring

38. AS A developer, I WANT `score(preds, golds)` to accept per-document predictions and gold keyed by docid, SO THAT scoring is independent of file formats.
39. AS A developer, I WANT mention normalisation as GTT does it (lower-case, punctuation removed, articles removed, whitespace collapsed) before comparison, SO THAT results are comparable to published numbers.
40. AS A developer, I WANT the subset match rule (a predicted entity is correct iff all its mentions belong to one gold entity of the same role), SO THAT the GTT entity comparison is reproduced.
41. AS A developer, I WANT alignment to exhaustively choose the one-to-one mapping of predicted to gold events that maximises document micro-F1, SO THAT the alignment matches GTT.
42. AS A developer, I WANT a predicted event to align only with a gold event of the same event type, SO THAT type errors are penalised as in GTT.
43. AS A developer, I WANT spurious events and missing events counted in every role including the event type, SO THAT precision and recall reflect over- and under-generation.
44. AS A developer, I WANT per-role P/R/F1 for the event type and the five roles plus micro-F1, SO THAT I can compare against published per-role tables.
45. AS A developer, I WANT the alignment search to fall back to a greedy alignment above a documented mapping-count threshold and count how often that happened, SO THAT a degenerate model emitting many events cannot hang evaluation.
46. AS A developer, I WANT the scorer to leave its inputs unmodified, SO THAT the same predictions can be scored, inspected, and logged without surprises (the original mutates in place).
47. AS A developer, I WANT the port verified against the vendored original `eval.py` on real dev gold and on perturbed predictions, SO THAT I trust that the numbers are comparable.
48. AS A developer, I WANT a document present in gold but absent from the predictions treated as an empty prediction, SO THAT partial prediction files still score.

### Diagnostics

49. AS A developer, I WANT relevance accuracy, SO THAT I can compare against the trivial always-`[]` baseline (~46 %).
50. AS A developer, I WANT event-count accuracy, SO THAT I can see whether the model over- or under-generates events.
51. AS A developer, I WANT event-type accuracy measured under a type-agnostic pairing, SO THAT type confusion is separable from role extraction quality (under the type-respecting alignment it would trivially be 100 %).
52. AS A developer, I WANT the parse-failure rate, SO THAT format failures are visible separately from extraction quality.
53. AS A developer, I WANT the diagnostics returned in the same result dict as the F1 metrics, SO THAT one `wandb.log` call records everything.

### Hub publication

54. AS A developer, I WANT the dataset pushed as the public `fnl-es/muc4-chat` with `train`, `dev`, `test` splits, SO THAT the notebook loads data by name.
55. AS A developer, I WANT the push to authenticate from `HF_TOKEN` or the cached CLI login, SO THAT the same command works locally and on Colab.
56. AS A developer, I WANT prepare to also write the examples as local JSONL, SO THAT I can inspect them and run evaluation offline.
57. AS A developer, I WANT the dataset card to state the source (`xinyadu/gtt`, MIT preprocessing, public-domain MUC-4), the target format, and the ordering/dedup rules, SO THAT the dataset is self-describing.

### Testing

58. AS A developer, I WANT all tests to run on CPU without a Hub token or W&B key, SO THAT the suite passes in any environment.
59. AS A developer, I WANT tokenizer-dependent tests to use the real Qwen tokenizer, SO THAT chat-template rendering is verified against the actual model family.
60. AS A developer, I WANT a small corpus-format fixture checked into the repo, SO THAT loader tests never hit the network.

## Implementation Decisions

### Layout and tooling

- Follow DESIGN §3 verbatim: `src/data/prepare.py`, `src/eval.py`, `tests/`,
  `configs/`. `src` is added to the import path via pytest and mypy
  configuration; `train.py` and the notebook are not created in this
  milestone.
- Core dependencies (CPU-installable): `transformers` + `tokenizers` (Qwen
  tokenizer for chat-template rendering and length checks), `datasets` and
  `huggingface_hub` (Hub push), `pyyaml` (configs), `wandb` (so `eval.py`
  imports cleanly locally; first `wandb.init` is milestone 2). Dev
  dependencies: `pytest`, `mypy`, `ruff`, `types-PyYAML`.
- Python `>=3.11,<3.13`. `ruff` with default rules plus import sorting;
  `mypy` strict on `src`.
- The corpus cache is a gitignored `data/raw/` directory; prepared JSONL
  goes to `data/prepared/`.
- The tokenizer used for length checks and rendering is the base model's
  (Qwen3-4B-Instruct); the exact Hub id is a module-level constant so a newer
  generation can be swapped in one place (DESIGN §5).

### Shared `Event` shape

- An event is a mapping with `incident_type: str` and the five roles
  `PerpInd`, `PerpOrg`, `Target`, `Victim`, `Weapon`, each `list[Entity]`
  where `Entity = list[str]` (mention strings). This mirrors the GTT
  scorer's internal shape so the port is line-for-line faithful. Role order
  is fixed as listed.
- **Wire-format key is `incident_type`**, not `event_type`: the target JSON,
  the prediction, and the scorer keep the corpus's key for parity with GTT
  and so the system prompt describes exactly what the model emits. Prose,
  identifiers and test names say *event type* per `CONTEXT.md`; only the
  JSON key and the scorer's result-dict key keep the corpus name.
- `Doc` carries `docid`, `text`, and the corpus events *with* offsets
  (`list[tuple[str, int]]` per entity); canonicalisation strips offsets.
- Typed as `TypedDict`/`dataclass` so `mypy --strict` checks both modules.

### Canonicalisation (prepare)

- Dedup: duplicate events (exact-equal after offset removal) within a
  document are collapsed to one, keeping first occurrence. This matches the
  original scorer's `if template not in templates` gold handling.
- Event order: primary key = minimum offset across all mentions (events
  with no mentions sort last); secondary key = event type; tertiary key =
  the compact JSON serialisation of the offset-free event. Fully
  deterministic regardless of source order.
- Mention order within an entity: by offset, then by string.
- Target JSON: compact separators, `ensure_ascii=False`, key order
  `incident_type` then roles in canonical order, empty roles omitted;
  irrelevant document → `[]`.

### Example

- Record per example: `docid`, `messages` (system/user/assistant),
  `n_input_tokens`, `truncated: bool` (a truncated document). Gold for
  evaluation is recovered by parsing the assistant message — the codec
  round-trip guarantees this is lossless minus offsets — so no separate
  gold column is stored.
- System prompt: a single module-level constant exported from the codec
  module (DESIGN §6). Used unchanged by prepare and, later, by baselines.
- Input budget: input tokens (system + user rendered with the Qwen chat
  template, `enable_thinking=False`, `add_generation_prompt=True`) must be
  ≤ 1800. Train documents above budget are dropped documents; dev/test
  documents have their text cut (by tokens, from the end) to fit and are
  flagged as truncated documents. Prepare prints kept/dropped/truncated
  counts per split.

### Parser (`parse_target`)

- Extracts the first JSON array from the output (tolerating prose and a
  markdown fence). On decode failure, treats the output as cut-off: scan
  for the longest prefix ending at an event boundary that parses; salvaged
  events form the prediction and the document is *not* a parse failure. If
  nothing is salvageable, return `[]` and flag a parse failure.
- Returns `(events, parse_ok)`; roles missing from an event are filled with
  `[]`; unknown keys are dropped; a bare-string entity becomes `[string]`;
  a missing/non-string `incident_type` becomes `"attack"` (GTT's own
  coercion).

### Scorer (`score`)

- Signature: `score(preds: dict[docid, list[Event]], golds: dict[docid,
  list[Event]], parse_failures: set[docid] = ()) -> Result`. Docs in
  `golds` but not in `preds` score as `[]`. Inputs are deep-copied before
  normalisation.
- Mention normalisation, subset match, one-to-one mapping enumeration with
  the `-1` "unaligned" sentinel, best-mapping-by-micro-F1, spurious/missing
  accounting, and the micro average are ported from the original, including
  counting `incident_type` as one slot per aligned event.
- Guard: if `(|gold|+1) ** |pred|` exceeds 1,000,000 for a document, use a
  greedy alignment (best-scoring pair first) and increment
  `diagnostics.greedy_alignment_docs`. Below the threshold, results are
  bit-identical to the original.
- Event-type accuracy: computed on a *separate, type-agnostic* pairing —
  the same alignment procedure with the type-equality constraint removed —
  as (number of pairs with equal type) / (number of pairs). It is a
  diagnostic only; the headline alignment and all P/R/F1 keep the
  type-equality constraint exactly as GTT. Documents with no pairs
  contribute nothing to this rate.
- Result dict: `{incident_type, PerpInd, PerpOrg, Target, Victim, Weapon,
  micro_avg}` each with `p_num, p_den, r_num, r_den, p, r, f1`, plus
  `diagnostics: {relevance_acc, event_count_acc, event_type_acc,
  parse_failure_rate, greedy_alignment_docs, n_docs}`. This dict is the
  W&B payload in later milestones: tracking is a sink above the seam, never
  part of it.
- Also expose a small CLI: `eval.py --pred <jsonl> --gold <jsonl>` printing
  the GTT-style table, for offline use on prepared JSONL.

### Hub publication

- `prepare.py --push` builds a `DatasetDict` with `train`/`dev`/`test` and
  pushes to `fnl-es/muc4-chat` (public). A README/dataset card is generated
  with source, licence, target format, and ordering rules. Without `--push`
  only local JSONL is written.

## Testing Decisions

- A good test drives a public function with hand-built inputs and asserts on
  its returned value; it does not inspect internals, private helpers, or
  I/O side effects. Test names use the `CONTEXT.md` vocabulary (document,
  event, event type, role, entity, mention, alignment, subset match,
  truncated document, cut-off output).
- **Scorer tests** (`tests/test_eval.py`): hand-built gold/prediction pairs
  covering exact match, mention normalisation, subset match (predicted
  mentions a subset of a gold entity → correct; an extra mention → wrong),
  wrong event type, spurious and missing events, empty gold vs empty
  prediction, and the multi-event alignment case where the naive
  order-preserving mapping is worse than the best mapping. Diagnostics
  tested on the same fixtures, including an event-type accuracy case where
  a prediction has the right roles but the wrong type (0 % under the
  diagnostic, spurious+missing under the headline alignment).
  Plus a **parity test**: the original `eval.py` is vendored under
  `tests/oracle/gtt_eval.py` (MIT, attributed) and both scorers are run on
  the dev-derived fixture gold against (a) itself, (b) gold with random
  entity/event drops and mention edits under a fixed seed; per-role
  `p_num/p_den/r_num/r_den` must be equal.
- **Codec tests** (`tests/test_prepare.py`): loader on a checked-in fixture
  of ~5 corpus documents (including an irrelevant document, a multi-event
  document with an offset tie, a duplicate event, an event with no
  mentions); round-trip `parse_target(render(canonical)) == canonical`;
  deterministic ordering under shuffled input; dedup; empty-role omission;
  cut-off output repair on the target cut at several byte positions (every
  complete event salvaged, never a raise); fence/prose tolerance; parse
  failure reporting; chat-template rendering with the real Qwen tokenizer
  (exactly three messages, no thinking block, generation prompt present,
  token count within the input budget for a short document); dropped
  document vs truncated document behaviour on an artificially long
  document.
- Tokenizer tests download the tokenizer once into the HF cache; they are
  marked `tokenizer` so they can be deselected offline.
- No tests for the download or Hub push; these are thin wrappers over
  `urllib`/`datasets` and are verified manually by the milestone's exit
  criterion (dataset visible on the Hub).
- Prior art: none in the repo yet — this milestone establishes the pattern.

## Out of Scope

- `train.py`, the LoRA config, collator, or any Unsloth/TRL code (milestone 3).
- The Colab notebook, vLLM/generation code, W&B `init`/`log` calls, baseline
  runs (milestone 2). Only the `wandb` dependency declaration is included.
- Adapter Hub repo `fnl-es/qwen3-4b-muc4-lora`.
- The DocEE loader; the dataset interface is kept small (`load`, `score`)
  so DocEE can drop in later, but no abstraction beyond that.
- Few-shot example selection for the 3-shot baseline.
- Sliding-window handling of long documents (round two).
- Any change to decisions in `docs/DESIGN.md`.

## Further Notes

- Facts verified 2026-09-20: corpus files total ~3.85 MB; 62 duplicate
  events across splits; 204/38/41 earliest-offset ties in train/dev/test;
  53/10/9 events without any mention; max events per document 14/9/6;
  every recorded offset matches its mention in the text.
- The original scorer's event-type test is a substring check
  (`pred_type in gold_type`). For the closed six-type vocabulary this equals
  equality, so the port uses equality; the oracle parity test covers real
  types only.
- `docs/DESIGN.md` predates `CONTEXT.md` and is kept as the interview
  record; where it says *template*, *incident type*, *any-mention
  matching*, or unqualified *truncation*, read the `CONTEXT.md` term.
- `docs/agents/issue-tracker.md` refers to a `triage-labels.md` that is not
  in the repo; the canonical label strings from the setup skill are used
  here (`ready-for-agent`).
- Execution guidance (from `docs/PROJECT_PLANNING.md`): TDD, scorer and
  round-trip tests first; stop at "tests green, dataset pushed"; commit only
  when asked.
