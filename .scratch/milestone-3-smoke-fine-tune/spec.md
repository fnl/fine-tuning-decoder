# Spec: Milestone 3 — smoke fine-tune of Qwen3-0.6B on 100 documents

Status: ready-for-agent
Created: 2026-09-22
Source: `map.md` and `issues/01`–`11` in this directory; `docs/DESIGN.md` §7, §9, §10, §11; `CONTEXT.md`

## Problem Statement

The project can prepare data, generate with a base model and score the result,
but it has never trained anything. Milestone 4 — the full QLoRA fine-tune that
defines "done" — would be the first time the training loop, the loss mask, the
checkpoint push and the eval callback ran at all, on a 4B model, on a T4, for
forty minutes a go. Every one of those four mechanisms has a silent failure
mode: a mask that trains on the document instead of the target, a push that
uploads nothing resumable, a callback that never fires, a run whose metrics
land in the wrong place. Discovering any of them during milestone 4 costs a
GPU session per attempt, as milestone 2 discovered three times over.

The user wants to prove the whole loop end to end on a model small enough that
a mistake costs minutes, on a subset small enough to iterate on, and to come
out of it with an adapter on the Hub that can be loaded back — so that
milestone 4 is a change of two YAML values and nothing else.

## Solution

A new training module fine-tunes a configured model on the first N documents
of the train split with completion-only loss, masking the prompt itself rather
than delegating that to the training library. Every ~½ epoch it scores the
first M dev documents with the real scorer by generating from the model under
training, logs the result beside the training loss in one W&B run, and saves a
checkpoint that is pushed to the Hub. One YAML defines the experiment; a new
notebook runs it on a free-tier T4 and, at the end, loads the *pushed* adapter
onto a freshly-loaded base model and generates one document.

The masking, the subset selection, the step arithmetic and the callback's
logging are pure functions tested on CPU with the real tokenizer, so the
mechanism that matters most is verified before a GPU is ever allocated.

## User Stories

1. AS A learner, I WANT the loss to be computed only over the target JSON, SO THAT the model learns to produce events rather than to reproduce news articles.
2. AS A learner, I WANT to see the exact token where the mask begins, SO THAT I can believe the completion-only loss is real rather than trust a library flag.
3. AS A learner, I WANT the masking to be a pure function I can run on my laptop, SO THAT a mistake costs a second rather than a Colab session.
4. AS A learner, I WANT the masking to work identically for the 0.6B and the 4B model, SO THAT the smoke run actually proves the path milestone 4 will take.
5. AS A learner, I WANT an example whose prompt is not a token-level prefix of its full rendering to fail loudly and name the document, SO THAT a chat-template change can never silently corrupt every label.
6. AS A learner, I WANT training examples that exceed the sequence budget to be dropped and counted rather than truncated or asserted, SO THAT one long document neither corrupts a target nor stops milestone 4's full run.
7. AS A learner, I WANT the model to be trained to emit its end-of-turn token, SO THAT generation stops on its own instead of running to the token limit.
8. AS A learner, I WANT the trailing newline the chat template emits after that token excluded from the loss, SO THAT the model is not trained on a token it can never produce.
9. AS A learner, I WANT one YAML to define an experiment, SO THAT a run is reproducible from a file I can read and diff.
10. AS A learner, I WANT the config to reject unknown keys by name, SO THAT a typo is an error rather than a silently ignored setting.
11. AS A learner, I WANT gradient accumulation derived from the effective and per-device batch sizes, SO THAT two numbers that must agree cannot disagree.
12. AS A learner, I WANT the compute dtype chosen from the GPU rather than stated in the config, SO THAT the same experiment file runs on a T4 and on a rented Ampere card.
13. AS A learner, I WANT quantization to stay a config key, SO THAT QLoRA versus plain LoRA remains a variable I can change per experiment.
14. AS A learner, I WANT the dev subset scored with the same scorer as the baselines, SO THAT the number means what the baselines' number means.
15. AS A learner, I WANT the dev score computed by actually generating, SO THAT parse failures and cut-off outputs are visible rather than hidden behind teacher forcing.
16. AS A learner, I WANT the evaluation metrics logged into the same run as the training loss on the same step axis, SO THAT I can see the loss fall and the F1 rise on one chart.
17. AS A learner, I WANT the dev metrics kept under a distinct prefix, SO THAT a 50-document training signal is never mistaken for a 200-document baseline result.
18. AS A learner, I WANT a checkpoint saved and pushed at every evaluation point, SO THAT a disconnected Colab session loses minutes rather than the whole run.
19. AS A learner, I WANT the pushed checkpoint to include what a resume would need, SO THAT "the adapter survived" is a fact rather than a hope.
20. AS A learner, I WANT only the adapter published and never a merged model, SO THAT the artifact stays small and the base model stays the base model.
21. AS A learner, I WANT the notebook to load the pushed adapter back onto a fresh base model and generate, SO THAT I have seen the artifact work from the Hub rather than from memory.
22. AS A learner, I WANT the run's config to record every resolved library version, SO THAT a future run that behaves differently can be diffed against this one.
23. AS A learner, I WANT every derived number in the run config, SO THAT I can reconstruct what actually ran without re-deriving it.
24. AS A learner, I WANT the run to point at the adapter repository rather than copy the adapter, SO THAT there is one home for the artifact.
25. AS A learner, I WANT a near-zero F1 to be an acceptable outcome, SO THAT I do not "fix" a working pipeline because a small model scored badly.
26. AS A learner, I WANT the diagnostics that indicate a broken pipeline named explicitly, SO THAT I know which numbers to actually look at.
27. AS A learner, I WANT enough optimizer steps for the loss trend to mean something, SO THAT "the loss went down" is evidence rather than noise.
28. AS A learner, I WANT a notebook cell that runs a handful of steps first, SO THAT an install or masking problem surfaces in one minute rather than ten.
29. AS A learner, I WANT the training notebook to need no runtime restart, SO THAT "run all" means run all.
30. AS A learner, I WANT the two notebooks named after what they do, SO THAT I never run baselines when I meant to train.
31. AS A learner, I WANT to be warned that re-running the training cell in one runtime abandons the W&B run, SO THAT I restart the runtime instead of losing a run's configuration.
32. AS A learner, I WANT the training module importable without a GPU, SO THAT the test suite runs on my laptop.
33. AS AN implementing agent, I WANT the step arithmetic to be a tested pure function, SO THAT the evaluation cadence is right before any GPU time is spent.
34. AS AN implementing agent, I WANT the callback testable with a fake engine and a fake trainer state, SO THAT its cadence and its logged keys are verified without a model.
35. AS AN implementing agent, I WANT the in-process engine to satisfy the existing engine type, SO THAT the callback is generation plus parsing plus scoring and nothing new.
36. AS A maintainer, I WANT milestone 4 to change only config values, SO THAT the full fine-tune is a run rather than a project.
37. AS A maintainer, I WANT the project's glossary to cover the training vocabulary, SO THAT the spec, the code and the docs use one set of words.
38. AS A maintainer, I WANT the design document corrected where this map superseded it, SO THAT the record does not quietly disagree with the code.

## Implementation Decisions

### Masking is ours, not the training library's

The prompt is rendered with the chat template and the generation prompt; the
full example is rendered with the same template; the first is asserted to be a
**token-level** prefix of the second; the labels are the full token sequence
with the prompt span set to the ignore index. The sequence is truncated after
the target's end-of-turn token, discarding the newline the template emits
after it.

This was prototyped against real documents and both tokenizers. The contract
it fixed, trimmed to the decision:

```python
prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True,
                                       add_generation_prompt=True,
                                       enable_thinking=False, return_dict=True)["input_ids"]
full   = tokenizer.apply_chat_template(messages, tokenize=True,
                                       enable_thinking=False, return_dict=True)["input_ids"]
if full[:len(prompt)] != prompt:
    raise MaskingError(...)                      # names the docid and both lengths
full   = full[: last_index_of(full, eos) + 1]    # drop the template's trailing newline
labels = [IGNORE] * len(prompt) + full[len(prompt):]
```

Rejected, with reasons recorded in the map: the library's assistant-only loss
(the pinned version cannot patch the Qwen templates, and the patched template
from a later version trains the empty thinking block that our inference prompt
already supplies) and the training framework's response-only helper (same
skew, and untestable without the GPU stack). The decisive property — that the
prompt is an exact token prefix — was verified across both models.

### The training module

A `train` module exposing: a config loader that validates; the masking
function; subset selection; dataset construction that tokenises, drops
over-budget examples and returns the count; the two step-arithmetic functions;
the scoring callback; a `train` entry point returning the run URL; and a thin
CLI taking a config path and an optional limit. Imports of the training and
quantization libraries are function-local so the module imports on CPU.

### The in-process engine joins the existing engine family

The generation module gains an engine backed by the model under training,
satisfying the existing engine type alongside the vLLM and constant engines.
It sets left padding itself, length-sorts its inputs, decodes greedily with
sampling parameters explicitly disabled, slices completions by prompt width
and derives the finish reason from the presence of the end-of-turn token. It
must not call the training library's inference-mode helper: the library
already wraps generation and calling the helper breaks its restore.

Consequently the callback is the existing generation call, the existing
parser, and the existing scorer — no new logic.

### Configuration

One YAML per experiment, in the project's own vocabulary; the module
translates to the training library's field names, so a pinned-library rename
does not reach the config. Keys: name, tags, model, dataset, adapter
repository, train and dev subset sizes, sequence budget, epochs, learning
rate, scheduler, warmup ratio, effective and per-device batch size, seed,
quantization, the LoRA block, evaluation cadence in epochs, a generation block
with token limit and batch size, and the W&B project.

- Accumulation is derived; an indivisible pair is an error.
- Compute dtype is detected from the GPU, not configured.
- Packing and padding-free are fixed off in code, not exposed.
- The evaluation cadence drives checkpointing as well as scoring.
- Absent subset sizes mean the whole split, which is how milestone 4 uses it.
- The smoke experiment runs **3 epochs**, not 1: at the 4B's effective batch
  size, 100 documents for 1 epoch is 6 optimizer steps and 2 evaluation
  points, too few for a loss trend to carry information.

### Over-budget examples are dropped and counted

Measured: one train document totals 2082 tokens, because the input budget
applied at data-preparation time bounds the prompt alone and a target can add
over 150 more. The training library does not enforce the sequence budget on a
pre-tokenised dataset, so nothing truncates silently; the module drops such
examples, counts them, and records the count in the run config. Dev documents
are unaffected and the generation path already refuses an over-budget input.

### Evaluation callback

Hooked on the end of a step with its own cadence, because the library's
evaluation hook never fires unless an evaluation strategy is configured. It
generates over the dev subset, parses, scores, logs, and retains the last
result for end-of-training artifacts. It restores nothing by hand: the
generation wrapper handles training-mode and cache restoration.

### One W&B run, ours

The module initialises the run — project, name, tags, job type, config —
before handing the training library its reporting switch, which must be set
explicitly because the pinned version defaults to none. Nothing in the
training stack finishes the run, so the module does.

Metrics: the library's own under its training prefix; the scorer's flattened
result under a dev prefix, logged together with the global step as an ordinary
metric and **without** an explicit step argument, since the integration pins
every metric to that step and an explicit step silently drops rows. The dev
prefix is deliberate: a 50-document training signal must not be averaged with
a 200-document baseline.

Config: the YAML verbatim, the resolved versions of the whole pinned chain,
the repository stamps, and every derived number. At the end of training: a
predictions table with the same columns as the baseline runs, and the
proof-criteria summary. The adapter is not copied into W&B — the run records
the repository and the final revision.

### Checkpointing and the Hub

The checkpoint strategy must be the one that includes the checkpoint directory,
not the root-mirror strategy: the latter explicitly excludes checkpoint folders
and therefore pushes nothing resumable, which is the opposite of what this
milestone is for. Pushes must be forced to complete, since an unforced push is
silently skipped when the previous upload is still running on the single shared
worker. The final push happens explicitly after training regardless of
strategy. The model card is written by overriding the trainer's card hook,
because the training library rewrites the card at every save and would
otherwise clobber a hand-written one and mislabel an adapter as a full model.

### Notebook and repo hygiene

A second notebook for training: runtime check, guarded install of the training
stack, clone at a ref and editable install, printed resolved versions,
secrets, a limited smoke run, the real run, and a final cell that loads the
pushed adapter onto a fresh base model and generates one document. **No
runtime restart** — the pinned stack is satisfied by Colab's current torch, so
none of milestone 2's restart workarounds apply. A markdown warning records
that re-running the training cell in one runtime abandons the configured W&B
run.

The baselines notebook is renamed to say what it does rather than where it
runs, with the README link, the design document's tree and the notebook test
updated with it.

### Documentation

The design document is corrected where this map superseded it — §9's "1 epoch"
for the smoke experiment, and §7 gains the masking mechanism, the padding-free
decision and the dropped-example policy — each as an in-place correction with
a note, not a silent edit. The README gains a fine-tuning section mirroring
the baseline section. The glossary has already gained the training vocabulary.

## Testing Decisions

A good test drives a public seam with real data shapes and asserts observable
behaviour — the labels produced, the counts returned, the keys logged — never
internals. Prior art: the data-preparation tests (fixture corpus, a
module-scoped tokenizer fixture behind the `tokenizer` marker), the generation
tests (a fake engine that records what it received and returns canned results)
and the scorer tests (hand-built gold/prediction pairs).

One new test module for the training module, plus one assertion added to the
existing notebook test. No new seam in the generation module: the in-process
engine satisfies the engine type the fake engine already stands in for.

1. **Masking**, both tokenizers, `tokenizer` marker: every prompt token
   ignored, every completion token kept, the boundary exactly at the prompt
   length, the sequence ending at the end-of-turn token, an empty document
   yielding exactly two kept tokens, and both tokenizers agreeing on the count
   of kept tokens.
2. **Prefix violation** raises the named error and names the document.
3. **Sequence budget**: an over-long example is dropped and counted; a dev
   example is untouched.
4. **Step arithmetic** over the real smoke numbers, plus fewer steps than one
   interval, a non-integer cadence, and a single epoch.
5. **Subset selection**: first N, N absent, N larger than the split.
6. **Config loading**: unknown key rejected by name, indivisible batch sizes
   rejected, the smoke experiment's every key accepted.
7. **The callback** with a fake engine and a fake trainer state: fires at the
   expected steps, logs exactly the dev-prefixed scorer keys plus the global
   step, and never passes an explicit step argument.
8. **Notebooks**: both load, neither carries outputs, every config path they
   mention exists, and the training notebook contains no runtime-restart call.

The in-process engine, the model loader and the trainer construction are
deliberately untested: they need a GPU, and the first smoke run is what
exercises them.

## Out of Scope

- Milestone 4: the 4B experiment file, its run, and evaluating a finished
  adapter over the full dev split (which raises its own question about
  fp16-base versus quantized-adapter numerics).
- Milestone 5: rented GPUs, bf16 LoRA, Colab paid compute.
- Milestone 6: `RESULTS.md` and the written comparison.
- A resume-after-disconnect path in the notebook. The checkpoint contents make
  it possible — the recovery requires downloading the pushed checkpoint first
  — but the path itself is not built here.
- Merged or GGUF export, rank and learning-rate sweeps, the plain-peft
  rewrite, the base-model variant, DocEE.
- Unifying the three modules' YAML readers into a shared config module.

## Further Notes

- **Done criterion.** One W&B run of the smoke experiment showing a loss curve
  whose last quarter is below its first, dev metrics at every expected
  evaluation point, the startup log of the first batch's kept-token count and
  decoded span, an adapter repository on the Hub, and a notebook cell that
  loaded that adapter back and generated. **No score gates this.** A near-zero
  F1 passes. The signals that would mean a broken pipeline are a parse-failure
  rate above the zero-shot baseline's 7.5 %, or a collapse to all-empty
  predictions together with a flat loss curve.
- **Order of work that de-risks earliest.** (1) Run the smoke experiment's
  first handful of steps on a T4 as soon as the masking function and a minimal
  trainer exist — confirm the labels reach the collator and the loss is
  finite, which is the one assumption taken from source rather than from a
  GPU. (2) The masking function and its tests. (3) Config loading, subset
  selection, step arithmetic and their tests. (4) The in-process engine. (5)
  The callback and its fake-driven test. (6) Checkpointing and the push. (7)
  The notebook, the rename and the notebook test. (8) Docs.
- **The assumption to check first.** That the training stack accepts a dataset
  carrying token ids and labels verbatim was established by reading the
  installed library's source, not by running it. If a real T4 disagrees, the
  fallback is the library's prompt-completion dataset form, which computes the
  same mask from the same prefix property.
- **Numbers worth knowing.** 5.9 % of tokens in the smoke subset carry loss;
  completions run 2 to 152 tokens, mean 34.6; 40 of the 100 documents are
  empty and contribute two kept tokens each. The adapter is under 40 MiB per
  push. Milestone 2's baselines to sit beside: zero-shot 18.6 micro-F1, 3-shot
  20.0, always-empty 0.
