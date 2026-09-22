# MUC-4 Event Extraction

Fine-tuning a small decoder LLM to read a news document and emit every event
it describes as structured JSON, scored against the MUC-4 annotations with
the GTT evaluation. This glossary fixes the words we use for that pipeline:
data, prompting, model output, scoring, and experiments.

## Language

### Corpus

**Corpus**:
The raw MUC-4 documents with their annotations, as preprocessed by GTT, in
three splits.
_Avoid_: raw data, source data, MUC-4 files

**Document**:
One MUC-4 news text, identified by a docid, from which zero or more events
are extracted.
_Avoid_: article, text, sample, doc (in prose)

**Split**:
One of the three fixed partitions of the corpus: `train`, `dev`, `test`.
_Avoid_: validation, holdout, eval set

**Relevant document**:
A document with at least one event. An irrelevant document has none and its
target is empty.
_Avoid_: positive document, negative document

**Dataset**:
The chat-formatted examples derived from the corpus and published on the
Hub, one example per document, keeping the three splits.
_Avoid_: training data, prepared data, corpus

### Events

**Event**:
One incident described in a document: an event type plus five roles. A
document yields zero or more events. Called a *template* in the MUC-4 data
and the GTT scorer.
_Avoid_: template, frame, record, incident (as the unit)

**Event type**:
The category of an event, one of six: attack, bombing, kidnapping, arson,
robbery, forced work stoppage. Stored as `incident_type` in the corpus.
_Avoid_: incident type, class, label

**Role**:
One of the five participant slots of an event: PerpInd (perpetrating
individual), PerpOrg (perpetrating organisation), Target (physical target),
Victim (human victim), Weapon. A role holds zero or more entities.
_Avoid_: slot, argument, field, key

**Entity**:
One participant filling a role, represented as the set of its coreferent
mentions.
_Avoid_: filler, argument, cluster, coreference chain

**Mention**:
A verbatim string from the document that refers to an entity. In the corpus
every mention carries an offset.
_Avoid_: span, surface form, string

**Offset**:
The character position of a mention in its document. Present in the corpus,
absent from targets and predictions.
_Avoid_: position, index, span start

**Empty role**:
A role holding no entities. Omitted from targets.
_Avoid_: null role, missing role

**Gold**:
The annotated events of a document, in canonical form, that predictions are
scored against.
_Avoid_: reference, ground truth, labels

**Canonical events**:
A document's gold events after removing duplicate events, ordering events
and mentions deterministically, and dropping offsets. The form used for
targets and for scoring.
_Avoid_: normalised events (normalisation is a mention operation), cleaned events

**Duplicate event**:
Two annotated events of one document that are identical once offsets are
removed. Only one is kept.
_Avoid_: repeated template

### Examples and prompts

**Example**:
The chat-formatted form of one document: a system prompt, a user message
holding the document text, and an assistant message holding the target.
_Avoid_: sample, row, conversation, instance, record

**System prompt**:
The single fixed instruction that names the event types, defines the roles,
states the exclusion rule and specifies the output format. Identical across
zero-shot, few-shot and fine-tuned runs.
_Avoid_: instruction, task description, preamble, prompt (on its own)

**Exclusion rule**:
The system prompt rule that military clashes between armed forces are not
events.
_Avoid_: military rule, negative rule

**Input**:
Everything the model sees before generating: the system prompt, any
exemplars, and the document's user message, as rendered by the chat template.
_Avoid_: prompt, context, query

**Exemplar**:
A train example whose document and target are placed in the input as a
demonstration turn pair; a few-shot baseline uses the same fixed exemplars in
every input.
_Avoid_: shot, demonstration, few-shot example, in-context example

**Input budget**:
The maximum number of input tokens an example may have.
_Avoid_: max length, context limit, sequence length

**Target**:
The JSON representation of a document's canonical events that the model is
trained to emit; empty for an irrelevant document.
_Avoid_: label, answer, completion, ground truth, output

**Truncated document**:
A dev or test document whose text was shortened to fit the input budget.
_Avoid_: truncated input, clipped document, truncation (unqualified)

**Dropped document**:
A train document excluded because it exceeds a budget: the input budget when
the dataset is prepared, or the sequence budget when a training run tokenises
it. Dev and test documents are never dropped. Always counted and reported.
_Avoid_: filtered document, skipped document

### Model output

**Output**:
The raw text a model generates for one input.
_Avoid_: generation, response, completion, answer

**Prediction**:
The events parsed from one output for one document.
_Avoid_: pred, hypothesis, system output, extracted templates

**Cut-off output**:
An output ended by the generation length limit before its JSON was
complete. The complete events in it are salvaged into the prediction.
_Avoid_: truncated output, truncation (unqualified), incomplete JSON

**Parse failure**:
An output from which no event could be recovered. Scored as an empty
prediction and counted in the diagnostics.
_Avoid_: invalid output, malformed output, parse error

### Scoring

**Scorer**:
Our port of the GTT evaluation: it turns predictions and gold into per-role
and micro precision, recall and F1.
_Avoid_: evaluator, metric, eval (as a noun)

**Evaluation**:
The full loop over a split: generate outputs, parse predictions, score,
compute diagnostics, record the results. Scoring is one step of it.
_Avoid_: testing, eval (as a noun), benchmark

**Mention normalisation**:
Lower-casing, removing punctuation and articles, and collapsing whitespace,
applied to every mention before comparison.
_Avoid_: cleaning, canonicalisation (reserved for events)

**Subset match**:
The entity comparison rule: a predicted entity is correct iff all of its
normalised mentions belong to one gold entity of the same role.
_Avoid_: any-mention match, overlap match, fuzzy match

**Alignment**:
The one-to-one assignment of predicted events to gold events (or to none)
that maximises a document's micro-F1. A predicted event can align only with
a gold event of the same event type.
_Avoid_: matching (reserved for entities), pairing, mapping

**Spurious event**:
A predicted event left unaligned. Counts against precision.
_Avoid_: false positive, hallucinated event, extra template

**Missing event**:
A gold event left unaligned. Counts against recall.
_Avoid_: false negative, missed template

**Micro-F1**:
The headline metric: F1 over all entities and event types pooled across
roles and documents.
_Avoid_: overall F1, average F1, macro-F1

**Per-role F1**:
Precision, recall and F1 for one role, or for the event type, pooled across
documents.
_Avoid_: slot F1, argument F1

**Diagnostics**:
The secondary metrics reported beside F1: relevance accuracy, event-count
accuracy, event-type accuracy and parse-failure rate.
_Avoid_: auxiliary metrics, stats, extra metrics

**Relevance accuracy**:
The share of documents where prediction and gold agree on whether the
document is relevant.
_Avoid_: detection accuracy, empty accuracy

**Event-count accuracy**:
The share of documents where the prediction has exactly as many events as
the gold.
_Avoid_: template-count match, count accuracy

**Event-type accuracy**:
The share of predicted events whose event type equals that of the gold
event they pair with when events are paired without regard to type.
_Avoid_: incident-type accuracy, type accuracy on aligned templates

### Training

**Prompt**:
The part of an example the model is not trained to produce: the system
prompt, any exemplars and the document, up to and including the generation
prompt. The *input* of an evaluation, seen from the training side.
_Avoid_: context, instruction part, source

**Completion**:
The part of an example the model is trained to produce: the target followed
by its end-of-turn token. Everything the chat template emits after that token
is excluded, because generation stops there.
_Avoid_: response, answer, assistant part, label span

**Masked token**:
A token excluded from the loss, its label set to -100. Every prompt token is
masked.
_Avoid_: ignored token, padded token, -100 token

**Completion-only loss**:
Training loss computed over the completion tokens alone. The project's one
non-negotiable training decision.
_Avoid_: response-only loss, assistant-only loss, masked loss

**Sequence budget**:
The maximum number of tokens of prompt and completion together that one
training example may have. Distinct from the *input budget*, which bounds the
prompt alone when the dataset is prepared.
_Avoid_: max length, max_seq_len, context limit

**Subset**:
The first N examples of a split, named by an experiment's config. Absent N
means the whole split.
_Avoid_: sample, slice, shard

**Eval point**:
A step in a run at which the dev subset is scored and a checkpoint is saved.
The two coincide by design.
_Avoid_: evaluation step, callback, validation step

**Checkpoint**:
What a run saves at an eval point: the adapter, and the optimizer state that
would let the run resume.
_Avoid_: snapshot, save, model

### Models and experiments

**Base model**:
The pre-trained instruct checkpoint that is fine-tuned and that also serves,
unchanged, as its own zero-shot and few-shot baseline.
_Avoid_: foundation model, backbone, original model

**Smoke model**:
The small same-family model used to prove the training loop end to end, not
to produce results.
_Avoid_: tiny model, test model

**Baseline**:
A result obtained without fine-tuning: zero-shot, few-shot, or the trivial
always-empty prediction.
_Avoid_: control, reference run

**Engine**:
The component that turns rendered inputs into outputs: an inference server
on a GPU, an in-process model, or the constant engine that answers `[]` for
the always-empty baseline.
_Avoid_: backend, generator, inference server, model (for the concept)

**Adapter**:
The LoRA weights produced by fine-tuning, published on their own and never
merged into the base model.
_Avoid_: fine-tuned model, merged model, weights

**Experiment**:
One named configuration of model, data subset and training knobs, described
by one config file.
_Avoid_: setup, trial, config (for the concept)

**Run**:
One execution of an experiment, tracked with its metrics.
_Avoid_: job, session, training

**Smoke run**:
A run of the smoke model on a small subset whose only purpose is to prove
that training, masking, checkpointing and evaluation work.
_Avoid_: test run, dry run

**Milestone**:
One of the six ordered project stages. Milestone 4 defines done.
_Avoid_: phase, sprint, step

**Round two**:
Work explicitly deferred past the six milestones.
_Avoid_: future work, backlog, v2
