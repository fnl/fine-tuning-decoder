# 04 Which Qwen3-4B instruct checkpoint?

Type: research
Status: resolved
Blocked by: —
HITL: no

## Question

DESIGN §5 says "Qwen3-4B-Instruct" and "check the Unsloth list, a newer
generation may exist". Today the Hub has at least `Qwen/Qwen3-4B` (hybrid
thinking, needs `enable_thinking=False`) and `Qwen/Qwen3-4B-Instruct-2507`
(non-thinking instruct). Determine, with sources:

1. The exact checkpoint ID to use for baselines *and* later fine-tuning
   (must be the same model), license, and whether a newer Qwen instruct
   generation ≤4B exists as of 2026-09.
2. Whether Unsloth publishes a matching repo (`unsloth/...`) and whether
   it lists the model as supported for QLoRA on T4.
3. Whether `enable_thinking=False` is needed/accepted by the chosen
   model's chat template, and what the template emits (any `<think>` block
   to strip from outputs?).
4. Tokenizer identity: is the tokenizer identical to the one
   `data/prepare.py` used for `n_input_tokens` (`Qwen/Qwen3-0.6B` family)?
   If not, does the prepared dataset's truncation flag still hold?

The answer decides `model:` in every YAML and the smoke config
`configs/qwen3-0.6b-smoke.yaml`.

## Answer

**Use `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) for baselines and fine-tuning; load weights from `unsloth/Qwen3-4B-Instruct-2507(-unsloth-bnb-4bit)` on Colab if desired, but keep the chat template from the official repo.**

### 1. Checkpoint, license, newer generations (as of 2026-09-20)
- `Qwen/Qwen3-4B-Instruct-2507`: created 2025-08-05, README 2025-08-06, last commit 2025-09-17 (tokenizer_config), `license:apache-2.0`, 4.0B params (3.6B non-emb.), 262k native context, needs `transformers>=4.51`. Card: "This model supports only non-thinking mode and does not generate `<think></think>` blocks in its output." https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 (commits: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/commits/main)
- `Qwen/Qwen3-4B` (2025-04-27, hybrid thinking) is the older sibling; not needed.
- Newer ≤4B text-capable Qwen: **`Qwen/Qwen3.5-4B`** (created 2026-02-27, public 2026-03-02, Apache-2.0). Rejected for this project: it is a vision-language model ("Causal Language Model with Vision Encoder") with a Gated-DeltaNet/attention hybrid, 248k vocab, "The latest `transformers` is required" (Unsloth: "use `transformers v5`"), custom Mamba Triton kernels, and Unsloth says "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models". Thinking default is contradictory across sources (card: "operate in thinking mode by default"; Unsloth: "For Qwen3.5 0.8B, 2B, 4B and 9B, reasoning is disabled by default") - not verified. https://huggingface.co/Qwen/Qwen3.5-4B , https://unsloth.ai/docs/models/qwen3.5/fine-tune.md
- Qwen3.6 (2026-04), Qwen3.7 (proprietary), Qwen3.8 (2026-08: 27B, 2.4T-A95B) have no ≤4B release. https://github.com/QwenLM/Qwen3.8 , https://en.wikipedia.org/wiki/Qwen
- Full `Qwen/*4B*` listing (HF API, newest first): Qwen-Drive-1.0-4B (2026-08-27), Qwen3.5-4B(-Base) (2026-02-27), Qwen3-VL-4B-* (2025-10), Qwen3-4B-{Instruct,Thinking}-2507 (2025-08-05), ... - no other text-only instruct ≤4B.

### 2. Unsloth support
- Repos exist: `unsloth/Qwen3-4B-Instruct-2507`, `-unsloth-bnb-4bit`, `-bnb-4bit`, `-FP8`, `-GGUF` (all Apache-2.0, created 2025-08-06). https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507
- Official notebook `nb/Qwen3_(4B)-Instruct.ipynb` (README row "Qwen3 (4B) | Conversational") uses `model_name = "unsloth/Qwen3-4B-Instruct-2507"`, `load_in_4bit = True`, `max_seq_length = 2048`, and states: 'press "Run all" on a **free** Tesla T4 Google Colab instance!'. Kaggle variant is pinned to `accelerator=nvidiaTeslaT4`. https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Instruct.ipynb
- Unsloth's Qwen3 docs page lists `unsloth/Qwen3-[SIZE]-unsloth-bnb-4bit` (0.6B..32B) but does not name the 2507 checkpoint explicitly; support is evidenced by the mirror repos and the notebook, not the docs list.

### 3. `enable_thinking` and `<think>` output
- Official 2507 template contains neither `enable_thinking` nor `<think>` (verified: downloaded `tokenizer_config.json`). Passing `enable_thinking=False` is accepted and silently ignored (rendered identically with `{}`, `False`, `True`: 198 tokens on a test prompt). Card: "specifying `enable_thinking=False` is no longer required." Nothing to strip from outputs.
- **Trap**: `unsloth/Qwen3-4B-Instruct-2507` (and its `-unsloth-bnb-4bit`) ship a *different* `chat_template.jinja` (hybrid Qwen3 template). Rendered on our messages it inserts `<think>\n\n</think>\n\n` before every assistant turn in *training* renders but not in the generation prompt -> train/inference mismatch. Unsloth's own notebook avoids this via `get_chat_template(tokenizer, chat_template="qwen3-instruct")`, whose assistant turns render without a think block (verified in `unsloth/chat_templates.py`). So: either load tokenizer from `Qwen/Qwen3-4B-Instruct-2507`, or call `get_chat_template(..., "qwen3-instruct")`, and add a test asserting no `<think>` in rendered training text.
- Smoke model `Qwen/Qwen3-0.6B` (config `qwen3-0.6b-smoke.yaml`) *does* use the hybrid template: with `enable_thinking=False` it appends `<think>\n\n</think>\n\n` to the generation prompt (+4 tokens) and to assistant turns; the model then answers without a think block. Keep `enable_thinking=False` in train.py: no-op for 2507, required for 0.6B. There is no 0.6B in the 2507 line.

### 4. Tokenizer identity vs `data/prepare.py`
- Ticket premise is stale: `prepare.py` sets `TOKENIZER_ID = "Qwen/Qwen3-4B-Instruct-2507"` (line 23) and renders with `enable_thinking=False`, so `n_input_tokens` already uses the recommended checkpoint's tokenizer and template. Truncation flag holds as-is.
- `tokenizer.json` sha256 `aeb13307...92dae4` and `vocab.json` are byte-identical across `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Instruct-2507`, `unsloth/Qwen3-4B-Instruct-2507(-unsloth-bnb-4bit)` (HF API `?blobs=true`); vocab len 151669 everywhere. `merges.txt` differs only by a `#version: 0.2` header line. Only chat templates differ.
- Prepared data: train max `n_input_tokens` 1767, dev 1375, test 1352, 0 truncated, budget 1800. Under the 0.6B template (+4 tokens) the max is 1771 < 1800, so the flag also holds for the smoke model. (`Qwen3.5-4B` has a different 248k-token vocab; the counts would not transfer.)

### Open caveats
- `src/data/prepare.py` is **not tracked by git**: `.gitignore` line `data/` matches `src/data/` too (`git check-ignore -v` confirms). Fix to `/data/` and commit.
- Unsloth mirror template mismatch (item 3) must be guarded in train.py + a test; DESIGN §5 should say "2507" and drop the "check the Unsloth list" note.
- Qwen3.5-4B thinking-default and T4 fine-tunability not verified (conflicting sources); revisit only if the project moves to transformers v5.
- Unsloth docs never state "Qwen3-4B-Instruct-2507 QLoRA on T4" verbatim; the claim rests on the notebook text and `load_in_4bit=True`.
- vLLM on T4 for this model is ticket 05, not covered here.

