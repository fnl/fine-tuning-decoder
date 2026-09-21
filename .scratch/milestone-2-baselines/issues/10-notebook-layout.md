# 10 Notebook cells and secrets handling

Type: grilling
Status: resolved
Blocked by: 02, 03, 09
HITL: yes

## Question

Layout of `notebooks/colab.ipynb` for milestone 2 (strictly M2 cells):

1. clone `fnl/fine-tuning-decoder` at a ref (how the ref is chosen: a
   form field, default `main`?) and `cd`;
2. install: `uv`? or `pip install -e .` plus the engine install line from 06;
3. secrets: `google.colab.userdata` → `os.environ` for `HF_TOKEN`,
   `WANDB_API_KEY`; fail loudly if missing;
4. `generate` and `eval` cells for each of the two GPU baselines (loop over
   configs vs. one cell per run);
5. output-stripping convention (`nbstripout`? a `ruff`-excluded dir already
   exists) and how the notebook is tested at all (a smoke `--limit 5` run).

## Answer

Grilled 2026-09-20. Facts: no `notebooks/` yet, no nbstripout, no
pre-commit; ruff already excludes `notebooks`; Colab has pip, not uv.

**Cells** (`notebooks/colab.ipynb`, strictly M2; markdown cell on top
explaining "Run all twice: the install cell restarts the runtime once"):
1. runtime check: `!nvidia-smi`, `torch.__version__`, `sys.version`
2. clone + checkout: `REF = "main"  # @param {type:"string"}`;
   `git clone https://github.com/fnl/fine-tuning-decoder` (skipped if the
   dir exists) → `git checkout $REF` → `%cd`
3. install, guarded by `importlib.util.find_spec("vllm") is None`:
   `pip install vllm==0.29.0 --extra-index-url https://wheels.vllm.ai/0.29.0/cu129
   --extra-index-url https://download.pytorch.org/whl/cu129` then
   `pip install -e .`; ends with `get_ipython().kernel.do_shutdown(restart=True)`
   only when it installed (auto-restart; cells 1–3 idempotent)
4. secrets: `google.colab.userdata.get` → `os.environ` for `HF_TOKEN` (set
   now, used by milestone 3) and `WANDB_API_KEY`; assert non-empty
5. gold: `load_dataset("fnl-es/muc4-chat")["dev"].to_json("data/prepared/dev.jsonl")`
6. smoke: `!python -m generate --config configs/qwen3-4b-zero-shot.yaml
   --limit 5 --out outputs/smoke` + `!head` of the JSONL; no W&B
7. zero-shot: `!python -m generate --config configs/qwen3-4b-zero-shot.yaml
   && python -m eval --config … --pred outputs/qwen3-4b-zero-shot/dev.jsonl
   --gold data/prepared/dev.jsonl --wandb`
8. 3-shot: same for `configs/qwen3-4b-3-shot.yaml`

One cell per run (each ends in its `run.url`). The always-`[]` baseline runs
locally, not in the notebook.

**Output stripping**: standard git filter. `nbstripout` as a dev dependency;
`uv run nbstripout --install --attributes .gitattributes` commits
`*.ipynb filter=nbstripout` in `.gitattributes`; the per-clone `.git/config`
part is documented in README setup ("run once after cloning"). The agent runs
`--install` in this clone during implementation and verifies with
`git check-attr filter notebooks/colab.ipynb`. The notebook is authored as
JSON via `nbformat`, so it is committed without outputs from the start.

**Testing**: CPU test parses the notebook with `nbformat`, asserts no cell
has outputs, and that every `configs/*.yaml` referenced exists. Cell 6 is the
GPU smoke; the three W&B runs are M2's validation.

**Opening**: README "Open in Colab" link
`https://colab.research.google.com/github/fnl/fine-tuning-decoder/blob/main/notebooks/colab.ipynb`;
`REF` defaults to `main`.

## Comments

**2026-09-21:** cell order changed after the first Colab runs. The runtime
restart resets the working directory, so with clone (cell 2) before install
(cell 3) every later cell ran in `/content` and `configs/…` was not found
(a `%cd` per cell was tried and rejected as litter). Now: runtime check →
vLLM install + `torchaudio` removal + restart → clone, checkout, `%cd`,
`pip install -e .` → import check → secrets → gold → smoke → zero-shot →
3-shot. The clone cell checks out `origin/$REF` (detached) so re-running it
picks up new pushes; "Run all twice" still holds.
