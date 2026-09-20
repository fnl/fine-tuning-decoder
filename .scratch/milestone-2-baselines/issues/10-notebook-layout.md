# 10 Notebook cells and secrets handling

Type: grilling
Status: open
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
