# 11 `train.ipynb` layout and the repo changes around it

Type: grilling
Status: open
Blocked by: 01, 07, 08
HITL: yes

## Question

Milestone 2's notebook is 8 cells and took three attempts to make work on a
T4; its lessons (auto-restart guard, `%cd` after every restart, chaining with
`&&`, secrets asserted loudly, a `--limit 5` smoke cell before the real run)
are the starting point, not a rediscovery. Decide:

- The cell list for `notebooks/train.ipynb`: runtime check → guarded Unsloth
  install + restart → clone@`REF` → `pip install -e .` → secrets → dev gold
  dump → a `--limit` smoke cell → the real training cell. Is there also a
  "load the pushed adapter back and generate one document" cell, given
  ticket 06's proof criteria?
- The rename `colab.ipynb` → `baselines.ipynb`: does it happen in this spec
  (and with it the README's Colab link, the `test_notebook.py` fixture and
  the DESIGN §3 tree), or is it left alone to keep the diff small?
- How `test_notebook.py` extends to two notebooks: same assertions (loads,
  no outputs, referenced config paths exist) parameterised over both?
- Whether `train.ipynb` needs anything from `data/prepared/` at all, given
  the dataset is on the Hub and the callback scores in-process.
- README: a "Fine-tuning" section mirroring "Running a baseline", and what
  the prerequisites table gains (a Hub token with write scope is already
  listed "from milestone 3").
- `.gitattributes` / nbstripout already cover a second notebook — confirm.

Fixes the spec's notebook and docs sections.
