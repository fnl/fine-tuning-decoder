"""Notebook tests: the Colab notebook parses, carries no outputs and names only existing configs."""

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "notebooks" / "colab.ipynb"


def code_cells() -> list[nbformat.NotebookNode]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_has_no_cell_outputs() -> None:
    assert all(cell.outputs == [] for cell in code_cells())


def test_every_config_the_notebook_names_exists() -> None:
    configs = {m for cell in code_cells() for m in re.findall(r"configs/\S+\.yaml", cell.source)}
    assert configs and all((ROOT / config).is_file() for config in configs)
