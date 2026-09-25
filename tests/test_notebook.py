"""Notebook tests: the Colab notebooks parse, carry no outputs and name only existing configs."""

import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOKS = ROOT / "notebooks"


def code_cells(name: str) -> list[nbformat.NotebookNode]:
    notebook = nbformat.read(NOTEBOOKS / name, as_version=4)
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


@pytest.mark.parametrize("name", ["baselines.ipynb", "train.ipynb"])
def test_notebook_has_no_cell_outputs(name: str) -> None:
    assert all(cell.outputs == [] for cell in code_cells(name))


@pytest.mark.parametrize("name", ["baselines.ipynb", "train.ipynb"])
def test_every_config_the_notebook_names_exists(name: str) -> None:
    cells = code_cells(name)
    configs = {m for cell in cells for m in re.findall(r"configs/\S+\.yaml", cell.source)}
    assert configs and all((ROOT / config).is_file() for config in configs)


def test_training_notebook_never_restarts_the_runtime() -> None:
    assert not any("do_shutdown" in cell.source for cell in code_cells("train.ipynb"))
