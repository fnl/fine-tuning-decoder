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


# The stack every GPU run so far resolved to (milestone-4 ticket 12); torch stays Colab's.
PINNED = {
    "unsloth": "2026.9.11",
    "unsloth_zoo": "2026.9.7",
    "trl": "0.24.0",
    "transformers": "5.5.0",
    "peft": "0.20.0",
    "bitsandbytes": "0.50.2",
}


def test_training_install_pins_every_stamped_package_but_torch() -> None:
    from train import VERSIONED

    pins = {
        package: pinned
        for cell in code_cells("train.ipynb")
        if "pip install" in cell.source
        for package, pinned in re.findall(r"([\w-]+)==([\w.+-]+)", cell.source)
    }
    assert pins == {package: PINNED[package] for package in VERSIONED if package != "torch"}


def test_every_resume_cell_skips_itself_while_its_run_id_is_empty() -> None:
    # "Run all" must never resume the run it just finished
    cells = [
        cell.source
        for cell in code_cells("train.ipynb")
        if re.search(r"-m train .*--resume", cell.source)
    ]
    assert cells and all(
        re.search(r'^(\w+) = ""  # @param.*^if \1:$', source, re.MULTILINE | re.DOTALL)
        for source in cells
    )
