# Taken from https://blog.iqmo.com/blog/python/jupyter_notebook_testing/
import os
from pathlib import Path

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path("/home/ioverho/prob_conf_mat/documentation/Getting Started")
SKIP_NOTEBOOKS = []
TIMEOUT = 600


@pytest.mark.parametrize(
    argnames="notebook",
    argvalues=[
        file for file in NOTEBOOK_DIR.glob("*.ipynb") if file.name not in SKIP_NOTEBOOKS
    ],
)
def test_notebook_execution(notebook: Path) -> None:
    # First set the absolute path of the notebook
    notebook_abs_fp = notebook.resolve()

    # Change the working directory to that of the notebook
    os.chdir(notebook.parent)

    with notebook_abs_fp.open() as f:
        nb = nbformat.read(fp=f, as_version=4)

    ep = ExecutePreprocessor(timeout=TIMEOUT)
    ep.preprocess(nb)
