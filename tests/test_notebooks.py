# Taken from https://blog.iqmo.com/blog/python/jupyter_notebook_testing/
from pathlib import Path

import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_DIR = Path("/home/ioverho/bayes_conf_mat/documentation/Getting Started")
SKIP_NOTEBOOKS = ["mnist_digits_class.ipynb"]
TIMEOUT = 600

@pytest.mark.parametrize(
    argnames="notebook", argvalues=[file for file in NOTEBOOK_DIR.glob('*.ipynb') if file.name not in SKIP_NOTEBOOKS]
)
def test_notebook_execution(notebook: Path) -> None:
    with open(file=notebook) as f:
        nb = nbformat.read(fp=f, as_version=4)

    ep = ExecutePreprocessor(timeout=TIMEOUT)
    ep.preprocess(nb)
