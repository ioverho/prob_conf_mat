import re
from pathlib import Path

import pytest

from bayes_conf_mat.io import get_io
from bayes_conf_mat.io.abc import ConfMatIOException, ConfMatIOWarning

class TestCSV:
    def test_file_not_exist(self):
        with pytest.raises(ValueError, match="No file found at:"):
            get_io(
                format="csv",
                location="fobbarbaz.csv",
                type="confusion_matrix",
            ).load()

    def test_nonexistent_type(self):
        with pytest.raises(ValueError, match="For CSV, `type` must be one of"):
            get_io(
                format="csv",
                location="./tests/data/confusion_matrices/sklearn_1.csv",
                type="foobarbaz",
            ).load()

    def test_incorrect_io_type(self):
        with pytest.raises(ValueError):
            get_io(
                format="csv",
                location="./tests/data/confusion_matrices/sklearn_face_classification.csv",
                type="cond_pred",
            ).load()

    def test_conf_mat_validation(self):
        test_conf_mats_dir = Path("./tests/data/malformed_confusion_matrices")

        with pytest.raises(
            ValueError, match="The requested array has an inhomogeneous shape"
        ):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "malformed_shape.csv",
                type="confusion_matrix",
            ).load()

        with pytest.raises(ConfMatIOException, match="Row contains values that cannot"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "nan_values.csv",
                type="confusion_matrix",
            ).load()

        with pytest.raises(ConfMatIOException, match="Row contains values that cannot"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "non_integer_data.csv",
                type="confusion_matrix",
            ).load()

        with pytest.raises(ConfMatIOException, match="Some rows contain no entries"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "zero_row_counts.csv",
                type="confusion_matrix",
            ).load()

        with pytest.warns(ConfMatIOWarning, match="Some columns contain no entries, meaning model never predicted it."):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "zero_col_counts.csv",
                type="confusion_matrix",
            ).load()

class TestInMemory:
    def test_no_data(self):
        with pytest.raises(TypeError, match=re.escape("InMemory.__init__() missing 1 required positional argument: 'data'")):
            get_io(
                format="in_memory",
                ).load()

    def test_array_like_conversion(self):
        with pytest.raises(ConfMatIOException, match="The constructed confusion matrix is not of integer type"):
            get_io(
                format="in_memory",
                data=[[1, "foo"], [0, 1]]
                ).load()

        with pytest.raises(TypeError, match="In-memory confusion matrix is of invalid type. Must a `np.ArrayLike`."):
            get_io(
                format="in_memory",
                data=[[1, 0], [0, 1], [0, 0, 1]]
                ).load()

    def test_conf_mat_validation(self):
        with pytest.raises(
            ConfMatIOException, match="The constructed confusion matrix is malformed."
        ):
            get_io(
                format="in_memory",
                data=[[1, 0], [0, 1], [0, 1]]
                ).load()

        with pytest.raises(
            ConfMatIOException, match="The constructed confusion matrix is malformed."
        ):
            get_io(
                format="in_memory",
                data=[[1, 0, 1], [0, 1, 0]]
                ).load()

        with pytest.raises(ConfMatIOException, match="Some rows contain no entries"):
            get_io(
                format="in_memory",
                data=[[0, 0], [0, 1]]
                ).load()

        with pytest.warns(ConfMatIOWarning, match="Some columns contain no entries, meaning model never predicted it."):
            get_io(
                format="in_memory",
                data=[[0, 1], [0, 1]]
                ).load()
