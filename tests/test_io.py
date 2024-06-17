from pathlib import Path

import pytest

from bayes_conf_mat.io import get_io
from bayes_conf_mat.io.base import ConfMatIOException, ConfMatIOWarning

class TestCSV:
    def test_file_not_exist(self):
        with pytest.raises(ValueError, match="No file found at:"):
            get_io(
                format="csv",
                location="fobbarbaz.csv",
                type="confusion_matrix",
            )

    def test_nonexistent_type(self):
        with pytest.raises(ValueError, match="For CSV, `type` must be one of"):
            get_io(
                format="csv",
                location="./tests/data/confusion_matrices/sklearn_1.csv",
                type="foobarbaz",
            )

    def test_incorrect_io_type(self):
        with pytest.raises(ValueError):
            get_io(
                format="csv",
                location="./tests/data/confusion_matrices/sklearn_face_classification.csv",
                type="cond_pred",
            )()

    def test_conf_mat_validation(self):
        test_conf_mats_dir = Path("./tests/data/malformed_confusion_matrices")

        with pytest.raises(
            ValueError, match="The requested array has an inhomogeneous shape"
        ):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "malformed_shape.csv",
                type="confusion_matrix",
            )()

        with pytest.raises(ConfMatIOException, match="Row contains values that cannot"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "nan_values.csv",
                type="confusion_matrix",
            )()

        with pytest.raises(ConfMatIOException, match="Row contains values that cannot"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "non_integer_data.csv",
                type="confusion_matrix",
            )()

        with pytest.raises(ConfMatIOException, match="Some rows contain no entries"):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "zero_row_counts.csv",
                type="confusion_matrix",
            )()

        with pytest.warns(ConfMatIOWarning):
            get_io(
                format="csv",
                location=test_conf_mats_dir / "zero_col_counts.csv",
                type="confusion_matrix",
            )()
