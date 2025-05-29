import numpy as np
import pytest

from bayes_conf_mat.io import (
    validate_confusion_matrix,
    load_csv,
    ConfMatIOException,
    ConfMatIOWarning,
)


class TestCSV:
    def test_file_not_exist(self):
        with pytest.raises(FileNotFoundError, match="No such file or directory:"):
            load_csv(
                location="foobarbaz.csv",
            )


class TestConfMatValidation:
    def test_dtype_conversion(self) -> None:
        # Integer should pass directly
        validate_confusion_matrix(confusion_matrix=[[1, 0], [0, 1]])

        # Object should fail
        with pytest.raises(
            ConfMatIOException,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1, "foo"], [0, 1]])

        # Float should fail
        with pytest.raises(
            ConfMatIOException,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1.0, 0], [0, 1]])

        # uint should not fail
        validate_confusion_matrix(
            confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.uint)
        )

        # complex float should fail
        with pytest.raises(
            ConfMatIOException,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(
                confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.complex128)
            )

        # bool should not fail
        validate_confusion_matrix(
            confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.bool)
        )

        with pytest.raises(
            ConfMatIOException,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=np.array([[1, 0], [np.inf, 1]]))

    def test_shape(self) -> None:
        # 2D Square matrix should not fail
        conf_mat = np.ones((2, 2), dtype=np.int64)
        validate_confusion_matrix(confusion_matrix=conf_mat)

        # Non 2D matrix should fail
        with pytest.raises(
            ConfMatIOException, match="The loaded confusion matrix is malformed."
        ):
            conf_mat = np.ones((3, 3, 3), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        # Non square matrix should fail
        with pytest.raises(
            ConfMatIOException, match="The loaded confusion matrix is malformed."
        ):
            conf_mat = np.ones((2, 4), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        with pytest.raises(
            ConfMatIOException, match="The loaded confusion matrix is malformed."
        ):
            conf_mat = np.ones((4, 2), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        with pytest.raises(
            ConfMatIOException, match="The loaded confusion matrix is malformed."
        ):
            conf_mat = np.ones((1, 1), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

    def test_empty(self) -> None:
        with pytest.raises(ConfMatIOException, match="Some rows contain no entries"):
            validate_confusion_matrix(confusion_matrix=[[0, 0], [0, 1]])

        with pytest.warns(
            ConfMatIOWarning,
            match="Some columns contain no entries, meaning model never predicted it.",
        ):
            validate_confusion_matrix(confusion_matrix=[[0, 1], [0, 1]])
