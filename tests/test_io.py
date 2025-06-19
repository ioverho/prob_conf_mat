import pathlib
import numpy as np
import pytest

from prob_conf_mat.io import (
    validate_confusion_matrix,
    load_csv,
    ConfMatIOWarning,
    ConfMatIOError,
)

DATA_DIR = pathlib.Path("./tests/data")


class TestCSV:
    def test_file_not_exist(self):
        with pytest.raises(FileNotFoundError, match="No such file or directory:"):
            load_csv(
                location="foobarbaz.csv",
            )

    @pytest.mark.parametrize(
        argnames="file_path",
        argvalues=list((DATA_DIR / "confusion_matrices").glob("*.csv")),
    )
    def test_valid_csv(self, file_path):
        load_csv(
            location=file_path,
        )

    @pytest.mark.parametrize(
        argnames="file_path",
        argvalues=list((DATA_DIR / "malformed_confusion_matrices").glob("*.csv")),
    )
    def test_malformed_csv(self, file_path):
        with pytest.raises(
            expected_exception=ConfMatIOError,
        ):
            load_csv(
                location=file_path,
            )


class TestConfMatValidation:
    def test_dtype_conversion(self) -> None:
        # Integer should pass directly
        validate_confusion_matrix(confusion_matrix=[[1, 0], [0, 1]])

        # Object should fail
        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1, "foo"], [0, 1]])

        # Float should fail
        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1.0, 0], [0, 1]])

        # uint should not fail
        validate_confusion_matrix(
            confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.uint),
        )

        # complex float should fail
        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(
                confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.complex128),
            )

        # bool should not fail
        validate_confusion_matrix(
            confusion_matrix=np.array([[1, 0], [0, 1]], dtype=np.bool),
        )

        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is not of type integer.",
        ):
            validate_confusion_matrix(confusion_matrix=np.array([[1, 0], [np.inf, 1]]))

    def test_shape(self) -> None:
        # 2D Square matrix should not fail
        conf_mat = np.ones((2, 2), dtype=np.int64)
        validate_confusion_matrix(confusion_matrix=conf_mat)

        # Non 2D matrix should fail
        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is malformed.",
        ):
            conf_mat = np.ones((3, 3, 3), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        # Non square matrix should fail
        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is malformed.",
        ):
            conf_mat = np.ones((2, 4), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is malformed.",
        ):
            conf_mat = np.ones((4, 2), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix is malformed.",
        ):
            conf_mat = np.ones((1, 1), dtype=np.int64)
            validate_confusion_matrix(confusion_matrix=conf_mat)

    def test_empty(self) -> None:
        with pytest.raises(ConfMatIOError, match="Some rows contain no entries"):
            validate_confusion_matrix(confusion_matrix=[[0, 0], [0, 1]])

        with pytest.warns(
            ConfMatIOWarning,
            match="Some columns contain no entries, meaning model never predicted it.",
        ):
            validate_confusion_matrix(confusion_matrix=[[0, 1], [0, 1]])

    def test_value_bounds(self) -> None:
        with pytest.raises(
            ConfMatIOError,
            #match="The loaded confusion matrix has non-finite elements.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1, np.nan], [0, 1]])

        with pytest.raises(
            ConfMatIOError,
            match="The loaded confusion matrix has negative elements.",
        ):
            validate_confusion_matrix(confusion_matrix=[[1, -1], [0, 1]])
