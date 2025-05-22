import warnings
import csv
import pathlib

import numpy as np
import jaxtyping as jtyping

DEFAULT_DTYPE = np.int64


class ConfMatIOWarning(Warning):
    pass


class ConfMatIOException(Exception):
    pass


def load_csv(
    location: str | pathlib.Path,
    encoding: str = "utf-8",
    newline: str = "\n",
    dialect: str = "excel",
    delimiter: str = ",",
    lineterminator: str = "\r\n",
    dtype: np.typing.DTypeLike = DEFAULT_DTYPE,
) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
    """Loads a CSV file into memory, and parses it as if it were a valid confusion matrix.

    Args:
        location (str | pathlib.Path): the location of csv file containing the confusion matrix
        encoding (str): the encoding of the confusion matrix file
        newline (str): the newline character used in the confusion matrix file
        dialect (str): the csv dialect, passed to `csv.reader`
        delimiter (str): the csv delimiter character, passed to `csv.reader`
        lineterminator (str): the csv lineterminator character, passed to `csv.reader`
        dtype (DTypeLike, optional): the desired dtype of the numpy array. Defaults to int64.

    Returns:
        Int[ndarray, 'num_classes num_classes']: the parsed confusion matrix
    """
    rows = []
    with open(
        location,
        "r",
        newline=newline,
        encoding=encoding,  # type: ignore
    ) as f:
        reader = csv.reader(
            f,
            dialect=dialect,
            delimiter=delimiter,
            lineterminator=lineterminator,
        )

        for i, row in enumerate(reader):
            try:
                row_vals = list(map(int, row))
            except ValueError:
                raise ConfMatIOException(
                    f"Row contains values that cannot be converted to int: Row number: {i}. File: {location}"
                )

            rows.append(row_vals)

    arr = np.array(rows, dtype=dtype)

    return arr


def validate_confusion_matrix(
    confusion_matrix: jtyping.Int[np.typing.ArrayLike, " num_classes num_classes"],
    dtype: np.typing.DTypeLike = DEFAULT_DTYPE,
) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
    """Validates a confusion matrix to prevent any future funny business.

    For a confusion matrix to be valid, it:
        1. Must be square matrix (i.e., arrays with 2 dimensions)
        2. Must contain only positive integers
        3. Must contain at least 2 classes
        4. Must have at least one record for each ground-truth class
        5. Should have at least one record for each prediction

    Args:
        confusion_matrix (jtyping.Int[np.ndarray, 'num_classes num_classes']): the confusion matrix
        dtype (DTypeLike, optional): the desired dtype of the numpy array. Defaults to int64.

    Returns:
        Int[ndarray, 'num_classes num_classes']: the validated confusion matrix as a numpy ndarray
    """

    #! Must be an np.ndarray
    if not isinstance(confusion_matrix, np.ndarray):
        try:
            confusion_matrix = np.array(object=confusion_matrix)
        except Exception as e:
            raise ConfMatIOException(
                f"While trying to convert a confusion matrix to a numpy array, encountered the following exception: {e}."
            )

    #! Must be 2-dimensional
    if not (len(confusion_matrix.shape) == 2):
        raise ConfMatIOException(
            f"The loaded confusion matrix is malformed. Shape: {confusion_matrix.shape}. A confusion matrix should have exactly 2 dimensions. Current dimensions: {confusion_matrix.shape}"
        )

    #! Must be square
    if not (confusion_matrix.shape[0] == confusion_matrix.shape[1]):
        raise ConfMatIOException(
            f"The loaded confusion matrix is malformed. Shape: {confusion_matrix.shape}. A confusion matrix should be square. Current dimensions: {confusion_matrix.shape}"
        )

    #! Must have at least 2 classes
    if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
        raise ConfMatIOException(
            f"The loaded confusion matrix is malformed. Shape: {confusion_matrix.shape}. A confusion matrix should have at least 2 classes. Current dimensions: {confusion_matrix.shape}"
        )

    #! Must be an array of only integers
    # Or at the very least, a dtype such that `dtype + float = float`
    if not np.issubdtype(confusion_matrix.dtype, np.integer):
        try:
            confusion_matrix = confusion_matrix.astype(dtype=dtype, casting="safe")
        except Exception as e:
            raise ConfMatIOException(
                f"The loaded confusion matrix is not of type integer. While trying to convert, encounterted the following exception: {e}. Confusion matrix: {confusion_matrix}"
            )

    #! All values must be finite
    if not np.all(np.isfinite(confusion_matrix)):
        raise ConfMatIOException(
            f"The loaded confusion matrix has non-finite elements. Confusion matrix: {confusion_matrix}"
        )

    #! Must have at least one record for each ground truth class
    cond_counts = confusion_matrix.sum(axis=1)
    if not np.all(cond_counts > 0):
        offenders = np.where(cond_counts == 0)[0].tolist()
        raise ConfMatIOException(
            f"Some rows contain no entries, meaning condition does not exist. Rows: {offenders}. Confusion matrix: {confusion_matrix}"
        )

    #! Should have at least one record for each predicted class
    # This is easily violated for poorly optimized models with class imbalance
    pred_counts = confusion_matrix.sum(axis=0)
    if not np.all(pred_counts > 0):
        offenders = np.where(pred_counts == 0)[0].tolist()
        warnings.warn(
            message=f"Some columns contain no entries, meaning model never predicted it. Columns: {offenders}. Confusion matrix: {confusion_matrix}",
            category=ConfMatIOWarning,
        )

    return confusion_matrix


def pred_cond_to_confusion_matrix(
    pred_cond: jtyping.Int[np.ndarray, " num_samples 2"],
    pred_first: bool = True,
) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
    """Converts an array-like of model prediction, ground truth pairs into an unnormalized confusion matrix.
    Confusion matrix *always* has predictions on the columns, condition on the rows.

    Raises:
        ValueError: _description_

    Args:
        pred_cond (jtyping.Int[np.ndarray, ' num_samples 2']): the arraylike collection of predictions
        pred_first (bool, optional): whether the model prediction is on the first column, or the ground truth label. Defaults to True.

    Returns:
        jtyping.Int[np.ndarray, ' num_classes num_classes']
    """

    support = np.unique(pred_cond)
    support_size = support.shape[0]
    if not (np.arange(support_size) == support).all():
        raise ValueError(
            f"Predictions file must contain all labels at least once. Found labels for {list(support)}"
        )

    locs, counts = np.unique(pred_cond, axis=0, return_counts=True)

    confusion_matrix = np.zeros((support_size, support_size), dtype=int)

    if pred_first:
        confusion_matrix[locs[:, 1], locs[:, 0]] = counts
    else:
        confusion_matrix[locs[:, 1], locs[:, 0]] = counts

    return confusion_matrix


def confusion_matrix_to_pred_cond(
    confusion_matrix: jtyping.Int[np.ndarray, " num_classes num_classes"],
    pred_first: bool = True,
) -> jtyping.Int[np.ndarray, " num_samples 2"]:
    """Converts an unnormalized confusion matrix into an array of model prediction, ground truth pairs.
    Assumes predictions on the columns, condition on the rows of the confusion matrix.

    Args:
        confusion_matrix (jtyping.Int[np.ndarray, ' num_classes num_classes']): the unnormalized confusion matrix
        pred_first (bool, optional): whether the model prediction should be on the first column, or the ground truth label. Defaults to True.

    Returns:
        jtyping.Int[np.ndarray, ' num_samples 2']
    """  # noqa: E501
    output = []
    for row_num, row in enumerate(confusion_matrix):
        for col_num, occurences in enumerate(row):
            if pred_first:
                output.extend([[col_num, row_num]] * occurences)
            else:
                output.extend([[row_num, col_num]] * occurences)

    output = np.array(output)

    return output
