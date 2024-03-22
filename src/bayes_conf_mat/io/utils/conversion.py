import numpy as np
import jaxtyping as jtyping


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
    """  # noqa: E501

    support = np.unique(pred_cond)
    support_size = support.shape[0]
    if not (np.arange(support_size) == support).all():
        raise ValueError(
            f"Predictions file must contain all labels at least once. Found labels for {list(support)}"  # noqa: E501
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
