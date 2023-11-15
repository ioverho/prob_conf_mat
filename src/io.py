import csv

import numpy as np
import jaxtyping as jtyping


def load_preds_file(
    fp: str,
    structure: str = "pred_target",
    encoding: str = "utf-8",
    newline: str = "\n",
    dialect: str = "excel",
    delimiter: str = ",",
    lineterminator: str = "\r\n",
) -> jtyping.Int[np.ndarray, "num_samples 2"]:
    pred_target = []
    with open(fp, "r", newline=newline, encoding=encoding) as f:
        reader = csv.reader(
            f,
            dialect=dialect,
            delimiter=delimiter,
            lineterminator=lineterminator,
        )

        for row in reader:
            row_vals = []
            for val in row:
                row_vals.append(int(val))

            pred_target.append(row_vals)

    pred_target = np.array(pred_target)
    if structure == "pred_target":
        pass
    elif structure == "target_pred":
        pred_target[:, [1, 0]] = pred_target[:, [0, 1]]

    return pred_target


def pred_target_to_confusion_matrix(
    pred_target: jtyping.Int[np.ndarray, "num_samples 2"]
):
    support = np.unique(pred_target)
    support_size = support.shape[0]
    if not (np.arange(support_size) == support).all():
        raise ValueError(
            f"Predictions file must contain all labels at least once. Found labels for {list(support)}"
        )

    locs, counts = np.unique(pred_target, axis=0, return_counts=True)

    confusion_matrix = np.zeros((support_size, support_size), dtype=int)
    confusion_matrix[locs[:, 1], locs[:, 0]] = counts

    return confusion_matrix
