import csv

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.io.base import IOBase, ConfMatIOException
from bayes_conf_mat.io.utils import pred_cond_to_confusion_matrix

_CSV_TYPES = {"confusion_matrix", "conf_mat", "pred_cond", "cond_pred"}


class CSV(IOBase):
    """Loads in a csv file as a confusion matrix, or a separated list of predictions and conditions.

    Args:
        location (str): the location of the CSV file
        type (str): the type of file. Must be one of `_CSV_TYPES`
        encoding (str, optional): the file's encoding. Defaults to "utf-8".
        newline (str, optional): the newline character. Defaults to "\\n".
        dialect (str, optional): the csv dialect. Defaults to "excel".
        delimiter (str, optional): the delimiter character. Defaults to ",".
        lineterminator (str, optional): the line terminator character. Defaults to "\\r\\n".
    """

    format = "csv"

    def __init__(
        self,
        location: str = None,
        type: str = None,
        encoding: str = "utf-8",
        newline: str = "\n",
        dialect: str = "excel",
        delimiter: str = ",",
        lineterminator: str = "\r\n",
    ):
        super().__init__(location)

        if type not in _CSV_TYPES:
            raise ValueError(f"For CSV, `type` must be one of {_CSV_TYPES}")

        self.type = type
        self.encoding = encoding
        self.newline = newline
        self.dialect = dialect
        self.delimiter = delimiter
        self.lineterminator = lineterminator

    def _load(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
        rows = []
        with open(
            self.location, "r", newline=self.newline, encoding=self.encoding
        ) as f:
            reader = csv.reader(
                f,
                dialect=self.dialect,
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
            )

            for i, row in enumerate(reader):
                try:
                    row_vals = list(map(int, row))
                except ValueError:
                    raise ConfMatIOException(
                        f"Row contains values that cannot be converted to int: Row number: {i}. File: {self.location}"
                    )

                rows.append(row_vals)

        arr = np.array(rows)

        if self.type == "pred_cond":
            arr = pred_cond_to_confusion_matrix(arr, pred_first=True)

        elif self.type == "cond_pred":
            arr = pred_cond_to_confusion_matrix(arr, pred_first=False)

        return arr
