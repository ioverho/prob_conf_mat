import csv

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.io.base import IOBase
from bayes_conf_mat.io.utils import pred_cond_to_confusion_matrix

_CSV_TYPES = {"confusion_matrix", "pred_cond", "cond_pred"}


class CSV(IOBase):
    format = "csv"

    def __init__(
        self,
        location: str,
        type: str,
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

    def load(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
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

            for row in reader:
                row_vals = list(map(int, row))

                rows.append(row_vals)

        arr = np.array(rows)

        if self.type == "pred_cond":
            arr = pred_cond_to_confusion_matrix(arr, pred_first=True)

        elif self.type == "cond_pred":
            arr = pred_cond_to_confusion_matrix(arr, pred_first=False)

        return arr
