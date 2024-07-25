import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.io.base import IOBase


class InMemory(IOBase):
    """Stores a confusion matrix in memory.

    Essentially just a wrapper around an existing confusion matrix.

    Args:
        data (Int[ndarray, "num_classes num_classes"])): the raw confusion matrix
        location (str | None): Ignored
    """

    format = "in_memory"

    def __init__(
        self, data: jtyping.Int[np.ndarray, "num_classes num_classes"], location=None
    ):
        super().__init__(location=None)

        self.data = data

    def _load(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
        return self.data
