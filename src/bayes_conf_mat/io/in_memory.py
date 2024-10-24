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
        self,
        data: jtyping.Int[np.typing.ArrayLike, "num_classes num_classes"],
        location=None,
    ):
        super().__init__(location=None)

        if isinstance(data, np.ndarray):
            self.data = data
        else:
            try:
                self.data = np.array(data)
            except Exception as e:
                raise TypeError(
                    f"In-memory confusion matrix is of invalid type. Must a `np.ArrayLike`. Currently: {type(data)}. While trying to convert, encountered the following exception: {e}"
                )

    def _load(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
        return self.data
