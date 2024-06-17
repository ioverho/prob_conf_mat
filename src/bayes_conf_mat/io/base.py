import inspect
from abc import ABCMeta, abstractmethod
from pathlib import Path
import warnings

import numpy as np
import jaxtyping as jtyping


IO_REGISTRY = dict()


class ConfMatIOWarning(Warning):
    pass


class ConfMatIOException(Exception):
    pass


class IOBase(metaclass=ABCMeta):
    def __init__(self, location: str):
        self.location = Path(location).resolve()

        if not (self.location.exists() and self.location.is_file()):
            raise ValueError(f"No file found at: {str(self.location)}")

        self.location = str(self.location)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        if cls.format in IO_REGISTRY:
            raise ValueError(
                f"Format '{cls.format}' already has an IO method assigned to it: {IO_REGISTRY[cls.format]}."  # noqa: E501
            )

        # Register =============================================================
        IO_REGISTRY[cls.format] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    @property
    @abstractmethod
    def format(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _load(self, *args, **kwargs):
        raise NotImplementedError

    def _validate_conf_mat(self, conf_mat: np.ndarray) -> None:
        if not (len(conf_mat.shape) == 2 and conf_mat.shape[0] == conf_mat.shape[1]):
            raise ConfMatIOException(
                f"The constructed confusion matrix is malformed. Shape: {conf_mat.shape}. File: {self.location}"
            )

        if not np.issubdtype(conf_mat.dtype.type, np.integer):
            raise ConfMatIOException(
                f"The constructed confusion matrix is not of integer type. File: {self.location}"
            )

        cond_counts = conf_mat.sum(axis=1)
        if not np.all(cond_counts > 0):
            offenders = np.where(cond_counts == 0)[0].tolist()
            raise ConfMatIOException(
                f"Some rows contain no entries, meaning condition does not exist. Rows: {offenders}. File: {self.location}"
            )

        pred_counts = conf_mat.sum(axis=0)
        if not np.all(pred_counts > 0):
            offenders = np.where(pred_counts == 0)[0].tolist()
            warnings.warn(
                f"Some columns contain no entries, meaning model never predicted it. Columns: {offenders}. File: {self.location}",
                ConfMatIOWarning,
            )

    def __call__(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
        conf_mat = self._load()

        self._validate_conf_mat(conf_mat=conf_mat)

        return conf_mat
