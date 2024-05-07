import inspect
import typing  # noqa: F401
from abc import ABCMeta, abstractmethod  # noqa: F401
from pathlib import Path

import numpy as np  # noqa: F401
import jaxtyping as jtyping  # noqa: F401


IO_REGISTRY = dict()


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

    def __call__(self) -> jtyping.Int[np.ndarray, " num_classes num_classes"]:
        return self.load()
