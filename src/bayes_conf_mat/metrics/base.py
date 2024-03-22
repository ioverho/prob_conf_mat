import inspect
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from inspect import signature
from itertools import product

import numpy as np
import jaxtyping as jtyping

# Root metrics are always computed, because they're (almost) always needed as
# intermediate variables
_ROOT_METRICS = {
    "norm_confusion_matrix",
    "p_condition",
    "p_pred_given_condition",
    "p_pred",
    "p_condition_given_pred",
}
METRIC_REGISTRY = dict()
AVERAGING_REGISTRY = dict()


@dataclass(frozen=True)
class RootMetric:
    name: str

    @property
    def full_name(self):
        return self.name

    @property
    def is_multiclass(self):
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def bounds(self):
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def dependencies(self):
        return ()

    @property
    def sklearn_equivalent(self):
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def aliases(self):
        return [self.name]


class Metric(metaclass=ABCMeta):
    """The abstract base class for metrics.

    Properties should be implemented as class attributes in derived metrics

    The `compute_metric` method needs to be implemented

    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        for alias in cls.aliases:
            if alias in METRIC_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by {METRIC_REGISTRY[alias]}."  # noqa: E501
                )

        # Make sure the parameters of the `compute_metric` function are actually
        # the ones listed as dependencies
        parameters = set(signature(cls.compute_metric).parameters.keys()) - {"self"}
        dependencies = set(cls.dependencies)
        if parameters != dependencies:
            raise TypeError(
                f"The input for the {cls.__name__}'s `compute_metric` method does not match the specified dependencies: {parameters} != {dependencies}"  # noqa: E501
            )

        # Register =============================================================
        for alias in cls.aliases:
            METRIC_REGISTRY[alias] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    @property
    @abstractmethod
    def full_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multiclass(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dependencies(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sklearn_equivalent(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def aliases(self):
        return [None]

    @abstractmethod
    def compute_metric(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(
        self, *args, **kwargs
    ) -> jtyping.Float[np.ndarray, " num_samples ..."]:
        return self.compute_metric(*args, **kwargs)

    @property
    def _metric_name(self):
        return self.aliases[0]

    @property
    def name(self):
        if hasattr(self, "_instantiation_name"):
            return self._instantiation_name
        else:
            metric_kwargs = "".join(
                [f"+{k}={getattr(self, k)}" for k, _ in self._kwargs.items()]
            )

            return self._metric_name + metric_kwargs

    def __repr__(self) -> str:
        return f"Metric({self.name})"

    def __str__(self) -> str:
        return f"Metric({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Averaging(metaclass=ABCMeta):
    """The abstract base class for metric averaging.

    Properties should be implemented as class attributes in derived metrics.

    The `compute_average` method needs to be implemented

    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        for alias in cls.aliases:
            if alias in AVERAGING_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by {AVERAGING_REGISTRY[alias]}."  # noqa: E501
                )

        for alias in cls.aliases:
            if alias in METRIC_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by {METRIC_REGISTRY[alias]}."  # noqa: E501
                )

        # Register =============================================================
        for alias in cls.aliases:
            AVERAGING_REGISTRY[alias] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    @property
    @abstractmethod
    def full_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def dependencies(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def sklearn_equivalent(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def aliases(self):
        raise NotImplementedError

    @abstractmethod
    def compute_average(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(
        self,
        metric_vals: jtyping.Float[np.ndarray, " num_samples num_classes"],
        *args,
        **kwargs,
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        return self.compute_average(metric_vals, *args, **kwargs)

    @property
    def _averaging_name(self):
        return self.aliases[0]

    @property
    def name(self):
        if hasattr(self, "_instantiation_name"):
            return self._instantiation_name
        else:
            kwargs = "".join(
                [f"+{k}={getattr(self, k)}" for k, _ in self._kwargs.items()]
            )

            return self._averaging_name + kwargs

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class AveragedMetric(metaclass=ABCMeta):
    """The base class for the composition of any instance of `Metric` with any instance of `Averaging`

    Args:
        metric (typing.Type[Metric])
        averaging (typing.Type[Averaging])
    """

    def __init__(self, metric: typing.Type[Metric], averaging: typing.Type[Averaging]):
        super().__init__()

        self.base_metric = metric

        if self.base_metric.is_multiclass:
            raise ValueError(
                f"Cannot aggregate a metric ({self.base_metric.name}) that is already multiclass."  # noqa: E501
            )

        self.averaging = averaging

    @property
    def aliases(self):
        return [
            f"{lhs}@{rhs}"
            for lhs, rhs in product(self.base_metric.aliases, self.averaging.aliases)
        ]

    @property
    def full_name(self):
        return f"{self.base_metric.full_name} with {self.averaging.full_name}"

    @property
    def is_multiclass(self):
        return True

    @property
    def bounds(self):
        return self.base_metric.bounds

    @property
    def dependencies(self):
        dependencies = (
            *self.base_metric.dependencies,
            *self.averaging.dependencies,
        )

        return dependencies

    @property
    def sklearn_equivalent(self):
        sklearn_equivalent = self.base_metric.sklearn_equivalent
        if self.averaging.sklearn_equivalent is not None:
            (
                sklearn_equivalent.sklearn_equivalent
                + f"with average={self.averaging.sklearn_equivalent}"
            )

        return sklearn_equivalent

    def compute_metric(self, *args, **kwargs):
        return self.base_metric(*args, **kwargs)

    def compute_average(self, *args, **kwargs):
        return self.averaging(*args, **kwargs)

    def __call__(self, **kwargs) -> jtyping.Float[np.ndarray, " num_samples"]:
        metric_vals = self.compute_metric(
            **{
                key: value
                for key, value in kwargs.items()
                if key == "samples" or key in self.base_metric.dependencies
            }
        )

        aggregated_metric_vals = self.averaging(
            metric_vals,
            **{
                key: value
                for key, value in kwargs.items()
                if key in self.averaging.dependencies
            },
        )

        return aggregated_metric_vals

    @property
    def _kwargs(self):
        kwargs = {
            "metric": self.base_metric._kwargs,
            "averaging": self.averaging._kwargs,
        }

        return kwargs

    @property
    def name(self):
        if hasattr(self, "_instantiation_name"):
            return self._instantiation_name
        else:
            return f"{self.base_metric.name}@{self.averaging.name}"

    def __repr__(self) -> str:
        return f"AveragedMetric({self.name})"

    def __str__(self) -> str:
        return f"AveragedMetric({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
