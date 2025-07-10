from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import inspect
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from inspect import signature
from itertools import product

import numpy as np


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
    """A metric that all other metrics depend upon, but with no dependencies."""

    name: str

    @property
    def full_name(self) -> str:  # noqa: D102
        return self.name

    @property
    def is_multiclass(self) -> bool:  # noqa: D102
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def bounds(self) -> tuple[float, float]:  # noqa: D102
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def dependencies(self) -> typing.Sequence[str]:  # noqa: D102
        return ()

    @property
    def sklearn_equivalent(self) -> str | None:  # noqa: D102
        raise TypeError("Root metrics are not directly interpretable.")

    @property
    def aliases(self) -> list[str]:  # noqa: D102
        return [self.name]


class Metric(metaclass=ABCMeta):
    """The abstract base class for metrics.

    Properties should be implemented as class attributes in derived metrics

    The `compute_metric` method needs to be implemented

    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        for alias in cls.aliases:  # type: ignore
            if alias in METRIC_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. "
                    f"Currently used by metric {METRIC_REGISTRY[alias]}.",
                )

        # Make sure the parameters of the `compute_metric` function are actually
        # the ones listed as dependencies
        parameters = set(signature(cls.compute_metric).parameters.keys()) - {"self"}
        dependencies = set(cls.dependencies)  # type: ignore
        if parameters != dependencies:
            raise TypeError(
                f"The input for the {cls.__name__}'s `compute_metric` method does not match the specified dependencies: {parameters} != {dependencies}",  # noqa: E501
            )

        # Make sure the dependencies actually exist
        if not hasattr(cls, "_validated_dependencies"):
            for dependency in cls.dependencies:
                if dependency in _ROOT_METRICS:
                    continue

        # Register =============================================================
        for alias in cls.aliases:  # type: ignore
            METRIC_REGISTRY[alias] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    def __init__(self):
        self._instantiation_name: str = ""

    @property
    @abstractmethod
    def full_name(self) -> str:  # type: ignore
        """A human-readable name for this metric."""
        raise NotImplementedError

    full_name: str

    @property
    @abstractmethod
    def is_multiclass(self) -> bool:  # type: ignore
        """Whether or not this metric computes a value for each class individually, or for all classes at once."""  # noqa: E501
        raise NotImplementedError

    is_multiclass: bool

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float]:  # type: ignore
        """A tuple of the minimum and maximum possible value for this metric to take.

        Can be infinite.
        """
        raise NotImplementedError

    bounds: tuple[float, float]

    @property
    @abstractmethod
    def dependencies(self) -> typing.Sequence[str]:  # type: ignore
        """All metrics upon which this metric depends.

        Used to generate a computation schedule, such that no metric is calculated before its
        dependencies. The dependencies **must** match the `compute_metric` signature.
        This is checked during class definition.
        """
        raise NotImplementedError

    dependencies: typing.Sequence[str]

    @property
    @abstractmethod
    def sklearn_equivalent(self) -> str | None:  # type: ignore
        """The `sklearn` equivalent function, if applicable"""
        raise NotImplementedError

    sklearn_equivalent: str | None

    @property
    @abstractmethod
    def aliases(self) -> typing.Sequence[str]:  # type: ignore
        """A list of all valid aliases for this metric. Can be used when creating metric syntax strings."""
        raise NotImplementedError

    aliases: typing.Sequence[str]

    @abstractmethod
    def compute_metric(self, *args, **kwargs):
        """Computes the metric values from its dependencies."""
        raise NotImplementedError()

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> jtyping.Float[np.ndarray, " num_samples ..."]:
        return self.compute_metric(*args, **kwargs)

    @property
    def _metric_name(self) -> str | None:
        return self.aliases[0]

    @property
    def name(self) -> str:
        if self._instantiation_name != "":
            return self._instantiation_name
        metric_kwargs = "".join(
            [f"+{k}={getattr(self, k)}" for k, _ in self._kwargs.items()],
        )

        return f"{self._metric_name}{metric_kwargs}"

    def __repr__(self) -> str:
        return f"Metric({self.name})"

    def __str__(self) -> str:
        return f"Metric({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self) -> int:
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
        for alias in cls.aliases:  # type: ignore
            if alias in AVERAGING_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by averaging method {AVERAGING_REGISTRY[alias]}.",
                )

        for alias in cls.aliases:  # type: ignore
            if alias in METRIC_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by metric {METRIC_REGISTRY[alias]}.",
                )

        # Register =============================================================
        for alias in cls.aliases:  # type: ignore
            AVERAGING_REGISTRY[alias] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    def __init__(self):
        self._instantiation_name: str = ""

    @property
    @abstractmethod
    def full_name(self) -> str:  # type: ignore
        raise NotImplementedError

    full_name: str

    @property
    @abstractmethod
    def dependencies(self) -> typing.Sequence[str]:  # type: ignore
        raise NotImplementedError

    dependencies: typing.Sequence[str]

    @property
    @abstractmethod
    def sklearn_equivalent(self) -> str | None:  # type: ignore
        raise NotImplementedError

    sklearn_equivalent: str | None

    @property
    @abstractmethod
    def aliases(self) -> typing.Sequence[str]:  # type: ignore
        raise NotImplementedError

    aliases: typing.Sequence[str]

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
    def _averaging_name(self) -> str:
        return self.aliases[0]

    @property
    def name(self) -> str:
        if self._instantiation_name != "":
            return self._instantiation_name
        kwargs = "".join(
            [f"+{k}={getattr(self, k)}" for k, _ in self._kwargs.items()],
        )

        return self._averaging_name + kwargs

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class AveragedMetric(metaclass=ABCMeta):
    """The abstract base class for the composition of any instance of `Metric` with any instance of `Averaging`.

    Args:
        metric (Metric): a binary metric
        averaging (Averaging): an averaging method
    """

    def __init__(self, metric: Metric, averaging: Averaging):
        super().__init__()

        self.base_metric = metric

        if self.base_metric.is_multiclass:
            raise ValueError(
                f"Cannot aggregate a metric ({self.base_metric.name}) that is already multiclass.",  # noqa: E501
            )

        self.averaging = averaging

        self._instantiation_name: str = ""

    @property
    def aliases(self) -> typing.Sequence[str]:
        """A list of all valid aliases for this metric.

        Constructed from the product of the all aliases of the Metric and Averaging methods.

        Can be used when creating metric syntax strings.
        """
        return [
            f"{lhs}@{rhs}"
            for lhs, rhs in product(self.base_metric.aliases, self.averaging.aliases)  # type: ignore
        ]

    @property
    def full_name(self) -> str:
        return f"{self.base_metric.full_name} with {self.averaging.full_name}"

    @property
    def is_multiclass(self) -> typing.Literal[True]:
        """Whether or not this metric computes a value for each class individually, or for all classes at once.

        An AveragedMetric is *always* multiclass.
        """
        return True

    @property
    def bounds(self) -> tuple[float, float]:
        """A tuple of the minimum and maximum possible value for this metric to take. Can be infinite."""
        return self.base_metric.bounds  # type: ignore

    @property
    def dependencies(self) -> typing.Sequence[str]:
        """All metrics upon which this AveragedMetric depends.

        Constructed from the union of all Metric and AveragingMethod dependencies.

        Used to generate a computation schedule, such that no metric is calculated before its dependencies.

        The dependencies **must** match the `compute_metric` signature.

        This is checked during class definition.
        """
        dependencies = (
            *self.base_metric.dependencies,  # type: ignore
            *self.averaging.dependencies,  # type: ignore
        )

        return dependencies

    @property
    def sklearn_equivalent(self) -> str | None:
        """The `sklearn` equivalent function, if applicable"""
        sklearn_equivalent = self.base_metric.sklearn_equivalent
        if self.averaging.sklearn_equivalent is not None:
            sklearn_equivalent = (
                sklearn_equivalent.sklearn_equivalent  # type: ignore
                + f"with average={self.averaging.sklearn_equivalent}"
            )

        return sklearn_equivalent  # type: ignore

    def compute_metric(
        self,
        *args,
        **kwargs,
    ) -> jtyping.Float[np.ndarray, " num_samples ..."]:
        return self.base_metric(*args, **kwargs)

    def compute_average(
        self,
        *args,
        **kwargs,
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        return self.averaging.__call__(*args, **kwargs)

    def __call__(self, **kwargs) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        metric_vals = self.compute_metric(
            **{
                key: value
                for key, value in kwargs.items()
                if key == "samples" or key in self.base_metric.dependencies
            },
        )

        aggregated_metric_vals = self.averaging(
            metric_vals=metric_vals,
            **{
                key: value
                for key, value in kwargs.items()
                if key in self.averaging.dependencies
            },
        )

        return aggregated_metric_vals[:, np.newaxis]

    @property
    def _kwargs(self) -> dict[str, dict[str, typing.Any]]:
        kwargs = {
            "metric": self.base_metric._kwargs,
            "averaging": self.averaging._kwargs,
        }

        return kwargs

    @property
    def name(self) -> str:
        if self._instantiation_name != "":
            return self._instantiation_name
        return f"{self.base_metric.name}@{self.averaging.name}"

    def __repr__(self) -> str:
        return f"AveragedMetric({self.name})"

    def __str__(self) -> str:
        return f"AveragedMetric({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
