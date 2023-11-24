from abc import ABCMeta, abstractmethod
import inspect
import typing

import numpy as np
import jaxtyping as jtyping

METRIC_REGISTRY = dict()
AGGREGATION_REGISTRY = dict()


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

        # Register =============================================================
        for alias in cls.aliases:
            METRIC_REGISTRY[alias] = cls

        cls._kwargs = inspect.getfullargspec(cls).annotations

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
    def range(self):
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
    def name(self):
        return self.aliases[0]


class Aggregation(metaclass=ABCMeta):
    """The abstract base class for metric aggregations.

    Properties should be implemented as class attributes in derived metrics

    The `compute_aggregation` method needs to be implemented

    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        for alias in cls.aliases:
            if alias in AGGREGATION_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by {AGGREGATION_REGISTRY[alias]}."  # noqa: E501
                )

        # Register =============================================================
        for alias in cls.aliases:
            AGGREGATION_REGISTRY[alias] = cls

        cls._kwargs = inspect.getfullargspec(cls).annotations

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
    def compute_aggregation(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(
        self,
        metric_vals: jtyping.Float[np.ndarray, " num_samples num_classes"],
        *args,
        **kwargs,
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        return self.compute_aggregation(metric_vals, *args, **kwargs)

    @property
    def name(self):
        return self.aliases[0]


class AggregatedMetric(metaclass=ABCMeta):
    def __init__(
        self, metric: typing.Type[Metric], aggregation: typing.Type[Aggregation]
    ):
        super().__init__()

        self.base_metric = metric

        if self.base_metric.is_multiclass:
            raise ValueError(
                f"Cannot aggregate a metric ({self.base_metric.name}) that is already multiclass."  # noqa: E501
            )

        self.aggregation = aggregation

    @property
    def full_name(self):
        return f"{self.base_metric.full_name} with {self.aggregation.full_name}"

    @property
    def is_multiclass(self):
        return True

    @property
    def range(self):
        return self.base_metric.range

    @property
    def dependencies(self):
        dependencies = (
            *self.base_metric.dependencies,
            *self.aggregation.dependencies,
        )

        return dependencies

    @property
    def sklearn_equivalent(self):
        sklearn_equivalent = self.base_metric.sklearn_equivalent
        if self.aggregation.sklearn_equivalent is not None:
            (
                sklearn_equivalent.sklearn_equivalent
                + f"with average={self.aggregation.sklearn_equivalent}"
            )

        return sklearn_equivalent

    def compute_metric(self, *args, **kwargs):
        return self.base_metric(*args, **kwargs)

    def compute_aggregation(self, *args, **kwargs):
        return self.aggregation(*args, **kwargs)

    def __call__(self, **kwargs) -> jtyping.Float[np.ndarray, " num_samples"]:
        metric_vals = self.compute_metric(
            **{
                key: value
                for key, value in kwargs.items()
                if key == "samples" or key in self.base_metric.dependencies
            }
        )

        aggregated_metric_vals = self.aggregation(
            metric_vals,
            **{
                key: value
                for key, value in kwargs.items()
                if key in self.aggregation.dependencies
            },
        )

        return aggregated_metric_vals

    @property
    def name(self):
        return f"{self.base_metric.name}@{self.aggregation.name}"

    @property
    def _kwargs(self):
        kwargs = {
            "metric": self.base_metric._kwargs,
            "aggregation": self.aggregation._kwargs,
        }

        return kwargs
