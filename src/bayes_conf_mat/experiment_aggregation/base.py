import inspect
import typing
from abc import ABCMeta, abstractmethod

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics import Metric, AveragedMetric
from bayes_conf_mat.experiment_aggregation.heterogeneity import (
    estimate_i2,
    HeterogeneityResult,
)

AGGREGATION_REGISTRY = dict()


# TODO: have this class output a result dataclass
class ExperimentAggregation(metaclass=ABCMeta):
    """The abstract base class for experiment aggregation methods.

    Properties should be implemented as class attributes in derived metrics

    The `compute_metric` method needs to be implemented

    """

    def __init__(self, rng: np.random.BitGenerator):
        self.rng = rng

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
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def aliases(self):
        raise NotImplementedError

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError

    def _mappable_aggregate(self, kwargs: dict):
        return self.aggregate(**kwargs)

    def __call__(
        self,
        metric: Metric | AveragedMetric,
        experiment_samples: typing.Annotated[typing.List, "num_experiments"],
    ) -> typing.Tuple[np.ndarray, HeterogeneityResult]:
        # Stack the experiment values
        stacked_experiment_values: jtyping.Float[
            np.ndarray, " num_experiments num_samples"
        ] = np.stack([result.values for result in experiment_samples], axis=0)

        aggregated_samples = self.aggregate(
            stacked_experiment_values, bounds=metric.bounds
        )
        aggregation_heterogeneity = estimate_i2(stacked_experiment_values)

        return (aggregated_samples, aggregation_heterogeneity)

    def __repr__(self) -> str:
        return f"ExperimentAggregator({self.name})"

    def __str__(self) -> str:
        return f"ExperimentAggregator({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)
