import inspect
import typing
from abc import ABCMeta, abstractmethod
from multiprocessing import Pool

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics import Metric, AggregatedMetric

AGGREGATION_REGISTRY = dict()


class ExperimentAggregation(metaclass=ABCMeta):
    """The abstract base class for metrics.

    Properties should be implemented as class attributes in derived metrics

    The `compute_metric` method needs to be implemented

    """

    def __init__(self, rng: np.random.BitGenerator, num_proc: int = 0):
        self.rng = rng

        self.num_proc = num_proc

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

    def __repr__(self) -> str:
        return f"ExperimentAggregator({self.name})"

    def __str__(self) -> str:
        return f"ExperimentAggregator({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError

    def _mappable_aggregate(self, kwargs: dict):
        return self.aggregate(**kwargs)

    def __call__(
        self,
        experiment_samples: typing.List[
            jtyping.Float[np.ndarray, " num_samples *num_classes"]
        ],
        metric: Metric | AggregatedMetric,
    ):
        # Stack and split the experiment values such that
        # [num_experiments [num_samples, num_classes]] -> [num_classes [num_experiments, num_samples]]
        stacked_values = np.stack(
            [metric_values for _, metric_values in experiment_samples.items()], axis=0
        )

        if metric.is_multiclass:
            aggregated_samples = self.aggregate(
                stacked_values, rng=self.rng, extrema=metric.range
            )

        else:
            num_classes = stacked_values.shape[2]

            split_values = np.split(
                stacked_values, indices_or_sections=num_classes, axis=2
            )

            # Prep for multiprocessing
            # Get the required number of RNGs
            child_rngs = self.rng.spawn(num_classes)

            fit_func_args = [
                {
                    "distribution_samples": np.squeeze(samples, axis=2),
                    "extrema": metric.range,
                    "rng": rng,
                }
                for samples, rng in zip(split_values, child_rngs)
            ]

            if self.num_proc > 0:
                with Pool(self.num_proc) as pool:
                    aggregated_samples = pool.map(
                        self._mappable_aggregate,
                        fit_func_args,
                    )
            else:
                aggregated_samples = [
                    self._mappable_aggregate(kwargs) for kwargs in fit_func_args
                ]

            aggregated_samples = np.stack(aggregated_samples, axis=1)

        return aggregated_samples
