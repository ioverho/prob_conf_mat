from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.experiment_group import ExperimentGroup
    from prob_conf_mat.experiment import ExperimentResult
    from prob_conf_mat.experiment_aggregation.heterogeneity import HeterogeneityResult
    from prob_conf_mat.utils import RNG
    from prob_conf_mat.utils.typing import MetricLike

import inspect
import typing
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from prob_conf_mat.experiment_aggregation.heterogeneity import estimate_i2

AGGREGATION_REGISTRY = dict()


class ExperimentAggregator(metaclass=ABCMeta):
    """The abstract base class for experiment aggregation methods.

    Properties should be implemented as class attributes in derived metrics

    The `compute_metric` method needs to be implemented

    """

    def __init__(self, rng: RNG) -> None:
        self.rng = rng
        self._init_params = dict()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Validate =============================================================
        # Make sure that all aliases are unique
        for alias in cls.aliases:  # type: ignore
            if alias in AGGREGATION_REGISTRY:
                raise ValueError(
                    f"Alias '{alias}' not unique. Currently used by {AGGREGATION_REGISTRY[alias]}.",  # noqa: E501
                )

        # Register =============================================================
        for alias in cls.aliases:  # type: ignore
            AGGREGATION_REGISTRY[alias] = cls

        cls._kwargs = {
            param.name: param.annotation
            for param in inspect.signature(cls).parameters.values()
        }

    @property
    @abstractmethod
    def full_name(self) -> str:  # type: ignore
        """A human-readable name for this experiment-aggregation method."""
        raise NotImplementedError

    full_name: str

    @property
    @abstractmethod
    def aliases(self) -> list[str]:  # type: ignore
        """A list of all valid aliases for this metric. Can be used in configuration files."""
        raise NotImplementedError

    aliases: list[str]

    @property
    def name(self) -> str:
        """The name of the experiment aggregation method."""
        init_name = self._init_params.get("aggregation", None)
        if init_name is not None:
            return init_name
        return self.aliases[0]

    @abstractmethod
    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_experiments"]:
        """Aggregates samples from many experiments.

        Args:
            experiment_samples (Float[nd.array, "num_samples num_experiments"]): the samples from
                the individual experiments
            bounds (tuple[float, float]): the maximum and minimum possible value that the samples
                might take

        Returns:
            Float[nd.array, "num_samples num_experiments"]: samples from the aggregate distribution
        """
        raise NotImplementedError

    def _mappable_aggregate(self, kwargs: dict):
        return self.aggregate(**kwargs)

    def __call__(
        self,
        experiment_group: ExperimentGroup,
        metric: MetricLike,
        experiment_results: typing.Annotated[list[ExperimentResult], "num_experiments"],
    ) -> ExperimentAggregationResult:
        """Aggregates a series of experiment results from a specific ExperimentGroup and Metric."""
        # Stack the experiment values
        stacked_experiment_results: jtyping.Float[
            np.ndarray,
            " num_samples #num_classes num_experiments",
        ] = np.stack(
            [experiment_result.values for experiment_result in experiment_results],
            axis=-1,
        )

        # For each class, get the aggregated values
        all_class_aggregated_experiment_result = []
        all_class_heterogeneity = []
        for class_label in range(stacked_experiment_results.shape[1]):
            per_class_stacked_experiment_results: jtyping.Float[
                np.ndarray,
                " num_samples num_experiments",
            ] = stacked_experiment_results[:, class_label, :]

            per_class_aggregated_experiment_result: jtyping.Float[
                np.ndarray,
                " num_samples",
            ] = self.aggregate(
                experiment_samples=per_class_stacked_experiment_results,
                bounds=metric.bounds,  # type: ignore
            )

            all_class_aggregated_experiment_result.append(
                per_class_aggregated_experiment_result,
            )

            all_class_heterogeneity.append(
                estimate_i2(individual_samples=per_class_stacked_experiment_results),
            )

        # Finally, stack everything back together
        aggregated_experiment_result: jtyping.Float[
            np.ndarray,
            " num_samples #num_classes",
        ] = np.vstack(per_class_aggregated_experiment_result)  # type: ignore

        result = ExperimentAggregationResult(
            experiment_group=experiment_group,
            aggregator=self,  # type: ignore
            metric=metric,
            heterogeneity_results=all_class_heterogeneity,
            values=aggregated_experiment_result,
        )

        return result

    def __repr__(self) -> str:  # noqa: D105
        return f"ExperimentAggregator({self.name})"

    def __str__(self) -> str:  # noqa: D105
        return f"ExperimentAggregator({self.name})"

    def __eq__(self, other: object) -> bool:  # noqa: D105
        return isinstance(other, self.__class__) and self.name == other.name

    def __hash__(self):  # noqa: D105
        return hash(self.name)


@dataclass(frozen=True)
class ExperimentAggregationResult:
    """Class containing results from performing experiment aggregation."""

    experiment_group: ExperimentGroup
    aggregator: type[ExperimentAggregator]
    metric: MetricLike
    heterogeneity_results: list[HeterogeneityResult]
    values: jtyping.Float[np.ndarray, " num_samples #num_classes"]

    @property
    def name(self) -> str:
        """The name of the experiment group."""
        return self.experiment_group.name

    def __repr__(self) -> str:
        return (
            f"ExperimentAggregationResult(experiment_group={self.experiment_group}, "
            f"metric={self.metric}, aggregator={self.aggregator})"
        )

    def __str__(self) -> str:
        return (
            f"ExperimentAggregationResult(experiment_group={self.experiment_group}, "
            f"metric={self.metric}, aggregator={self.aggregator})"
        )
