from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import jaxtyping as jtyping

from prob_conf_mat.metrics.abc import Averaging
from prob_conf_mat.stats import (
    numpy_batched_arithmetic_mean,
    numpy_batched_convex_combination,
    numpy_batched_geometric_mean,
)


class MacroAverage(Averaging):
    """Computes the arithmetic mean over all classes, also known as macro-averaging."""

    full_name = "Macro Averaging"
    dependencies = ()
    sklearn_equivalent = "macro"
    aliases = ["macro", "macro_average", "mean"]

    def compute_average(
        self,
        metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        scalar_array = numpy_batched_arithmetic_mean(
            metric_values,
            axis=1,
            keepdims=False,
        )

        return scalar_array


class WeightedAverage(Averaging):
    """Computes the class prevalence weighted mean over all classes, also known as weighted averaging."""

    full_name = "Class Prevalence Weighted Averaging"
    dependencies = ("p_condition",)
    sklearn_equivalent = "weighted"
    aliases = ["weighted", "weighted_average", "micro", "micro_average"]

    def compute_average(
        self,
        metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        scalar_array = numpy_batched_convex_combination(
            metric_values,
            convex_weights=p_condition,
            axis=1,
            keepdims=False,
        )

        return scalar_array


class SelectPositiveClass(Averaging):
    """Selects only the positive class, also known as binary averaging."""

    full_name = "Select Positive Class"
    dependencies = ()
    sklearn_equivalent = "binary"
    aliases = ["select_positive", "binary", "select"]

    def __init__(self, positive_class: int = -1) -> None:
        super().__init__()

        self.positive_class = positive_class

    def compute_average(
        self,
        metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        if metric_values.shape[1] < self.positive_class:
            raise IndexError(
                f"Passed metric values contain fewer classes than 'positive_class' argument: `{self.positive_class}`.",  # noqa: E501
            )

        scalar_array = metric_values[:, self.positive_class]

        return scalar_array


class HarmonicMean(Averaging):
    """Computes the harmonic mean over all classes."""

    full_name = "Harmonic Mean Averaging"
    dependencies = ()
    sklearn_equivalent = None
    aliases = ["harmonic", "harm"]

    def compute_average(
        self,
        metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        scalar_array = numpy_batched_geometric_mean(
            metric_values,
            axis=1,
            keepdims=False,
        )

        return scalar_array


class GeometricMean(Averaging):
    """Computes the geometric mean over all classes."""

    full_name = "Geometric Mean Averaging"
    dependencies = ()
    sklearn_equivalent = None
    aliases = ["geometric", "geom"]

    def compute_average(
        self,
        metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        scalar_array = numpy_batched_geometric_mean(
            metric_values,
            axis=1,
            keepdims=False,
        )

        return scalar_array
