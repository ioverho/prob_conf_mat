import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics.registration import register_metric_aggregation
from bayes_conf_mat.math.batched_aggregation import (
    numpy_batched_arithmetic_mean,
    numpy_batched_convex_combination,
    numpy_batched_geometric_mean,
    numpy_batched_harmonic_mean,
)


@register_metric_aggregation(
    identifier="macro",
    full_name="Macro Averaging",
    required_simple_metrics=(),
    sklearn_equivalent="macro",
)
def macro_average(
    metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """_summary_

    Args:
        array (jtyping.Float[np.ndarray, ' num_samples num_classes']): _description_
        axis (int, optional): _description_. Defaults to -1.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """  # noqa: E501
    scalar_array = numpy_batched_arithmetic_mean(
        metric_values,
        axis=1,
        keepdims=False,
    )

    return scalar_array


@register_metric_aggregation(
    identifier="weighted",
    full_name="Prevalence Weighted Averaging",
    required_simple_metrics=("p_condition",),
    sklearn_equivalent="weighted",
)
def weighted_average(
    metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """_summary_

    Args:
        array (jtyping.Float[np.ndarray, ' num_samples num_classes']): _description_
        axis (int, optional): _description_. Defaults to -1.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """  # noqa: E501
    scalar_array = numpy_batched_convex_combination(
        metric_values,
        convex_weights=p_condition,
        axis=1,
        keepdims=False,
    )

    return scalar_array


@register_metric_aggregation(
    identifier="binary",
    full_name="Positive Class Averaging",
    required_simple_metrics=(),
    sklearn_equivalent="binary",
)
def positive_class_average(
    metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
    positive_class: int = 1,
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """_summary_

    Args:
        array (jtyping.Float[np.ndarray, ' num_samples num_classes']): _description_
        axis (int, optional): _description_. Defaults to -1.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """  # noqa: E501
    scalar_array = metric_values[:, positive_class]

    return scalar_array


@register_metric_aggregation(
    identifier="geometric",
    full_name="Geometric Mean Averaging",
    required_simple_metrics=(),
    sklearn_equivalent=None,
)
def geometric_average(
    metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """_summary_

    Args:
        array (jtyping.Float[np.ndarray, ' num_samples num_classes']): _description_
        axis (int, optional): _description_. Defaults to -1.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """  # noqa: E501
    scalar_array = numpy_batched_geometric_mean(
        metric_values,
        axis=1,
        keepdims=False,
    )

    return scalar_array


@register_metric_aggregation(
    identifier="harmonic",
    full_name="Harmonic Mean Averaging",
    required_simple_metrics=(),
    sklearn_equivalent=None,
)
def harmonic_average(
    metric_values: jtyping.Float[np.ndarray, " num_samples num_classes"],
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """_summary_

    Args:
        array (jtyping.Float[np.ndarray, ' num_samples num_classes']): _description_
        axis (int, optional): _description_. Defaults to -1.
        keepdims (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """  # noqa: E501
    scalar_array = numpy_batched_harmonic_mean(
        metric_values,
        axis=1,
        keepdims=False,
    )

    return scalar_array
