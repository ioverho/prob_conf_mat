from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from prob_conf_mat.utils.typing import MetricLike

import re
from functools import cache

from prob_conf_mat.metrics.abc import (
    _ROOT_METRICS,
    METRIC_REGISTRY,
    AVERAGING_REGISTRY,
    AveragedMetric,
    RootMetric,
)

RESERVED_CHARACTERS = {
    "@",  # Denotes the boundary between describing the metric and its averaging method
    "+",  # Denotes a kwarg key
    "=",  # Denotes a kwarg value
}
# TODO: document these regex strings
# Reminder: always document regex immediately after writing down...
NAME_REGEX = re.compile(r"([^\+\@\=]+)[\+\@\=]?")
ARGUMENT_REGEX = re.compile(r"\+([^\+\@\=]+)\=([^\+\@\=]+)[^\+\@]?")
UNTERMINATED_ARGUMENT_REGEX = re.compile(r"\+([^\+\@\=]+)")


def _parse_kwargs(kwargs: dict[str, str]) -> dict[str, typing.Any]:
    """Parse any passed kwargs."""
    for k, v in kwargs.items():
        # First try to eval
        try:
            val = eval(v)

            kwargs[k] = val

        # If it fails, assume a string
        except:  # noqa: E722
            continue

    return kwargs


@cache
def get_metric(syntax_string: str) -> MetricLike:
    """Takes a metric syntax string and returns a metric class instance, potentially with included averaging.

    Args:
        syntax_string (str): a valid metric syntax string

    Returns:
        callable: a metric class instance
    """  # noqa: E501
    # Split on the averaging
    syntax_components = syntax_string.split("@")

    if len(syntax_components) > 2:
        raise ValueError(
            f"Multiple averaging methods found in metric string `{syntax_string}`. Make sure to include only one `@` character",  # noqa: E501
        )

    # Parse the metric name ====================================================
    metric_string = syntax_components[0]

    metric_name = NAME_REGEX.findall(metric_string)[0]

    if metric_name in METRIC_REGISTRY:
        metric_class = METRIC_REGISTRY[metric_name]

    elif metric_name in _ROOT_METRICS:
        return RootMetric(name=metric_name)

    else:
        raise ValueError(
            f"Metric alias must be registered. "
            f"Currently: {metric_name}. "
            f"Must be one of {set(METRIC_REGISTRY.keys())}",
        )

    # Parse and pass the kwargs for the metric function ========================
    metric_kwargs = ARGUMENT_REGEX.findall(metric_string)
    metric_kwargs = dict(metric_kwargs)
    metric_kwargs = _parse_kwargs(metric_kwargs)

    # Check if there are more started kwargs than kwargs with values
    # If so, assume an unterminated kwargs expression
    unterminated_kwargs = UNTERMINATED_ARGUMENT_REGEX.findall(metric_string)
    if len(unterminated_kwargs) != len(metric_kwargs):
        raise ValueError(
            f"Found potentially unterminated kwarg in: {metric_string}. "
            "Make sure kwargs are written as '+foo=bar'",
        )

    metric_instance = metric_class(**metric_kwargs)

    metric_instance._instantiation_name = metric_string

    # Parse the averaging name ===============================================
    if len(syntax_components) == 2:
        averaging_string = syntax_components[1]

        if metric_instance.is_multiclass:
            raise ValueError(
                "Metric is already multivariate and does not need to be averaged. Please remove the `@` specification",  # noqa: E501
            )

        averaging_name = NAME_REGEX.findall(averaging_string)[0]

        try:
            averaging_class = AVERAGING_REGISTRY[averaging_name]
        except KeyError:
            raise ValueError(
                f"Averaging alias must be registered. "
                f"Currently: {averaging_name}. "
                f"Must be one of {set(AVERAGING_REGISTRY.keys())}",
            )

        # Parse and pass the kwargs for the metric function ========================
        averaging_kwargs = ARGUMENT_REGEX.findall(averaging_string)
        averaging_kwargs = dict(averaging_kwargs)
        averaging_kwargs = _parse_kwargs(averaging_kwargs)

        # Check if there are more started kwargs than kwargs with values
        # If so, assume an unterminated kwargs expression
        unterminated_kwargs = UNTERMINATED_ARGUMENT_REGEX.findall(averaging_string)
        if len(unterminated_kwargs) != len(averaging_kwargs):
            raise ValueError(
                f"Found potentially unterminated kwarg in: {averaging_string}. "
                "Make sure kwargs are written as '+foo=bar'",
            )

        averaging_instance = averaging_class(**averaging_kwargs)

        # Compose the metric & averaging function ============================
        composed_metric_instance = AveragedMetric(
            metric=metric_instance,
            averaging=averaging_instance,
        )

        composed_metric_instance._instantiation_name = syntax_string  # type: ignore

        return composed_metric_instance

    return metric_instance
