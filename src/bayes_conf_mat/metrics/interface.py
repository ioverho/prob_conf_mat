import re
from functools import partial
from copy import deepcopy

from bayes_conf_mat.metrics.registration import (
    IMPLEMENTED_AGGREGATION_FUNCTIONS,
    IMPLEMENTED_COMPLEX_METRICS,
    IMPLEMENTED_SIMPLE_METRICS,
)
from bayes_conf_mat.metrics.composition import compose_metric_and_aggregation

RESERVED_CHARACTERS = {
    "@",
    "+",
    "=",
}
NAME_REGEX = re.compile(r"([^\+\@\=]+)[\+\@\=]?")
ARGUMENT_REGEX = re.compile(r"\+([^\+\@\=]+)\=([^\+\@\=]+)[^\+\@]?")


def parse_and_pass_kwargs_to_func(func, kwargs):
    # Parse any passed kwargs ==================================================
    # Only accepting numeric, None or str as argument values
    for k, v in kwargs.items():
        try:
            # Check if value is numeric
            val = float(v)
            # Check if float is a whole number
            if val.is_integer():
                val = int(val)

            kwargs[k] = val

        except ValueError:
            # Check for None
            if v == "None":
                kwargs[k] = None

            # Else we assume a string
            else:
                continue

    if len(kwargs) > 0:
        wrapped_func = partial(func, **kwargs)
        wrapped_func.__dict__ = deepcopy(func.__dict__)
    else:
        wrapped_func = func

    return wrapped_func


def get_metric(syntax_string: str) -> callable:
    """Takes a metric syntax string and returns a metric function, potentially with included aggregation.

    Args:
        syntax_string (str): a valid metric syntax string

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        callable: a valid metric function
    """  # noqa: E501
    syntax_components = syntax_string.split("@")

    if len(syntax_components) > 2:
        raise ValueError(
            f"Multiple aggregations found in metric string `{syntax_string}`. Make sure to include only one `@` character"  # noqa: E501
        )

    # Parse the metric name ====================================================
    metric_string = syntax_components[0]

    metric_name = NAME_REGEX.findall(metric_string)[0]

    if metric_name in IMPLEMENTED_COMPLEX_METRICS:
        metric_func = IMPLEMENTED_COMPLEX_METRICS[metric_name]

    elif metric_name in IMPLEMENTED_SIMPLE_METRICS:
        metric_func = IMPLEMENTED_SIMPLE_METRICS[metric_name]

    else:
        raise ValueError(
            f"Metric `{metric_name}` not found. Please choose one of: {set((IMPLEMENTED_SIMPLE_METRICS | IMPLEMENTED_COMPLEX_METRICS).keys())}"  # noqa: E501
        )

    # Parse and pass the kwargs for the metric function ========================
    metric_kwargs = ARGUMENT_REGEX.findall(metric_string)
    metric_kwargs = dict(metric_kwargs)

    wrapped_metric_func = parse_and_pass_kwargs_to_func(metric_func, metric_kwargs)

    # Parse the aggregation name ===============================================
    if len(syntax_components) == 2:
        aggregation_string = syntax_components[1]

        if metric_func.is_multiclass:
            raise ValueError(
                "Metric is already multivariate and does not need to be aggregated. Please remove the `@` specification"  # noqa: E501
            )

        aggregation_name = NAME_REGEX.findall(aggregation_string)[0]

        try:
            aggregation_func = IMPLEMENTED_AGGREGATION_FUNCTIONS[aggregation_name]
        except KeyError:
            raise ValueError(
                f"Aggregation `{aggregation_name}` not found. Please choose one of: {set(IMPLEMENTED_AGGREGATION_FUNCTIONS.keys())}"  # noqa: E501
            )

        # Parse and pass the kwargs for the metric function ========================
        aggregation_kwargs = ARGUMENT_REGEX.findall(aggregation_string)
        aggregation_kwargs = dict(aggregation_kwargs)

        wrapped_aggregation_func = parse_and_pass_kwargs_to_func(
            aggregation_func, aggregation_kwargs
        )

        # Compose the metric & aggregation function ============================
        wrapped_metric_func = compose_metric_and_aggregation(
            wrapped_metric_func, wrapped_aggregation_func
        )

    return wrapped_metric_func
