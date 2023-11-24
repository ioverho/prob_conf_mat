import re

from bayes_conf_mat.metrics.base import (
    METRIC_REGISTRY,
    AGGREGATION_REGISTRY,
    AggregatedMetric,
)

RESERVED_CHARACTERS = {
    "@",
    "+",
    "=",
}
NAME_REGEX = re.compile(r"([^\+\@\=]+)[\+\@\=]?")
ARGUMENT_REGEX = re.compile(r"\+([^\+\@\=]+)\=([^\+\@\=]+)[^\+\@]?")


def _parse_kwargs(kwargs):
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

    return kwargs


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

    if metric_name in METRIC_REGISTRY:
        metric_class = METRIC_REGISTRY[metric_name]

    else:
        raise ValueError(
            f"Metric `{metric_name}` not found. Please choose one of: {set(METRIC_REGISTRY.keys())}"  # noqa: E501
        )

    # Parse and pass the kwargs for the metric function ========================
    metric_kwargs = ARGUMENT_REGEX.findall(metric_string)
    metric_kwargs = dict(metric_kwargs)
    metric_kwargs = _parse_kwargs(metric_kwargs)

    metric_instance = metric_class(**metric_kwargs)

    # Parse the aggregation name ===============================================
    if len(syntax_components) == 2:
        aggregation_string = syntax_components[1]

        if metric_instance.is_multiclass:
            raise ValueError(
                "Metric is already multivariate and does not need to be aggregated. Please remove the `@` specification"  # noqa: E501
            )

        aggregation_name = NAME_REGEX.findall(aggregation_string)[0]

        try:
            aggregation_class = AGGREGATION_REGISTRY[aggregation_name]
        except KeyError:
            raise ValueError(
                f"Aggregation `{aggregation_name}` not found. Please choose one of: {set(AGGREGATION_REGISTRY.keys())}"  # noqa: E501
            )

        # Parse and pass the kwargs for the metric function ========================
        aggregation_kwargs = ARGUMENT_REGEX.findall(aggregation_string)
        aggregation_kwargs = dict(aggregation_kwargs)
        aggregation_kwargs = _parse_kwargs(aggregation_kwargs)

        aggregation_instance = aggregation_class(**aggregation_kwargs)

        # Compose the metric & aggregation function ============================
        composed_metric_instance = AggregatedMetric(
            metric=metric_instance,
            aggregation=aggregation_instance,
        )

        return composed_metric_instance

    else:
        return metric_instance
