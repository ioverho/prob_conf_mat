import typing
from functools import wraps


def validate_composed_metric(composed_metric_func):
    @wraps(composed_metric_func)
    def validator(*args, **kwargs):
        passed_kwargs = set(kwargs.keys())
        required_kwargs = set(composed_metric_func.required_simple_metrics)

        if len(required_kwargs - passed_kwargs) > 0:
            missing_kwargs = required_kwargs - passed_kwargs
            raise TypeError(
                f"{composed_metric_func.identifier} missing {len(missing_kwargs)} required keyword arguments: {tuple(missing_kwargs)}"  # noqa: E501
            )

        elif len(passed_kwargs - required_kwargs) > 0:
            excess_kwargs = passed_kwargs - required_kwargs
            raise TypeError(
                f"{composed_metric_func.identifier} got {len(excess_kwargs)} unexpected keyword arguments {tuple(excess_kwargs)}"  # noqa: E501
            )

        return composed_metric_func(*args, **kwargs)

    return validator


def compose_metric_and_aggregation(
    metric_func,
    aggregation_func,
) -> typing.Callable:
    if metric_func.is_multiclass:
        raise ValueError(
            "A multivariate metric like `{metric_func.__name__}` does not need aggregating, per definition."  # noqa: E501
        )

    def composed_metric_aggregation(**kwargs):
        metric_vals = metric_func(
            **{k: kwargs[k] for k in metric_func.required_simple_metrics}
        )
        aggregated_metric_vals = aggregation_func(
            metric_values=metric_vals,
            **{k: kwargs[k] for k in aggregation_func.required_simple_metrics},
        )

        return aggregated_metric_vals

    composed_metric_aggregation.identifier = (
        f"{metric_func.identifier}@{aggregation_func.identifier}"
    )
    composed_metric_aggregation.full_name = (
        f"{metric_func.full_name} with {aggregation_func.full_name}"
    )
    composed_metric_aggregation.is_multiclass = True
    composed_metric_aggregation.range = metric_func.range

    composed_metric_aggregation.required_simple_metrics = (
        *metric_func.required_simple_metrics,
        *aggregation_func.required_simple_metrics,
    )

    composed_metric_aggregation.sklearn_equivalent = metric_func.sklearn_equivalent
    if aggregation_func.sklearn_equivalent is not None:
        (
            composed_metric_aggregation.sklearn_equivalent
            + f"with average={aggregation_func.sklearn_equivalent}"
        )

    # TODO: decide whether to keep this property
    # composed_metric_aggregation.composition = (metric_func, aggregation_func)

    return validate_composed_metric(composed_metric_aggregation)
