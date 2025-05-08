#! The order matters, do not change
from .abc import (
    _ROOT_METRICS,
    METRIC_REGISTRY,
    AVERAGING_REGISTRY,
    RootMetric,
    Metric,
    AveragedMetric,
)
from ._metrics import *
from .averaging import *
from .interface import get_metric
from .collection import MetricCollection

# Check that all metrics have valid dependencies
for metric in METRIC_REGISTRY:
    for dependency in METRIC_REGISTRY[metric].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{metric}` not valid because: {e}"  # noqa: E501
            )

for aggregation in AVERAGING_REGISTRY:
    for dependency in AVERAGING_REGISTRY[aggregation].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{aggregation}` not valid because: {e}"  # noqa: E501
            )
