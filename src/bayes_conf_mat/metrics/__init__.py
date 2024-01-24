#! The order matters, do not change
from .base import (
    _ROOT_METRICS,
    METRIC_REGISTRY,
    AGGREGATION_REGISTRY,
    Metric,  # noqa: F401, F403
    AggregatedMetric,  # noqa: F401, F403
)
from .simple_metrics import *  # noqa: F401, F403
from .complex_metrics import *  # noqa: F401, F403
from .aggregations import *  # noqa: F401, F403
from .interface import get_metric  # noqa: F401
from .collection import MetricCollection  # noqa: F401

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

for aggregation in AGGREGATION_REGISTRY:
    for dependency in AGGREGATION_REGISTRY[aggregation].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{aggregation}` not valid because: {e}"  # noqa: E501
            )
