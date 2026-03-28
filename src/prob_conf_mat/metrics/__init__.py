# IMPORTANT: The order matters, do not change
from ._metrics import *
from .abc import (
    _ROOT_METRICS,  # pyright: ignore[reportPrivateUsage]
    AVERAGING_REGISTRY,
    METRIC_REGISTRY,
    AveragedMetric,
    Metric,
    RootMetric,
)
from .averaging import *
from .collection import MetricCollection
from .experimental_metrics import *
from .interface import get_metric

# Check that all metrics have valid dependencies
for metric in METRIC_REGISTRY:
    for dependency in METRIC_REGISTRY[metric].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{metric}` not valid because: {e}",
            ) from e

for aggregation in AVERAGING_REGISTRY:
    for dependency in AVERAGING_REGISTRY[aggregation].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{aggregation}` not valid because: {e}",
            ) from e
