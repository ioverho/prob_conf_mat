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
from .experimental import *
from .interface import get_metric

# Check that all metrics have valid dependencies
for metric in METRIC_REGISTRY:
    for dependency in METRIC_REGISTRY[metric].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            dep = get_metric(dependency)
            if is_experimental(dep) and not is_experimental(METRIC_REGISTRY[metric]):  # pyright: ignore[reportArgumentType]
                raise TypeError(
                    f"Non-experimental metric `{metric}` depends on experimental metric `{dependency}`."
                )

        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{metric}` not valid because: {e}",
            ) from e

for averaging in AVERAGING_REGISTRY:
    for dependency in AVERAGING_REGISTRY[averaging].dependencies:
        if dependency in _ROOT_METRICS:
            continue

        try:
            get_metric(dependency)
            dep = get_metric(dependency)
            if is_experimental(dep) and not is_experimental(AVERAGING_REGISTRY[averaging]):  # pyright: ignore[reportArgumentType]
                raise TypeError(
                    f"Non-experimental averaging method `{averaging}` depends on experimental metric `{dependency}`."
                )

        except Exception as e:
            raise KeyError(
                f"Dependency `{dependency}` of `{averaging}` not valid because: {e}",
            ) from e
