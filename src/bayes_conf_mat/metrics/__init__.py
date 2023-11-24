#! DO NOT ALTER
# The import order is deliberate
# from .registration import IMPLEMENTED_SIMPLE_METRICS, IMPLEMENTED_COMPLEX_METRICS, IMPLEMENTED_AGGREGATION_FUNCTIONS
# from .simple import *
# from .binary import *
# from .multiclass import *
# from .aggregation import *
# from .interface import get_metric

from .base import METRIC_REGISTRY, AGGREGATION_REGISTRY, AggregatedMetric
from .simple_metrics import *
from .complex_metrics import *
from .aggregations import *
from .interface import get_metric

# Check that all metrics have valid dependencies
for metric in METRIC_REGISTRY:
    for dependency in METRIC_REGISTRY[metric].dependencies:
        if dependency not in METRIC_REGISTRY:
            raise KeyError(
                f"Dependency `{dependency}` of `{metric}` not found in metric registry."  # noqa: E501
            )

for aggregation in AGGREGATION_REGISTRY:
    for dependency in AGGREGATION_REGISTRY[aggregation].dependencies:
        if dependency not in METRIC_REGISTRY:
            raise KeyError(
                f"Dependency `{dependency}` of `{aggregation}` not found in metric registry."  # noqa: E501
            )
