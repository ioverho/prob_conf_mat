from collections import OrderedDict

#! DO NOT ALTER
# The import order is deliberate
from .registration import IMPLEMENTED_SIMPLE_METRICS, IMPLEMENTED_COMPLEX_METRICS
from .simple import *
from .binary import *
from .multiclass import *
from .aggregation import *


def compute_all_simple_metrics(samples):
    simple_metrics = dict()
    for indentifier, simple_metric in filter(
        lambda x: x[1],
        list(IMPLEMENTED_SIMPLE_METRICS.items()),
    ):
        simple_metrics[indentifier] = simple_metric(samples)

    return simple_metrics


IMPLEMENTED_CLASS_AGGREGATIONS = OrderedDict(
    [
        (
            "macro",
            dict(
                name="Macro Average", function=aggregation.numpy_batched_arithmetic_mean
            ),
        ),
        (
            "weighted",
            dict(
                name="Weighted Micro Average",
                function=aggregation.numpy_batched_convex_combination,
            ),
        ),
        (
            "harmonic",
            dict(
                name="Harmonic Mean", function=aggregation.numpy_batched_harmonic_mean
            ),
        ),
        (
            "geometric",
            dict(
                name="Geometric Mean", function=aggregation.numpy_batched_geometric_mean
            ),
        ),
    ]
)
