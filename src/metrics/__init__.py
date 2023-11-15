import typing
from collections import OrderedDict

#! DO NOT ALTER
# The import order is deliberate
from .registration import IMPLEMENTED_SIMPLE_METRICS, IMPLEMENTED_COMPLEX_METRICS
from .simple import *
from .binary import *
from .multiclass import *
from .aggregation import *

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
