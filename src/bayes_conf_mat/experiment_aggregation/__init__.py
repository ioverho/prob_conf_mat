#! The order matters, do not change
from .base import (
    AGGREGATION_REGISTRY,  # noqa: F401
)
from .interface import get_experiment_aggregator  # noqa: F401
from .beta import BetaAggregator  # noqa: F401
from .fe_gaussian import FEGaussianAggregator  # noqa: F401
from .re_gaussian import REGaussianAggregator  # noqa: F401
from .histogram import HistogramAggregator  # noqa: F401
