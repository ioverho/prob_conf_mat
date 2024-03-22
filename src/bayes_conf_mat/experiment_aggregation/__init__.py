#! The order matters, do not change
from .base import AGGREGATION_REGISTRY, ExperimentAggregation
from .interface import get_experiment_aggregator
from .aggregators.beta import BetaAggregator
from .aggregators.fe_gaussian import FEGaussianAggregator
from .aggregators.re_gaussian import REGaussianAggregator
from .aggregators.histogram import HistogramAggregator
from .aggregators.gamma import GammaAggregator
