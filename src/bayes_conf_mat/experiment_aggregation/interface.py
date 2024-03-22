import numpy as np

from bayes_conf_mat.experiment_aggregation.base import AGGREGATION_REGISTRY


def get_experiment_aggregator(
    aggregation: str, rng: np.random.BitGenerator, **kwargs
):
    aggregator_instance = AGGREGATION_REGISTRY[aggregation](
        rng=rng, **kwargs
    )
    return aggregator_instance
