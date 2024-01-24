import numpy as np

from bayes_conf_mat.experiment_aggregation.base import AGGREGATION_REGISTRY


def get_experiment_aggregator(
    aggregation: str, rng: np.random.BitGenerator, num_proc: int, **kwargs
):
    aggregator_instance = AGGREGATION_REGISTRY[aggregation](
        rng=rng, num_proc=num_proc, **kwargs
    )
    return aggregator_instance
