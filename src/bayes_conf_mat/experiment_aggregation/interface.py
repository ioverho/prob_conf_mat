import numpy as np

from bayes_conf_mat.experiment_aggregation.base import AGGREGATION_REGISTRY


def get_experiment_aggregator(
    aggregation: str, rng: int | np.random.BitGenerator, **kwargs
):
    if aggregation not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Parameter `aggregation` must be a registered aggregation method. Currently: {aggregation}. Must be one of {set(AGGREGATION_REGISTRY.keys())}"
        )

    aggregator_instance = AGGREGATION_REGISTRY[aggregation](rng=rng, **kwargs)

    aggregator_instance._init_params = dict(aggregation=aggregation, **kwargs)

    return aggregator_instance
