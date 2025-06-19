from prob_conf_mat.experiment_aggregation.abc import (
    AGGREGATION_REGISTRY,
    ExperimentAggregator,
)
from prob_conf_mat.utils import RNG


def get_experiment_aggregator(
    aggregation: str, rng: RNG, **kwargs
) -> ExperimentAggregator:
    if aggregation not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Parameter `aggregation` must be a registered aggregation method. Currently: {aggregation}. Must be one of {set(AGGREGATION_REGISTRY.keys())}"
        )

    aggregator_instance = AGGREGATION_REGISTRY[aggregation](rng=rng, **kwargs)

    aggregator_instance._init_params = dict(aggregation=aggregation, **kwargs)

    return aggregator_instance
