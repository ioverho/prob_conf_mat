from prob_conf_mat.experiment_aggregation.abc import (
    AGGREGATION_REGISTRY,
    ExperimentAggregator,
)
from prob_conf_mat.utils import RNG


def get_experiment_aggregator(
    aggregation: str,
    rng: RNG,
    **kwargs,
) -> ExperimentAggregator:
    """Fetches an `ExperimentAggregator` from its registered name.

    Args:
        aggregation (str): the name of the aggregator
        rng (RNG): the RNG this aggregator should use

    Keyword Args:
        **kwargs: any keyword arguments that get passed to the
            `__init__` method of the ExperimentAgrgegator

    """
    if aggregation not in AGGREGATION_REGISTRY:
        raise ValueError(
            f"Parameter `aggregation` must be a registered aggregation method. "
            f"Currently: {aggregation}. Must be one of {set(AGGREGATION_REGISTRY.keys())}",
        )

    aggregator_instance = AGGREGATION_REGISTRY[aggregation](rng=rng, **kwargs)

    aggregator_instance._init_params = dict(aggregation=aggregation, **kwargs)

    return aggregator_instance
