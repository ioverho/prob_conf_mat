from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import numpy as np
    import jaxtyping as jtyping


from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator


class SingletonAggregator(ExperimentAggregator):
    """An aggregation to apply to an ExperimentGroup that needs no aggregation.

    For example, the ExperimentGroup only contains one Experiment.

    Essentially just the [identity function](https://en.wikipedia.org/wiki/Identity_function):

    $$f(x)=x$$

    """

    full_name = "Singleton experiment aggregation"
    aliases = ["singleton", "identity"]

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        if num_experiments > 1:
            raise ValueError(
                f"Parameter `num_experiments` > 1. Currently {num_experiments}",
            )

        return experiment_samples[:, 0]
