from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import typing

import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation


class SingletonAggregator(ExperimentAggregation):
    name = "singleton"
    full_name = "Singleton experiment aggregation"
    aliases = ["singleton", "identity"]

    def __init__(self, rng: np.random.BitGenerator) -> None:
        super().__init__(rng=rng)

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        if num_experiments > 1:
            raise ValueError(
                f"Parameter `num_experiments` > 1. Currently {num_experiments}"
            )

        return experiment_samples[:, 0]
