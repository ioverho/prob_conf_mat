import typing

import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.utils.rng import RNG

def truncated_sample(
    sampling_distribution: scipy.stats.rv_continuous,
    bounds: tuple[float, float],
    rng: RNG,
    num_samples: int,
) -> jtyping.Float[np.ndarray, " num_samples"]:
    u = rng.uniform(low=0.0, high=1.0, size=(num_samples,))

    truncated_u = u * (
        sampling_distribution.cdf(bounds[1]) - sampling_distribution.cdf(bounds[0])
    ) + sampling_distribution.cdf(bounds[0])

    truncated_samples = sampling_distribution.ppf(q=truncated_u)

    return truncated_samples
