import typing

import scipy
import numpy as np


def truncated_sample(
    sampling_distribution: typing.Type[scipy.stats.rv_continuous],
    bounds: typing.Tuple[int],
    rng: np.random.BitGenerator,
    num_samples: int,
):
    u = rng.uniform(0.0, 1.0, size=(num_samples,))

    truncated_u = u * (
        sampling_distribution.cdf(bounds[1]) - sampling_distribution.cdf(bounds[0])
    ) + sampling_distribution.cdf(bounds[0])

    truncated_samples = sampling_distribution.ppf(q=truncated_u)

    return truncated_samples
