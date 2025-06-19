from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import scipy
    import numpy as np
    import jaxtyping as jtyping

    from prob_conf_mat.utils.rng import RNG


def truncated_sample(
    sampling_distribution: scipy.stats.rv_continuous,
    bounds: tuple[float, float],
    rng: RNG,
    num_samples: int,
) -> jtyping.Float[np.ndarray, " num_samples"]:
    """Generates a bounded sample from an unbouded continuous Scipy distribution.

    Uses [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    to draw samples from the unbounded distribution.

    The quantiles sampled uniformly are bounded, such that their transform is also implicitly
    bounded.

    Args:
        sampling_distribution (scipy.stats.rv_continuous): the unbouded continuous
            Scipy distribution
        bounds (tuple[float, float]): the bounds
        rng (RNG): the random number generator
        num_samples (int): the number of samples to draw

    Returns:
        jtyping.Float[np.ndarray, " num_samples"]: the samples from the bounded distribution
    """
    u = rng.uniform(low=0.0, high=1.0, size=(num_samples,))

    truncated_u = u * (
        sampling_distribution.cdf(bounds[1]) - sampling_distribution.cdf(bounds[0])
    ) + sampling_distribution.cdf(bounds[0])

    truncated_samples = sampling_distribution.ppf(q=truncated_u)

    return truncated_samples
