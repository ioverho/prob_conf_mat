import typing

import numpy as np
import scipy
import jaxtyping as jtyping

from bayes_conf_mat.math.truncated_sampling import truncated_sample


def gaussian_approximated_conflation(
    distribution_samples: typing.List[jtyping.Float[np.ndarray, " num_samples"]],
    range: typing.Tuple[int],
    rng: np.random.BitGenerator,
):
    max_samples = 0
    means = []
    variances = []
    for samples in distribution_samples:
        loc, scale = scipy.stats.norm.fit(samples)

        means.append(np.mean(loc))
        variances.append(np.power(scale, 2))

        if samples.shape[0] > max_samples:
            max_samples = samples.shape[0]

    means = np.array(means)
    variances = np.array(variances)

    weights = 1 / variances

    agg_mu = np.sum(weights * means) / np.sum(weights)
    agg_var = 1 / np.sum(weights)

    conflated_distribution = scipy.stats.norm(
        loc=agg_mu,
        scale=np.sqrt(agg_var),
    )

    conflated_distribution_samples = truncated_sample(
        sampling_distribution=conflated_distribution,
        range=range,
        rng=rng,
        num_samples=max_samples,
    )

    return conflated_distribution_samples
