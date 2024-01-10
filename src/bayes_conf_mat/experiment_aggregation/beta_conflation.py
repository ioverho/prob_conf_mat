import typing

import numpy as np
import scipy
import jaxtyping as jtyping

def beta_approximated_conflation(
    distribution_samples: typing.List[jtyping.Float[np.ndarray, " num_samples"]],
    extrema: typing.Tuple[int],
    rng: np.random.BitGenerator,
    estimation_method: str = "mle",
):
    num_experiments, num_samples = distribution_samples.shape

    alphas = []
    betas = []
    for samples in distribution_samples:
        alpha, beta, _, _ = scipy.stats.beta.fit(
            samples, method=estimation_method, floc=extrema[0], fscale=extrema[1]
        )

        alphas.append(alpha)
        betas.append(beta)

    conflated_alpha = sum(alphas) - (num_experiments - 1)
    conflated_beta = sum(betas) - (num_experiments - 1)

    conflated_distribution_samples = scipy.stats.beta.rvs(
        a=conflated_alpha,
        b=conflated_beta,
        size=num_samples,
        random_state=rng,
    )

    return conflated_distribution_samples
