from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.utils import RNG

import numpy as np


_DIRICHLET_PRIOR_STRATEGIES = {
    "bayes-laplace": 1.0,
    "bayes": 1.0,
    "laplace": 1.0,
    "ones": 1.0,
    "one": 1.0,
    "jeffrey": 0.5,
    "jeffreys": 0.5,
    "halves": 0.5,
    "half": 0.5,
    "haldane": 0.0,
    "zeros": 0.0,
    "zero": 0.0,
}


# TODO: add support for other dtypes
def dirichlet_prior(
    strategy: str | float | int | jtyping.Float[np.typing.ArrayLike, " ..."],
    shape: tuple[int, ...],
) -> jtyping.Float[np.ndarray, " ..."]:
    """Creates a prior array for a Dirichlet distribution.

    Returns:
        jtyping.Float[np.ndarray, " ..."]: the prior vector
    """
    if isinstance(strategy, float | int):
        if strategy < 0:
            raise ValueError(
                f"A Dirichlet prior must contain positive values. Received: {strategy}",
            )

        prior = np.full(shape, fill_value=strategy, dtype=np.float64)

    elif isinstance(strategy, str):
        if strategy not in _DIRICHLET_PRIOR_STRATEGIES:
            raise ValueError(
                f"Prior strategy `{strategy}` not recognized. "
                f"Choose one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}",
            )

        strategy_fill_value = _DIRICHLET_PRIOR_STRATEGIES[strategy]
        prior = np.full(shape, fill_value=strategy_fill_value, dtype=np.float64)

    else:
        try:
            # TODO: control dtype
            prior = np.array(strategy, dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            raise ValueError(
                f"While trying to convert {strategy} to a numpy array, "
                f"received the following error:\n{e}",
            )

        if prior.shape != shape:
            raise ValueError(
                f"Prior does not match required shape, {prior.shape} != {shape}. "
                f"Parsed {prior} of type {type(prior)} from {strategy} fo type {type(strategy)}.",
            )

    return prior


def dirichlet_sample(
    rng: RNG,
    alphas: jtyping.Float[np.ndarray, " ..."],
    num_samples: int,
) -> jtyping.Float[np.ndarray, " num_samples ..."]:
    """Generate Dirichlet distributed samples from an array of Gamma distributions.

    A Dirichlet distribution can be constructed by [dividing a set of Gamma distributions by their sum](https://en.wikipedia.org/wiki/Dirichlet_distribution#Related_distributions).

    For some reason the Numpy implementation of the [Dirichlet distribution](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.dirichlet.html#numpy.random.Generator.dirichlet)
    is not vectorized, while the implementation of the [Gamma distribution](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.gamma.html).
    is.

    Adapted from [StackOverflow](https://stackoverflow.com/a/15917312).

    This function is the performance bottleneck for this package.
    Need to make sure it's performant.

    Args:
        rng (RNG): the random number generator
        alphas (jtyping.Float[np.ndarray, "..."]): the Dirichlet parameters
        num_samples (int): the number of samples to retrieve

    Returns:
        jtyping.Float[np.ndarray, " num_samples ..."]: samples from the specified Dirichlet distribution
    """  # noqa: E501
    # TODO: add support for other dtypes
    alphas = alphas.astype(np.float64)

    # Broadcast alphas to the desired shape
    alphas = np.broadcast_to(alphas, (num_samples, *alphas.shape))

    # Generate independent gamma-distributed samples
    # This bit dominates the run-time
    arr = rng.standard_gamma(alphas)

    # Normalize the sampled gamma distributed variables
    # This bit is probably as fast as it can get
    np.divide(arr, np.einsum("...i->...", arr)[..., np.newaxis], out=arr)

    return arr
