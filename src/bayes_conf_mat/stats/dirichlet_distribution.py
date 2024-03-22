import typing
# from collections.abc import Iterable

import numpy as np
import jaxtyping as jtyping

_DIRICHLET_PRIOR_STRATEGIES = {
    "bayes-laplace": 1.0,
    "bayes": 1.0,
    "laplace": 1.0,
    "ones": 1.0,
    "jeffreys": 0.5,
    "halves": 0.5,
    "haldane": 0.0,
    "zeros": 0.0,
}


def dirichlet_prior(
    strategy: str | float | int | jtyping.Float[np.ndarray, " ..."],
    shape: typing.Tuple[int],
):
    if isinstance(strategy, float) or isinstance(strategy, int):
        prior = np.full(shape, fill_value=strategy)

    elif isinstance(strategy, str):
        if strategy not in _DIRICHLET_PRIOR_STRATEGIES:
            raise ValueError(
                f"Prior strategy `{strategy}` not recognized. Choose one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"  # noqa: E501
            )

        strategy_fill_value = _DIRICHLET_PRIOR_STRATEGIES[strategy]
        prior = np.full(shape, fill_value=strategy_fill_value)

    else:
        try:
            prior = np.array(strategy)
        except Exception as e:
            raise ValueError(
                f"While trying to convert {strategy} to a numpy array, received the following error:\n{e}"
            )

        if prior.shape != shape:
            raise ValueError(
                f"Prior does not match required shape, {prior.shape} != {shape}"
            )

    return prior


def dirichlet_sample(
    rng: np.random.RandomState,
    alphas: jtyping.Float[np.ndarray, " ..."],
    num_samples: int,
) -> jtyping.Float[np.ndarray, " num_samples ..."]:
    """
    Generate Dirichlet distributed samples from an array of Gamma distributions.

    Unlike the numpy implementation, this can be vectorized.

    Taken from: https://stackoverflow.com/a/15917312

    Args:
        rng (np.random.RandomState)
        alphas (jtyping.Float[np.ndarray, "..."]): the Dirichlet parameters
        num_samples (int): the number of samples to retrieve

    Returns:
        _type_: _description_

    """
    # Check if the array is already batched
    # And to try to batch it if not
    alphas = np.broadcast_to(alphas, (num_samples, *alphas.shape))

    r = rng.standard_gamma(alphas)

    d = r / r.sum(-1, keepdims=True)

    return d
