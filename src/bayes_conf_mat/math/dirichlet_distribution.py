import typing

import numpy as np

_PRIOR_STRATEGIES = {
    "bayes-laplace": 1,
    "bayes": 1,
    "laplace": 1,
    "ones": 1,
    "jeffreys": 0.5,
    "halves": 0.5,
    "haldane": 0,
    "zeros": 0,
}


def dirichlet_prior(strategy: str, shape: typing.Union[int, typing.Tuple[int]]):
    if isinstance(strategy, int):
        prior = np.full(shape, fill_value=strategy)

    elif isinstance(strategy, str):
        if strategy not in _PRIOR_STRATEGIES:
            raise ValueError(
                f"Prior strategy `{strategy}` not recognized. Choose one of: {set(_PRIOR_STRATEGIES.keys())}"  # noqa: E501
            )

        strategy_fill_value = _PRIOR_STRATEGIES[strategy]
        prior = np.full(shape, fill_value=strategy_fill_value)

    else:
        raise ValueError(f"Prior strategy *format* `{type(strategy)}` not recognized.")

    return prior


def dirichlet_sample(
    rng: np.random.RandomState, alphas: np.ndarray, num_samples: int = 1
):
    """
    Generate Dirichlet distributed samples from an array of Gamma distributions.

    Unlike the numpy implementation, this can be batched.

    Taken from: https://stackoverflow.com/a/15917312
    """
    # Check if the first dimension of the array is `num_samples`
    if alphas.shape != num_samples:
        # If not, add the 'batch' dimension
        alphas = np.broadcast_to(alphas, (num_samples, *alphas.shape))

    r = rng.standard_gamma(alphas)

    return r / r.sum(-1, keepdims=True)
