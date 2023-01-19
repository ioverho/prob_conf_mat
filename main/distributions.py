import typing

import numpy as np

IMPLEMENTED_DIRICHLET_PRIOR_STRATEGIES = {"zeros", "ones"}


def init_dirichlet_prior(
    strategy: typing.Union[str, float], num_categories: int, verbose: bool = False
):
    """Constructs a Dirichlet prior from a given strategy

    Args:
        strategy (typing.Union[str, int]): _description_
        num_categories (int): _description_
    """

    if isinstance(strategy, int):
        strategy = float(strategy)

    if isinstance(strategy, float):
        prior = np.full(shape=(num_categories,), fill_value=strategy)

        if verbose:
            print(f"Dirichlet prior initialized with all {strategy}.")

    elif isinstance(strategy, str):

        if strategy not in IMPLEMENTED_DIRICHLET_PRIOR_STRATEGIES:
            raise ValueError(
                f"Strategy must be one of {IMPLEMENTED_DIRICHLET_PRIOR_STRATEGIES}"
            )

        elif strategy == "zeros":
            prior = np.zeros(shape=(num_categories,))

            if verbose:
                print(f"Dirichlet prior initialized with all zeros")

        elif strategy == "ones":
            prior = np.zeros(shape=(num_categories,))

            if verbose:
                print(f"Dirichlet prior initialized with all ones")

    return prior
