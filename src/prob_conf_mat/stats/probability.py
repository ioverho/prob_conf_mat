import numpy as np


def odds(prob: np.floating) -> np.floating:
    """Computes the [odds](https://en.wikipedia.org/wiki/Odds) from a probability.

    Args:
        prob (float): the probability

    Returns:
        odds: float
    """
    return prob / (1 - prob)
