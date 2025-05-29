from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import math

import numpy as np


def hdi_estimator(
    samples: jtyping.Float[np.ndarray, " num_samples"],
    prob: float,
) -> tuple[
    float | jtyping.Float[np.ndarray, ""],
    float | jtyping.Float[np.ndarray, ""],
]:
    """Computes the highest density interval (HDI) of an array of samples for a given probability.

    Adapted from [arviz](https://python.arviz.org/en/stable/_modules/arviz/stats/stats.html#_hdi).

    Guaranteed to contain the median if `prob > 0.5`, and if the distribution is unimodal, also
    contains the mode.

    Args:
        samples (jtyping.Float[np.ndarray, " num_samples"]): the array of samples
        prob (float): the probability

    Returns:
        tuple[float, float]: the lower and upper bound of the HDI
    """
    # Sort the samples
    sorted_posterior_samples = np.sort(samples)

    # Figure out how many samples are included and excluded
    n_samples = samples.shape[0]
    n_included = math.floor(prob * n_samples)
    n_excluded = n_samples - n_included

    # Find smallest interval
    # Largest excluded values minus smallest included values
    idx_min_interval = np.argmin(
        sorted_posterior_samples[n_included:] - sorted_posterior_samples[:n_excluded],
    )

    # Compute bounds
    lb = sorted_posterior_samples[idx_min_interval]
    ub = sorted_posterior_samples[n_included + idx_min_interval]

    return (lb, ub)
