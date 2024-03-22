import math

import numpy as np
import jaxtyping as jtyping


def hdi_estimator(samples: jtyping.Float[np.ndarray, " num_samples"], prob: float):
    # Sort the samples
    sorted_posterior_samples = np.sort(samples)

    # Figure out how many samples are included and excluded
    n_samples = samples.shape[0]
    n_included = math.floor(prob * n_samples)
    n_excluded = n_samples - n_included

    # Find smallest interval
    # Largest excluded values minus smallest included values
    idx_min_interval = np.argmin(
        sorted_posterior_samples[n_included:] - sorted_posterior_samples[:n_excluded]
    )

    # Compute bounds
    lb = sorted_posterior_samples[idx_min_interval]
    ub = sorted_posterior_samples[n_included + idx_min_interval]

    return (lb, ub)
