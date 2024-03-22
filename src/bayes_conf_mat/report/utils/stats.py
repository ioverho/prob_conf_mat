import typing

import math
import numpy as np
import jaxtyping as jtyping
import scipy.stats as stats


def histogram_mode_estimator(
    samples: jtyping.Float[np.ndarray, " num_samples"],
    range: typing.Optional[typing.Tuple[float, float]] = None,
):
    bin_counts, bin_edges = np.histogram(samples, bins="auto", range=range)
    modal_bin = np.argmax(bin_counts)

    mode = (bin_edges[modal_bin] + bin_edges[modal_bin + 1]) / 2

    return mode


def kde_mode_estimator(
    samples: jtyping.Float[np.ndarray, " num_samples"],
    range: typing.Optional[typing.Tuple[float, float]] = None,
):
    from lightkde import kde_1d

    kernel_densities, evaluation_points = kde_1d(
        samples,
        x_min=range[0] if range is not None else None,
        x_max=range[1] if range is not None else None,
    )

    modal_point = np.argmax(kernel_densities)

    mode = evaluation_points[modal_point]

    return mode


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


def summarize_posterior(
    posterior_samples: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
):
    hdi = hdi_estimator(posterior_samples, prob=ci_probability)

    summary = [
        ("Median", np.median(posterior_samples)),
        ("Mode", histogram_mode_estimator(posterior_samples)),
        (f"{ci_probability*100:.1f}% HDI", hdi),
        ("Skew", stats.skew(posterior_samples)),
        ("Kurt", stats.kurtosis(posterior_samples)),
    ]

    return summary
