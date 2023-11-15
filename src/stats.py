import math
from collections import OrderedDict

import numpy as np


def hdi(array, hdi_prob):
    """
    Compute hpi over the flattened array.

    Taken from ArviZ's `_hdi` function:
    https://python.arviz.org/en/stable/_modules/arviz/stats/stats.html
    """
    array = array.flatten()
    n = len(array)

    array = np.sort(array)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = np.subtract(
        array[interval_idx_inc:], array[:n_intervals], dtype=np.float_
    )

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)
    hdi_min = array[min_idx]
    hdi_max = array[min_idx + interval_idx_inc]

    hdi_interval = np.array([hdi_min, hdi_max])

    return interval_width[min_idx], hdi_interval


def estimate_median_and_mode(array):
    # Estimates the binwidth using Freedman-Diaconis rule
    # https://stats.stackexchange.com/a/862

    quantiles = np.quantile(array, [0, 0.25, 0.50, 0.75, 1])

    median = quantiles[2]

    sample_min = quantiles[0]
    sample_max = quantiles[4]
    iqr = quantiles[3] - quantiles[1]
    hist_width = 2 * iqr / math.pow(array.shape[0], 1 / 3)
    num_bins = int(np.round((sample_max - sample_min) / hist_width))
    counts, bin_edges = np.histogram(array, num_bins)
    modal_bin = np.argmax(counts)
    mode = (bin_edges[modal_bin] + bin_edges[modal_bin + 1]) / 2

    return median, mode, (quantiles[1], quantiles[3])


def summarize_posterior_samples(samples, alpha: float = 0.05):
    # Empirical mean and std. dev.
    mean = np.mean(samples)
    std_dev = np.std(samples)

    # Estimate median and mode by taking quantiles
    median, mode, (q25, q75) = estimate_median_and_mode(samples)

    # Estimate HDI as the CI
    # Use length as 'metric uncertainty'
    hdi_length, (hdi_lb, hdi_ub) = hdi(samples, hdi_prob=(1 - alpha / 2))

    summary = OrderedDict(
        [
            ("MAP", mode),
            ("HDI LB", hdi_lb),
            ("HDI UB", hdi_ub),
            ("Uncert", hdi_length),
            ("Mean", mean),
            ("StdDev", std_dev),
            ("Median", median),
            ("Q25", q25),
            ("Q75", q75),            
        ]
    )

    return summary
