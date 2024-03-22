import typing

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation


class HistogramAggregator(ExperimentAggregation):
    name = "histogram"
    full_name = "Histrogram approximated conflation experiment aggregation"
    aliases = ["hist", "histogram"]

    def __init__(
        self,
        rng: np.random.BitGenerator,
        pseudo_count_weight: float = 0.1,
    ) -> None:
        super().__init__(rng=rng)

        # This is super arbitrary and should probably be tuned
        self.pseudo_count_weight = pseudo_count_weight

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_experiments, num_samples = distribution_samples.shape

        # Find the smallest recommended bin width for all experiments
        min_bin_width = float("inf")
        for samples in distribution_samples:
            distribution_bins = np.histogram_bin_edges(samples, bins="auto")

            bin_width = distribution_bins[2] - distribution_bins[1]

            if bin_width < min_bin_width:
                min_bin_width = bin_width

        # Find the support for the aggregated histogram
        # Avoids having lots of zero-count bins
        min_min = np.min(distribution_samples)
        max_max = np.max(distribution_samples)

        found_bins = np.arange(
            start=max(min_min - bin_width, bounds[0]),
            stop=min(max_max + 2 * bin_width, bounds[1]),
            step=bin_width,
        )
        num_bins = found_bins.shape[0]

        # The pseudo-counts should have `pseudo_count_weight` the weight of the true samples
        smoothing_coeff = 1 / num_bins * self.pseudo_count_weight * num_samples

        conflated_distribution = np.zeros(shape=(num_bins - 1,))
        for samples in distribution_samples:
            # Re-compute the histograms along the support of the aggregated distribution
            binned_distribution, bins = np.histogram(
                samples,
                bins=found_bins,
                bounds=bounds,
            )

            # Estimate the bin probabilities
            log_p_hat = np.log(binned_distribution + smoothing_coeff) - np.log(
                num_samples + smoothing_coeff * num_bins
            )

            conflated_distribution += log_p_hat

        conflated_distribution = np.exp(
            conflated_distribution - np.logaddexp.reduce(conflated_distribution)
        )

        # Resample the conflated distribution
        # Samples at the midpoint of each bin
        conflated_distribution_samples = self.rng.choice(
            (bins[:-1] + bins[1:]) / 2,
            size=num_samples,
            p=conflated_distribution,
        )

        # Jitter the values so they 'fall' off the bin midpoints
        bin_noise = self.rng.uniform(
            low=-min_bin_width / 2, high=min_bin_width / 2, size=num_samples
        )

        conflated_distribution_samples = conflated_distribution_samples + bin_noise

        return conflated_distribution_samples
