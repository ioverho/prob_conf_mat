import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from bayes_conf_mat.utils import RNG


class HistogramAggregator(ExperimentAggregator):
    """Samples from a histogram approximate conflation distribution.

    First bins all individual experiment groups, and then computes the product of the probability masses
    across individual experiments.

    Unlike other methods, this does not make a parametric assumption. However, the resulting distribution can
    'look' unnatural, and requires overlapping supports within the sample. If any experiment assigns 0
    probability mass to any bin, the conflated bin will also contain 0 probability mass.

    As such, inter-experiment heterogeneity can be a significant problem.

    Uses [numpy.histogram_bin_edges](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html)
    to estimate the number of bin edges needed per experiment, and takes the smallest across all experiments
    for the aggregate distribution.

    Danger: Assumptions:
        - the individual experiment distributions' supports overlap

    References: Read more:
        1. [Hill, T. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)

    """

    full_name = "Histrogram approximated conflation experiment aggregation"
    aliases = ["hist", "histogram"]

    def __init__(
        self,
        rng: RNG,
        pseudo_count_weight: float = 0.1,
    ) -> None:
        super().__init__(rng=rng)

        # This is super arbitrary and should probably be tuned
        self.pseudo_count_weight = pseudo_count_weight

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Find the smallest recommended bin width for all experiments
        min_bin_width = float("inf")
        for per_experiment_samples in experiment_samples.T:
            distribution_bins = np.histogram_bin_edges(
                per_experiment_samples, bins="auto"
            )

            bin_width = distribution_bins[2] - distribution_bins[1]

            if bin_width < min_bin_width:
                min_bin_width = bin_width

        # Find the support for the aggregated histogram
        # Avoids having lots of zero-count bins
        min_min = np.min(experiment_samples)
        max_max = np.max(experiment_samples)

        found_bins = np.arange(
            start=max(min_min - min_bin_width, bounds[0]),
            stop=min(max_max + 2 * min_bin_width, bounds[1]),
            step=min_bin_width,
        )
        num_bins = found_bins.shape[0]

        # The pseudo-counts should have `pseudo_count_weight` times the weight of the true samples
        smoothing_coeff = 1 / num_bins * self.pseudo_count_weight * num_samples

        conflated_distribution = np.zeros(shape=(num_bins - 1,))
        for per_experiment_samples in experiment_samples.T:
            # Re-compute the histograms along the support of the aggregated distribution
            binned_distribution, bins = np.histogram(
                per_experiment_samples,
                bins=found_bins,
                range=bounds,
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
            (bins[:-1] + bins[1:]) / 2,  # type: ignore
            size=num_samples,
            p=conflated_distribution,
        )

        # Jitter the values so they fall off the bin midpoints
        bin_noise = self.rng.uniform(
            low=-min_bin_width / 2, high=min_bin_width / 2, size=num_samples
        )

        conflated_distribution_samples = np.clip(
            conflated_distribution_samples + bin_noise, a_min=bounds[0], a_max=bounds[1]
        )

        return conflated_distribution_samples
