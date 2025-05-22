import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from bayes_conf_mat.stats import truncated_sample


class FELogGaussianAggregator(ExperimentAggregator):
    """Samples from the Log-Gaussian conflated distribution.

    Uses the inverse variance weighted mean and standard errors, calculated on the log-transformed
    samples. Specifically, the aggregate distribution
    $\\mathcal{N}(\\tilde{\\mu}, \\tilde{\\sigma})$ is estimated as:

    $$\\begin{aligned}
        w_{i}&=\\dfrac{\\sigma_{i}^{-2}}{\\sum_{j}^{M}\\sigma_{j}^{-2}} \\\\
        \\tilde{\\mu}&=\\sum_{i}^{M}w_{i}\\mu_{i} \\\\
        \\tilde{\\sigma^2}&=\\dfrac{1}{\\sum_{i}^{M}\\sigma_{i}^{-2}}
    \\end{aligned}$$

    where $M$ is the total number of experiments.

    Danger: Assumptions:
        - the individual experiment distributions are normally (Gaussian) distributed
        - there **is no** inter-experiment heterogeneity present

    References: Read more:
        1. [Hill, T. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)
        3. [Higgins, J., & Thomas, J. (Eds.). (2023). Cochrane handbook for systematic reviews of interventions.](https://training.cochrane.org/handbook/current/chapter-10#section-10-3)
        4. [Borenstein et al. (2021). Introduction to meta-analysis.](https://www.wiley.com/en-us/Introduction+to+Meta-Analysis%2C+2nd+Edition-p-9781119558354)
        5. ['Meta-analysis' on Wikipedia](https://en.wikipedia.org/wiki/Meta-analysis#Statistical_models_for_aggregate_data)

    """

    full_name = "Fixed-effect Log Gaussian conflated experiment aggregator"
    aliases = ["fe_log_gaussian", "log_gaussian", "log_normal"]

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        transformed_experiment_samples = np.log(experiment_samples)

        # Estimate the means and variances for each distribution
        means = np.mean(transformed_experiment_samples, axis=0)
        variances = np.var(transformed_experiment_samples, axis=0, ddof=1)

        # Compute the aggregated mean and variance
        # i.e. the inverse-variance weighted mean
        weights = 1 / variances

        agg_mu = np.sum(weights * means) / np.sum(weights)
        agg_var = 1 / np.sum(weights)

        # Redefine the sampling distribution
        conflated_distribution = scipy.stats.norm(
            loc=agg_mu,
            scale=np.sqrt(agg_var),
        )

        # Sample from the distribution, truncating at the bounds of the metric
        conflated_distribution_samples = truncated_sample(
            sampling_distribution=conflated_distribution,
            bounds=bounds,
            rng=self.rng,
            num_samples=num_samples,
        )

        transformed_conflated_distribution_samples = np.exp(
            conflated_distribution_samples
        )

        return transformed_conflated_distribution_samples
