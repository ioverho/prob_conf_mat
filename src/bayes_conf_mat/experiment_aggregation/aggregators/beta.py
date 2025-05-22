import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from bayes_conf_mat.utils import RNG


class BetaAggregator(ExperimentAggregator):
    """Samples from the beta-conflated distribution.

    Specifically, the aggregate distribution $\\text{Beta}(\\tilde{\\alpha}, \\tilde{\\beta})$ is estimated as:

    $$\\begin{aligned}
        \\tilde{\\alpha}&=\\left[\\sum_{i=1}^{M}\\alpha_{i}\\right]-\\left(M-1\\right) \\\\
        \\tilde{\\beta}&=\\left[\\sum_{i=1}^{M}\\beta_{i}\\right]-\\left(M-1\\right)
    \\end{aligned}$$

    where $M$ is the total number of experiments.

    Uses [`scipy.stats.beta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html) class to fit beta-distributions.

    Danger: Assumptions:
        - the individual experiment distributions are beta distributed
        - the metrics are bounded, although the range need not be (0, 1)

    References: Read more:
        1. [Hill, T. P. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T. P., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)
        3. ['Beta distribution' on Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)

    Args:
        estimation_method (str): method for estimating the parameters of the individual experiment distributions. Options are 'mle' for maximum-likelihood estimation, or 'mome' for the method of moments estimator. MLE tends be more efficient but is difficult to estimate

    """

    full_name = "Beta conflated experiment aggregation"
    aliases = ["beta", "beta_conflation"]

    def __init__(self, rng: RNG, estimation_method: str = "mle") -> None:
        super().__init__(rng=rng)

        # Honestly should get rid of this
        # MLE is more efficient than MoME, and difference is small
        self.estimation_method = estimation_method

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        if bounds[0] == -float("inf") or bounds[1] == float("inf"):
            raise NotImplementedError(
                "Beta aggregation does not (yet) support metrics with infite bounds."
            )

        # Tranform the data to lie in the bounds of the beta distribution
        # if bounds[0] != 0.0 or bounds[1] != 1.0:
        #    if bounds[0] == -float("inf") or bounds[1] == float("inf"):
        #        raise NotImplementedError(
        #            "Beta aggregation does not (yet) support metrics with infite bounds."
        #        )
        #
        #    transformed_distribution_samples = (distribution_samples - bounds[0]) / (
        #        bounds[1] - bounds[0]
        #    )
        #
        # else:
        #    transformed_distribution_samples = distribution_samples
        #
        ## Try to shift any values at the bounds just off the bounds using 'prior'
        ## https://stats.stackexchange.com/a/31313
        ## Otherwise MLE breaks
        # if (
        #    np.min(transformed_distribution_samples) == 0.0
        #    or np.max(transformed_distribution_samples) == 1.0
        # ):
        #    mome_mean = np.mean(transformed_distribution_samples)
        #
        #    transformed_distribution_samples = (
        #        transformed_distribution_samples * (num_samples - 1) + mome_mean
        #    ) / num_samples

        alphas = []
        betas = []
        for per_experiment_samples in experiment_samples.T:
            alpha, beta, _, _ = scipy.stats.beta.fit(
                np.clip(
                    per_experiment_samples,
                    a_min=bounds[0] + 1e-9,
                    a_max=bounds[1] - 1e-9,
                ),
                method=self.estimation_method,
                floc=bounds[0],
                fscale=bounds[1] - bounds[0],
            )

            alphas.append(alpha)
            betas.append(beta)

        conflated_alpha = sum(alphas) - (num_experiments - 1)
        conflated_beta = sum(betas) - (num_experiments - 1)

        # Scipy distributions won't accept the RNG wrapper
        # So pass rng.rng
        conflated_distribution_samples = scipy.stats.beta.rvs(
            a=conflated_alpha,
            b=conflated_beta,
            size=num_samples,
            loc=bounds[0],
            scale=bounds[1] - bounds[0],
            random_state=self.rng.rng,
        )

        # conflated_distribution_samples = (
        #    bounds[1] - bounds[0]
        # ) * conflated_distribution_samples + bounds[0]

        return conflated_distribution_samples
