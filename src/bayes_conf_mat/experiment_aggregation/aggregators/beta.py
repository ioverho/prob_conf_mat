import typing

import scipy
import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment_aggregation.base import ExperimentAggregation


class BetaAggregator(ExperimentAggregation):
    name = "beta"
    full_name = "Beta conflated experiment aggregation"
    aliases = ["beta", "beta_conflation"]

    def __init__(
        self, rng: np.random.BitGenerator, estimation_method: str = "mle"
    ) -> None:
        super().__init__(rng=rng)

        # Honestly should get rid of this
        # MLE is more efficient than MoME, and difference is small
        self.estimation_method = estimation_method

    def aggregate(
        self,
        distribution_samples: jtyping.Float[np.ndarray, " num_experiments num_samples"],
        bounds: typing.Tuple[int],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_experiments, num_samples = distribution_samples.shape

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
        for samples in distribution_samples:
            alpha, beta, _, _ = scipy.stats.beta.fit(
                np.clip(samples, a_min=bounds[0] + 1e-9, a_max=bounds[1] - 1e-9),
                method=self.estimation_method,
                floc=bounds[0],
                fscale=bounds[1] - bounds[0],
            )

            alphas.append(alpha)
            betas.append(beta)

        conflated_alpha = sum(alphas) - (num_experiments - 1)
        conflated_beta = sum(betas) - (num_experiments - 1)

        conflated_distribution_samples = scipy.stats.beta.rvs(
            a=conflated_alpha,
            b=conflated_beta,
            size=num_samples,
            loc=bounds[0],
            scale=bounds[1] - bounds[0],
            random_state=self.rng,
        )

        # conflated_distribution_samples = (
        #    bounds[1] - bounds[0]
        # ) * conflated_distribution_samples + bounds[0]

        return conflated_distribution_samples
