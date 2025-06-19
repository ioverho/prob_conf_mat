from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.utils import RNG

import scipy
import numpy as np

from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from prob_conf_mat.experiment_aggregation.heterogeneity import (
    heterogeneity_DL,
    heterogeneity_PM,
)
from prob_conf_mat.stats import truncated_sample


class REGaussianAggregator(ExperimentAggregator):
    """Samples from the Random Effects Meta-Analytical Estimator.

    First uses the standard the inverse variance weighted mean and standard errors as model parameters, before
    debiasing the weights to incorporate inter-experiment heterogeneity. As a result, studies with larger
    standard errors will be upweighted relative to the fixed-effects model.

    Specifically, starting with a Fixed-Effects model $\\mathcal{N}(\\tilde{\\mu_{\\text{FE}}}, \\tilde{\\sigma_{\\text{FE}}})$,

    $$\\begin{aligned}
        w_{i}&=\\dfrac{\\left(\\sigma_{i}^2+\\tau^2\\right)^{-1}}{\\sum_{j}^{M}\\left(\\sigma_{j}^2+\\tau^2\\right)^{-1}} \\\\
        \\tilde{\\mu}&=\\sum_{i}^{M}w_{i}\\mu_{i} \\\\
        \\tilde{\\sigma^2}&=\\dfrac{1}{\\sum_{i}^{M}\\sigma_{i}^{-2}}
    \\end{aligned}$$

    where $\\tau$ is the estimated inter-experiment heterogeneity, and $M$ is the total number of experiments.

    Uses the Paule-Mandel iterative heterogeneity estimator, which does not make a parametric assumption. The
    more common (but biased) DerSimonian-Laird estimator can also be used by setting
    `paule_mandel_heterogeneity: bool = False`.

    If `hksj_sampling_distribution: bool = True`, the aggregated distribution is a more conservative
    $t$-distribution, with degrees of freedom equal to $M-1$. This is especially more conservative when there
    are only a few experiments available, and can substantially increase the aggregated distribution's variance.

    Danger: Assumptions:
        - the individual experiment distributions are normally (Gaussian) distributed
        - there **is** inter-experiment heterogeneity present

    References: Read more:
        3. [Higgins, J., & Thomas, J. (Eds.). (2023). Cochrane handbook for systematic reviews of interventions.](https://training.cochrane.org/handbook/current/chapter-10#section-10-3)
        4. [Borenstein et al. (2021). Introduction to meta-analysis.](https://www.wiley.com/en-us/Introduction+to+Meta-Analysis%2C+2nd+Edition-p-9781119558354)
        5. ['Meta-analysis' on Wikipedia](https://en.wikipedia.org/wiki/Meta-analysis#Statistical_models_for_aggregate_data)
        4. [IntHout, J., Ioannidis, J. P., & Borm, G. F. (2014). The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method.](https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-14-25)
        5. [Langan et al. (2019). A comparison of heterogeneity variance estimators in simulated random‐effects meta‐analyses.](https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1316?casa_token=NcK51p09KsYAAAAA%3A_ZkOpRymLWcDTOK5uv6UCJah6MLuEZ430pJJAENiRq2HF9_K4AlGQqhJ7_akJUig5DxkoiKec1Hdp60)

    Args:
        paule_mandel_heterogeneity (bool): whether to use the Paule-Mandel method for estimating inter-experiment heterogeneity, or fallback to the DerSimonian-Laird estimator. Defaults to True.
        hksj_sampling_distribution (bool): whether to use the Hartung-Knapp-Sidik-Jonkman corrected $t$-distribition as the aggregate sampling distribution. Defaults to False.
    """

    full_name = "Random-effects Gaussian meta-analytical experiment aggregator"
    aliases = ["re", "random_effect", "re_gaussian", "re_normal"]

    def __init__(
        self,
        rng: RNG,
        paule_mandel_heterogeneity: bool = True,
        hksj_sampling_distribution: bool = False,
    ) -> None:
        super().__init__(rng=rng)
        self.paule_mandel_heterogeneity = paule_mandel_heterogeneity
        self.hksj_sampling_distribution = hksj_sampling_distribution

    def aggregate(
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Estimate the means and variances for each distribution
        means = np.mean(experiment_samples, axis=0)
        variances = np.var(experiment_samples, axis=0, ddof=1)

        # Estimate the between-experiment variance
        tau2 = heterogeneity_DL(means, variances)
        if self.paule_mandel_heterogeneity:
            tau2 = heterogeneity_PM(
                means,
                variances,
                init_tau2=tau2,
                maxiter=100,
                # This is still *very* new, experimental
                # Leave False for now
                use_viechtbauer_correction=False,
            )

        weights = 1 / (variances + tau2)

        agg_variance = 1 / np.sum(weights)
        agg_mean = np.sum(weights * means) / np.sum(weights)

        if self.hksj_sampling_distribution:
            # Uses t-distrbution instead of normal distribution
            # More conservative for small number of studies
            q = np.sum(weights * np.power(means - agg_mean, 2)) / (num_experiments - 1)

            # HKSJ factor with correction
            # Strictly more conservative than FE model
            # hksj_factor = np.sqrt(q)
            hksj_factor = max(1.0, np.sqrt(q))

            aggregated_distribution = scipy.stats.t(
                df=num_experiments - 1,
                loc=agg_mean,
                scale=hksj_factor * np.sqrt(agg_variance),
            )

        else:
            # Uses Gaussian distrbution
            aggregated_distribution = scipy.stats.norm(
                loc=agg_mean,
                scale=np.sqrt(agg_variance),
            )

        aggregated_distribution_samples = truncated_sample(
            sampling_distribution=aggregated_distribution,
            bounds=bounds,
            rng=self.rng,
            num_samples=num_samples,
        )

        return aggregated_distribution_samples
