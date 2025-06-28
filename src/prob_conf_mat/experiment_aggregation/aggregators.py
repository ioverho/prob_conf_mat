from __future__ import annotations
import typing
import warnings

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.utils import RNG

import numpy as np
import scipy
import scipy.stats

from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from prob_conf_mat.experiment_aggregation.heterogeneity import (
    heterogeneity_dl,
    heterogeneity_pm,
)
from prob_conf_mat.stats import truncated_sample

__all__ = [
    "SingletonAggregator",
    "BetaAggregator",
    "GammaAggregator",
    "FEGaussianAggregator",
    "REGaussianAggregator",
    "HistogramAggregator",
]


class SingletonAggregator(ExperimentAggregator):
    """An aggregation to apply to an ExperimentGroup that needs no aggregation.

    For example, the ExperimentGroup only contains one Experiment.

    Essentially just the [identity function](https://en.wikipedia.org/wiki/Identity_function):

    $$f(x)=x$$

    """

    full_name = "Singleton experiment aggregation"
    aliases = ["singleton", "identity"]

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        if num_experiments > 1:
            raise ValueError(
                f"Parameter `num_experiments` > 1. Currently {num_experiments}",
            )

        return experiment_samples[:, 0]


class BetaAggregator(ExperimentAggregator):
    r"""Samples from the beta-conflated distribution.

    Specifically, the aggregate distribution $\text{Beta}(\tilde{\alpha}, \tilde{\beta})$ is
    estimated as:

    $$\begin{aligned}
        \tilde{\alpha}&=\left[\sum_{i=1}^{M}\alpha_{i}\right]-\left(M-1\right) \\
        \tilde{\beta}&=\left[\sum_{i=1}^{M}\beta_{i}\right]-\left(M-1\right)
    \end{aligned}$$

    where $M$ is the total number of experiments.

    Uses [`scipy.stats.beta`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)
    class to fit beta-distributions.

    Danger: Assumptions:
        - the individual experiment distributions are beta distributed
        - the metrics are bounded, although the range need not be (0, 1)

    References: Read more:
        1. [Hill, T. P. (2008). Conflations Of Probability Distributions: An Optimal Method For
            Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T. P., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)
        3. ['Beta distribution' on Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)

    Args:
        estimation_method (str): method for estimating the parameters of the individual
            experiment distributions. Options are 'mle' for maximum-likelihood estimation, or 'mome'
            for the method of moments estimator. MLE tends be more efficient but is difficult
            to estimate

    """

    full_name = "Beta conflated experiment aggregation"
    aliases = ["beta", "beta_conflation"]

    def __init__(self, rng: RNG, *, estimation_method: str = "mle") -> None:
        super().__init__(rng=rng)

        # Honestly should get rid of this
        # MLE is more efficient than MoME, and difference is small
        self.estimation_method = estimation_method

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        if bounds[0] == -float("inf") or bounds[1] == float("inf"):
            raise NotImplementedError(
                "Beta aggregation does not (yet) support metrics with infite bounds.",
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
        conflated_distribution_samples: jtyping.Float[np.ndarray, " num_samples"] = (
            scipy.stats.beta.rvs(
                a=conflated_alpha,
                b=conflated_beta,
                size=num_samples,
                loc=bounds[0],
                scale=bounds[1] - bounds[0],
                random_state=self.rng.rng,
            )
        )  # type: ignore

        # conflated_distribution_samples = (
        #    bounds[1] - bounds[0]
        # ) * conflated_distribution_samples + bounds[0]

        return conflated_distribution_samples


class GammaAggregator(ExperimentAggregator):
    r"""Samples from the Gamma-conflated distribution.

    Specifically, the aggregate distribution $\\text{Gamma}(\\tilde{\\alpha}, \\tilde{\\beta})$
    ($\\alpha$ is the shape, $\\beta$ the rate parameter) is estimated as:

    $$\\begin{aligned}
        \\tilde{\\alpha}&=\\left[\\sum_{i}^{M}\\alpha_{i}\\right]-(M-1) \\\\
        \\tilde{\\beta}&=\\dfrac{1}{\\sum_{i}^{M}\\beta_{i}^{-1}}
    \\end{aligned}$$

    where $M$ is the total number of experiments.

    An optional `shifted: bool` argument exists to dynamically estimate the support for the
    distribution. Can help fit to individual experiments, but likely minimally impacts the
    aggregate distribution.

    Danger: Assumptions:
        - the individual experiment distributions are gamma distributed

    References: Read more:
        1. [Hill, T. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)
        3. ['Gamma distribution' on Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution)

    """  # noqa: E501

    full_name = "Gamma conflated experiment aggregator"
    aliases = ["gamma", "gamma_conflation"]

    def __init__(self, rng: RNG, *, shifted: bool = False) -> None:
        super().__init__(rng=rng)

        self.shifted = shifted

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        if bounds[0] < 0 or bounds[1] < 0:
            raise ValueError(
                "Gamma aggregator does not support metrics with negative bounds.",
            )
        num_samples, num_experiments = experiment_samples.shape

        # Estimate the 'loc' variable, i.e. the minimum of the support
        # Improves fit to individual experiment distributions
        # Minimal impact on the conflated distribution
        loc_estimate = np.min(experiment_samples) - 1e-09 if self.shifted else bounds[0]

        # Estimate the shape and rate for each distribution
        alphas = []
        betas = []
        for per_experiment_samples in experiment_samples.T:
            finite_samples = per_experiment_samples[np.isfinite(per_experiment_samples)]

            if finite_samples.shape[0] < 0.9 * per_experiment_samples.shape[0]:
                warnings.warn(
                    "An experiment sample has more than 10% non-finite values.",
                )

            alpha, _, beta = scipy.stats.gamma.fit(
                finite_samples,
                floc=loc_estimate,
            )

            alphas.append(alpha)
            betas.append(beta)

        alphas = np.array(alphas)
        betas = np.array(betas)

        # Estimate the parameters of the conflated distribution
        conflated_alpha = np.sum(alphas) - (num_experiments - 1)
        conflated_beta = 1 / np.sum(1 / betas)

        # Redefine the sampling distribution
        conflated_distribution = scipy.stats.gamma(
            a=conflated_alpha,
            scale=conflated_beta,
            loc=loc_estimate,
        )

        # Sample from the distribution, truncating at the bounds of the metric
        conflated_distribution_samples = truncated_sample(
            sampling_distribution=conflated_distribution,
            bounds=(loc_estimate, bounds[1]),
            rng=self.rng,
            num_samples=num_samples,
        )

        return conflated_distribution_samples


class FEGaussianAggregator(ExperimentAggregator):
    r"""Samples from the Gaussian-conflated distribution.

    This is equivalent to the fixed-effects meta-analytical estimator.

    Uses the inverse variance weighted mean and standard errors. Specifically, the aggregate
    distribution $\\mathcal{N}(\\tilde{\\mu}, \\tilde{\\sigma})$ is estimated as:

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

    """  # noqa: E501

    full_name = "Fixed-effect Gaussian meta-analytical experiment aggregator"
    aliases = ["fe", "fixed_effect", "fe_gaussian", "gaussian", "normal", "fe_normal"]

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Estimate the means and variances for each distribution
        means = np.mean(experiment_samples, axis=0)
        variances = np.var(experiment_samples, axis=0, ddof=1)

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

        return conflated_distribution_samples


class REGaussianAggregator(ExperimentAggregator):
    r"""Samples from the Random Effects Meta-Analytical Estimator.

    First uses the standard the inverse variance weighted mean and standard errors as model
    parameters, before debiasing the weights to incorporate inter-experiment heterogeneity.
    As a result, studies with larger standard errors will be upweighted relative to the
    fixed-effects model.

    Specifically, starting with a Fixed-Effects model
    $\\mathcal{N}(\\tilde{\\mu_{\\text{FE}}}, \\tilde{\\sigma_{\\text{FE}}})$,

    $$\\begin{aligned}
        w_{i}&=\\dfrac{\\left(\\sigma_{i}^2+\\tau^2\\right)^{-1}}{\\sum_{j}^{M}\\left(\\sigma_{j}^2+\\tau^2\\right)^{-1}} \\\\
        \\tilde{\\mu}&=\\sum_{i}^{M}w_{i}\\mu_{i} \\\\
        \\tilde{\\sigma^2}&=\\dfrac{1}{\\sum_{i}^{M}\\sigma_{i}^{-2}}
    \\end{aligned}$$

    where $\\tau$ is the estimated inter-experiment heterogeneity, and $M$ is the total number
    of experiments.

    Uses the Paule-Mandel iterative heterogeneity estimator, which does not make a parametric
    assumption. The more common (but biased) DerSimonian-Laird estimator can also be used by setting
    `paule_mandel_heterogeneity: bool = False`.

    If `hksj_sampling_distribution: bool = True`, the aggregated distribution is a more conservative
    $t$-distribution, with degrees of freedom equal to $M-1$. This is especially more conservative
    when there are only a few experiments available, and can substantially increase the aggregated
    distribution's variance.

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
        paule_mandel_heterogeneity (bool): whether to use the Paule-Mandel method for estimating
            inter-experiment heterogeneity, or fallback to the DerSimonian-Laird estimator.
            Defaults to True.
        hksj_sampling_distribution (bool): whether to use the Hartung-Knapp-Sidik-Jonkman corrected
            $t$-distribition as the aggregate sampling distribution.
            Defaults to False.
    """  # noqa: E501

    full_name = "Random-effects Gaussian meta-analytical experiment aggregator"
    aliases = ["re", "random_effect", "re_gaussian", "re_normal"]

    def __init__(
        self,
        rng: RNG,
        *,
        paule_mandel_heterogeneity: bool = True,
        hksj_sampling_distribution: bool = False,
    ) -> None:
        super().__init__(rng=rng)
        self.paule_mandel_heterogeneity = paule_mandel_heterogeneity
        self.hksj_sampling_distribution = hksj_sampling_distribution

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Estimate the means and variances for each distribution
        means = np.mean(experiment_samples, axis=0)
        variances = np.var(experiment_samples, axis=0, ddof=1)

        # Estimate the between-experiment variance
        tau2 = heterogeneity_dl(means, variances)
        if self.paule_mandel_heterogeneity:
            tau2 = heterogeneity_pm(
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


class HistogramAggregator(ExperimentAggregator):
    r"""Samples from a histogram approximate conflation distribution.

    First bins all individual experiment groups, and then computes the product of the probability
    masses across individual experiments.

    Unlike other methods, this does not make a parametric assumption. However, the resulting
    distribution can 'look' unnatural, and requires overlapping supports within the sample.
    If any experiment assigns 0 probability mass to any bin, the conflated bin will also
    contain 0 probability mass.

    As such, inter-experiment heterogeneity can be a significant problem.

    Uses [numpy.histogram_bin_edges](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html)
    to estimate the number of bin edges needed per experiment, and takes the smallest across all
    experiments for the aggregate distribution.

    Danger: Assumptions:
        - the individual experiment distributions' supports overlap

    References: Read more:
        1. [Hill, T. (2008). Conflations Of Probability Distributions: An Optimal Method For Consolidating Data From Different Experiments.](http://arxiv.org/abs/0808.1808)
        2. [Hill, T., & Miller, J. (2011). How to combine independent data sets for the same quantity.](https://arxiv.org/abs/1005.4978)

    """  # noqa: E501

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

    def aggregate(  # noqa: D102
        self,
        experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
        bounds: tuple[float, float],
    ) -> jtyping.Float[np.ndarray, " num_samples"]:
        num_samples, num_experiments = experiment_samples.shape

        # Find the smallest recommended bin width for all experiments
        min_bin_width = float("inf")
        for per_experiment_samples in experiment_samples.T:
            distribution_bins = np.histogram_bin_edges(
                per_experiment_samples,
                bins="auto",
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
                num_samples + smoothing_coeff * num_bins,
            )

            conflated_distribution += log_p_hat

        conflated_distribution = np.exp(
            conflated_distribution - np.logaddexp.reduce(conflated_distribution),
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
            low=-min_bin_width / 2,
            high=min_bin_width / 2,
            size=num_samples,
        )

        conflated_distribution_samples = np.clip(
            conflated_distribution_samples + bin_noise,
            a_min=bounds[0],
            a_max=bounds[1],
        )

        return conflated_distribution_samples
