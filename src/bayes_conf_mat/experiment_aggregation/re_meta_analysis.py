import typing

import jaxtyping as jtyping
import numpy as np
import scipy

from bayes_conf_mat.math.truncated_sampling import truncated_sample
from bayes_conf_mat.experiment_aggregation.heterogeneity import heterogeneity_DL, heterogeneity_PM

def re_meta_analysis(
    distribution_samples: typing.List[jtyping.Float[np.ndarray, " num_samples"]],
    range: typing.Tuple[int],
    rng: np.random.BitGenerator,
    use_paule_mandel_heterogeneity_estimate: bool = True,
    use_viechtbauer_correction: bool = True,
    use_hksj_sampling_distribution: bool = True,
):
    num_experiments, num_samples = distribution_samples.shape

    means = np.mean(distribution_samples, axis=1)
    variances = np.var(distribution_samples, axis=1, ddof=1)

    tau2 = heterogeneity_DL(means, variances)
    if use_paule_mandel_heterogeneity_estimate:
        tau2 = heterogeneity_PM(
            means,
            variances,
            init_tau2=tau2,
            maxiter=100,
            use_viechtbauer_correction=use_viechtbauer_correction,
        )

    weights = 1 / (variances + tau2)

    agg_variance = 1 / np.sum(weights)
    agg_mean = np.sum(weights * means) / np.sum(weights)

    if use_hksj_sampling_distribution:
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

        aggregated_distribution_samples = truncated_sample(
            sampling_distribution=aggregated_distribution,
            range=range,
            rng=rng,
            num_samples=num_samples,
        )

    else:
        aggregated_distribution = scipy.stats.norm(
            loc=agg_mean,
            scale=np.sqrt(agg_variance),
        )

        aggregated_distribution_samples = truncated_sample(
            sampling_distribution=aggregated_distribution,
            range=range,
            rng=rng,
            num_samples=num_samples,
        )

    return aggregated_distribution_samples
