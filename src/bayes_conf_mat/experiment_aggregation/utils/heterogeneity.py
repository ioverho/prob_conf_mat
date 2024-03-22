from dataclasses import dataclass

import jaxtyping as jtyping
import numpy as np
import scipy


def heterogeneity_DL(
    means,
    variances,
):
    """Compute the DerSimonian-Laird estimate of between-experiment heterogeneity.

    Args:
        means (jtyping.Float[np.ndarray, " num_experiments"]): the experiment means
        variances (jtyping.Float[np.ndarray, " num_experiments"]): the experiment variances

    Returns:
        float: estimate of the between-experiment heterogeneity
    """  # noqa: E501

    num_studies = means.shape[0]

    w_fe = 1 / variances

    mu_fe = np.sum(w_fe * means) / np.sum(w_fe)

    q = np.sum(w_fe * np.power(means - mu_fe, 2))
    df = num_studies - 1
    denom = np.sum(w_fe) - (np.sum(np.power(w_fe, 2)) / np.sum(w_fe))
    tau2 = max(0.0, (q - df) / denom)

    return tau2


def heterogeneity_PM(
    means: jtyping.Float[np.ndarray, " num_experiments"],
    variances: jtyping.Float[np.ndarray, " num_experiments"],
    init_tau2: float = 0.0,
    atol: float = 1e-5,
    maxiter: int = 50,
    use_viechtbauer_correction: bool = False,
):
    """Compute the Paule-Mandel estimate of between-experiment heterogeneity.

    Based on the `_fit_tau_iterative` function from `stats_models`.

    Original implementation is based on Appendix A of
        'Random-effects model for meta-analysis of clinical trials: An update' (DerSimonian and Kacker, 2007).

    We make two modifications:
        1. instead of stopping iteration if F(tau_2) < 0, we back-off to the midpoint between the current and previous estimate
        2. optionally, we apply the Viechtbauer correction to the root. Instead of converging to the mean, converge to the median

    Args:
        means (jtyping.Float[np.ndarray, " num_experiments"]): the experiment means
        variances (jtyping.Float[np.ndarray, " num_experiments"]): the experiment variances
        init_tau2 (float, optional): the inital tau2 estimate. Defaults to 0.0.
        atol (float, optional): when to assume convergence. Defaults to 1e-5.
        maxiter (int, optional): the maximum number of iterations needed. Defaults to 50.
        use_viechtbauer_correction (bool, optional): whether to use the Viechtbauer correction. Defaults to True.

    Returns:
        float: estimate of the between-experiment heterogeneity
    """  # noqa: E501

    prev_tau2 = 0.0
    tau2 = init_tau2
    num_studies = means.shape[0]
    patience = maxiter

    if use_viechtbauer_correction:
        root = scipy.stats.chi2(df=num_studies - 1).median()
    else:
        root = num_studies - 1

    while patience > 0:
        # Estimate RE summary stat
        w_PM = 1 / (variances + tau2)
        mu_PM = np.sum(w_PM * means) / np.sum(w_PM)

        # Compute generalised Q values
        # i.e. the residual squares
        q_PM = np.sum(w_PM * np.power(means - mu_PM, 2))

        # Check if residual has converged
        F_tau2 = q_PM - root
        if F_tau2 < 0:
            # If negative, reduce tau2 estimate
            # and try again
            # This is different from other implementations
            tau2 = (prev_tau2 + tau2) / 2

            # Allow for an extra iteration for the correction to take effect
            patience -= 1
            continue
            # break

        if np.allclose(F_tau2, 0.0, atol=atol):
            # Check convergence
            break

        # Otherwise, update tau2
        delta_denom = np.sum(np.power(w_PM, 2) * np.power(means - mu_PM, 2))

        prev_tau2 = tau2
        tau2 += F_tau2 / delta_denom

        patience -= 1

    if tau2 < 0.0:
        tau2 = 0

    return tau2


@dataclass(frozen=True)
class HeterogeneityResult:
    i2: float
    within_experiment_variance: float
    between_experiment_variance: float
    i2_interpretation: str


def estimate_i2(
    individual_samples: jtyping.Float[np.ndarray, " num_classes num_samples"],
) -> float:
    """Estimates a generalised I^2 metric, as suggested by Bowden et al. [1], using a Paule-Mandel tau2 estimator.

    Measures the amopunt of variance attributable to within-experiment variation vs. between-experiment variation.

    References:
    [1] Bowden, J., Tierney, J. F., Copas, A. J., & Burdett, S. (2011). Quantifying, displaying and accounting for heterogeneity in the meta-analysis of RCTs using standard and generalised Qstatistics. BMC medical research methodology, 11(1), 1-12.

    Args:
        individual_samples (jtyping.Float[np.ndarray, " num_classes num_samples"])

    Returns:
        float: the I^2 estimate
    """  # noqa: E501
    num_samples = [samples.shape[0] for samples in individual_samples]
    means = np.mean(individual_samples, axis=1)
    variances = np.var(individual_samples, axis=1)

    # Pooled intra-experiment variance
    pooled_var = sum((n - 1) * var for n, var in zip(num_samples, variances)) / (
        sum(num_samples) - len(num_samples)
    )

    # Estimate of inter-experiment variance
    tau2_dl = heterogeneity_DL(means, variances)
    tau2_pm = heterogeneity_PM(
        means, variances, init_tau2=tau2_dl, use_viechtbauer_correction=False
    )

    # Proportion of variance attributable to inter-experiment variance
    i2 = tau2_pm / (tau2_pm + pooled_var)

    result = HeterogeneityResult(
        i2=i2,
        within_experiment_variance=pooled_var,
        between_experiment_variance=tau2_pm,
        i2_interpretation=interpret_i2(i2),
    )

    return result


def interpret_i2(i2_score: float):
    """'Interprets I^2 values using the guideline prescribed by the Cochrane Handbook [1]

    References:
    [1] Higgins, J. P., & Green, S. (Eds.). (2008). Cochrane handbook for systematic reviews of interventions.

    Args:
        i2_score (float): the I2 estimate

    Returns:
        str: a rough interpretation of the magnitude of I2
    """  # noqa: E501
    if i2_score < 0.0 or i2_score > 1.0:
        raise ValueError(
            "I^2 should be in the range (0.0, 1.0). Currently out-of-bounds."
        )
    elif i2_score < 0.3:
        het_sig = "insignificant"
    elif i2_score >= 0.3 and i2_score < 0.4:
        het_sig = "borderline moderate"
    elif i2_score >= 0.4 and i2_score < 0.5:
        het_sig = "moderate"
    elif i2_score >= 0.5 and i2_score < 0.6:
        het_sig = "borderline substantial"
    elif i2_score >= 0.6 and i2_score < 0.75:
        het_sig = "borderline considerable"
    elif i2_score >= 0.75 and i2_score <= 1.0:
        het_sig = "considerable"

    het_sig += " heterogeneity"

    return het_sig
