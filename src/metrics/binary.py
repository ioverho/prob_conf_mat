import typing

import numpy as np
import jaxtyping as jtyping

from confusion_matrix import ConfusionMatrixSamples
from aggregation import numpy_batched_harmonic_mean


def compute_positive_likelihood_ratio(conf_mats: ConfusionMatrixSamples):
    return conf_mats.true_positive_rate / conf_mats.false_positive_rate


def compute_negative_likelihood_ratio(conf_mats: ConfusionMatrixSamples):
    return conf_mats.false_negative_rate / conf_mats.true_negative_rate


def compute_diagnostic_odds_ratio(
    conf_mats: ConfusionMatrixSamples,
    positive_likelihood_ratio: typing.Optional[
        jtyping.Float[jtyping.Array, "num_samples support_size"]
    ] = None,
    negative_likelihood_ratio: typing.Optional[
        jtyping.Float[jtyping.Array, "num_samples support_size"]
    ] = None,
):
    if positive_likelihood_ratio is None:
        positive_likelihood_ratio = compute_positive_likelihood_ratio(conf_mats)

    if negative_likelihood_ratio is None:
        negative_likelihood_ratio = compute_negative_likelihood_ratio(conf_mats)

    return positive_likelihood_ratio / negative_likelihood_ratio


def compute_fbeta(
    conf_mats: ConfusionMatrixSamples,
    beta: typing.Optional[float] = 1,
):
    beta_2 = beta**2

    f1 = (
        (1 + beta_2)
        * (conf_mats.positive_predictive_value * conf_mats.true_positive_rate)
        / (beta_2 * conf_mats.positive_predictive_value + conf_mats.true_positive_rate)
    )

    # In case one of the ratios is nan (most likely due to 0 division), set to 0
    f1 = np.nan_to_num(
        f1,
        nan=0.0,
    )

    return f1


def compute_f1(
    conf_mats: ConfusionMatrixSamples,
):
    return compute_fbeta(conf_mats, beta=1)


def compute_informedness(
    conf_mats: ConfusionMatrixSamples,
):
    return conf_mats.true_positive_rate + conf_mats.true_negative_rate - 1


def compute_markedness(
    conf_mats: ConfusionMatrixSamples,
):
    return conf_mats.positive_predictive_value + conf_mats.negative_predictive_value - 1


def compute_p4(
    conf_mats: ConfusionMatrixSamples,
):
    values = np.stack(
        [
            conf_mats.positive_predictive_value,
            conf_mats.true_positive_rate,
            conf_mats.true_negative_rate,
            conf_mats.negative_predictive_value,
        ],
        axis=2,
    )

    return numpy_batched_harmonic_mean(values, keepdims=False)


def compute_jaccard_index(conf_mats: ConfusionMatrixSamples):
    return conf_mats.diag_mass / (
        conf_mats.p_pred + conf_mats.p_condition - conf_mats.diag_mass
    )


def compute_prevalence_threshold(conf_mats: ConfusionMatrixSamples):
    num = (
        np.sqrt(conf_mats.true_positive_rate * conf_mats.false_positive_rate)
        - conf_mats.false_positive_rate
    )
    denom = conf_mats.true_positive_rate - conf_mats.false_positive_rate

    return num / denom
