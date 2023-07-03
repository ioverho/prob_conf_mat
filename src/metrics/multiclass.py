import typing

import numpy as np
import jaxtyping as jtyping

from confusion_matrix import ConfusionMatrixSamples


def compute_accuracy(
    conf_mats: ConfusionMatrixSamples,
):
    return np.sum(conf_mats.diag_mass, axis=1)


def compute_balanced_accuracy(conf_mats: ConfusionMatrixSamples):
    """
    Computes the multi-class balanced accuracy, as used by scikit-learn.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

    For more information, see: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """
    balanced_accuracy = np.nanmean(
        np.diagonal(
            conf_mats.p_pred_given_condition,
            axis1=1,
            axis2=2,
        ),
        axis=-1,
    )

    return balanced_accuracy


def compute_adjusted_balanced_accuracy(
    conf_mats: ConfusionMatrixSamples,
    balanced_accuracy: typing.Optional[
        jtyping.Float[jtyping.Array, "num_samples support_size"]
    ] = None,
):
    """
    Computes the chance-corrected multi-class balanced accuracy, as used by scikit-learn.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

    For more information, see: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """  # noqa: E501

    if balanced_accuracy is None:
        balanced_accuracy = compute_balanced_accuracy(conf_mats)

    chance = 1 / (conf_mats.p_condition != 0).sum(axis=1)

    adjusted_balanced_accuracy = (balanced_accuracy - chance) / (1 - chance)

    return adjusted_balanced_accuracy


def compute_cohens_kappa(
    conf_mats: ConfusionMatrixSamples,
):
    p_agreement = np.sum(conf_mats.diag_mass, axis=1)

    p_chance = np.einsum("bc, bc->b", conf_mats.p_condition, conf_mats.p_pred)

    return (p_agreement - p_chance) / (1 - p_chance)


def compute_mcc(
    conf_mats: ConfusionMatrixSamples,
):
    marginals_inner_prod = np.einsum(
        "bc, bc->b", conf_mats.p_condition, conf_mats.p_pred
    )
    numerator = np.sum(conf_mats.diag_mass, axis=1) - marginals_inner_prod

    mcc = numerator / np.sqrt(
        (1 - np.power(conf_mats.p_condition, 2).sum(axis=1))
        * (1 - np.power(conf_mats.p_pred, 2).sum(axis=1))
    )

    return mcc
