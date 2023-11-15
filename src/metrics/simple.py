import typing
import warnings

import numpy as np
import jaxtyping as jtyping

from src.confusion_matrix import ConfusionMatrixSamples
from src.metrics.registration import register_simple_metric


@register_simple_metric(
    identifier="p_condition",
    full_name="Prevalence",
    range=(0.0, 1.0),
)
def compute_p_condition(
    samples: ConfusionMatrixSamples,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the marginal probability distribution of conditions, i.e. prevalence.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    p_condition = samples.p_condition

    return p_condition


@register_simple_metric(
    identifier="p_pred",
    full_name="Marginal Distribution of Predictions",
    range=(0.0, 1.0),
)
def compute_p_pred(
    samples: ConfusionMatrixSamples,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the marginal probability distribution of predictions.

    The probability that the model predicts a class
        `(TP + FN) / N`
    where TP are the true positives, FN are the falsely predicted negatives, and N are the total number of predictions.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    p_pred = samples.norm_confusion_matrix.sum(axis=1)

    if (p_pred == 0).any():
        warnings.warn("Simulated model neglects class, `p_pred' contains 0.")

    return p_pred


@register_simple_metric(
    identifier="p_condition_given_pred",
    full_name="Conditional Distribution of Conditions",
    range=(0.0, 1.0),
)
def compute_p_condition_given_pred(
    samples: ConfusionMatrixSamples,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
    """Computes the marginal probability distribution of predictions.

    The probability that the model predicts a class
        `(TP + FN) / N`
    where TP are the true positives, FN are the falsely predicted negatives, and N are the total number of predictions.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501
    if p_pred is None:
        p_pred = compute_p_pred(samples)

    p_condition_given_pred = samples.norm_confusion_matrix / p_pred[:, np.newaxis, :]

    return p_condition_given_pred


@register_simple_metric(
    identifier="diag_mass",
    full_name="Diagonal of Normalized Confusion Matrix",
    range=(0.0, 1.0),
)
def compute_diag_mass(
    samples: ConfusionMatrixSamples,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the mass on the diagonal of the normalized confusion matrix.

    The rate of true positives to *all* entries:
        `TP / N`
    where TP are the true positives, and N are the total number of predictions.

    Not to be confused with the True Positive Rate.

    Returns:
        np.ndarray [num_samples, num_classes]
    """

    # The mass on the diagonal (TP)
    diag_mass = np.diagonal(
        a=samples.norm_confusion_matrix,
        axis1=1,
        axis2=2,
    )

    return diag_mass


@register_simple_metric(
    identifier="tpr",
    full_name="True Positive Rate",
    range=(0.0, 1.0),
)
def compute_true_positive_rate(
    samples: ConfusionMatrixSamples,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the True Positive Rate, i.e. recall, sensitivity.

    The ratio of true positives to condition positives:
        `TP / (TP + FN)`
    where TP are the true positives, and FN are the false negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    true_positive_rate = np.diagonal(
        a=samples.p_pred_given_condition,
        axis1=1,
        axis2=2,
    )

    return true_positive_rate


@register_simple_metric(
    identifier="fnr",
    full_name="False Negative Rate",
    range=(0.0, 1.0),
)
def compute_false_negative_rate(
    samples: ConfusionMatrixSamples,
    true_positive_rate: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the False Negative Rate, i.e. the miss-rate

    The ratio of false negatives to condition positives:
        `FN / (TP + FN)`
    where TP are the true positives, and FN are the false negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    if true_positive_rate is None:
        true_positive_rate = compute_true_positive_rate(samples)

    false_negative_rate = 1 - true_positive_rate

    return false_negative_rate


@register_simple_metric(
    identifier="ppv",
    full_name="Positive Predictive Value",
    range=(0.0, 1.0),
)
def compute_positive_predictive_value(
    samples: ConfusionMatrixSamples,
    p_condition_given_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the Positive Predictive Value, i.e. precision.

    The ratio of true positives to predicted positives:
        `TP / (TP + FP)`
    where TP are the true positives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    if p_condition_given_pred is None:
        p_condition_given_pred = compute_p_condition_given_pred(samples, p_pred=p_pred)

    positive_predictive_value = np.diagonal(
        a=p_condition_given_pred,
        axis1=1,
        axis2=2,
    )

    return positive_predictive_value


@register_simple_metric(
    identifier="fdr",
    full_name="False Discovery Rate",
    range=(0.0, 1.0),
)
def compute_false_discovery_rate(
    samples: ConfusionMatrixSamples,
    positive_predictive_value: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    p_condition_given_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the False Discovery Rate.

    The ratio of falsely predicted positives to predicted positives:
        `FP / (TP + FP)`
    where TP are the true positives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    if positive_predictive_value is None:
        positive_predictive_value = compute_positive_predictive_value(
            samples, p_condition_given_pred=p_condition_given_pred, p_pred=p_pred
        )

    false_discovery_rate = 1 - positive_predictive_value

    return false_discovery_rate


@register_simple_metric(
    identifier="fpr",
    full_name="False Positiv Rate",
    range=(0.0, 1.0),
)
def compute_false_positive_rate(
    samples: ConfusionMatrixSamples,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    diag_mass: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the False Positive Rate, the probability of false alarm.

    The ratio of falsely predicted positives to condition negatives:
        `FP / (TN + FP)`
    where TN are the true negatives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    # This requires reasoning about true negatives in a multi-class setting
    # This is somewhat involved, hence the unintuitive formula
    if diag_mass is None:
        diag_mass = compute_diag_mass(samples)

    if p_pred is None:
        p_pred = compute_p_pred(samples)

    false_positive_rate = (p_pred - diag_mass) / (1 - samples.p_condition)

    return false_positive_rate


@register_simple_metric(
    identifier="tnr",
    full_name="True Negative Rate",
    range=(0.0, 1.0),
)
def compute_true_negative_rate(
    samples: ConfusionMatrixSamples,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    diag_mass: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    false_positive_rate: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the True Negative Rate, i.e. specificity, selectivity.

    The ratio of true predicted negatives to condition negatives:
        `TN / (TN + FP)`
    where TN are the true negatives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    # Much easier to compute this from the False Positive Rate
    if false_positive_rate is None:
        false_positive_rate = compute_false_positive_rate(
            samples, p_pred=p_pred, diag_mass=diag_mass
        )

    true_negative_rate = 1 - false_positive_rate

    return true_negative_rate


@register_simple_metric(
    identifier="for",
    full_name="False Omission Rate",
    range=(0.0, 1.0),
)
def compute_false_omission_rate(
    samples: ConfusionMatrixSamples,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    diag_mass: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the False Omission Rate.

    The ratio of falsely predicted negatives to predicted negatives:
        `FN / (TN + FN)`
    where TN are the true negatives, and FN are the falsely predicted negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    # This requires reasoning about true negatives in a multi-class setting
    # This is somewhat involved, hence the unintuitive formula
    if p_pred is None:
        p_pred = compute_p_pred(samples)

    if diag_mass is None:
        diag_mass = compute_diag_mass(samples)

    false_omission_rate = (samples.p_condition - diag_mass) / (1 - p_pred)

    return false_omission_rate


@register_simple_metric(
    identifier="npv",
    full_name="Negative Predictive Value",
    range=(0.0, 1.0),
)
def compute_negative_predictive_value(
    samples: ConfusionMatrixSamples,
    p_pred: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    diag_mass: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
    false_omission_rate: typing.Optional[
        jtyping.Float[np.ndarray, " num_samples num_classes"]
    ] = None,
) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
    """Computes the Negative Predicitive Value.

    The ratio of true negatives to predicted negatives:
        `TN / (TN + FN)`
    where TN are the true negatives, and FN are the falsely predicted negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """
    # Much easier to compute this from the False Omission Rate
    if false_omission_rate is None:
        false_omission_rate = compute_false_omission_rate(
            samples, p_pred=p_pred, diag_mass=diag_mass
        )

    negative_predictive_value = 1 - false_omission_rate

    return negative_predictive_value
