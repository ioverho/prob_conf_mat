"""
This contains relatively simple metrics, usually only used as intermediate variables.

The simple/complex dichotomy is pretty arbitrary though
"""


import jaxtyping as jtyping
import numpy as np

from bayes_conf_mat.metrics.base import Metric


class Prevalence(Metric):
    """Computes the marginal distribution of condition occurence, i.e. prevalence.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Marginal Distribution of Condition"
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("p_condition",)
    sklearn_equivalent = None
    aliases = ["prevalence"]

    def compute_metric(
        self,
        p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        return p_condition


class ModelBias(Metric):
    """Computes the marginal probability distribution of predictions.

    The probability that the model predicts a class
        `(TP + FN) / N`
    where TP are the true positives, FN are the falsely predicted negatives, and N are the total number of predictions.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Marginal Distribution of Predictions"
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("p_pred",)
    sklearn_equivalent = None
    aliases = ["model_bias"]

    def compute_metric(
        self,
        p_pred: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        # TODO: check confusion matrix before metric computation
        # if (p_pred == 0).any():
        #    warnings.warn("Simulated model neglects class, `p_pred' contains 0.")

        return p_pred


class DiagMass(Metric):
    """Computes the mass on the diagonal of the normalized confusion matrix.

    The rate of true positives to *all* entries:
        `TP / N`
    where TP are the true positives, and N are the total number of predictions.

    Not to be confused with the True Positive Rate.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Diagonal of Normalized Confusion Matrix"
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = None
    aliases = ["diag_mass"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        diag_mass = np.diagonal(
            a=norm_confusion_matrix,
            axis1=1,
            axis2=2,
        )

        return diag_mass


class TruePositiveRate(Metric):
    """Computes the True Positive Rate, i.e. recall, sensitivity.

    The ratio of true positives to condition positives:
        `TP / (TP + FN)`
    where TP are the true positives, and FN are the false negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "True Positive Rate"
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("p_pred_given_condition",)
    sklearn_equivalent = None
    aliases = ["true_positive_rate", "sensitivity", "recall", "hit_rate", "tpr"]

    def compute_metric(
        self,
        p_pred_given_condition: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        true_positive_rate = np.diagonal(
            a=p_pred_given_condition,
            axis1=1,
            axis2=2,
        )

        return true_positive_rate


class FalseNegativeRate(Metric):
    """Computes the False Negative Rate, i.e. miss-rate

    The ratio of false negatives to condition positives:
        `FN / (TP + FN)`
    where TP are the true positives, and FN are the false negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "False Negative Rate"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("true_positive_rate",)
    sklearn_equivalent = None
    aliases = ["false_negative_rate", "miss_rate", "fnr"]

    def compute_metric(
        self,
        true_positive_rate: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        false_negative_rate = 1 - true_positive_rate

        return false_negative_rate


class PositivePredictiveValue(Metric):
    """Computes the Positive Predictive Value, i.e. precision.

    The ratio of true positives to predicted positives:
        `TP / (TP + FP)`
    where TP are the true positives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Positive Predictive Value"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("p_condition_given_pred",)
    sklearn_equivalent = None
    aliases = ["positive_predictive_value", "precision", "ppv"]

    def compute_metric(
        self,
        p_condition_given_pred: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        positive_predictive_value = np.diagonal(
            a=p_condition_given_pred,
            axis1=1,
            axis2=2,
        )

        return positive_predictive_value


class FalseDiscoveryRate(Metric):
    """Computes the False Discovery Rate.

    The ratio of falsely predicted positives to predicted positives:
        `FP / (TP + FP)`
    where TP are the true positives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "False Discovery Rate"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("positive_predictive_value",)
    sklearn_equivalent = None
    aliases = ["false_discovery_rate", "fdr"]

    def compute_metric(
        self,
        positive_predictive_value: jtyping.Float[
            np.ndarray, " num_samples num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        false_discovery_rate = 1 - positive_predictive_value

        return false_discovery_rate


class FalsePositiveRate(Metric):
    """Computes the False Positive Rate, the probability of false alarm.

    The ratio of falsely predicted positives to condition negatives:
        `FP / (TN + FP)`
    where TN are the true negatives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "False Positive Rate"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("diag_mass", "p_pred", "p_condition")
    sklearn_equivalent = None
    aliases = ["false_positive_rate", "fall-out", "fall_out", "fpr"]

    def compute_metric(
        self,
        diag_mass: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        false_positive_rate = (p_pred - diag_mass) / (1 - p_condition)

        return false_positive_rate


class TrueNegativeRate(Metric):
    """Computes the True Negative Rate, i.e. specificity, selectivity.

    The ratio of true predicted negatives to condition negatives:
        `TN / (TN + FP)`
    where TN are the true negatives, and FP are the falsely predicted positives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "True Negative Rate"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("false_positive_rate",)
    sklearn_equivalent = None
    aliases = ["true_negative_rate", "specificity", "selectivity", "tnr"]

    def compute_metric(
        self,
        false_positive_rate: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        true_negative_rate = 1 - false_positive_rate

        return true_negative_rate


class FalseOmissionRate(Metric):
    """Computes the False Omission Rate.

    The ratio of falsely predicted negatives to predicted negatives:
        `FN / (TN + FN)`
    where TN are the true negatives, and FN are the falsely predicted negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "False Omission Rate"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("p_condition", "p_pred", "diag_mass")
    sklearn_equivalent = None
    aliases = ["false_omission_rate", "for"]

    def compute_metric(
        self,
        p_condition: jtyping.Float[np.ndarray, " num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, " num_samples num_classes"],
        diag_mass: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        # This requires reasoning about true negatives in a multi-class setting
        # This is somewhat involved, hence the unintuitive formula
        false_omission_rate = (p_condition - diag_mass) / (1 - p_pred)

        return false_omission_rate


class NegativePredictiveValue(Metric):
    """Computes the Negative Predicitive Value.

    The ratio of true negatives to predicted negatives:
        `TN / (TN + FN)`
    where TN are the true negatives, and FN are the falsely predicted negatives.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Negative Predictive Value"
    is_complex = True
    is_multiclass = False
    range = (0.0, 1.0)
    dependencies = ("false_omission_rate",)
    sklearn_equivalent = None
    aliases = ["negative_predictive_value", "npv"]

    def compute_metric(
        self,
        false_omission_rate: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        # This requires reasoning about true negatives in a multi-class setting
        # This is somewhat involved, hence the unintuitive formula
        negative_predictive_value = 1 - false_omission_rate

        return negative_predictive_value
