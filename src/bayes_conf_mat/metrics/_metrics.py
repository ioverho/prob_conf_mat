"""
This monolithic file contains all pre-defined metrics.

Honestly, just use the search feature when going through this.
"""

import jaxtyping as jtyping
import numpy as np

from bayes_conf_mat.stats import numpy_batched_harmonic_mean
from bayes_conf_mat.metrics.base import Metric


class Prevalence(Metric):
    """Computes the marginal distribution of condition occurence, i.e. prevalence.

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Marginal Distribution of Condition"
    is_multiclass = False
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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
    bounds = (0.0, 1.0)
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


class PositiveLikelihoodRatio(Metric):
    # TODO: write documentation

    full_name = "Positive Likelihood Ratio"
    is_multiclass = False
    bounds = (0.0, float("inf"))
    dependencies = ("tpr", "fpr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["plr", "positive_likelihood_ratio"]

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return tpr / fpr


class NegativeLikelihoodRatio(Metric):
    # TODO: write documentation

    full_name = "Negative Likelihood Ratio"
    is_multiclass = False
    bounds = (0.0, float("inf"))
    dependencies = ("fnr", "tnr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["negative_likelihood_ratio", "nlr"]

    def compute_metric(
        self,
        fnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return fnr / tnr


class DiagnosticOddsRatio(Metric):
    # TODO: write documentation

    full_name = "Negative Likelihood Ratio"
    is_multiclass = False
    bounds = (0.0, float("inf"))
    dependencies = ("nlr", "plr")
    sklearn_equivalent = None
    aliases = ["dor", "diagnostic_odds_ratio"]

    def compute_metric(
        self,
        plr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        nlr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        dor = plr / nlr

        return dor


class F1(Metric):
    # TODO: write documentation

    full_name = "F1-score"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("ppv", "tpr")
    sklearn_equivalent = "f1_score"
    aliases = ["f1"]

    def compute_metric(
        self,
        ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        f1 = 2 * (ppv * tpr) / (ppv + tpr)

        # In case one of the ratios is nan (most likely due to 0 division), set to 0
        f1 = np.nan_to_num(
            f1,
            nan=0.0,
        )

        return f1


class FBeta(Metric):
    # TODO: write documentation

    full_name = "FBeta-score"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("ppv", "tpr")
    sklearn_equivalent = "fbeta_score"
    aliases = ["fbeta"]

    def __init__(self, beta: float = 1.0):
        super().__init__()

        self.beta = beta

    def compute_metric(
        self,
        ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        beta_2 = self.beta**2

        f1 = (1 + beta_2) * (ppv * tpr) / (beta_2 * ppv + tpr)

        # In case one of the ratios is nan (most likely due to 0 division), set to 0
        f1 = np.nan_to_num(
            f1,
            nan=0.0,
        )

        return f1


class Informedness(Metric):
    # TODO: write documentation

    full_name = "Informedness"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("tpr", "tnr")
    sklearn_equivalent = None
    aliases = ["informedness", "youdenj", "youden_j", "bookmaker_informedness"]

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return tpr + tnr - 1


class Markedness(Metric):
    # TODO: write documentation

    full_name = "Markedness"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("ppv", "npv")
    sklearn_equivalent = None
    aliases = ["markedness", "delta_p"]

    def compute_metric(
        self,
        ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
        npv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return ppv + npv - 1


class P4(Metric):
    # TODO: write documentation

    full_name = "P4-score"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("ppv", "tpr", "tnr", "npv")
    sklearn_equivalent = None
    aliases = ["p4"]

    def compute_metric(
        self,
        ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        npv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        values = np.stack(
            [
                ppv,
                tpr,
                tnr,
                npv,
            ],
            axis=2,
        )

        return numpy_batched_harmonic_mean(values, axis=2, keepdims=False)


class JaccardIndex(Metric):
    # TODO: write documentation

    full_name = "Jaccard Index"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("diag_mass", "p_pred", "p_condition")
    sklearn_equivalent = "jaccard_score"
    aliases = ["jaccard", "jaccard_index", "threat_score", "critical_success_index"]

    def compute_metric(
        self,
        diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return diag_mass / (p_pred + p_condition - diag_mass)


class PrevalenceThreshold(Metric):
    # TODO: write documentation

    full_name = "Prevalence Threshold"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("tpr", "fpr")
    sklearn_equivalent = None
    aliases = ["prev_thresh", "prevalence_threshold"]

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        num = np.sqrt(tpr * fpr) - fpr
        denom = tpr - fpr

        return num / denom


class Accuracy(Metric):
    """Computes the (multiclass) accuracy score.

    The rate of correct classifications to all classifications:

    $$(TP + TN) / N$$

    where $TP$ are the true positives, $TN$ the true negatives and $N$ the total number of predictions.

    Note: Read more:
        1. [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification)
    """  # noqa: E501

    full_name = "Accuracy"
    is_multiclass = True
    bounds = (0.0, 1.0)
    dependencies = ("diag_mass",)
    sklearn_equivalent = "accuracy_score"
    aliases = ["acc", "accuracy"]

    def compute_metric(
        self,
        diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        """
        Args:
            diag_mass:

        Returns:
            :
        """
        return np.sum(diag_mass, axis=1)


class BalancedAccuracy(Metric):
    """Computes the (multiclass) balanced accuracy score.

    Uses the scikit-learn definition, but is equivalent to Wikipedia.
    The macro-average of the per-class TPR:
        `1/|C|\\sum TPR_{c}`

    scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

    If `adjusted=True`, the chance-corrected variant is computed instead.

    For more information, see: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score

    Args:
        tpr (np.ndarray [num_samples, num_classes])

    Returns:
        np.ndarray [num_samples, num_classes]
    """

    full_name = "Balanced Accuracy"
    is_multiclass = True
    bounds = (0.0, 1.0)
    dependencies = ("tpr", "p_condition")
    sklearn_equivalent = "balanced_accuracy_score"
    aliases = ["ba", "balanced_accuracy"]

    def __init__(self, adjusted: bool = False) -> None:
        super().__init__()

        self.adjusted = adjusted

    def _compute_ba(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        balanced_accuracy = np.nanmean(
            tpr,
            axis=-1,
        )

        return balanced_accuracy

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        ba = self._compute_ba(tpr)

        if self.adjusted:
            chance = 1 / (p_condition != 0).sum(axis=1)

            ba = (ba - chance) / (1 - chance)

        return ba


class CohensKappa(Metric):
    # TODO: write documentation

    full_name = "Cohen's Kappa"
    is_multiclass = True
    bounds = (-1.0, 1.0)
    dependencies = ("diag_mass", "p_condition", "p_pred")
    sklearn_equivalent = "cohen_kappa_score"
    aliases = ["kappa", "cohen_kappa"]

    def compute_metric(
        self,
        diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        p_agreement = np.sum(diag_mass, axis=1)

        p_chance = np.einsum("bc, bc->b", p_condition, p_pred)

        return (p_agreement - p_chance) / (1 - p_chance)


class MatthewsCorrelationCoefficient(Metric):
    """Computes Matthew's Correlation Coefficient (MCC).

    A metric that holistically combines many different classification metrics.

    A perfect classifier scores `1.0`, a random classifier `0.0`. Smaller values indicate worse than random performance.

    It is related to Pearson's Chi-square test.

    Quoting Wikipedia:
    'Some scientists claim the Matthews correlation coefficient to be the most informative single score to establish the quality of a binary classifier prediction in a confusion matrix context.'

    scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient
    Wikipedia: https://en.wikipedia.org/wiki/Phi_coefficient

    Args:
        diag_mass (np.ndarray [num_samples, num_classes]): a simple metric
        p_condition (np.ndarray [num_samples, num_classes]): a simple metric
        p_pred (np.ndarray [num_samples, num_classes]): a simple metric

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    full_name = "Matthews Correlation Coefficient"
    is_multiclass = True
    bounds = (-1.0, 1.0)
    dependencies = ("diag_mass", "p_condition", "p_pred")
    sklearn_equivalent = "matthews_corrcoef"
    aliases = [
        "mcc",
        "matthews_corrcoef",
        "matthews_correlation_coefficient",
        "phi",
        "phi_coefficient",
    ]

    def compute_metric(
        self,
        diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        marginals_inner_prod = np.einsum("bc, bc->b", p_condition, p_pred)
        numerator = np.sum(diag_mass, axis=1) - marginals_inner_prod

        mcc = numerator / np.sqrt(
            (1 - np.power(p_condition, 2).sum(axis=1))
            * (1 - np.power(p_pred, 2).sum(axis=1))
        )

        return mcc
