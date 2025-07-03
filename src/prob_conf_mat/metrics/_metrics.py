from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import numpy as np

from prob_conf_mat.metrics.abc import Metric
from prob_conf_mat.stats import numpy_batched_harmonic_mean


# ==============================================================================
# Fundamental metrics
# Pretty much all metrics are going to rely on these
# ==============================================================================
class DiagMass(Metric):
    r"""Computes the mass on the diagonal of the normalized confusion matrix.

    It is defined as the rate of true positives to all entries:

    $$\mathtt{diag}(\mathbf{CM})=TP / N$$

    where $TP$ are the true positives, and $N$ are the total number of predictions.

    This is a metric primarily used as a intermediate value for other metrics, and says relatively
    little on its own.

    Not to be confused with the True Positive Rate.

    """

    full_name = "Diagonal of Normalized Confusion Matrix"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = None
    aliases = ["diag_mass"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray,
            " num_samples num_classes num_classes",
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        diag_mass = np.diagonal(
            a=norm_confusion_matrix,
            axis1=1,
            axis2=2,
        )

        return diag_mass


class Prevalence(Metric):
    r"""Computes the marginal distribution of condition occurence. Also known as the prevalence.

    It can be defined as the rate of positives to all predictions:

    $$\mathtt{Prev}=P / N$$

    where $P$ is the count of condition positives, and $N$ are the total number of predictions.

    This is a metric primarily used as a intermediate value for other metrics, and say relatively little
    on its own.

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
    r"""Computes the marginal distribution of prediction occurence. Also known as the model bias.

    It can be defined as the rate of predicted positives to all predictions:

    $$\mathtt{Bias}=PP / N$$

    where $PP$ is the count of predicted positives, and $N$ are the total number of predictions.

    This is a metric primarily used as a intermediate value for other metrics, and say relatively little
    on its own.

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


# ==============================================================================
# Simple metrics
# These tell us a little about model performanec, but not the whole story
# Can be computed directly on the fundametal metrics, but are still usually
# used as intermediate values
# ==============================================================================
class TruePositiveRate(Metric):
    r"""Computes the True Positive Rate, also known as recall, sensitivity.

    It is defined as the ratio of correctly predited positives to all condition positives:

    $$\mathtt{TPR}=TP / P$$

    where $TP$ are the true positives, and $TN$ are true negatives and $N$ the number of predictions.

    Essentially, out of all condition positives, how many were correctly predicted. Can be seen as a metric
    measuring retrieval.

    Examples:
        - `tpr`
        - `recall@macro`

    Note: Read more:
        1. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
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
            np.ndarray,
            " num_samples num_classes num_classes",
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        true_positive_rate = np.diagonal(
            a=p_pred_given_condition,
            axis1=1,
            axis2=2,
        )

        return true_positive_rate


class FalseNegativeRate(Metric):
    r"""Computes the False Negative Rate, also known as the miss-rate.

    It is defined as the ratio of false negatives to condition positives:

    $$\mathtt{FNR}=FN / (TP + FN)$$

    where $TP$ are the true positives, and $FN$ are the false negatives.

    Examples:
        - `fnr`
        - `false_negative_rate@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/False_positives_and_false_negatives#Related_terms)
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
    r"""Computes the Positive Predictive Value, also known as precision.

    It is defined as the ratio of true positives to predicted positives:

    $$\mathtt{PPV}=TP / (TP + FP)$$

    where $TP$ is the count of true positives, and $FP$ the count falsely predicted positives.

    It is the complement of the False Discovery Rate, $PPV=1-FDR$.

    Examples:
        - `ppv`
        - `precision@macro`

    Note: Read more:
        1. [scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
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
    r"""Computes the False Discovery Rate.

    It is defined as the ratio of falsely predicted positives to predicted positives:

    $$\mathtt{FDR}=FP / (TP + FP)$$

    where $TP$ is the count of true positives, and $FP$ the count of falsely predicted positives.

    It is the complement of the Positive Predictve Value, $FDR=1-PPV$.

    Examples:
        - `fdr`
        - `false_discovery_rate@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/False_discovery_rate)
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
            np.ndarray,
            " num_samples num_classes",
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        false_discovery_rate = 1 - positive_predictive_value

        return false_discovery_rate


class FalsePositiveRate(Metric):
    r"""Computes the False Positive Rate, the probability of false alarm.

    Also known as the fall-out.

    It is defined as the ratio of falsely predicted positives to condition negatives:

    $$\mathtt{FPR}=FP / (TN + FP)$$

    where $TN$ is the count of true negatives, and $FP$ the count of falsely predicted positives.

    It is the complement of the True Negative Rate, $FPR=1-TNR$.

    Examples:
        - `fpr`
        - `fall-out@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/False_positive_rate)
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
    r"""Computes the True Negative Rate, i.e. specificity, selectivity.

    It is defined as the ratio of true predicted negatives to condition negatives:

    $$\mathtt{TNR}=TN / (TN + FP)$$

    where $TN$ is the count of true negatives, and FP the count of falsely predicted positives.

    It is the complement of the False Positive Rate, $TNR=1-FPR$.

    Examples:
        - `tnr`
        - `selectivity@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
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
    r"""Computes the False Omission Rate.

    It is defined as the ratio of falsely predicted negatives to all predicted negatives:

    $$\mathtt{FOR}=FN / (TN + FN)$$

    where $$TN$$ is the count of true negatives, and $$FN$$ the count of falsely predicted negatives.

    It is the complement of the Negative Predictive Value, $FOR=1-NPV$.

    Examples:
        - `for`
        - `false_omission_rate@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values#false_omission_rate)
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
    r"""Computes the Negative Predicitive Value.

    It is defined as the ratio of true negatives to all predicted negatives:

    $$\mathtt{NPV}=TN / (TN + FN)$$

    where TN are the true negatives, and FN are the falsely predicted negatives.

    It is the complement of the False Omission Rate, $NPV=1-FOR$.

    Examples:
        - `npv`
        - `negative_predictive_value@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values#Negative_predictive_value_(NPV))
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


# ==============================================================================
# Complex metrics
# These actually tell you something interesting about model performance
# ==============================================================================
class Accuracy(Metric):
    r"""Computes the multiclass accuracy score.

    It is defined as the rate of correct classifications to all classifications:

    $$\mathtt{Acc}=(TP + TN) / N$$

    where $TP$ are the true positives, $TN$ the true negatives and $N$ the total number of predictions.

    Possible values lie in the range [0.0, 1.0], with larger values denoting better performance. The value
    of a random classifier is dependent on the label distribution, which makes accuracy especially susceptible
    to class imbalance. It is also not directly comparable across datasets.

    Examples:
        - `acc`
        - `accuracy@macro`

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
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        acc = np.sum(diag_mass, axis=1)
        return acc[:, np.newaxis]


class BalancedAccuracy(Metric):
    r"""Computes the balanced accuracy score.

    It is defined as the the arithmetic average of the per-class true-positive rate:

    $$\mathtt{BA}=\frac{1}{|C|}\sum TPR_{c}$$

    where $TPR$ is the true positive rate (precision).

    Possible values lie in the range [0.0, 1.0], with larger values denoting better performance. Unlike
    accuracy, balanced accuracy can be 'chance corrected', such that random performance is yield a score
    of 0.0. This can be achieved by setting `adjusted=True`.

    Examples:
        - `ba`
        - `balanced_accuracy@macro`
        - `ba+adjusted=True`

    Note: Read more:
        1. [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)

    Args:
        adjusted (bool): whether the chance-corrected variant is computed. Defaults to `False`.
    """  # noqa: E501

    full_name = "Balanced Accuracy"
    is_multiclass = True
    bounds = (0.0, 1.0)
    dependencies = ("tpr", "p_condition")
    sklearn_equivalent = "balanced_accuracy_score"
    aliases = ["ba", "balanced_accuracy"]

    def __init__(self, *, adjusted: bool = False) -> None:
        super().__init__()
        self.adjusted = adjusted

    def _compute_ba(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        balanced_accuracy = np.nanmean(
            tpr,
            axis=-1,
        )
        return balanced_accuracy

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        ba = self._compute_ba(tpr)
        if self.adjusted:
            chance = 1 / (p_condition != 0).sum(axis=1)
            ba = (ba - chance) / (1 - chance)

        return ba[:, np.newaxis]


class MatthewsCorrelationCoefficient(Metric):
    """Computes the multiclass Matthew's Correlation Coefficient (MCC), also known as the phi coefficient.

    Goes by a variety of names, depending on the application scenario.

    A metric that holistically combines many different classification metrics.

    A perfect classifier scores 1.0, a random classifier 0.0. Values smaller than 0
    indicate worse than random performance.

    It's absolute value is proportional to the square root of the Chi-square test statistic.

    Quoting Wikipedia:
    > Some scientists claim the Matthews correlation coefficient to be the most informative
    single score to establish the quality of a binary classifier prediction in a confusion matrix context.

    Examples:
        - `mcc`
        - `phi`

    Note: Read more:
        1. [scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Phi_coefficient)

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
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        marginals_inner_prod = np.einsum("bc, bc->b", p_condition, p_pred)
        numerator = np.sum(diag_mass, axis=1) - marginals_inner_prod

        mcc = numerator / np.sqrt(
            (1 - np.power(p_condition, 2).sum(axis=1))
            * (1 - np.power(p_pred, 2).sum(axis=1)),
        )

        return mcc[:, np.newaxis]


class CohensKappa(Metric):
    r"""Computes the multiclass Cohen's Kappa coefficient.

    Commonly used to quantify inter-annotator agreement, Cohen's kappa can also
    be used to quantify the quality of a predictor.

    It is defined as

    $$\kappa=\frac{p_o-p_e}{1-p_e}$$

    where $p_o$ is the observed agreement and $p_e$ the expected agreement
    due to chance. Perfect agreement yields a score of 1, with a score of
    0 corresponding to random performance. Several guidelines exist to interpret
    the magnitude of the score.

    Examples:
        - `kappa`
        - `cohen_kappa`

    Note: Read more:
        1. [sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-kappa)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    """

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
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        p_agreement = np.sum(diag_mass, axis=1)

        p_chance = np.einsum("bc, bc->b", p_condition, p_pred)

        kappa = (p_agreement - p_chance) / (1 - p_chance)

        return kappa[:, np.newaxis]


class F1(Metric):
    r"""Computes the univariate $F_{1}$-score.

    It is defined as:

    $$\mathtt{F}_{1}=2\dfrac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

    or simply put, the harmonic mean between precision (PPV) and recall (TPR).

    It is an exceedingly common metric used to evaluate machine learning performance. It is closely
    related to the Precision-Recall curve, an anlysis with varying thresholds.

    The 1 in the name from an unseen $\beta$ parameter that weights precision and recall.
    See the `FBeta` metric.

    The $F_{1}$-score is susceptible to class imbalance. Values fall in the range [0, 1]. A random
    classifier which predicts a class with a probability $p$, achieves a performance of,

    $$2\dfrac{\text{prevalence}\cdot p}{\text{prevalence}+p}.$$

    Since this value is maximized for $p=1$, [Flach & Kull](https://proceedings.neurips.cc/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html)
    recommend comparing performance not to a random classifier, but the 'always-on' classifier
    (perfect recall but poor precision). See the `F1Gain` metric.

    Examples:
        - `f1`
        - `f1@macro`

    Note: Read more:
        1. [sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
        2. [Wikipedia](https://en.wikipedia.org/wiki/F-score)
    """

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
    r"""Computes the univariate $F_{\beta}$-score.

    Commonly used to quantify inter-annotator agreement, Cohen's kappa can also
    be used to quantify the quality of a predictor.

    It is defined as:

    $$\mathtt{F}_{\beta}=(1+\beta^2)\dfrac{\text{precision} \cdot \text{recall}}{\beta^2\cdot\text{precision} + \text{recall}}$$

    or simply put, the weighted harmonic mean between precision (PPV) and recall (TPR).

    The value of $\beta$ determines to which degree a user deems recall more important than
    precision. Larger values (x > 1) weight recall more, whereas lower values weight precision more.
    A value of 1 corresponds to equal weighting, see the `F1` metric.

    The $F_{\beta}$-score is susceptible to class imbalance. Values fall in the range [0, 1]. A
    random classifier which predicts a class with a probability $p$, achieves a performance of,

    $$(1+\beta^2)\dfrac{\text{prevalence}\cdot p}{\beta^2\cdot\text{prevalence}+p}.$$

    Since this value is maximized for $p=1$, [Flach & Kull](https://proceedings.neurips.cc/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html)
    recommend comparing performance not to a random classifier, but the 'always-on' classifier
    (perfect recall but poor precision). See the `FBetaGain` metric.

    Examples:
        - `fbeta+beta=2`
        - `fbeta+beta=0.5@macro`

    Note: Read more:
        1. [sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
        2. [Wikipedia](https://en.wikipedia.org/wiki/F-score)
    """  # noqa: E501

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
        ppv: jtyping.Float[np.ndarray, " num_samplesnum_classes"],
        tpr: jtyping.Float[np.ndarray, " num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        beta_2 = self.beta**2

        fbeta = (1 + beta_2) * (ppv * tpr) / (beta_2 * ppv + tpr)

        # In case one of the ratios is nan (most likely due to 0 division), set to 0
        fbeta = np.nan_to_num(
            fbeta,
            nan=0.0,
        )

        return fbeta


class Informedness(Metric):
    r"""Computes the Informedness metric, also known Youden's J.

    It is defined as:

    $$\mathtt{J}=\text{sensitivity}+\text{specificity}-1$$

    where sensitivity is the True Positive Rate (TPR), and specificity is the
    True Negative Rate (TNR).

    Values fall in the range [-1, 1], with higher values corresponding to better performance and 0
    corresponding to random performance.

    In the binary case, this metric is equivalent to the adjusted balanced accuracy, `ba+adj=True`.

    It is commonly used in conjunction with a Reciever-Operator Curve analysis.

    Examples:
        - `informedness`
        - `youdenj@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Youden%27s_J_statistic)
    """

    full_name = "Informedness"
    is_multiclass = False
    bounds = (-1.0, 1.0)
    dependencies = ("tpr", "tnr")
    sklearn_equivalent = None
    aliases = ["informedness", "youdenj", "youden_j", "bookmaker_informedness", "bm"]

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        return tpr + tnr - 1


class Markedness(Metric):
    r"""Computes the markedness metric, also known as $\Delta p$.

    It is defined as:

    $$\Delta p=\text{precision}+NPV-1$$

    where precision is the Positive Predictive Value (PPV).

    Values fall in the range [-1, 1], with higher values corresponding to better performance and 0
    corresponding to random performance.

    It is commonly used in conjunction with a Reciever-Operator Curve analysis.

    Examples:
        - `markedness`
        - `delta_p@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Markedness#)
    """

    full_name = "Markedness"
    is_multiclass = False
    bounds = (-1.0, 1.0)
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
    r"""Computes the P4 metric.

    It is defined as:

    $$\mathtt{P4}=4\left(\dfrac{1}{\text{precision}}+\dfrac{1}{\text{recall}}+\dfrac{1}{\text{specificity}}+\dfrac{1}{NPV}\right)^{-1}$$

    where precision corresponds to the Positive Predictive Value (PPV), recall to the
    True Positive Rate (TPR), and specificity to the True Negative Rate (TNR). Put otherwise, it is
    the harmonic mean of the 4 listed metrics.

    Introduced in 2022 by [Sitarz](https://arxiv.org/abs/2210.11997), it is meant to extend the
    properties of the F1, Markedness and Informedness metrics. It is one of few defined metrics
    that incorporates the Negative Predictive Value.

    Possible values lie in the range [0, 1], with a score of 0 implying one of the intermediate
    metrics is 0, and a 1 requiring perfect classification.

    Relative to MCC, the author notes different behaviour at extreme values, but otherwise the
    metrics are meant to provide a similar amount of information with a single value.

    Examples:
        - `p4`
        - `p4@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/P4-metric)

    """

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
    r"""Computes the Jaccard Index, also known as the threat score.

    It is defined as:

    $$\mathtt{Jaccard}=\dfrac{TP}{TP+FP+FN}$$

    where $TP$ is the count of true positives, $FP$ the count of false positives and $FN$ the count
    of false negatives.

    Alternatively, it may be defined as the area of overlap between predicted and conditions,
    divided by the area of all predicted and condition positives.

    Due to the alternative definition, it is commonly used when labels are not readily present, for
    example in evaluating clustering performance.

    Examples:
        - `jaccard`
        - `critical_success_index@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index#Jaccard_index_in_binary_classification_confusion_matrices)

    """

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


class PositiveLikelihoodRatio(Metric):
    r"""Computes the positive likelihood ratio.

    It is defined as

    $$\mathtt{LR}^{+}=\dfrac{\text{sensitivity}}{1-\text{specificity}}$$

    where sensitivity is the True Positive Rate (TPR), and specificity is the
    True Negative Rate (TNR).

    Simply put, it is the ratio of the probabilities of the model predicting a positive when the
    condition is positive and negative, respectively.

    Possible values lie in the range [0.0, $\infty$], with 0.0 corresponding to no true positives,
    and infinity corresponding to no false positives. Larger values indicate better performance,
    with a score of 1 corresponding to random performance.

    Examples:
        - `plr`
        - `positive_likelihood_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio)

    """

    full_name = "Positive Likelihood Ratio"
    is_multiclass = False
    bounds = (0.0, float("inf"))
    dependencies = ("tpr", "fpr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["plr", "positive_likelihood_ratio"]

    def __init__(self, *, clamp: bool = False) -> None:
        super().__init__()
        self.clamp = clamp

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        if self.clamp:
            fpr = np.where(fpr == 0, np.min(fpr[fpr != 0.0]), fpr)

        return tpr / fpr


class LogPositiveLikelihoodRatio(Metric):
    r"""Computes the positive likelihood ratio.

    It is defined as

    $$\mathtt{LogLR}{+}=\log\dfrac{\text{sensitivity}}{1-\text{specificity}}$$

    where sensitivity is the True Positive Rate (TPR), and specificity is the
    True Negative Rate (TNR).

    Simply put, it is logarithm of the ratio of the probabilities of the model predicting a
    positive when the condition is positive and negative, respectively.

    Possible values lie in the range ($-\infty$, $\infty$), with $-\infty$ corresponding to no
    true positives, and infinity corresponding to no false positives. Larger values indicate better
    performance, with a score of 0 corresponding to random performance.

    Examples:
        - `log_plr`
        - `lplr`
        - `log_positive_likelihood_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#positive_likelihood_ratio)

    """

    full_name = "Log Positive Likelihood Ratio"
    is_multiclass = False
    bounds = (-float("inf"), float("inf"))
    dependencies = ("tpr", "fpr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["log_plr", "lplr", "log_positive_likelihood_ratio"]

    def __init__(self, *, clamp: bool = False) -> None:
        super().__init__()
        self.clamp = clamp

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        if self.clamp:
            fpr = np.where(fpr == 0, np.min(fpr[fpr != 0.0]), fpr)

        lplr = np.log(tpr) - np.log(fpr)

        return lplr


class NegativeLikelihoodRatio(Metric):
    r"""Computes the negative likelihood ratio.

    It is defined as

    $$\mathtt{LR}^{-}=\dfrac{1-\text{sensitivity}}{\text{specificity}}$$

    where sensitivity is the True Positive Rate (TPR), and specificity is the
    True Negative Rate(TNR).

    Simply put, it is the ratio of the probabilities of the model predicting a negative when the
    condition is positive and negative, respectively.

    Possible values lie in the range [0.0, $\infty$], with 0.0 corresponding to no false negatives,
    and infinity corresponding to no true negatives. Smaller values indicate better performance,
    with a score of 1 corresponding to random performance.

    Examples:
        - `nlr`
        - `negative_likelihood_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#negative_likelihood_ratio)

    """

    full_name = "Negative Likelihood Ratio"
    is_multiclass = False
    bounds = (0.0, float("inf"))
    dependencies = ("fnr", "tnr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["negative_likelihood_ratio", "nlr"]

    def __init__(self, *, clamp: bool = False) -> None:
        super().__init__()
        self.clamp = clamp

    def compute_metric(
        self,
        fnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        if self.clamp:
            tnr = np.where(tnr == 0, np.min(tnr[tnr != 0.0]), tnr)

        return fnr / tnr


class LogNegativeLikelihoodRatio(Metric):
    r"""Computes the negative likelihood ratio.

    It is defined as

    $$\mathtt{LogLR}{-}=\log \dfrac{1-\text{sensitivity}}{\text{specificity}}$$

    where sensitivity is the True Positive Rate (TPR), and specificity is the
    True Negative Rate (TNR).

    Simply put, it is the logarithm of the ratio of the probabilities of the model predicting a
    negative when the condition is positive and negative, respectively.

    Possible values lie in the range ($-\infty$, $\infty$), with $-\infty$ corresponding to no
    true positives, and infinity corresponding to no true negatives. Smaller values indicate better
    performance, with a score of 0 corresponding to random performance.

    Examples:
        - `log_nlr`
        - `lnlr`
        - `log_negative_likelihood_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Likelihood_ratios_in_diagnostic_testing#negative_likelihood_ratio)

    """

    full_name = "Log Negative Likelihood Ratio"
    is_multiclass = False
    bounds = (-float("inf"), float("inf"))
    dependencies = ("fnr", "tnr")
    sklearn_equivalent = "class_likelihood_ratios"
    aliases = ["lnlr", "log_negative_likelihood_ratio", "log_nlr"]

    def __init__(self, *, clamp: bool = False) -> None:
        super().__init__()
        self.clamp = clamp

    def compute_metric(
        self,
        fnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        if self.clamp:
            tnr = np.where(tnr == 0, np.min(tnr[tnr != 0.0]), tnr)

        lnlr = np.log(fnr) - np.log(tnr)

        return lnlr


class DiagnosticOddsRatio(Metric):
    r"""Computes the diagnostic odds ratio.

    It is defined as:

    $$\mathtt{DOR}=\dfrac{\mathtt{LR}^{+}=}{\mathtt{LR}^{-}=}$$

    where $\mathtt{LR}^{+}=$ and $\mathtt{LR}^{-}=$ are the positive and
    negative likelihood ratios, respectively.

    Possible values lie in the range [0.0, $\infty$]. Larger values indicate better performance,
    with a score of 1 corresponding to random performance.

    To make experiment aggregation easier, you can log transform this metric by specifying
    `log_transform=true`. This makes the sampling distribution essentially Gaussian.

    Examples:
        - `dor`
        - `diagnostic_odds_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Diagnostic_odds_ratio)

    """

    full_name = "Diagnostic Odds Ratio"
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
        return plr / nlr


class LogDiagnosticOddsRatio(Metric):
    r"""Computes the diagnostic odds ratio.

    It is defined as:

    $$\mathtt{LogDOR}=\mathtt{LogLR}^{+}-\mathtt{LogLR}^{-}$$

    where $\mathtt{LR}^{+}$ and $\mathtt{LR}^{-}=$ are the positive and
    negative likelihood ratios, respectively.

    Possible values lie in the range (-$\infty$, $\infty$). Larger values indicate better
    performance, with a score of 0 corresponding to random performance.

    Examples:
        - `log_dor`
        - `ldor`
        - `log_diagnostic_odds_ratio@macro`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Diagnostic_odds_ratio)

    """

    full_name = "Log Diagnostic Odds Ratio"
    is_multiclass = False
    bounds = (-float("inf"), float("inf"))
    dependencies = ("log_plr", "log_nlr")
    sklearn_equivalent = None
    aliases = ["log_dor", "ldor", "log_diagnostic_odds_ratio"]

    def compute_metric(
        self,
        log_plr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        log_nlr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        log_dor = log_plr - log_nlr

        return log_dor


class PrevalenceThreshold(Metric):
    r"""Computes the prevalence threshold.

    It is defined as:

    $$\phi \mathtt{e}=\frac{\sqrt{\mathtt{TPR}\cdot(1-\mathtt{TNR})}+\mathtt{TNR}-1}{\mathtt{TPR}+\mathtt{TNR}-1}$$

    where $\mathtt{TPR}$ and $\mathtt{TNR}$ are the true positive and negative rates, respectively.

    Possible values lie in the range (0, 1). Larger values indicate *worse* performance, with a
    score of 0 corresponding to perfect classification, and a score of 1 to perfect
    misclassifcation.

    It representents the inflection point in a sensitivity and specificity curve (ROC), beyond which
    a classifiers positive predictive value drops sharply. See [Balayla (2020)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0240215#sec002)
    for more information.

    Examples:
        - `pt`
        - `prevalence_threshold`

    Note: Read more:
        1. [Balayla, J. (2020). Prevalence threshold ($\phi \mathtt{e}$) and the geometry of screening curves. Plos one, 15(10), e0240215.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7540853/)

    """  # noqa: E501

    full_name = "Prevalence Threshold"
    is_multiclass = False
    bounds = (0, 1)
    dependencies = ("tpr", "tnr")
    sklearn_equivalent = None
    aliases = ["prev_thresh", "pt", "prevalence_threshold"]

    def compute_metric(
        self,
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        numerator = np.sqrt(tpr * (1 - tnr)) + tnr - 1
        denominator = tpr + tnr - 1

        pt = numerator / denominator

        return pt
