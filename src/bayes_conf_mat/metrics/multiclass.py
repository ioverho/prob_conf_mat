import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics.registration import register_complex_metric


@register_complex_metric(
    identifier="acc",
    full_name="Accuracy",
    is_multiclass=True,
    range=(0.0, 1.0),
    required_simple_metrics=("diag_mass",),
    sklearn_equivalent="accuracy_score",
)
def compute_accuracy(
    diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    """Computes the (multiclass) accuracy score.

    The rate of correct classifications to all classifications:
        `(TP + TN) / N`
    where TP are the true positives, TN the true negatives and N the total number of predictions.

    scikit-learn: https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
    Wikipedia: https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification

    Args:
        diag_mass (np.ndarray [num_samples, num_classes]): a simple metric

    Returns:
        np.ndarray [num_samples, num_classes]
    """  # noqa: E501

    return np.sum(diag_mass, axis=1)


@register_complex_metric(
    identifier="ba",
    full_name="Balanced Accuracy",
    is_multiclass=True,
    range=(0.0, 1.0),
    required_simple_metrics=("tpr",),
    sklearn_equivalent="balanced_accuracy_score",
)
def compute_balanced_accuracy(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    """Computes the (multiclass) balanced accuracy score.

    Uses the scikit-learn definition, but is equivalent to Wikipedia.
    The macro-average of the per-class TPR:
        `1/|C|\\sum TPR_{c}`

    scikit-learn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

    Args:
        tpr (np.ndarray [num_samples, num_classes])

    Returns:
        np.ndarray [num_samples, num_classes]
    """

    balanced_accuracy = np.nanmean(
        tpr,
        axis=-1,
    )

    return balanced_accuracy


@register_complex_metric(
    identifier="adjba",
    full_name="Adjusted Balanced Accuracy",
    is_multiclass=True,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "tpr",
        "p_condition",
    ),
    sklearn_equivalent="balanced_accuracy_score",
)
def compute_adjusted_balanced_accuracy(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    """Computes the chance-corrected multi-class balanced accuracy, as used by scikit-learn.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score

    For more information, see: https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    """  # noqa: E501

    balanced_accuracy = compute_balanced_accuracy(tpr=tpr)

    chance = 1 / (p_condition != 0).sum(axis=1)

    adjusted_balanced_accuracy = (balanced_accuracy - chance) / (1 - chance)

    return adjusted_balanced_accuracy


@register_complex_metric(
    identifier="kappa",
    full_name="Cohen's Kappa",
    is_multiclass=True,
    range=(-1.0, 1.0),
    required_simple_metrics=(
        "diag_mass",
        "p_condition",
        "p_pred",
    ),
    sklearn_equivalent="cohen_kappa_score",
)
def compute_cohens_kappa(
    diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    p_agreement = np.sum(diag_mass, axis=1)

    p_chance = np.einsum("bc, bc->b", p_condition, p_pred)

    return (p_agreement - p_chance) / (1 - p_chance)


@register_complex_metric(
    identifier="mcc",
    full_name="Matthews Correlation Coefficient",
    is_multiclass=True,
    range=(-1.0, 1.0),
    required_simple_metrics=(
        "diag_mass",
        "p_condition",
        "p_pred",
    ),
    sklearn_equivalent="matthews_corrcoef",
)
def compute_mcc(
    diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    """Computes the (multiclass) Matthew's Correlation Coefficient (MCC).

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
    marginals_inner_prod = np.einsum("bc, bc->b", p_condition, p_pred)
    numerator = np.sum(diag_mass, axis=1) - marginals_inner_prod

    mcc = numerator / np.sqrt(
        (1 - np.power(p_condition, 2).sum(axis=1))
        * (1 - np.power(p_pred, 2).sum(axis=1))
    )

    return mcc
