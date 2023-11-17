import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.metrics.registration import register_complex_metric
from bayes_conf_mat.metrics.aggregation import numpy_batched_harmonic_mean


@register_complex_metric(
    identifier="plr",
    full_name="Positive Likelihood Ratio",
    is_multiclass=False,
    range=(0.0, float("inf")),
    required_simple_metrics=(
        "tpr",
        "fpr",
    ),
    sklearn_equivalent="class_likelihood_ratios",
)
def compute_positive_likelihood_ratio(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return tpr / fpr


@register_complex_metric(
    identifier="nlr",
    full_name="Negative Likelihood Ratio",
    is_multiclass=False,
    range=(0.0, float("inf")),
    required_simple_metrics=(
        "fnr",
        "tnr",
    ),
    sklearn_equivalent="class_likelihood_ratios",
)
def compute_negative_likelihood_ratio(
    fnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return fnr / tnr


@register_complex_metric(
    identifier="dor",
    full_name="Diagnostic Odds Ratio",
    is_multiclass=False,
    range=(0, float("inf")),
    required_simple_metrics=(
        "tpr",
        "fpr",
        "fnr",
        "tnr",
    ),
    sklearn_equivalent=None,
)
def compute_diagnostic_odds_ratio(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    fnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    positive_likelihood_ratio = compute_positive_likelihood_ratio(
        tpr=tpr,
        fpr=fpr,
    )

    negative_likelihood_ratio = compute_negative_likelihood_ratio(
        fnr=fnr,
        tnr=tnr,
    )

    diagnostic_odds_ratio = positive_likelihood_ratio / negative_likelihood_ratio

    return diagnostic_odds_ratio


@register_complex_metric(
    identifier="fbeta",
    full_name="F-beta Score",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "ppv",
        "tpr",
    ),
    sklearn_equivalent="fbeta_score",
)
def compute_fbeta(
    ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    beta: float,
):
    beta_2 = beta**2

    f1 = (1 + beta_2) * (ppv * tpr) / (beta_2 * ppv + tpr)

    # In case one of the ratios is nan (most likely due to 0 division), set to 0
    f1 = np.nan_to_num(
        f1,
        nan=0.0,
    )

    return f1


@register_complex_metric(
    identifier="f1",
    full_name="F1",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "ppv",
        "tpr",
    ),
    sklearn_equivalent="f1_score",
)
def compute_f1(
    ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return compute_fbeta(ppv=ppv, tpr=tpr, beta=1)


@register_complex_metric(
    identifier="informedness",
    full_name="Informedness",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "tpr",
        "tnr",
    ),
    sklearn_equivalent=None,
)
def compute_informedness(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return tpr + tnr - 1


@register_complex_metric(
    identifier="markedness",
    full_name="Markedness",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "ppv",
        "npv",
    ),
    sklearn_equivalent=None,
)
def compute_markedness(
    ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    npv: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return ppv + npv - 1


@register_complex_metric(
    identifier="p4",
    full_name="P4",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "ppv",
        "tpr",
        "tnr",
        "npv",
    ),
    sklearn_equivalent=None,
)
def compute_p4(
    ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    tnr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    npv: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    values = np.stack(
        [
            ppv,
            tpr,
            tnr,
            npv,
        ],
        axis=2,
    )

    return numpy_batched_harmonic_mean(values, keepdims=False)


@register_complex_metric(
    identifier="jaccard",
    full_name="Jaccard Index",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "diag_mass",
        "p_pred",
        "p_condition",
    ),
    sklearn_equivalent="jaccard_score",
)
def compute_jaccard_index(
    diag_mass: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
    p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    return diag_mass / (p_pred + p_condition - diag_mass)


@register_complex_metric(
    identifier="prev_thresh",
    full_name="Prevalence Threshold",
    is_multiclass=False,
    range=(0.0, 1.0),
    required_simple_metrics=(
        "tpr",
        "fpr",
    ),
    sklearn_equivalent=None,
)
def compute_prevalence_threshold(
    tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
    fpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
):
    num = np.sqrt(tpr * fpr) - fpr
    denom = tpr - fpr

    return num / denom
