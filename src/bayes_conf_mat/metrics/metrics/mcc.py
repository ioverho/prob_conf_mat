import jaxtyping as jtyping
import numpy as np

from bayes_conf_mat.metrics.abc import Metric


class MatthewsCorrelationCoefficient(Metric):
    """Computes the multiclass Matthew's Correlation Coefficient (MCC).

    Goes by a variety of names, depending on the application scenario.

    A metric that holistically combines many different classification metrics.

    A perfect classifier scores 1.0, a random classifier 0.0. Values smaller than 0
    indicate worse than random performance.

    It is related to Pearson's Chi-square test.

    Quoting Wikipedia:
    > Some scientists claim the Matthews correlation coefficient to be the most informative
    single score to establish the quality of a binary classifier prediction in a confusion matrix context.

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
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        marginals_inner_prod = np.einsum("bc, bc->b", p_condition, p_pred)
        numerator = np.sum(diag_mass, axis=1) - marginals_inner_prod

        mcc = numerator / np.sqrt(
            (1 - np.power(p_condition, 2).sum(axis=1))
            * (1 - np.power(p_pred, 2).sum(axis=1))
        )

        return mcc
