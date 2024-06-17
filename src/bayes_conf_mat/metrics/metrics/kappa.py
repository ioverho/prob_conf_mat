import jaxtyping as jtyping
import numpy as np

from bayes_conf_mat.metrics.base import Metric


class CohensKappa(Metric):
    """Computes the multiclass Cohen's Kappa coefficient.

    Commonly used to quantify inter-annotator agreement, Cohen's kappa can also
    be used to quantify the quality of a predictor.

    It is defined as

    $$\\frac{p_o-p_e}{1-p_e}$$

    where $p_o$ is the observed agreement and $p_e$ the expected agreement
    due to chance. Perfect agreement yields a score of 1, with a score of
    0 corresponding to random performance. Several guidelines exist to interpret
    the magnitude of the score.

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
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        p_agreement = np.sum(diag_mass, axis=1)

        p_chance = np.einsum("bc, bc->b", p_condition, p_pred)

        return (p_agreement - p_chance) / (1 - p_chance)
