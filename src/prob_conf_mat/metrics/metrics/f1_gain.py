import jaxtyping as jtyping
import numpy as np

from prob_conf_mat.metrics.abc import Metric


class PrecisionGain(Metric):
    # TODO: write documentation

    full_name = "Precision Gain"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("precision", "prevalence")
    sklearn_equivalent = None
    aliases = ["precision_gain"]

    def compute_metric(
        self,
        precision: jtyping.Float[np.ndarray, "num_samples num_classes"],
        prevalence: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        precision_gain = (precision - prevalence) / ((1 - prevalence) * precision)

        precision_gain = np.clip(precision_gain, a_min=0.0, a_max=1.0)

        return precision_gain


class RecallGain(Metric):
    # TODO: write documentation

    full_name = "Recall Gain"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("recall", "prevalence")
    sklearn_equivalent = None
    aliases = ["recall_gain"]

    def compute_metric(
        self,
        recall: jtyping.Float[np.ndarray, "num_samples num_classes"],
        prevalence: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        recall_gain = (recall - prevalence) / ((1 - prevalence) * recall)

        recall_gain = np.clip(recall_gain, a_min=0.0, a_max=1.0)

        return recall_gain


class F1Gain(Metric):
    # TODO: write documentation

    full_name = "F1 Gain"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = ("f1", "prevalence")
    sklearn_equivalent = None
    aliases = ["f1_gain"]

    def compute_metric(
        self,
        f1: jtyping.Float[np.ndarray, "num_samples num_classes"],
        prevalence: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        f1_gain = (f1 - prevalence) / ((1 - prevalence) * f1)

        f1_gain = np.clip(f1_gain, a_min=0.0, a_max=1.0)

        return f1_gain


class FBetaGain(Metric):
    # TODO: write documentation

    full_name = "FBeta Gain"
    is_multiclass = False
    bounds = (0.0, 1.0)
    dependencies = (
        "ppv",
        "tpr",
        "prevalence",
    )
    sklearn_equivalent = None
    aliases = ["fbeta_gain"]

    def __init__(self, beta: float = 1.0):
        super().__init__()

        self.beta = beta

    def compute_metric(
        self,
        ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
        tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        prevalence: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes num_classes"]:
        beta_2 = self.beta**2

        fbeta = (1 + beta_2) * (ppv * tpr) / (beta_2 * ppv + tpr)

        # In case one of the ratios is nan (most likely due to 0 division), set to 0
        fbeta = np.nan_to_num(
            fbeta,
            nan=0.0,
        )

        fbeta_gain = (fbeta - prevalence) / ((1 - prevalence) * fbeta)

        fbeta_gain = np.clip(fbeta_gain, a_min=0.0, a_max=1.0)

        return fbeta_gain
