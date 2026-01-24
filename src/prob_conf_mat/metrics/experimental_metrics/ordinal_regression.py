import numpy as np
import jaxtyping as jtyping

from prob_conf_mat.metrics.abc import Metric

class MeanAbsoluteError(Metric):
    r"""Computes the Mean Absolute Error (MAE).

    It is defined as:

    $$\frac{1}{N}\sum_{i=1}^N |y_{i}-\hat{y}_{i}|$$

    where $y_{i}$ is the ground-truth condition, and $\hat{y}_{i}$ the predicted class for sample
    $i$, respectively.

    Note that this is an **ordinal** classification metric. It assumes that the different classes
    have an ordering. In other words, the magnitude of the misclassification risk increases with
    the amount of classes in between the true and predicted class.

    Examples:
        - `mae`
        - `mean_absolute_error`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
    """

    full_name = "Mean Absolute Error"
    is_multiclass = True
    bounds = (0.0, float("inf"))
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = "mean_absolute_error"
    aliases = ["mean_absolute_error", "mae"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        _, num_classes, _ = norm_confusion_matrix.shape

        # Generate the distance matrix
        # 0 along the main diagonal, increases with distance to main diagonal
        dist_matrix = np.add.reduce(
            [
                np.diag(np.full((num_classes - np.abs(k),), np.abs(k)), k)
                for k in range(-(num_classes - 1), num_classes)
                if k != 0
            ],
        )

        mae = np.sum(norm_confusion_matrix * dist_matrix[np.newaxis, :, :], axis=(1, 2), keepdims=True)

        mae = mae.reshape(-1, 1)

        return mae

class MeanSquaredError(Metric):
    r"""Computes the Mean Squared Error (MSE).

    It is defined as:

    $$\frac{1}{N}\sum_{i=1}^N (y_{i}-\hat{y}_{i})^2$$

    where $y_{i}$ is the ground-truth condition, and $\hat{y}_{i}$ the predicted class for sample
    $i$, respectively.

    Note that this is an **ordinal** classification metric. It assumes that the different classes
    have an ordering. In other words, the magnitude of the misclassification risk increases with
    the amount of classes in between the true and predicted class.

    Examples:
        - `mse`
        - `mean_squared_error`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)
    """

    full_name = "Mean Squared Error"
    is_multiclass = True
    bounds = (0.0, float("inf"))
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = "mean_squared_error"
    aliases = ["mean_squared_error", "mean_square_error", "mse"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        _, num_classes, _ = norm_confusion_matrix.shape

        # Generate the distance matrix
        # 0 along the main diagonal, increases with distance to main diagonal
        dist_matrix = np.add.reduce(
            [
                np.diag(np.full((num_classes - np.abs(k),), np.abs(k)), k)
                for k in range(-(num_classes - 1), num_classes)
                if k != 0
            ],
        )

        dist_matrix = np.power(dist_matrix, 2)

        mse = np.sum(norm_confusion_matrix * dist_matrix[np.newaxis, :, :], axis=(1, 2), keepdims=True)

        mse = mse.reshape(-1, 1)

        return mse

class RootMeanSquaredError(Metric):
    r"""Computes the Root Mean Squared Error (RMSE).

    It is defined as:

    $$\sqrt{\frac{1}{N}\sum_{i=1}^N (y_{i}-\hat{y}_{i})^2}$$

    where $y_{i}$ is the ground-truth condition, and $\hat{y}_{i}$ the predicted class for sample
    $i$, respectively.

    Note that this is an **ordinal** classification metric. It assumes that the different classes
    have an ordering. In other words, the magnitude of the misclassification risk increases with
    the amount of classes in between the true and predicted class.

    Examples:
        - `rmse`
        - `root_mean_squared_error`

    Note: Read more:
        1. [Wikipedia](https://en.wikipedia.org/wiki/Root_mean_squared_deviation)
    """

    full_name = "Root Mean Squared Error"
    is_multiclass = True
    bounds = (0.0, float("inf"))
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = "root_mean_squared_error"
    aliases = ["root_mean_squared_error", "root_mean_square_error", "rmse"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        _, num_classes, _ = norm_confusion_matrix.shape

        # Generate the distance matrix
        # 0 along the main diagonal, increases with distance to main diagonal
        dist_matrix = np.add.reduce(
            [
                np.diag(np.full((num_classes - np.abs(k),), np.abs(k)), k)
                for k in range(-(num_classes - 1), num_classes)
                if k != 0
            ],
        )

        dist_matrix = np.power(dist_matrix, 2)

        mse = np.sum(norm_confusion_matrix * dist_matrix[np.newaxis, :, :], axis=(1, 2), keepdims=True)

        mse = mse.reshape(-1, 1)

        rmse = np.sqrt(mse)

        return rmse

class OffByOneAccuracy(Metric):
    r"""Computes the off-by-1 or adjacent accuracy.

    It is defined as:

    $$\frac{1}{N}\sum_{i=1}^N 1(|y_{i}-\hat{y}_{i}|\leq 1)$$

    where $y_{i}$ is the ground-truth condition, and $\hat{y}_{i}$ the predicted class for sample
    $i$, respectively.

    It measures the proportion of classifications which were correct, or whose class is within 1
    off the true class.

    Note that this is an **ordinal** classification metric. It assumes that the different classes
    have an ordering. In other words, the magnitude of the misclassification risk increases with
    the amount of classes in between the true and predicted class.

    Examples:
        - `offby1`
        - `offbyone`
        - `1off`
        - `adjacc`

    Note: Read more:
        1. Vargas, V. M., Duran-Rosal, A. M., Guijo-Rubio, D., Gutierrez, P. A., & Hervas-Martinez,
        C. (2023). Generalised triangular distributions for ordinal deep learning: Novel proposal
        and optimisation. Information Sciences, 648, 119606.
    """

    full_name = "Off-by-One Accuracy"
    is_multiclass = True
    bounds = (0.0, 1.0)
    dependencies = ("norm_confusion_matrix",)
    sklearn_equivalent = None
    aliases = ["offby1", "offbyone", "1off", "1off", "oneoff", "adjacc", "adjacent_accuracy"]

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
    ) -> jtyping.Float[np.ndarray, " num_samples 1"]:
        off_by_1_acc = (
            np.sum(np.diagonal(norm_confusion_matrix, offset=0, axis1=1, axis2=2), axis=1)
            + np.sum(np.diagonal(norm_confusion_matrix, offset=-1, axis1=1, axis2=2), axis=1)
            + np.sum(np.diagonal(norm_confusion_matrix, offset=1, axis1=1, axis2=2), axis=1)
        )

        off_by_1_acc = off_by_1_acc.reshape(-1, 1)

        return off_by_1_acc


class PolynomialWeightedKappa(Metric):
    r"""Computes the Polynomial Weighted Kappa Coefficient.

    It is defined as:

    $$1-\frac{\sum_{i,j}\omega_{i,j} C_{i,j}}{\sum_{i,j}\omega_{i,j} C_{i,\bullet}C_{\bullet,j}^{\intercal}}$$

    where $C$ is the normalized confusion matrix, $C_{i,\bullet}$ and $C_{\bullet,j}$ the predicted
    and ground truth marginals (respectively), and $\omega$ is the distance weighting matrix.

    Specifically, $\omega$ is defined as:

    $$\omega_{ij}=\frac{|i-j|^p}{(K-1)^p}$$

    This approach increases the weight on a misclassification based on the distance of classes $i$
    and $j$ by interpolating between $0$ ($i=j$) to $1$ ($i=1\wedge j=K$) using a
    polynomial of order $p$. When $p=1$, this is just a linear interpolation. Most commonly, $p=2$.

    It is related to [Cohen's Kappa][prob_conf_mat.metrics.CohensKappa].
    Perfect agreement yields a score of 1, with a score of
    0 corresponding to random performance. Several guidelines exist to interpret
    the magnitude of the score.

    Note that this is an **ordinal** classification metric. It assumes that the different classes
    have an ordering. In other words, the magnitude of the misclassification risk increases with
    the amount of classes in between the true and predicted class.

    Args:
        power (float, optional): the degree of the interpolating polynomial.
            A common value is 2, or Quadratic Weighted Kappa.
            Defaults to 1, in which case this is just linear.

    Examples:
        - `pwkappa`
        - `polynomial_weighted_kappa`

    Note: Read more:
        1. Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement
        or partial credit. Psychological bulletin, 70(4), 213.
        2. de La Torre, J., Puig, D., & Valls, A. (2018). Weighted kappa loss function for
        multi-class classification of ordinal data in deep learning. Pattern Recognition Letters,
        105, 144-154.
    """

    full_name = "Polynomial Weighted Kappa"
    is_multiclass = True
    bounds = (-1.0, 1.0)
    dependencies = ("norm_confusion_matrix", "p_condition", "p_pred")
    sklearn_equivalent = None
    aliases = ["polynomial_weighted_kappa", "pwkappa"]

    def __init__(self, power: float = 1.0):
        super().__init__()

        self.power = power

    def compute_metric(
        self,
        norm_confusion_matrix: jtyping.Float[
            np.ndarray, " num_samples num_classes num_classes"
        ],
        p_condition: jtyping.Float[np.ndarray, "num_samples num_classes"],
        p_pred: jtyping.Float[np.ndarray, "num_samples num_classes"],
    ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
        _, num_classes = p_condition.shape

        # Create the omega matrix
        omega_numerator = np.add.reduce(
            [
                np.diag(np.full((num_classes - np.abs(k),), np.abs(k)), k)
                for k in range(-(num_classes - 1), num_classes)
                if k != 0
            ],
        )

        omega_denominator = num_classes - 1

        omega = (omega_numerator**self.power) / (omega_denominator**self.power)

        #
        numerator = omega[np.newaxis, :, :] * norm_confusion_matrix
        denominator = omega[np.newaxis, :, :] * np.einsum(
            "bc, bd->bcd", p_condition, p_pred
        )

        kappa = 1 - np.sum(numerator, axis=(1, 2)) / np.sum(denominator, axis=(1, 2))

        kappa = kappa.reshape(-1, 1)

        return kappa
