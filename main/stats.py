import math

import numpy as np

def compute_confusion_matrix(
    preds: np.typing.ArrayLike, labels: np.typing.ArrayLike, num_classes: int
):

    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(labels)):
        conf_mat[labels[i]][preds[i]] += 1

    return conf_mat


def get_batched_confusion_matrix_stats(confusion_matrices: np.typing.ArrayLike):
    """
    Expects a batched set of confusion matrixes as a single array of dimension [batch_size, num_classes, num_classes].
    Returns a dict with the computed statistics, of dimension [batch_size, 1] or [batch_size, num_classes].

    This function is fully vectorized and should scale nicely.

    """

    num_classes = confusion_matrices.shape[-1]

    stats = dict()

    n_label = np.sum(confusion_matrices, axis=2)
    n_predicted = np.sum(confusion_matrices, axis=1)
    n_correct = np.trace(confusion_matrices, axis1=1, axis2=2)
    n = np.sum(confusion_matrices, axis=(1, 2))
    n2 = np.power(n, 2)

    stats["accuracy"] = n_correct / n

    tp = np.diagonal(confusion_matrices, axis1=1, axis2=2)
    fp = n_predicted - tp
    tn = n_correct[:, np.newaxis] - tp
    fn = n_label - tp

    stats["precision"] = tp / (tp + fp)
    stats["recall"] = tp / (tp + fn)

    stats["f1"] = (
        2
        * (stats["precision"] * stats["recall"])
        / (stats["precision"] + stats["recall"])
    )

    stats["f1_micro"] = np.einsum("bl,bl->b", n_label, stats["f1"]) / n
    stats["f1_macro"] = np.sum(stats["f1"], axis=1) / num_classes

    mcc_num = n_correct * n - np.einsum("bl,bl->b", n_label, n_predicted)
    mcc_denom = np.sqrt(n2 - np.einsum("bl,bl->b", n_predicted, n_predicted)) * np.sqrt(
        n2 - np.einsum("bl,bl->b", n_label, n_label)
    )

    stats["mcc"] = mcc_num / mcc_denom

    return stats


def summarize_posterior_samples(samples, alpha: float = 0.05):

    quantiles = np.quantile(samples, [0, 0.25, 0.75, 1, alpha / 2, 1 - alpha / 2])

    # Estimates the binwidth using Freedman-Diaconis rule
    # https://stats.stackexchange.com/a/862
    sample_min = quantiles[0]
    sample_max = quantiles[3]
    iqr = quantiles[2] - quantiles[1]
    hist_width = 2 * iqr / math.pow(samples.shape[0], 1 / 3)
    num_bins = int(np.round((sample_max - sample_min) / hist_width))
    counts, bin_edges = np.histogram(samples, num_bins)
    modal_bin = np.argmax(counts)
    mode = (bin_edges[modal_bin] + bin_edges[modal_bin + 1]) / 2

    summary = {
        "MAP": mode,
        "Mean": np.mean(samples),
        "Std. Dev": np.std(samples),
        f"CI LB": quantiles[4],
        f"CI UB": quantiles[5],
    }

    for k, v in summary.items():
        summary[k] = f"{v:.4f}"

    return summary
