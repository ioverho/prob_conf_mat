from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.utils import MetricLike

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ListwiseComparisonResult:
    """Class to store results from a listwise comparison between experiments."""

    experiment_names: list[str]
    metric: MetricLike

    p_experiment_given_rank: jtyping.Float[
        np.ndarray,
        " num_experiments num_experiments",
    ]
    p_rank_given_experiment: jtyping.Float[
        np.ndarray,
        " num_experiments num_experiments",
    ]


def listwise_compare(
    experiment_scores_dict: dict[str, jtyping.Float[np.ndarray, " num_samples"]],
    metric: MetricLike,
) -> ListwiseComparisonResult:
    """Compares all experiments against each other in a listwise manner.

    This essentially tries to estimate to the expected ranking of the experiments.

    Args:
        experiment_scores_dict (dict[str, Float[ndarray, "num_samples" ]]): a mapping of experiment
            name to sampled metric values
        metric (MetricLike): the metric used. This parameter is only used for reporting

    Returns:
        ListwiseComparisonResult: the comparison result
    """
    # Convert the experiment values to a list of lists
    all_experiment_scores = list(experiment_scores_dict.items())

    # Transpose the list of lists
    experiment_names, experiment_scores = list(
        map(list, zip(*all_experiment_scores)),
    )

    # Stack the experiments into a [num_samples, num_experiments] array
    experiment_scores = np.stack(
        arrays=experiment_scores,
        axis=1,
    )

    num_samples, num_experiments = experiment_scores.shape

    # TODO: pre-sort the arrays by the mean scores
    # *Might* speed up quicksort
    # Very low priority
    # Rank the arrays
    experiment_inv_ranks = np.argsort(np.argsort(experiment_scores, axis=1), axis=1)

    # Invert the ranking (largest value gets smallest rank)
    experiment_ranks = experiment_scores.shape[1] - experiment_inv_ranks

    # Count the number of times each experiment achieved a certain rank
    rank_count_matrix = np.zeros((num_experiments, num_experiments))
    for i in range(num_experiments):
        ranks, counts = np.unique(experiment_ranks[:, i], axis=0, return_counts=True)

        for rank, rank_count in zip(ranks, counts):
            rank_count_matrix[i, rank - 1] = rank_count

    # Normalize the matrix
    rank_prob_matrix = rank_count_matrix / num_samples

    # Sort the table by MRR
    reciprocal_rank = 1 / (np.arange(rank_prob_matrix.shape[0]) + 1)

    mrr = np.sum(
        reciprocal_rank[np.newaxis, :] * rank_prob_matrix,
        axis=1,
    )

    idx = np.argsort(1 - mrr)
    rank_prob_matrix = rank_prob_matrix[idx, :]

    result = ListwiseComparisonResult(
        experiment_names=[experiment_names[i] for i in idx],
        metric=metric,
        p_rank_given_experiment=rank_prob_matrix,
        p_experiment_given_rank=rank_prob_matrix.T,
    )

    return result
