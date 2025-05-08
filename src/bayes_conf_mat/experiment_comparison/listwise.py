from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np
import jaxtyping as jtyping


@dataclass(frozen=True)
class ListwiseComparisonResult:
    experiment_names: typing.List[str]
    metric_name: str

    p_rank_given_experiment: jtyping.Float[
        np.ndarray, " num_experiments num_experiments"
    ]
    p_experiment_given_rank: jtyping.Float[
        np.ndarray, " num_experiments num_experiments"
    ]


def listwise_compare(
    experiment_values_dict: typing.Dict[str, jtyping.Float[np.ndarray, " num_samples"]],
    metric_name: str,
) -> ListwiseComparisonResult:
    experiment_names, experiment_values = list(
        map(list, zip(*list(experiment_values_dict.items())))
    )

    # Stack the experiments into a [num_samples, num_experiments] array
    stacked_metric_values = np.stack(
        experiment_values,
        axis=1,
    )

    num_samples, num_experiments = stacked_metric_values.shape

    # Pre-sort the arrays
    # Should speed up sorting, as arrays are already 'nearly' sorted
    pre_sort_indices = np.argsort(np.mean(stacked_metric_values, axis=0))
    inv_pre_sort_indices = np.argsort(pre_sort_indices)

    # Sort the arrays
    ranked_metric_values = np.argsort(
        stacked_metric_values[:, pre_sort_indices],
        axis=1,
    )

    # Invert the ranking (largest value gets smallest rank)
    ranked_metric_values = num_experiments - ranked_metric_values - 1

    # Reduce the ranked arrays to the distinct combinations of ranks
    rank_combinations, rank_comb_count = np.unique(
        ranked_metric_values, return_counts=True, axis=0
    )

    # Count the number of times each experiment achieved a certain rank
    idx_vector = np.arange(num_experiments)
    rank_count_matrix = np.zeros((num_experiments, num_experiments))
    for combi, count in zip(rank_combinations, rank_comb_count):
        rank_count_matrix[idx_vector, combi] += count

    # Invert the pre-sorting
    rank_count_matrix = rank_count_matrix[inv_pre_sort_indices, :]

    # Compute the tables of interest
    p_rank_given_experiment = rank_count_matrix / np.sum(rank_count_matrix, axis=1)
    p_experiment_given_rank = p_rank_given_experiment.T

    result = ListwiseComparisonResult(
        experiment_names=experiment_names,
        metric_name=metric_name,
        p_rank_given_experiment=p_rank_given_experiment,
        p_experiment_given_rank=p_experiment_given_rank,
    )

    return result
