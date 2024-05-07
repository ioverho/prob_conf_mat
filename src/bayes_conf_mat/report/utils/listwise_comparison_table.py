import typing
from tabulate import tabulate


import numpy as np
import jaxtyping as jtyping


def listwise_comparison_table(listwise_comparison_result, precision: int = 4) -> str:
    # Compute the expected rank of each experiment
    expected_rank = np.dot(
        listwise_comparison_result.p_rank_given_experiment,
        np.arange(listwise_comparison_result.p_rank_given_experiment.shape[0]) + 1,
    )

    # Sort by expected rank
    rank_index = np.argsort(expected_rank)

    column_headers = [
        str(listwise_comparison_result.experiment_names[i]) for i in rank_index
    ]

    row_headers = list(
        range(1, listwise_comparison_result.p_rank_given_experiment.shape[0] + 1)
    )

    cell_values = (listwise_comparison_result.p_rank_given_experiment * 100)[
        rank_index, :
    ]

    table = tabulate(
        cell_values,
        headers=["Rank"] + column_headers,
        showindex=row_headers,
        stralign="left",
        numalign="decimal",
        floatfmt=f".{precision-2}f",
        tablefmt="github",
    )

    return table


def expected_reward_table(
    expected_reward: jtyping.Float[np.ndarray, " num_experiments"],
    names: typing.List[str],
    precision: int = 4,
) -> str:
    idx = np.argsort(expected_reward)[::-1]

    expected_reward = tabulate(
        expected_reward[idx][:, np.newaxis],
        headers=["Experiment", "E[Reward]"],
        showindex=[names[i] for i in idx],
        stralign="left",
        numalign="decimal",
        floatfmt=f".{precision-2}f",
        tablefmt="github",
    )

    return expected_reward