import numpy as np
import pandas as pd
import pytest

from prob_conf_mat import Study


@pytest.fixture(scope="module")
def study() -> Study:
    totsch_table_3 = [
        [1, "3467175", 0.99763, 15087, 36, 7544, 18, 18, 7544],
        [2, "3394520", 0.99672, 15073, 50, 7537, 25, 25, 7537],
        [3, "3338942", 0.99596, 15062, 61, 7531, 31, 31, 7531],
        [4, "3339018", 0.99512, 15049, 74, 7525, 37, 37, 7525],
        [5, "3338836", 0.99498, 15047, 76, 7524, 38, 38, 7524],
        [6, "3429037", 0.9938, 15029, 94, 7515, 47, 47, 7515],
        [7, "3346448", 0.99296, 15017, 106, 7509, 53, 53, 7509],
        [8, "3338664", 0.99296, 15017, 106, 7509, 53, 53, 7509],
        [9, "3338358", 0.99282, 15014, 109, 7507, 55, 55, 7507],
        [10, "3339624", 0.9924, 15008, 115, 7504, 58, 58, 7504],
    ]

    totsch_table_3_df = pd.DataFrame.from_records(
        totsch_table_3,
        columns=["Rank", "TeamId", "Score", "TP+TN", "FP+FN", "TP", "FN", "FP", "TN"],
        index="Rank",
    )

    study = Study(
        seed=0,
        num_samples=100000,
        ci_probability=0.95,
    )

    study.add_metric(metric="acc")

    for rank, row in totsch_table_3_df.iterrows():
        study.add_experiment(
            experiment_name=f"{rank}/{row.TeamId}",
            confusion_matrix=[[row.TP, row.FN], [row.FP, row.TN]],
            confusion_prior=1,
            prevalence_prior=1,
        )

    return study


def test_list_wise_ranks(study) -> None:
    """For this trivial example, make sure that the listwise ordering sees consecutive decreases."""

    listwise_experiment_result = study.get_listwise_comparsion_result(
        metric="acc",
    )

    metric_result: pd.DataFrame = study.report_metric_summaries(
        metric="acc",
        table_fmt="pd",
    )  # type: ignore

    metric_result = metric_result.set_index("Group")

    experiment_ranks = [
        int(experiment_name.split("/")[0])
        for experiment_name in listwise_experiment_result.experiment_names
    ]

    for rank_l, rank_r in zip(experiment_ranks[:-1], experiment_ranks[1:]):
        metric_diff = (
            metric_result.loc[str(rank_l)].Median
            - metric_result.loc[str(rank_r)].Median
        )

        assert metric_diff >= 0.0, (experiment_ranks, rank_l, rank_r, metric_diff)  # type: ignore


def test_expected_reward_computation(study) -> None:
    study.report_expected_reward(metric="acc")
    study.report_expected_reward(metric="acc", rewards=1.0)
    study.report_expected_reward(metric="acc", rewards=np.ones((10,)))

    # Test too many rewards
    with pytest.raises(ValueError):
        study.report_expected_reward(metric="acc", rewards=np.ones((11,)))

    # Test rewards shape
    with pytest.raises(ValueError):
        study.report_expected_reward(metric="acc", rewards=np.ones((2, 10, 2)))
