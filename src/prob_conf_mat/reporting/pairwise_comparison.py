from __future__ import annotations


import typing

from prob_conf_mat.experiment_comparison import pairwise_compare

if typing.TYPE_CHECKING:
    from prob_conf_mat.metrics import Metric, AveragedMetric


def pairwise_comparison(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    experiment_group_a: str,
    experiment_group_b: str,
    experiment_a: str | typing.Literal["aggregated"],
    experiment_b: str | typing.Literal["aggregated"],
    min_sig_diff: typing.Optional[float] = None,
    precision: int = 4,
) -> None:
    lhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_a,
        experiment=experiment_a,
        sampling_method="posterior",
    ).values[:, class_label]

    rhs_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_b,
        experiment=experiment_b,
        sampling_method="posterior",
    ).values[:, class_label]

    lhs_random_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_a,
        experiment=experiment_a,
        sampling_method="random",
    ).values[:, class_label]

    rhs_random_samples = study.get_metric_samples(
        metric=metric.name,
        experiment_group=experiment_group_b,
        experiment=experiment_b,
        sampling_method="random",
    ).values[:, class_label]

    if experiment_a != "aggregated" and experiment_b != "aggregated":
        lhs_observed = study.get_metric_samples(
            metric=metric.name,
            experiment_group=experiment_group_a,
            experiment=experiment_a,
            sampling_method="input",
        ).values[:, class_label]

        rhs_observed = study.get_metric_samples(
            metric=metric.name,
            experiment_group=experiment_group_b,
            experiment=experiment_b,
            sampling_method="input",
        ).values[:, class_label]

        observed_diff = lhs_observed - rhs_observed
    else:
        observed_diff = None

    comparison_result = pairwise_compare(
        metric=metric,
        diff_dist=lhs_samples - rhs_samples,
        random_diff_dist=lhs_random_samples - rhs_random_samples,
        ci_probability=study.ci_probability,
        min_sig_diff=min_sig_diff,
        observed_difference=observed_diff,
        lhs_name=f"{experiment_group_a}/{experiment_a}",
        rhs_name=f"{experiment_group_b}/{experiment_b}",
    )

    print(comparison_result.template_sentence(precision=precision))
