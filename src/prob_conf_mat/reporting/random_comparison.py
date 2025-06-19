from __future__ import annotations

import typing

from prob_conf_mat.utils import fmt
from prob_conf_mat.experiment_comparison import pairwise_compare

if typing.TYPE_CHECKING:
    from prob_conf_mat.metrics import Metric, AveragedMetric


def pairwise_comparison_to_random(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    min_sig_diff: typing.Optional[float] = None,
    precision: int = 4,
    tablefmt: str = "html",
    **tabulate_kwargs,
) -> str:
    # Import optional dependencies only now
    import tabulate

    table = []
    for experiment_group_name, experiment_group in study.experiment_groups.items():
        for experiment_name, experiment in experiment_group.experiments.items():
            sampled_values = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="posterior",
            ).values[:, class_label]

            sampled_random_values = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="random",
            ).values[:, class_label]

            result = pairwise_compare(
                metric=metric,
                diff_dist=sampled_values - sampled_random_values,
                random_diff_dist=None,
                ci_probability=study.ci_probability,
                min_sig_diff=min_sig_diff,
                lhs_name=f"{experiment_group.name}/{experiment.name}",
                rhs_name="random",
                observed_difference=None,
            )

            rope = [-result.min_sig_diff, result.min_sig_diff]

            if rope[1] - rope[0] > 1e-4:
                rope_str = f"[{fmt(rope[0], precision=precision, mode='f')}, {fmt(rope[1], precision=precision, mode='f')}]"
            else:
                rope_str = f"[{fmt(rope[0], precision=precision, mode='e')}, {fmt(rope[1], precision=precision, mode='e')}]"

            row = [
                experiment_group.name,
                experiment.name,
                result.diff_dist_summary.median,
                result.p_direction,
                rope_str,
                result.p_rope,
                result.p_bi_sig,
                result.p_sig_pos,
                result.p_sig_neg,
            ]

            table.append(row)

    headers = [
        "Group",
        "Experiment",
        "Median Δ",
        "$p_{dir}$",
        "ROPE",
        "$p_{ROPE}$",
        "$p_{sig}$",
        "$p_{sig}^+$",
        "$p_{sig}^-$",
    ]

    summary_table = tabulate.tabulate(
        tabular_data=table,
        headers=headers,
        floatfmt=f".{precision}f",
        colalign=["left", "left"] + ["decimal" for _ in headers[2:]],
        tablefmt=tablefmt,
        **tabulate_kwargs,
    )

    return summary_table
