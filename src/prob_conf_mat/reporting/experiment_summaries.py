from __future__ import annotations
import typing

from prob_conf_mat.stats import summarize_posterior
from prob_conf_mat.utils import fmt

if typing.TYPE_CHECKING:
    from prob_conf_mat.metrics import Metric, AveragedMetric
    from tabulate import tabulate


def experiment_summaries(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    precision: int = 4,
    table_fmt: str = "html",
    **tabulate_kwargs,
) -> str:
    # Import optional dependencies only now
    import tabulate

    table = []
    for experiment_group_name, experiment_group in study.experiment_groups.items():
        for experiment_name, experiment in experiment_group.experiments.items():
            observed_experiment_result = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="input",
            )

            sampled_experiment_result = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="posterior",
            )

            distribution_summary = summarize_posterior(
                sampled_experiment_result.values[:, class_label],
                ci_probability=study.ci_probability,
            )

            if distribution_summary.hdi[1] - distribution_summary.hdi[0] > 1e-4:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='f')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='f')}]"
            else:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='e')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='e')}]"

            table_row = [
                experiment_group_name,
                experiment_name,
                observed_experiment_result.values[:, class_label],
                distribution_summary.median,
                distribution_summary.mode,
                hdi_str,
                distribution_summary.metric_uncertainty,
                distribution_summary.skew,
                distribution_summary.kurtosis,
            ]

            table.append(table_row)

    headers = ["Group", "Experiment", "Observed", *distribution_summary.headers]  # type: ignore

    table = tabulate.tabulate(  # type: ignore
        tabular_data=table,
        headers=headers,
        floatfmt=f".{precision}f",
        colalign=["left", "left"] + ["decimal" for _ in headers[2:]],
        tablefmt=table_fmt,
        **tabulate_kwargs,
    )

    return table


def random_experiment_summaries(
    study,
    metric: Metric | AveragedMetric,
    class_label: int,
    precision: int = 4,
    table_fmt: str = "html",
    **tabulate_kwargs,
):
    table = []
    for experiment_group_name, experiment_group in study.experiment_groups.items():
        for experiment_name, experiment in experiment_group.experiments.items():
            sampled_experiment_result = study.get_metric_samples(
                metric=metric.name,
                experiment_group=experiment_group_name,
                experiment=experiment_name,
                sampling_method="random",
            )

            distribution_summary = summarize_posterior(
                sampled_experiment_result.values[:, class_label],
                ci_probability=study.ci_probability,
            )

            if distribution_summary.hdi[1] - distribution_summary.hdi[0] > 1e-4:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='f')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='f')}]"
            else:
                hdi_str = f"[{fmt(distribution_summary.hdi[0], precision=precision, mode='e')}, {fmt(distribution_summary.hdi[1], precision=precision, mode='e')}]"

            table_row = [
                experiment_group_name,
                experiment_name,
                distribution_summary.median,
                distribution_summary.mode,
                hdi_str,
                distribution_summary.metric_uncertainty,
                distribution_summary.skew,
                distribution_summary.kurtosis,
            ]

            table.append(table_row)

    headers = ["Group", "Experiment", *distribution_summary.headers]  # type: ignore

    table = tabulate.tabulate(  # type: ignore
        tabular_data=table,
        headers=headers,
        floatfmt=f".{precision}f",
        colalign=["left", "left"] + ["decimal" for _ in headers[2:]],
        tablefmt=table_fmt,
        **tabulate_kwargs,
    )

    return table
