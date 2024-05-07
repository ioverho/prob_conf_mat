from tabulate import tabulate, SEPARATING_LINE
import numpy as np

from bayes_conf_mat.stats import summarize_posterior
from bayes_conf_mat.report.utils.formatting import fmt


def values_to_table_row(
    values, point_estimate, ci_probability, name: str = None, precision: int = 4
):
    posterior_summary = summarize_posterior(
        posterior_samples=values,
        ci_probability=ci_probability,
    )

    if posterior_summary.hdi[1] - posterior_summary.hdi[0] > 1e-4:
        hdi_str = f"[{fmt(posterior_summary.hdi[0], precision=precision, mode='f')}, {fmt(posterior_summary.hdi[1], precision=precision, mode='f')}]"
    else:
        hdi_str = f"[{fmt(posterior_summary.hdi[0], precision=precision, mode='e')}, {fmt(posterior_summary.hdi[1], precision=precision, mode='e')}]"

    table_row_value = (
        name,
        point_estimate,
        posterior_summary.median,
        posterior_summary.mode,
        hdi_str,
        posterior_summary.metric_uncertainty,
        posterior_summary.skew,
        posterior_summary.kurtosis,
    )

    headers = ["Experiment", "Point"] + posterior_summary.headers

    return table_row_value, headers


def aggregation_summary_table(
    point_estimates,
    individual_results,
    aggregated_results,
    ci_probability: float,
    table_fmt: str,
):
    table_row_values = []
    for point_estimate, posterior_result in zip(point_estimates, individual_results):
        point_estimate = np.squeeze(point_estimate.values)

        table_row_value, _ = values_to_table_row(
            name=posterior_result.experiment.name,
            values=posterior_result.values,
            point_estimate=point_estimate,
            ci_probability=ci_probability,
        )

        table_row_values.append(table_row_value)

    if table_fmt == "simple":
        table_row_values.append(SEPARATING_LINE)

    table_row_value, table_row_headers = values_to_table_row(
        name="Aggregate",
        values=aggregated_results.values,
        point_estimate=None,
        ci_probability=ci_probability,
    )

    table_row_values.append(table_row_value)

    aggregation_summary_table = tabulate(
        tabular_data=table_row_values,
        headers=table_row_headers,
        floatfmt=".4f",
        colalign=["left"] + ["decimal" for _ in table_row_headers[1:]],
        tablefmt=table_fmt,
    )

    return aggregation_summary_table
