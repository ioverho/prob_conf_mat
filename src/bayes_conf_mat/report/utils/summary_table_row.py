from bayes_conf_mat.stats import summarize_posterior
from bayes_conf_mat.utils.formatting import fmt


def values_to_table_row(values, point_estimate, ci_probability, precision: int = 4):
    posterior_summary = summarize_posterior(
        posterior_samples=values,
        ci_probability=ci_probability,
    )

    if posterior_summary.hdi[1] - posterior_summary.hdi[0] > 1e-4:
        hdi_str = f"[{fmt(posterior_summary.hdi[0], precision=precision, mode='f')}, {fmt(posterior_summary.hdi[1], precision=precision, mode='f')}]"
    else:
        hdi_str = f"[{fmt(posterior_summary.hdi[0], precision=precision, mode='e')}, {fmt(posterior_summary.hdi[1], precision=precision, mode='e')}]"

    table_row_value = [
        point_estimate,
        posterior_summary.median,
        posterior_summary.mode,
        hdi_str,
        posterior_summary.metric_uncertainty,
        posterior_summary.skew,
        posterior_summary.kurtosis,
    ]

    headers = ["Point"] + posterior_summary.headers

    return table_row_value, headers
