from pathlib import Path

import pytest
import matplotlib.pyplot as plt

import prob_conf_mat as pcm
from prob_conf_mat.io import load_csv

BASIC_METRICS = [
    "acc",
    "f1",
    "f1@macro",
]

FORMATS = [
    "simple",  # tabulate
    "github",  # tabulate
    "html",  # tabulate
    "records",  # records
    "pandas",  # pandas
    "pd",  # pandas
]


@pytest.fixture(scope="module")
def study() -> pcm.Study:
    study = pcm.Study(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    # Add a bucnh of metrics
    for metric in BASIC_METRICS:
        study.add_metric(metric=metric, aggregation="fe_gaussian")

    # Add a bunch of experiments
    conf_mat_paths = Path(
        "./documentation/Getting Started/mnist_digits",
    )
    for file_path in sorted(conf_mat_paths.glob("*.csv")):
        # Split the file name to recover the model and fold
        model, fold = file_path.stem.split("_")

        # Load in the confusion matrix using the utility function
        confusion_matrix = load_csv(location=file_path)

        # Add the experiment to the study
        study.add_experiment(
            experiment_name=f"{model}/fold_{fold}",
            confusion_matrix=confusion_matrix,
            prevalence_prior=0,
            confusion_prior=0,
        )

    return study


class TestReportingMethods:
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_plot_metric_summaries(self, study, metric):
        study.plot_metric_summaries(
            metric=metric,
            class_label=0,
        )

        plt.close()

    @pytest.mark.parametrize(argnames="format", argvalues=FORMATS)
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_report_random_metric_summaries(self, study, metric, format):
        study.report_random_metric_summaries(
            metric=metric,
            class_label=0,
            table_fmt=format,
        )

    @pytest.mark.parametrize(argnames="format", argvalues=FORMATS)
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_report_aggregated_metric_summaries(self, study, metric, format):
        study.report_aggregated_metric_summaries(
            metric=metric,
            class_label=0,
            table_fmt=format,
        )

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_plot_experiment_aggregation(self, study, metric):
        study.plot_experiment_aggregation(
            metric=metric,
            class_label=0,
            experiment_group="mlp",
        )

        study.plot_experiment_aggregation(
            metric=metric,
            class_label=0,
            experiment_group="svm",
        )

        plt.close()

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_forest_plot(self, study, metric):
        study.plot_forest_plot(metric=metric, class_label=0)

        plt.close()

    @pytest.mark.parametrize(argnames="format", argvalues=FORMATS)
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_pairwise_comparison(self, study, metric, format):
        study.report_pairwise_comparison(
            metric=metric,
            class_label=0,
            experiment_a="mlp/aggregated",
            experiment_b="svm/aggregated",
            min_sig_diff=0.005,
        )

        with pytest.warns(match="does not produce a table"):
            study.report_pairwise_comparison(
                metric=metric,
                class_label=0,
                experiment_a="mlp/aggregated",
                experiment_b="svm/aggregated",
                min_sig_diff=0.005,
                table_fmt=format,
            )

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_pairwise_comparison_plot(self, study, metric):
        study.plot_pairwise_comparison(
            metric=metric,
            class_label=0,
            experiment_a="mlp/aggregated",
            experiment_b="svm/aggregated",
            min_sig_diff=0.005,
        )

        study.plot_pairwise_comparison(
            metric=metric,
            class_label=0,
            method="histogram",
            experiment_a="mlp/aggregated",
            experiment_b="svm/aggregated",
            min_sig_diff=0.005,
        )

        study.plot_pairwise_comparison(
            metric=metric,
            class_label=0,
            experiment_a="mlp/fold_0",
            experiment_b="svm/fold_0",
            min_sig_diff=0.005,
        )

        plt.close()

    @pytest.mark.parametrize(argnames="format", argvalues=FORMATS)
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_report_pairwise_comparison_to_random(self, study, metric, format):
        study.report_pairwise_comparison_to_random(
            metric=metric,
            class_label=0,
            table_fmt=format,
        )

    @pytest.mark.parametrize(argnames="format", argvalues=FORMATS)
    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_report_listwise_comparison(self, study, metric, format):
        study.report_listwise_comparison(
            metric=metric,
            class_label=0,
            table_fmt=format,
        )
