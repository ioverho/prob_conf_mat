from pathlib import Path

import pytest

from bayes_conf_mat import Study
from bayes_conf_mat.io import load_csv

BASIC_METRICS = [
    "acc",
    "f1",
    "f1@macro",
]


class TestBasicExample:
    @pytest.fixture(scope="class")
    def study(self):
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
        )

        # Add a bucnh of metrics
        for metric in BASIC_METRICS:
            study.add_metric(metric=metric, aggregation="fe_gaussian")

        # Add a bunch of experiments
        conf_mat_paths = Path(
            "/home/ioverho/bayes_conf_mat/documentation/Getting Started/mnist_digits",
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

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_aggregation_reporting(self, study, metric):
        study.report_aggregated_metric_summaries(metric="f1@macro", class_label=0)

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_forest_plot(self, study, metric):
        study.plot_forest_plot(metric=metric, class_label=0)

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_pairwise_comparison(self, study, metric):
        # Compare the aggregated F1 macro scores
        study.report_pairwise_comparison(
            metric=metric,
            class_label=0,
            experiment_a="mlp/aggregated",
            experiment_b="svm/aggregated",
            min_sig_diff=0.005,
        )

    @pytest.mark.parametrize(argnames="metric", argvalues=BASIC_METRICS)
    def test_pairwise_comparison_plot(self, study, metric):
        # Compare the aggregated F1 macro scores
        study.plot_pairwise_comparison(
            metric=metric,
            class_label=0,
            experiment_a="mlp/aggregated",
            experiment_b="svm/aggregated",
            min_sig_diff=0.005,
        )
