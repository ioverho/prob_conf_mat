from pathlib import Path
from itertools import product

import pytest

import prob_conf_mat as pcm
from prob_conf_mat.metrics import METRIC_REGISTRY, AVERAGING_REGISTRY, get_metric
from prob_conf_mat.metrics.abc import AveragedMetric, Metric, RootMetric
from prob_conf_mat.metrics.interface import _parse_kwargs
from prob_conf_mat.io import load_csv


def test_get_metric_values() -> None:
    all_metrics_avgs_combinations = [
        v.aliases[0] for v in METRIC_REGISTRY.values() if v.is_multiclass
    ] + [
        f"{m}@{a}"
        for m, a in product(
            [v.aliases[0] for v in METRIC_REGISTRY.values() if not v.is_multiclass],
            [v.aliases[0] for v in AVERAGING_REGISTRY.values()],
        )
    ]

    study = pcm.Study(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

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

    for metric_str in all_metrics_avgs_combinations:
        # Add a bucnh of metrics
        study.add_metric(metric=metric_str, aggregation="fe_gaussian")

    for metric_str in all_metrics_avgs_combinations:
        study.get_metric_samples(
            metric=metric_str,
            experiment_name="mlp/aggregated",
            sampling_method="posterior",
        )


class TestInterface:
    def test_implemented_metrics(self):
        all_metrics_avgs_combinations = [
            (v.aliases[0], v, None) for v in METRIC_REGISTRY.values()
        ] + [
            (f"{metric_name}@{avg_name}", metric, avg_method)
            for (metric_name, metric), (avg_name, avg_method) in product(
                [
                    (v.aliases[0], v)
                    for v in METRIC_REGISTRY.values()
                    if not v.is_multiclass
                ],
                [(v.aliases[0], v) for v in AVERAGING_REGISTRY.values()],
            )
        ]

        for metric_str, metric, averaging_method in all_metrics_avgs_combinations:
            fetched_metric = get_metric(metric_str)

            # The metric must be an instance of the generating metric
            if isinstance(fetched_metric, Metric):
                assert isinstance(fetched_metric, metric), (fetched_metric, metric)

            # If an averaged metric:
            #   - the base metric must be an instance of the generating metric
            #   - the base averaging method must be an instance of the generating averaging method
            elif isinstance(fetched_metric, AveragedMetric):
                assert isinstance(fetched_metric.base_metric, metric), (
                    fetched_metric.base_metric,
                    metric,
                )

                assert isinstance(fetched_metric.averaging, averaging_method), (
                    fetched_metric.averaging,
                    averaging_method,
                )

    def test_get_metric(self):
        # Test metric syntax string with multiple averaging methods
        with pytest.raises(
            ValueError,
            match="Multiple averaging methods found in metric string",
        ):
            get_metric("foo@bar@baz")

        # Check RootMetric fetch
        assert isinstance(get_metric("norm_confusion_matrix"), RootMetric)

        # Test unknown metric
        with pytest.raises(
            ValueError,
            match="Metric alias must be registered.",
        ):
            get_metric("foobarbaz")

        # Test applying averaging metric to an already multiclass metric
        with pytest.raises(
            ValueError,
            match="Metric is already multivariate and does",
        ):
            get_metric("acc@macro")

        # Test unknown averaging method
        with pytest.raises(
            ValueError,
            match="Averaging alias must be registered.",
        ):
            get_metric("f1@foobarbaz")

    def test_kwargs_parsing(self):
        # Test the conversion of kwargs dict
        assert _parse_kwargs({"foo": "None"})["foo"] is None
        assert (out := _parse_kwargs({"foo": "0"})["foo"]) == 0 and isinstance(out, int)
        assert (out := _parse_kwargs({"foo": "1.5"})["foo"]) == 1.5 and isinstance(
            out,
            float,
        )
        assert isinstance(
            (out := _parse_kwargs({"foo": "[0.5]"})["foo"]),
            list,
        ) and isinstance(out[0], float)
        assert isinstance(
            _parse_kwargs({"foo": "This is a string"})["foo"],
            str,
        )
        assert isinstance(
            _parse_kwargs({"foo": "True"})["foo"],
            bool,
        )

        # Test finding kwargs in metric syntax string
        metric: pcm.metrics._metrics.BalancedAccuracy = get_metric("ba+adjusted=True")
        assert metric.adjusted == True

        metric: pcm.metrics._metrics.AveragedMetric = get_metric(
            "f1@binary+positive_class=0",
        )
        assert metric.averaging.positive_class == 0

        # Test finding unterminated kwargs in metric syntax string
        with pytest.raises(
            ValueError,
            match="Found potentially unterminated kwarg in",
        ):
            get_metric("ba+adjusted")

        with pytest.raises(
            ValueError,
            match="Found potentially unterminated kwarg in",
        ):
            get_metric("f1@binary+positive_class")
