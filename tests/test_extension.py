import numpy as np

from prob_conf_mat import Study
from prob_conf_mat.metrics import get_metric
from prob_conf_mat.metrics.abc import Metric, Averaging
from prob_conf_mat.experiment_aggregation import get_experiment_aggregator
from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from prob_conf_mat.utils.rng import RNG


class TestExtension:
    def test_defining_new_metric(self) -> None:
        # First define the metric
        class FowlkesMallows(Metric):
            full_name = "Fowlkes Mallows Index"
            is_multiclass = False
            bounds = (0.0, 1.0)
            dependencies = ("ppv", "tpr")
            sklearn_equivalent = "fowlkes_mallows_index"
            aliases = ["fowlkes_mallows", "fm", "fmi"]

            def compute_metric(self, ppv, tpr):
                return np.sqrt(ppv * tpr)

        # Then make sure it is accessible through the metric interface
        get_metric("fowlkes_mallows")

        # Finally, make sure a standard study example works
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=1,
            confusion_prior=1,
        )

        study.add_metric("fowlkes_mallows")

        study.report_metric_summaries("fowlkes_mallows", class_label=0)

    def test_defining_new_averaging_method(self) -> None:
        # First define the averaging method
        class Take2ndClass(Averaging):
            """Always takes the value of the 2nd class.

            ...
            """

            full_name = "Takes 2nd Class Value"
            dependencies = ()
            sklearn_equivalent = "binary, with positive_class=1"
            aliases = ["2nd_class", "two"]

            def compute_average(self, metric_values):
                scalar_array = metric_values[:, 1]

                return scalar_array

        # Then make sure it is accessible through the interface
        get_metric("f1@two")

        # Finally, make sure a standard study example works
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=1,
            confusion_prior=1,
        )

        study.add_metric("f1@two")

        study.report_metric_summaries("f1@two", class_label=0)

    def test_defining_new_composed_metric(self) -> None:
        # Make sure it is accessible through the metric interface
        get_metric("fowlkes_mallows@two")

        # Finally, make sure a standard study example works
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=1,
            confusion_prior=1,
        )

        study.add_metric("fowlkes_mallows@two")

        study.report_metric_summaries("fowlkes_mallows@two", class_label=0)

    def test_defining_new_experiment_aggregator(self) -> None:
        # First define the experiment aggregator
        class Take1stExperiment(ExperimentAggregator):
            full_name: str = "Always Takes 1st Experiment Result as Aggregate"
            aliases: list[str] = ["first", "1st"]

            def aggregate(
                self,
                experiment_samples,
                bounds,
            ):
                return experiment_samples[:, 1]

        # Then make sure it is accessible through the interface
        get_experiment_aggregator(aggregation="first", rng=RNG(0))

        # Finally, make sure a standard study example works
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=1,
            confusion_prior=1,
        )

        study.add_experiment(
            "test/test_b",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=1,
            confusion_prior=1,
        )

        study.add_metric("f1", aggregation="first")

        study.report_aggregated_metric_summaries("f1", class_label=0)
