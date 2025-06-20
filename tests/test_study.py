import pytest

from prob_conf_mat.study import Study
from prob_conf_mat.experiment_group import ExperimentGroup
from prob_conf_mat.experiment import Experiment


class TestStudy:
    def test_getitem(self):
        # First define a common setup
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )
        study.add_experiment(
            "test/foo",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Should fetch at the right level
        assert isinstance(study["test"], ExperimentGroup)
        assert isinstance(study["test/test_a"], Experiment)
        assert isinstance(study["test/foo"], Experiment)

        # Should raise an error when trying to fetch non-existent experiment group
        with pytest.raises(
            KeyError,
            match=r"No experiment group with name .* is currently present",
        ):
            study["foo"]

        with pytest.raises(
            KeyError,
            match=r"No experiment with name .* is currently present",
        ):
            study["test/bar"]

        # Should only accept strings
        with pytest.raises(
            TypeError,
        ):
            study[0]  # type: ignore

        # Should only accept strings
        with pytest.raises(
            ValueError,
        ):
            study["foo/bar/baz"]

        # Test unknown experiment group
        with pytest.raises(
            KeyError,
        ):
            study["foo/bar"]

    def test_init(self):
        # First experiment, then metric
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        # First metric, then experiment
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Experiment in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            experiments={
                "test": {
                    "test": {
                        "confusion_matrix": [[1, 0], [0, 1]],
                        "prevalence_prior": 0,
                        "confusion_prior": 0,
                    },
                },
            },
        )

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        # Metric in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            metrics={
                "acc": {"aggregation": "fe_gaussian"},
            },
        )

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Experiment and metric in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            experiments={
                "test": {
                    "test": {
                        "confusion_matrix": [[1, 0], [0, 1]],
                        "prevalence_prior": 0,
                        "confusion_prior": 0,
                    },
                },
            },
            metrics={
                "acc": {"aggregation": "fe_gaussian"},
            },
        )

    def test_metric_label_validation(self):
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="acc", aggregation="normal")
        study.add_metric(metric="f1", aggregation="normal")

        # Test valid multiclass label
        study._validate_metric_class_label_combination(
            metric="acc",
            class_label=0,
        )

        study._validate_metric_class_label_combination(
            metric="acc",
            class_label=None,
        )

        # Test accessing invalid class
        with pytest.warns(match="Metric is multiclass, ignoring class label."):
            study._validate_metric_class_label_combination(
                metric="acc",
                class_label=1,
            )

        # Test valid binary label
        study._validate_metric_class_label_combination(
            metric="f1",
            class_label=0,
        )

        study._validate_metric_class_label_combination(
            metric="f1",
            class_label=1,
        )

        # Test accessing invalid class
        with pytest.raises(
            ValueError,
            match="is not multiclass. You must provide a class label.",
        ):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=None,
            )

        with pytest.raises(ValueError, match="Class label must be in range"):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=-1,
            )

        with pytest.raises(ValueError, match="Class label must be in range"):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=3,
            )

        # Test unknown metric
        with pytest.raises(KeyError, match="Could not find metric"):
            study._validate_metric_class_label_combination(
                metric="foobarbaz",
                class_label=None,
            )
