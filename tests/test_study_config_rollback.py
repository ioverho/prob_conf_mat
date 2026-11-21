import pytest

from prob_conf_mat.config import ConfigError
from prob_conf_mat.study import Study


def _make_study() -> Study:
    study = Study(seed=0, num_samples=10000, ci_probability=0.95)
    study.add_experiment(
        "test/a",
        confusion_matrix=[[1, 0], [0, 1]],
        prevalence_prior=0,
        confusion_prior=0,
    )
    return study


class TestAddMetricRollback:
    def test_rejected_metric_not_in_config(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_metric(metric="not_a_real_metric")

        assert "not_a_real_metric" not in study.metrics

    def test_valid_metric_after_rejected_metric(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_metric(metric="not_a_real_metric")

        study.add_metric(metric="acc")

        assert list(study.metrics) == ["acc"]


class TestAddExperimentRollback:
    def test_rejected_experiment_in_existing_group_not_in_config(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_experiment(
                "test/b",
                confusion_matrix=[[1, 0], [0, 1]],
                prevalence_prior=-1,
                confusion_prior=0,
            )

        assert "b" not in study.experiments["test"]

    def test_rejected_experiment_in_new_group_not_in_config(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_experiment(
                "other/b",
                confusion_matrix=[[1, 0], [0, 1]],
                prevalence_prior=-1,
                confusion_prior=0,
            )

        assert "other" not in study.experiments

    def test_valid_experiment_after_rejected_experiment(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_experiment(
                "test/b",
                confusion_matrix=[[1, 0], [0, 1]],
                prevalence_prior=-1,
                confusion_prior=0,
            )

        study.add_experiment(
            "test/c",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        assert sorted(study.experiments["test"]) == ["a", "c"]
