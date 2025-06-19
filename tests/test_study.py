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
