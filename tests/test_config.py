import typing

import pytest
import numpy as np

from prob_conf_mat import Study
from prob_conf_mat.config import ConfigError, ConfigWarning


class TestConfig:
    base_config = dict(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    def fetch_base_config(self, *args) -> dict[str, typing.Any]:
        return {k: v for k, v in self.base_config.items() if k not in args}

    def test_seed(self) -> None:
        # Test a valid seed
        Study(seed=0, **self.fetch_base_config("seed"))

        # Test a negative int
        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be a positive integer.",
        ):
            Study(seed=-1, **self.fetch_base_config("seed"))

        # Test a non-convertible non-int class
        with pytest.raises(
            TypeError,
            match="Parameter `seed` must be a positive integer",
        ):
            Study(seed="foo", **self.fetch_base_config("seed"))  # type: ignore

        # Test a convertible non-int class
        with pytest.warns(
            ConfigWarning,
            match="Parameter `seed` must be a positive integer",
        ):
            Study(seed=0.1, **self.fetch_base_config("seed"))  # type: ignore

        # Test a negative convertible non-int class
        with (
            pytest.warns(
                ConfigWarning,
                match="Parameter `seed` must be a positive integer",
            ),
            pytest.raises(
                ConfigError,
                match="Parameter `seed` must be a positive integer.",
            ),
        ):
            Study(seed=-10.1, **self.fetch_base_config("seed"))  # type: ignore

        # Test a 'None'
        with pytest.warns(
            ConfigWarning,
            match="Recieved `None` as seed. Defaulting to fractional seconds:",
        ):
            Study(seed=None, **self.fetch_base_config("seed"))

    def test_num_samples(self) -> None:
        # Test a valid num_samples
        Study(num_samples=10000, **self.fetch_base_config("num_samples"))

        # Test a low value
        with pytest.warns(
            ConfigWarning,
            match="Parameter `num_samples` should be large to reduce variability",
        ):
            Study(num_samples=1, **self.fetch_base_config("num_samples"))

        # Test a negative int
        with pytest.raises(
            ConfigError,
            match="Parameter `num_samples` must be greater than 0. Currently:",
        ):
            Study(num_samples=-10000, **self.fetch_base_config("num_samples"))

        # Test a None
        with pytest.warns(
            ConfigWarning,
            match="Parameter `num_samples` is `None`. Setting to default value of 10000.",
        ):
            Study(num_samples=None, **self.fetch_base_config("num_samples"))

        # Test a non-convertible non-int class
        with pytest.raises(
            TypeError,
            match="Parameter `num_samples` must be a strictly positive integer",
        ):
            Study(num_samples="foo", **self.fetch_base_config("num_samples"))  # type: ignore

        # Test a convertible non-int class
        with pytest.warns(
            ConfigWarning,
            match="Parameter `num_samples` must be a strictly positive integer",
        ):
            Study(num_samples=10e5, **self.fetch_base_config("num_samples"))  # type: ignore

        # Test a negative convertible non-int class
        with (
            pytest.warns(
                ConfigWarning,
                match="Parameter `num_samples` must be a strictly positive integer",
            ),
            pytest.raises(
                ConfigError,
                match="Parameter `num_samples` must be greater than 0. Currently:",
            ),
        ):
            Study(num_samples=(-10e5), **self.fetch_base_config("num_samples"))  # type: ignore

    def test_ci_probability(self) -> None:
        # Test a valid num_samples
        Study(ci_probability=0.95, **self.fetch_base_config("ci_probability"))

        # Test CI probability bounds
        with pytest.raises(
            ConfigError,
            match=r"Parameter `ci_probability` must be within \(0.0, 1.0\]",
        ):
            Study(ci_probability=0.0, **self.fetch_base_config("ci_probability"))

        with pytest.raises(
            ConfigError,
            match=r"Parameter `ci_probability` must be within \(0.0, 1.0\]",
        ):
            Study(ci_probability=-0.5, **self.fetch_base_config("ci_probability"))

        with pytest.raises(
            ConfigError,
            match=r"Parameter `ci_probability` must be within \(0.0, 1.0\]",
        ):
            Study(ci_probability=1.5, **self.fetch_base_config("ci_probability"))

        # Test a None
        with pytest.warns(
            ConfigWarning,
            match="Parameter `ci_probability` is `None`. Setting to default value of 0.95",
        ):
            Study(ci_probability=None, **self.fetch_base_config("ci_probability"))

        # Test a non-convertible non-float class
        with pytest.raises(
            TypeError,
            match="Parameter `ci_probability` must be a float",
        ):
            Study(ci_probability="foo", **self.fetch_base_config("ci_probability"))  # type: ignore

        with pytest.raises(
            TypeError,
            match="Parameter `ci_probability` must be a float",
        ):
            Study(
                ci_probability=np.array([[0.95], [0.95]]),  # type: ignore
                **self.fetch_base_config("ci_probability"),
            )

        # Test a convertible non-float class
        with pytest.warns(
            ConfigWarning,
            match="Parameter `ci_probability` must be a float",
        ):
            Study(
                ci_probability=np.array(0.95),  # type: ignore
                **self.fetch_base_config("ci_probability"),
            )  # type: ignore

            with pytest.warns(
                DeprecationWarning,
                match="Conversion of an array with ndim > 0 to a scalar is deprecated",
            ):
                Study(
                    ci_probability=np.array([0.95]),  # type: ignore
                    **self.fetch_base_config("ci_probability"),
                )

    def test_prevalence_prior(self) -> None:
        # ======================================================================
        # Prevalence prior
        # ======================================================================
        # Test default prior
        with pytest.warns(
            ConfigWarning,
            match=r"Defaulting to the 0 \(Haldane\) prior\.",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                confusion_prior=0,
            )

        # Test None as prior
        with pytest.warns(
            ConfigWarning,
            match=r"Defaulting to the 0 \(Haldane\) prior\.",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=None,
                confusion_prior=0,
            )

        # Test numeric as prior
        # Integer
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=1,
            confusion_prior=0,
        )

        assert np.all(
            study["test/test"].prevalence_prior == 1.0,  # type: ignore
        )

        # Float
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=1.0,
            confusion_prior=0,
        )

        assert np.all(
            a=study["test/test"].prevalence_prior  # type: ignore
            == 1.0,
        )

        # Out of bounds numeric
        with pytest.raises(
            ConfigError,
            match=r"If numeric, must be greater than 0",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=-1.0,
                confusion_prior=0,
            )

        # Arraylike ============================================================
        # Direct
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=[1.0, 1.0],
            confusion_prior=0,
        )

        assert np.all(
            a=study["test/test"].prevalence_prior  # type: ignore
            == 1.0,
        )

        # Needs reshaping
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=[[1.0, 1.0]],
            confusion_prior=0,
        )

        assert np.all(
            a=study["test/test"].prevalence_prior  # type: ignore
            == 1.0,
        )

        # Malformed - too many columns
        with pytest.raises(
            ConfigError,
            match=r"prevalence prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=[[1.0, 1.0, 1.0]],
                confusion_prior=0,
            )

        # Malformed - too many rows
        with pytest.raises(
            ConfigError,
            match=r"prevalence prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=[[1.0, 1.0], [1.0, 1.0]],
                confusion_prior=0,
            )

        # Malformed - too few columns
        with pytest.raises(
            ConfigError,
            match=r"prevalence prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=[
                    [
                        1.0,
                    ],
                ],
                confusion_prior=0,
            )

        # Malformed - too few rows
        with pytest.raises(
            ConfigError,
            match=r"prevalence prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=[],
                confusion_prior=0,
            )

        # Invalid values in an arraylike
        with pytest.raises(
            ConfigError,
            match=r"all values must be positive",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=[-1.0, 1.0],
                confusion_prior=0,
            )

        with pytest.raises(
            ConfigError,
            match=r"prevalence prior is invalid",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=["one", 1.0],
                confusion_prior=0,
            )

    def test_confusion_matrix(self) -> None:
        # ======================================================================
        # Confusion prior
        # ======================================================================
        # Test default prior
        with pytest.warns(
            ConfigWarning,
            match=r"Defaulting to the 0 \(Haldane\) prior\.",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
            )

        # Test None as prior
        with pytest.warns(
            ConfigWarning,
            match=r"Defaulting to the 0 \(Haldane\) prior\.",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=None,
            )

        # Test numeric as prior
        # Integer
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=0,
            confusion_prior=1,
        )

        assert np.all(
            study["test/test"].confusion_prior == 1.0,  # type: ignore
        )

        # Float
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=0,
            confusion_prior=1.0,
        )

        assert np.all(
            a=study["test/test"].confusion_prior == 1.0,  # type: ignore
        )

        # Out of bounds numeric
        with pytest.raises(
            ConfigError,
            match=r"If numeric, must be greater than 0",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=-1.0,
            )

        # Arraylike ============================================================
        # Direct
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=0,
            confusion_prior=[[1.0, 1.0], [1.0, 1.0]],
        )

        assert np.all(
            a=study["test/test"].confusion_prior == 1.0,  # type: ignore
        )

        # Needs reshaping
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=0,
            confusion_prior=[[[1.0, 1.0], [1.0, 1.0]]],
        )

        assert np.all(
            a=study["test/test"].confusion_prior == 1.0,  # type: ignore
        )

        # Malformed - too many columns
        with pytest.raises(
            ConfigError,
            match=r"confusion prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            )

        # Malformed - too many rows
        with pytest.raises(
            ConfigError,
            match=r"confusion prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            )

        # Malformed - too few columns
        with pytest.raises(
            ConfigError,
            match=r"confusion prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[
                    [
                        1.0,
                    ],
                    [
                        1.0,
                    ],
                ],
            )

        # Malformed - too few rows
        with pytest.raises(
            ConfigError,
            match=r"confusion prior is malformed",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[[1.0, 1.0]],
            )

        # Invalid values in an arraylike
        with pytest.raises(
            ConfigError,
            match=r"all values must be positive",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[[-1.0, 1.0], [1.0, 1.0]],
            )

        with pytest.raises(
            ConfigError,
            match=r"confusion prior is invalid",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=[["one", 1.0], [1.0, 1.0]],
            )

        # TODO: multiple confusion matrices with different num_classes

    def test_adding_experiments(self) -> None:
        study = Study(**self.fetch_base_config())

        study.add_experiment(
            experiment_name="test/test",
            confusion_matrix=np.array([[1, 0], [0, 1]]),
            prevalence_prior=0,
            confusion_prior=0,
        )

        with pytest.warns(
            UserWarning,
            match=r"Experiment .* already exists. Overwriting",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test_1",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=0,
            )

            study.add_experiment(
                experiment_name="test/test_1",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=0,
            )

        with pytest.raises(
            ConfigError,
            match=r"Experiment group .* has incongruent confusion matrices",
        ):
            study = Study(**self.fetch_base_config())

            study.add_experiment(
                experiment_name="test/test_1",
                confusion_matrix=np.array([[1, 0], [0, 1]]),
                prevalence_prior=0,
                confusion_prior=0,
            )

            study.add_experiment(
                experiment_name="test/test_2",
                confusion_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                prevalence_prior=0,
                confusion_prior=0,
            )
