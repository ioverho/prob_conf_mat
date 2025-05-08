import typing

import pytest
import numpy as np

from bayes_conf_mat import Study
from bayes_conf_mat.config import ConfigError, ConfigWarning


class TestConfig:
    base_config = dict(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    def fetch_base_config(self, parameter: str) -> typing.Dict[str, typing.Any]:
        return {k: v for k, v in self.base_config.items() if k != parameter}

    def test_seed(self) -> None:
        # Test a valid seed
        Study(seed=0, **self.fetch_base_config("seed"))

        # Test a negative int
        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be a positive int.",
        ):
            Study(seed=-1, **self.fetch_base_config("seed"))

        # Test a non-convertible non-int class
        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be an instance of `<class 'int'>`, but got",
        ):
            Study(seed="foo", **self.fetch_base_config("seed"))  # type: ignore

        # Test a convertible non-int class
        # TODO: make this raise a warning
        Study(seed=0.1, **self.fetch_base_config("seed")) # type: ignore

        # Test a negative convertible non-int class
        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be a positive int.",
        ):
            Study(seed=-10.1, **self.fetch_base_config("seed")) # type: ignore

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
            ConfigError,
            match="Parameter `num_samples` must be an instance of `<class 'int'>`, but got",
        ):
            Study(num_samples="foo", **self.fetch_base_config("num_samples"))  # type: ignore

        # Test a convertible non-int class
        Study(num_samples=float(10e5), **self.fetch_base_config("num_samples"))  # type: ignore

        # Test a negative convertible non-int class
        with pytest.raises(
            ConfigError,
            match="Parameter `num_samples` must be greater than 0. Currently:",
        ):
            Study(num_samples=float(-10e5), **self.fetch_base_config("num_samples"))  # type: ignore

    def test_ci_probability(self) -> None:
        # Test a valid num_samples
        Study(ci_probability=0.95, **self.fetch_base_config("ci_probability"))

        # Test CI probability bounds
        with pytest.raises(
            ConfigError,
            match=r"Parameter `ci_probability` must be within \(0.0, 1.0\]",
        ):
            Study(ci_probability=0, **self.fetch_base_config("ci_probability"))

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
            ConfigError,
            match="Parameter `ci_probability` must be an instance of `<class 'float'>`",
        ):
            Study(ci_probability="foo", **self.fetch_base_config("ci_probability"))  # type: ignore

        with pytest.raises(
            ConfigError,
            match="Parameter `ci_probability` must be an instance of `<class 'float'>`",
        ):
            Study(
                ci_probability=np.array([[0.95], [0.95]]),  # type: ignore
                **self.fetch_base_config("ci_probability"),
            )

        # Test a convertible non-float class
        Study(ci_probability=np.array(0.95), **self.fetch_base_config("ci_probability"))  # type: ignore

        with pytest.warns(
            DeprecationWarning,
            match="Conversion of an array with ndim > 0 to a scalar is deprecated",
        ):
            Study(
                ci_probability=np.array([0.95]),  # type: ignore
                **self.fetch_base_config("ci_probability"),
            )
