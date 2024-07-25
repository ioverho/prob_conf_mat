from copy import deepcopy

import pytest

from bayes_conf_mat.config import Config
from bayes_conf_mat.config.config import ConfigError, ConfigWarning


class TestConfig:
    # This config should work
    base_config = dict(
        name="test",
        seed=0,
        num_samples=10000,
        ci_probability=1.0,
        experiments={
            "group_1": {
                "__default__": {
                    "prevalence_prior": 0.0,
                    "confusion_prior": 0.0,
                    "format": "csv",
                    "type": "confusion_matrix",
                },
                "a": {
                    "location": "./tests/data/confusion_matrices/sklearn_1.csv",
                },
                "b": {
                    "location": "./tests/data/confusion_matrices/sklearn_1.csv",
                },
            },
            "group_2": {
                "__default__": {
                    "prevalence_prior": 1.0,
                    "confusion_prior": 10.0,
                    "format": "csv",
                    "type": "confusion_matrix",
                },
                "a": {
                    "location": "./tests/data/confusion_matrices/sklearn_1.csv",
                },
            },
        },
        metrics={
            "__default__": {"aggregation": "hist"},
            "f1": {},
            "mcc": {"aggregation": "fe_gaussian"},
        },
    )

    def test_base_config(self):
        Config(**self.base_config)

    def test_seed(self):
        base_config = deepcopy(self.base_config)

        # ======================================================================
        # Type =================================================================
        # ======================================================================
        base_config["seed"] = "foo"

        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be an instance of `<class 'int'>`, but got",
        ):
            Config(**base_config)

        # ======================================================================
        # Bounds ===============================================================
        # ======================================================================
        base_config["seed"] = -1

        with pytest.raises(
            ConfigError,
            match="Parameter `seed` must be a positive int",
        ):
            Config(**base_config)

    def test_num_samples(self):
        base_config = deepcopy(self.base_config)

        # ======================================================================
        # Type =================================================================
        # ======================================================================
        base_config["num_samples"] = "foo"

        with pytest.raises(
            ConfigError,
            match="Parameter `num_samples` must be an instance of `<class 'int'>`, but got",
        ):
            Config(**base_config)

        # ======================================================================
        # Bounds ===============================================================
        # ======================================================================
        base_config["num_samples"] = -1

        with pytest.raises(
            ConfigError,
            match="Parameter `num_samples` must be greater than 0",
        ):
            Config(**base_config)

        base_config["num_samples"] = 1

        with pytest.warns(
            ConfigWarning,
        ):
            Config(**base_config)

    def test_ci_probability(self):
        base_config = deepcopy(self.base_config)

        # ======================================================================
        # Type =================================================================
        # ======================================================================
        base_config["ci_probability"] = "foo"

        with pytest.raises(
            ConfigError,
            match="Parameter `ci_probability` must be an instance of `<class 'float'>`, but got",
        ):
            Config(**base_config)

        # ======================================================================
        # Bounds ===============================================================
        # ======================================================================
        base_config["ci_probability"] = -1

        with pytest.raises(
            ConfigError,
            match="Parameter `ci_probability` must be within [0.0, 1.0]*",
        ):
            Config(**base_config)

        base_config["ci_probability"] = 2

        with pytest.raises(
            ConfigError,
            match="Parameter `ci_probability` must be within [0.0, 1.0]*",
        ):
            Config(**base_config)

    def test_experiments(self):
        base_config = deepcopy(self.base_config)

        # ======================================================================
        # Type =================================================================
        # ======================================================================
        base_config["experiments"] = [("foo", None)]

        with pytest.raises(
            ConfigError,
            match="Experiment group configuration must be a `collections.abc.Mapping` instance",
        ):
            Config(**base_config)

        # ======================================================================
        # Constraints ==========================================================
        # ======================================================================
        base_config = deepcopy(self.base_config)

        base_config["experiments"]["group_1"]["a"]["location"] = (
            "./tests/data/confusion_matrices/this_does_not_exist!!!.csv"
        )

        with pytest.raises(
            ConfigError,
        ):
            Config(**base_config)

        base_config = deepcopy(self.base_config)

        base_config["experiments"]["group_1"]["a"]["format"] = "foo"

        with pytest.raises(
            ConfigError,
            match=" Parameter `aggregation` must be a registered IO method",
        ):
            Config(**base_config)

        base_config = deepcopy(self.base_config)

        del base_config["experiments"]["group_1"]["__default__"]["type"]

        with pytest.raises(
            ConfigError,
        ):
            Config(**base_config)

        base_config = deepcopy(self.base_config)

        del base_config["experiments"]["group_1"]["__default__"]["prevalence_prior"]

        with pytest.warns(
            ConfigWarning,
        ):
            Config(**base_config)

    def test_wrong_metrics(self):
        base_config = deepcopy(self.base_config)

        base_config["metrics"] = [("foo", None)]

        with pytest.raises(
            ConfigError,
            match="The values in metrics must be a `collections.abc.Mapping` instance",
        ):
            Config(**base_config)
