from collections import OrderedDict
import hashlib
import pickle
import time
import typing
import warnings

import numpy as np

from prob_conf_mat.metrics import get_metric
from prob_conf_mat.experiment_aggregation import get_experiment_aggregator
from prob_conf_mat.io import validate_confusion_matrix
from prob_conf_mat.stats import _DIRICHLET_PRIOR_STRATEGIES
from prob_conf_mat.utils import RNG


# TODO: document this class
class ConfigWarning(Warning):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


# TODO: document this class
class ConfigError(Exception):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


# TODO: document this class
class Config:
    def __init__(
        self,
        seed: int | None = None,
        num_samples: int | None = None,
        ci_probability: float | None = None,
        experiments: dict[str, dict[str, dict[str, typing.Any]]] = {},
        metrics: dict[str, dict[str, typing.Any]] = {},
    ) -> None:
        # Set the RNG
        # Allows for potentially updating the seed
        self.rng = RNG(seed=None)

        # Set the initial values
        self.__setattr__("seed", seed)
        self.__setattr__("num_samples", num_samples)
        self.__setattr__("ci_probability", ci_probability)
        self.__setattr__("experiments", experiments)
        self.__setattr__("metrics", metrics)

    @property
    def seed(self) -> int:
        return self._seed

    def _validate_seed(self, value: int | None) -> int:
        # Handle default seed
        if value is None:
            value = int(time.time() * 256)

            warnings.warn(
                f"Recieved `None` as seed. Defaulting to fractional seconds: {value}",
                category=ConfigWarning,
            )

        # Handle seed of wrong type
        else:
            if not isinstance(value, int):
                try:
                    initial_value_type = type(value)
                    value = int(value)

                    warnings.warn(
                        (
                            f"Parameter `seed` must be a positive integer. "
                            f"Received: {initial_value_type}. Parsed as: {value}"
                        ),
                        category=ConfigWarning,
                    )

                except Exception as e:  # noqa: BLE001
                    raise TypeError(
                        (
                            f"Parameter `seed` must be a positive integer. "
                            f"Currently: {type(value)}. While trying to convert"
                            f"encountered the following exception: {e}"
                        ),
                    )

        # Validate seed value
        if value < 0:
            raise ConfigError(
                f"Parameter `seed` must be a positive integer. Currently: {value}.",
            )

        return value

    @seed.setter
    def seed(self, value: int | None) -> None:
        value_: int = self._validate_seed(value=value)

        self._seed = value_
        self.rng.seed = self._seed

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def _validate_num_samples(self, value: int | None) -> int:
        # Handle default parameter
        if value is None:
            warnings.warn(
                message=(
                    "Parameter `num_samples` is `None`. "
                    "Setting to default value of 10000. "
                    "This value is arbitrary, however, and should be carefully considered."
                ),
                category=ConfigWarning,
            )

            value = 10000

        # Handle parameter of wrong type
        else:
            if not isinstance(value, int):
                try:
                    initial_value_type = type(value)
                    value = int(value)

                    warnings.warn(
                        (
                            f"Parameter `num_samples` must be a strictly positive integer. "
                            f"Received: {initial_value_type}. Parsed as: {value}"
                        ),
                        category=ConfigWarning,
                    )

                except Exception as e:  # noqa: BLE001
                    raise TypeError(
                        (
                            f"Parameter `num_samples` must be a strictly positive integer. "
                            f"Currently: {type(value)}. While trying to convert encountered "
                            f"the following exception: {e}"
                        ),
                    )

        # Validate num_samples value
        if value <= 0:
            raise ConfigError(
                f"Parameter `num_samples` must be greater than 0. Currently: {value}",
            )

        # TODO: consider increasing the recommended value to from 10e+4 to 10e+5
        if value < 10000:
            warnings.warn(
                message=(
                    f"Parameter `num_samples` should be large to reduce variability. "
                    f"Consider increasing. Currently: {value}"
                ),
                category=ConfigWarning,
            )

        return value

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        value_ = self._validate_num_samples(value=value)

        self._num_samples = value_

    @property
    def ci_probability(self) -> float:
        return self._ci_probability

    def _validate_ci_probability(self, value: float | None) -> float:
        # Handle default parameter
        if value is None:
            warnings.warn(
                message=(
                    "Parameter `ci_probability` is `None`. Setting to default value of 0.95. "
                    "This value is arbitrary, however, and should be carefully considered."
                ),
                category=ConfigWarning,
            )

            value = 0.95

        # Handle parameter of wrong type
        else:
            if not isinstance(value, float):
                try:
                    initial_value_type = type(value)
                    value = float(value)

                    warnings.warn(
                        (
                            f"Parameter `ci_probability` must be a float. "
                            f"Received: {initial_value_type}. Parsed as: {value}"
                        ),
                        category=ConfigWarning,
                    )

                except Exception as e:  # noqa: BLE001
                    raise TypeError(
                        (
                            "Parameter `ci_probability` must be a float. "
                            f"Currently: {type(value)}. "
                            f"While trying to convert encountered the following exception: {e}"
                        ),
                    )

        # Validate ci_probability value
        if not (value > 0.0 and value <= 1.0):
            raise ConfigError(
                f"Parameter `ci_probability` must be within (0.0, 1.0]. Currently: {value}",
            )

        return value

    @ci_probability.setter
    def ci_probability(self, value: float) -> None:
        value_ = self._validate_ci_probability(value=value)

        self._ci_probability = value_

    @property
    def experiments(self) -> dict[str, dict[str, dict[str, typing.Any]]]:
        return self._experiments

    def _validate_experiments(
        self,
        value: dict[str, dict[str, dict[str, typing.Any]]] | None,
    ) -> dict[str, dict[str, dict[str, typing.Any]]]:
        def _validate_single_experiment(
            experiment_name: str,
            experiment_config: dict[str, typing.Any],
        ) -> dict[str, typing.Any]:
            updated_experiment_config = dict()

            # ==========================================================================
            # Confusion matrix
            # Expected type: Int[ArrayLike,'num_classes num_classes']
            # ==========================================================================
            if "confusion_matrix" not in experiment_config:
                raise ConfigError(
                    f"Experiment '{experiment_name}' has no confusion matrix. Please add one.",
                )
            confusion_matrix = experiment_config["confusion_matrix"]

            confusion_matrix = validate_confusion_matrix(
                confusion_matrix=confusion_matrix,
            )

            updated_experiment_config["confusion_matrix"] = confusion_matrix

            # ==========================================================================
            # Prevalence prior
            # Expected type: str | float | Float[ArrayLike,'num_classes']
            # ==========================================================================
            # First check if the key is in there, and fall back to standard default if not
            prevalence_prior = experiment_config.get("prevalence_prior", None)

            # Handle default parameter
            if prevalence_prior is None:
                warnings.warn(
                    (
                        f"Experiment '{experiment_name}'s prevalence prior is `None`. "
                        f"Defaulting to the 0 (Haldane) prior."
                    ),
                    category=ConfigWarning,
                )

                prevalence_prior = 0.0

            # Check if the string is a valid strategy
            elif isinstance(prevalence_prior, str):
                if prevalence_prior not in _DIRICHLET_PRIOR_STRATEGIES:
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s prevalence prior is invalid. "
                            f"Currently: {prevalence_prior}. If `str`, must be one of: "
                            f"{set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                        ),
                    )

            # Accept positive integer and float values
            elif isinstance(prevalence_prior, int | float):
                if prevalence_prior < 0:
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s prevalence prior is invalid. "
                            f"Currently: {prevalence_prior}. If numeric, must be greater than 0."
                        ),
                    )

            # Try to convert anything else to a numpy array
            else:
                try:
                    prevalence_prior = np.array(prevalence_prior, dtype=np.float64)
                except Exception as e:  # noqa: BLE001
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s prevalence prior is invalid. "
                            f"Expecting a type of str | float | Float[ArrayLike,'num_classes']. "
                            f"Currently: {type(prevalence_prior)}. "
                            f"While trying to convert to `np.ndarray`, the following exception "
                            f"was encountered: {e}"
                        ),
                    )

                # Additional numpy array validation
                # Check shape against confusion matrix
                if (
                    len(prevalence_prior.shape) != 1
                    or prevalence_prior.shape[0] != confusion_matrix.shape[0]
                ):
                    try:
                        prevalence_prior = prevalence_prior.reshape(
                            (confusion_matrix.shape[0],),
                        )
                    except Exception as e:  # noqa: BLE001
                        raise ConfigError(
                            (
                                f"Experiment '{experiment_name}'s prevalence prior is malformed. "
                                f"Expecting a 1D vector with length equal to the number of classes "
                                f"in the experiment. Current shape: {prevalence_prior.shape}. "
                                f"Expecting shape: {(confusion_matrix.shape[0],)}. "
                                f"While trying to reshape, encountered the following exception: {e}"
                            ),
                        )

                # Check that values are all positive
                if not np.all(prevalence_prior > 0.0):
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s prevalence prior is invalid. "
                            f"If providing an arraylike of values, all values must be positive."
                        ),
                    )

            updated_experiment_config["prevalence_prior"] = prevalence_prior

            # ==========================================================================
            # Confusion prior
            # Expected type: str | float | Float[ArrayLike,'num_classes num_classes']
            # ==========================================================================
            # First check if the key is in there, and fall back to standard default if not
            confusion_prior = experiment_config.get("confusion_prior", None)

            # Handle default parameter value
            if confusion_prior is None:
                warnings.warn(
                    (
                        f"Experiment '{experiment_name}'s confusion prior is `None`. "
                        f"Defaulting to the 0 (Haldane) prior."
                    ),
                    category=ConfigWarning,
                )

                confusion_prior = 0.0

            elif isinstance(confusion_prior, str):
                if confusion_prior not in _DIRICHLET_PRIOR_STRATEGIES:
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s confusion prior is invalid. "
                            f"Currently: {confusion_prior}. If `str`, must be one of: "
                            f"{set(_DIRICHLET_PRIOR_STRATEGIES.keys())}",
                        ),
                    )

            # Accept positive integer and float values
            elif isinstance(confusion_prior, int | float):
                if confusion_prior < 0:
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s confusion prior is invalid. "
                            f"Currently: {confusion_prior}. If numeric, must be greater than 0."
                        ),
                    )

            else:
                try:
                    confusion_prior = np.array(confusion_prior, dtype=np.float64)
                except Exception as e:  # noqa: BLE001
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s confusion prior is invalid. "
                            f"Expecting a type of str | float | Float[ArrayLike,'num_classes']. "
                            f"Currently: {type(confusion_prior)}. While trying to convert to "
                            f"`np.ndarray`, the following exception was encountered: {e}"
                        ),
                    )

                # Check shape against confusion matrix
                if (
                    (len(confusion_prior.shape) != 2)
                    or (confusion_prior.shape[0] != confusion_matrix.shape[0])
                    or (confusion_prior.shape[1] != confusion_matrix.shape[1])
                    or (confusion_prior.shape[0] != confusion_prior.shape[1])
                ):
                    try:
                        confusion_prior = confusion_prior.reshape(
                            confusion_matrix.shape,
                        )
                    except Exception as e:  # noqa: BLE001
                        raise ConfigError(
                            (
                                f"Experiment '{experiment_name}'s confusion prior is malformed. "
                                f"Expecting a square 2D matrix with length equal to the number of "
                                f"classes in the experiment. "
                                f"Current shape: {confusion_prior.shape}. "
                                f"Expecting shape: {confusion_matrix.shape}. "
                                f"While trying to reshape, encountered the following exception: {e}"
                            ),
                        )

                # Check that values are all positive
                if not np.all(confusion_prior > 0.0):
                    raise ConfigError(
                        (
                            f"Experiment '{experiment_name}'s confusion prior is invalid. "
                            f"If providing an arraylike of values, all values must be positive."
                        ),
                    )

            updated_experiment_config["confusion_prior"] = confusion_prior

            # ==========================================================================
            # Kwargs
            # Not expecting any of these for now
            # ==========================================================================
            kwargs = {
                k: v
                for k, v in experiment_config.items()
                if k not in ["confusion_matrix", "prevalence_prior", "confusion_prior"]
            }

            if len(kwargs) != 0:
                warnings.warn(
                    (
                        f"Experiment '{experiment_name}'s received the following "
                        f"superfluous parameters: {set(kwargs.keys())}. "
                        f"These are currently just ignored."
                    ),
                    category=ConfigWarning,
                )

            updated_experiment_config.update(kwargs)

            return updated_experiment_config

        # Handle default value
        experiments_config = dict() if value is None else value

        # Duck type to make sure it matches dict protocol
        if not (
            hasattr(experiments_config, "get") and hasattr(experiments_config, "items")
        ):
            raise ConfigError(
                (
                    f"The experiments configuration must implement the `get` and `items` "
                    f"attributes like a `dict`. Current type: {type(value)}"
                ),
            )

        updated_experiments_config = dict()
        for experiment_group_name, experiment_group in experiments_config.items():
            # ==================================================================
            # Experiment group name
            # ==================================================================
            if experiment_group_name == "aggregated":
                raise ConfigError("An experiment group may not be named 'aggregated'.")

            if not isinstance(experiment_group_name, str):
                try:
                    experiment_group_name = str(experiment_group_name)
                except Exception as e:  # noqa: BLE001
                    raise ConfigError(
                        (
                            f"Experiment group `{experiment_group_name}` must be an instance of "
                            f"`str`, but got `{type(experiment_group_name)}`. "
                            f"While trying to convert, ran into the following exception: {e}"
                        ),
                    )

            # Duck type to make sure it matches dict protocol
            if not (
                hasattr(experiment_group, "get") and hasattr(experiment_group, "items")
            ):
                raise ConfigError(
                    (
                        f"The experiment group configuration must implement the `get` and `items` "
                        f"attributes like a `dict`. Currently: {type(experiment_group)}"
                    ),
                )

            # ==================================================================
            # Experiment group
            # ==================================================================
            confusion_matrix_shapes = set()
            updated_experiment_group_config = dict()
            for experiment_name, experiment_config in experiment_group.items():
                # ==============================================================
                # Experiment name
                # ==============================================================
                if experiment_name == "aggregated":
                    raise ConfigError("An experiment may not be named 'aggregated'.")

                if not isinstance(experiment_name, str):
                    try:
                        experiment_name = str(experiment_name)
                    except Exception as e:  # noqa: BLE001
                        raise ConfigError(
                            (
                                f"The key for `{experiment_group_name}/{experiment_name}` "
                                f"must be an instance of `str`, but got "
                                f"`{type(experiment_group_name)}`. "
                                f"While trying to convert, ran into the following exception: {e}"
                            ),
                        )

                # ==============================================================
                # Experiment
                # ==============================================================
                full_experiment_name = f"{experiment_group_name}/{experiment_name}"
                updated_experiment_config = _validate_single_experiment(
                    experiment_name=full_experiment_name,
                    experiment_config=experiment_config,
                )

                updated_experiment_group_config[experiment_name] = (
                    updated_experiment_config
                )

                confusion_matrix_shapes.add(
                    updated_experiment_config["confusion_matrix"].shape,
                )

            if len(confusion_matrix_shapes) != 1:
                raise ConfigError(
                    (
                        f"Experiment group '{experiment_group_name}' has "
                        f"incongruent confusion matrices. "
                        f"Found shapes: {confusion_matrix_shapes}"
                    ),
                )

            updated_experiments_config[experiment_group_name] = (
                updated_experiment_group_config
            )

        return updated_experiments_config

    @experiments.setter
    def experiments(
        self,
        value: dict[str, dict[str, dict[str, typing.Any]]],
    ) -> None:
        value = self._validate_experiments(value)

        self._experiments = value

    @property
    def num_experiments(self) -> int:
        return sum(map(len, self.experiments.values()))

    @property
    def num_experiment_groups(self) -> int:
        return len(self.experiments)

    @property
    def metrics(self) -> dict[str, dict[str, typing.Any]]:
        return self._metrics

    def _validate_metrics(
        self,
        value: dict[str, dict[str, typing.Any]],
    ) -> dict[str, dict[str, typing.Any]]:
        def validate_metric_configuration(key: str, configuration: dict) -> None:
            # Empty configuration is allowed
            if len(configuration) == 0:
                return

            #! If non-empty, must contain an aggregation key
            if "aggregation" not in configuration:
                raise ConfigError(
                    (
                        f"The metric configuration for {key} must contain an `aggregation` key. "
                        f"Currently: {configuration}"
                    ),
                )

            #! Aggregation key must map to registered aggregation
            #! Aggregation config must be valid
            try:
                kwargs = {k: v for k, v in configuration.items() if k != "aggregation"}
                get_experiment_aggregator(
                    configuration["aggregation"],
                    rng=RNG(None),
                    **kwargs,
                )

            except Exception as e:  # noqa: BLE001
                raise ConfigError(
                    (
                        f"The aggregation configuration for metric {key} is invalid. "
                        f"Currently: {configuration}. "
                        f"While trying to parse, the following exception was encountered: {e}"
                    ),
                )

        # If not yet initialized, initialize to an empty dictionary
        if value is None:
            return dict()

        if not (hasattr(value, "get") and hasattr(value, "items")):
            raise ConfigError(
                (
                    f"Metrics configuration must implement the `get` and `items` "
                    f"attributes like a `dict`. "
                    f"Current type: {type(value)}"
                ),
            )

        default_config = value.get("__default__", dict())

        # Validate the default config
        if len(default_config) != 0:
            validate_metric_configuration("__default__", default_config)

        updated_metrics_config = OrderedDict()
        for metric_key, metric_config in value.items():
            # Do not validate the __default__ config
            if metric_key == "__default__":
                continue

            # Validate type ====================================================
            if not isinstance(metric_key, str):
                try:
                    metric_key = str(metric_key)
                except Exception as e:  # noqa: BLE001
                    raise ConfigError(
                        (
                            f"The keys in metrics must of type `str`. "
                            f"Currently: {type(metric_key)}. "
                            f"While trying to convert, the following exception was encountered: {e}"
                        ),
                    )

            if not (hasattr(value, "get") and hasattr(value, "items")):
                raise ConfigError(
                    (
                        f"Configuration for metric {metric_key} must implement the "
                        f"`get` and `items` attributes like a `dict`. "
                        f"Current type: {type(value)}"
                    ),
                )

            # Validate the key of the metric config ============================
            try:
                get_metric(metric_key)
            except Exception as e:  # noqa: BLE001
                raise ConfigError(
                    (
                        f"The following metric is an invalid metric syntax string: `{metric_key}`. "
                        f"While trying to parse, the following exception was encountered: {e}"
                    ),
                )

            # Validate the metric config =======================================
            if len(metric_config) == 0 and len(default_config) == 0:
                # If no metric aggregation config has been passed, make sure
                # there are not more than 1 experiment groups
                if (
                    len(self.experiments) > 0
                    and max(map(len, self.experiments.values())) > 1
                ):
                    # Check for when requesting to aggregate
                    # Allow for studies where the user does not want to aggregate
                    warnings.warn(
                        message=(
                            f"There is an experiment group with multiple experiments, "
                            f"but no aggregation method is provided for metric `{metric_key}`."
                        ),
                        category=ConfigWarning,
                    )

                updated_metrics_config[metric_key] = dict()

            elif len(metric_config) == 0 and len(default_config) != 0:
                # No metric aggregation config has been passed, but a default
                # metric aggregation dict does exist
                updated_metrics_config[metric_key] = default_config

            else:
                # Otherwise, validate each metric aggregation configuration
                validate_metric_configuration(
                    key=metric_key,
                    configuration=metric_config,
                )

                updated_metrics_config[metric_key] = metric_config

        return updated_metrics_config

    @metrics.setter
    def metrics(self, value: dict[str, dict[str, typing.Any]]) -> None:
        value = self._validate_metrics(value=value)

        self._metrics = value

    def to_dict(self) -> dict[str, typing.Any]:
        """Returns the Config as a Pythonic dict."""
        # Necessary to make sure all types are Pythonic
        updated_experiments_config = dict()
        for experiment_group_name, experiment_group in self.experiments.items():
            updated_experiment_group_config = dict()
            for experiment_name, experiment_config in experiment_group.items():
                updated_experiment_config = dict()

                updated_experiment_config["confusion_matrix"] = experiment_config[
                    "confusion_matrix"
                ].tolist()

                if isinstance(experiment_config["prevalence_prior"], np.ndarray):
                    updated_experiment_config["prevalence_prior"] = experiment_config[
                        "prevalence_prior"
                    ].tolist()
                else:
                    updated_experiment_config["prevalence_prior"] = experiment_config[
                        "prevalence_prior"
                    ]

                if isinstance(experiment_config["confusion_prior"], np.ndarray):
                    updated_experiment_config["confusion_prior"] = experiment_config[
                        "confusion_prior"
                    ].tolist()
                else:
                    updated_experiment_config["confusion_prior"] = experiment_config[
                        "confusion_prior"
                    ]

                updated_experiment_group_config[experiment_name] = (
                    updated_experiment_config
                )

            updated_experiments_config[experiment_group_name] = (
                updated_experiment_group_config
            )

        # Final state dict
        state_dict = {
            "seed": int(self.seed),
            "num_samples": int(self.num_samples),
            "ci_probability": float(self.ci_probability),
            "experiments": updated_experiments_config,
            "metrics": dict(self.metrics),
        }

        return state_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> typing.Self:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(pickle.dumps(self.to_dict()))

        fingerprint = hasher.hexdigest()

        return fingerprint

    def __hash__(self) -> int:
        return self.fingerprint.__hash__()
