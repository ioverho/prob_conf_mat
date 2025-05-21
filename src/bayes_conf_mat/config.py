import typing
import warnings
from collections import OrderedDict
import hashlib
import pickle
import time

import numpy as np

from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator
from bayes_conf_mat.stats import _DIRICHLET_PRIOR_STRATEGIES
from bayes_conf_mat.utils import RNG


class ConfigWarning(Warning):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ConfigError(Exception):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class Config:
    def __init__(
        self,
        seed: typing.Optional[int] = None,
        num_samples: typing.Optional[int] = None,
        ci_probability: typing.Optional[float] = None,
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

    def _validate_type(self, parameter: str, value: typing.Any) -> None:
        # Get the type we're expecting to see for this parameter
        expected_type: type = Config.__init__.__annotations__[parameter]

        # Check if optional type
        allows_none: bool = isinstance(expected_type, type(typing.Optional[float]))

        # If Optional, use non-optional type as expected type
        if allows_none:
            expected_type = expected_type.__args__[0]  # type: ignore

        if (value is None) and allows_none:
            return value
        elif isinstance(value, expected_type):
            return value

        # Try to convert to the correct value
        try:
            value = expected_type(value)
        except Exception as e:
            raise ConfigError(
                f"Parameter `{parameter}` must be an instance of `{expected_type}`, but got `{type(value)}`. While trying to convert, the following exception was encountered: {e}"
            )

        return value

    @property
    def seed(self) -> int:
        return self._seed

    def _validate_seed(self, value: int) -> int:
        if value is None:
            value = int(time.time() * 256)

            warnings.warn(
                f"Recieved `None` as seed. Defaulting to fractional seconds: {value}",
                category=ConfigWarning,
            )

            self.seed = value

        if value < 0:
            raise ConfigError(
                f"Parameter `seed` must be a positive int. Currently: {self.seed}"
            )

        return value

    @seed.setter
    def seed(self, value: int) -> None:
        value_ = self._validate_type(parameter="seed", value=value)
        value_ = self._validate_seed(value=value)

        self._seed = value_
        self.rng.seed = self._seed

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def _validate_num_samples(self, value) -> int:
        if value is None:
            warnings.warn(
                message="Parameter `num_samples` is `None`. Setting to default value of 10000. This value is arbitrary, however, and should be carefully considered.",
                category=ConfigWarning,
            )

            value = 10000

        if value <= 0:
            raise ConfigError(
                f"Parameter `num_samples` must be greater than 0. Currently: {value}"
            )

        if value < 10000:
            warnings.warn(
                message=f"Parameter `num_samples` should be large to reduce variability. Consider increasing. Currently: {value}",
                category=ConfigWarning,
            )

        return value

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        value_ = self._validate_type(parameter="num_samples", value=value)
        value_ = self._validate_num_samples(value=value)

        self._num_samples = value_

    @property
    def ci_probability(self) -> float:
        return self._ci_probability

    def _validate_ci_probability(self, value: float) -> float:
        if value is None:
            warnings.warn(
                message="Parameter `ci_probability` is `None`. Setting to default value of 0.95. This value is arbitrary, however, and should be carefully considered.",
                category=ConfigWarning,
            )

            value = 0.95

        if not (value > 0.0 and value <= 1.0):
            raise ConfigError(
                f"Parameter `ci_probability` must be within (0.0, 1.0]. Currently: {value}"
            )

        return value

    @ci_probability.setter
    def ci_probability(self, value: float) -> None:
        value_ = self._validate_type(parameter="ci_probability", value=value)
        value_ = self._validate_ci_probability(value=value)

        self._ci_probability = value_

    @property
    def experiments(self) -> dict[str, dict[str, dict[str, typing.Any]]]:
        return self._experiments

    def _validate_experiments(
        self,
        value: dict[str, dict[str, dict[str, typing.Any]]],
    ) -> dict[str, dict[str, dict[str, typing.Any]]]:
        if value is None:
            return dict()

        if not (hasattr(value, "get") and hasattr(value, "items")):
            raise ConfigError(
                f"The experiments configuration must implement the `get` and `items` attributes like a `dict`. Current type: {type(value)}"
            )

        updated_experiments_config = dict()
        for experiment_group_name, experiment_group in value.items():
            if experiment_group_name == "aggregated":
                raise ConfigError("An experiment group may not be named 'aggregated'.")

            if not isinstance(experiment_group_name, str):
                try:
                    experiment_group_name = str(experiment_group_name)
                except Exception as e:
                    raise ConfigError(
                        f"Experiment group `{experiment_group_name}` must be an instance of `str`, but got `{type(experiment_group_name)}`. While trying to convert, ran into the following exception: {e}"
                    )

            if not (
                hasattr(experiment_group, "get") and hasattr(experiment_group, "items")
            ):
                raise ConfigError(
                    f"The experiment group configuration must implement the `get` and `items` attributes like a `dict`. Currently: {type(experiment_group)}"
                )

            # Check for a __default__ config option
            # Use it to initialise missing values
            default_config = experiment_group.get("__default__", {})

            updated_experiment_group_config = dict()
            for experiment_name, experiment_config in experiment_group.items():
                if experiment_name == "aggregated":
                    raise ConfigError("An experiment may not be named 'aggregated'.")

                if experiment_name == "__default__":
                    if len(experiment_group) == 1:
                        raise ConfigError(
                            f"Experiment group {experiment_group_name} has no experiments, only a '__default__' key: {experiment_group}"
                        )

                    continue

                updated_experiment_config = dict()

                # Put everything from the '__default__' config into the experiment configs
                if len(default_config) > 0:
                    for k, v in default_config.items():
                        updated_experiment_config[k] = v

                # Check the experiment key
                if not isinstance(experiment_name, str):
                    try:
                        experiment_name = str(experiment_name)
                    except Exception as e:
                        raise ConfigError(
                            f"The key for `{experiment_group_name}/{experiment_name}` must be an instance of `str`, but got `{type(experiment_group_name)}`. While trying to convert, ran into the following exception: {e}"
                        )

                # Put all remaining experiment configuration items into the config
                # Does not perform validation right now
                updated_experiment_config.update(experiment_config)

                # ==============================================================
                # Validate location ============================================
                # ==============================================================
                # TODO: check if location is valid/exists??
                # if (
                #   "location" not in experiment_config
                #   and experiment_config.get("format", "") != "in_memory"
                # ):
                #   raise ConfigError(
                #       f"Experiment `{experiment_group_name}/{experiment_name}` must contain a `location` key. Currently: {experiment_config}."
                #   )

                # elif isinstance(
                #   experiment_config.get("location", ""),
                #   (str | bytes | os.PathLike)
                #   ):
                #   updated_experiment_config["location"] = experiment_config[
                #       "location"
                #   ]

                # else:
                #   raise ConfigError(
                #       f"Experiment `{experiment_group_name}/{experiment_name}` location is of invalid type. Must be one of {{str, bytes, os.PathLike}}. Currently: {type(experiment_config['location'])}."
                #   )

                # ==============================================================
                # Validate format ==============================================
                # ==============================================================
                # if ("format" not in experiment_config) and (
                #    "format" not in default_config
                # ):
                #    raise ConfigError(
                #        f"Either experiment `{experiment_group_name}/{experiment_name}` or `{experiment_group_name}/__default__` must contain a `format` key. Currently: {updated_experiment_config}."
                #    )
                # else:
                #    updated_experiment_config["format"] = experiment_config["format"]

                # ==============================================================
                # Validate form and type of the prevalence_prior ===============
                # ==============================================================
                # First check if the key is in there, and fall back to standard default if not
                if ("prevalence_prior" not in experiment_config) and (
                    "prevalence_prior" not in default_config
                ):
                    warnings.warn(
                        f"Experiment `{experiment_group_name}/{experiment_name}` or `{experiment_group_name}/__default__` must contain a `prevalence_prior` key. Defaulting to the 0 (Haldane) prior.",
                        category=ConfigWarning,
                    )
                    warnings.warn(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s prevalence prior is `None`. Defaulting to the 0 (Haldane) prior.",
                        category=ConfigWarning,
                    )

                    prevalence_prior = 0.0

                elif "prevalence_prior" in default_config:
                    prevalence_prior = default_config["prevalence_prior"]

                else:
                    prevalence_prior = experiment_config["prevalence_prior"]

                # Then check format
                if prevalence_prior is None:
                    prevalence_prior = 0.0

                elif (
                    isinstance(prevalence_prior, str)
                    and prevalence_prior not in _DIRICHLET_PRIOR_STRATEGIES
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s prevalence prior is invalid. Currently: {prevalence_prior}. If `str`, must be one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                    )

                elif isinstance(prevalence_prior, list):
                    try:
                        prevalence_prior = np.array(prevalence_prior)
                    except Exception as e:
                        raise ConfigError(
                            f"Experiment `{experiment_group_name}/{experiment_name}`'s prevalence prior is invalid. Currently: {prevalence_prior}. While trying to convert to `np.ndarray`, the following exception was encountered: {e}"
                        )

                elif not (
                    isinstance(prevalence_prior, int)
                    or isinstance(prevalence_prior, float)
                    or isinstance(prevalence_prior, np.ndarray)
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s prevalence prior is of an invalid type. Currently: {type(prevalence_prior)}. Should be one of {{str, int, float, jtyping.Float[np.ArrayLike, 'num_classes']}}"
                    )

                updated_experiment_config["prevalence_prior"] = prevalence_prior

                # ==============================================================
                # Validate form and type of the confusion_prior ================
                # ==============================================================
                # First check if the key is in there, and fall back to standard default if not
                if ("confusion_prior" not in experiment_config) and (
                    "confusion_prior" not in default_config
                ):
                    warnings.warn(
                        f"Experiment `{experiment_group_name}/{experiment_name}` or `{experiment_group_name}/__default__` must contain a `confusion_prior` key. Defaulting to the 0 (Haldane) prior.",
                        category=ConfigWarning,
                    )

                    confusion_prior = 0.0

                elif "confusion_prior" in default_config:
                    confusion_prior = default_config["confusion_prior"]

                else:
                    confusion_prior = experiment_config["confusion_prior"]

                # Then check format
                if confusion_prior is None:
                    warnings.warn(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s confusion prior is `None`. Defaulting to the 0 (Haldane) prior.",
                        category=ConfigWarning,
                    )

                    confusion_prior = 0.0

                elif (
                    isinstance(confusion_prior, str)
                    and confusion_prior not in _DIRICHLET_PRIOR_STRATEGIES
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s confusion prior is invalid. Currently: {confusion_prior}. If `str`, must be one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                    )

                elif isinstance(confusion_prior, list):
                    try:
                        confusion_prior = np.array(confusion_prior)
                    except Exception as e:
                        raise ConfigError(
                            f"Experiment `{experiment_group_name}/{experiment_name}`'s confusion prior is invalid. Currently: {confusion_prior}. While trying to convert to `np.ndarray`, the following exception was encountered: {e}"
                        )

                elif not (
                    isinstance(confusion_prior, int)
                    or isinstance(confusion_prior, float)
                    or isinstance(confusion_prior, np.ndarray)
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s confusion prior is of an invalid type. Currently: {type(confusion_prior)}. Should be one of {{str, int, float, jtyping.Float[np.ArrayLike, 'num_classes']}}"
                    )

                updated_experiment_config["confusion_prior"] = confusion_prior

                # ==============================================================
                # Validate misc IO kwargs ======================================
                # ==============================================================
                io_kwargs = {
                    k: v
                    for k, v in experiment_config.items()
                    if k not in ["prevalence_prior", "confusion_prior"]
                }

                for k, v in io_kwargs.items():
                    updated_experiment_config[k] = v

                updated_experiment_group_config[experiment_name] = (
                    updated_experiment_config
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
        self, value: dict[str, dict[str, typing.Any]]
    ) -> dict[str, dict[str, typing.Any]]:
        def validate_metric_configuration(key: str, configuration: dict) -> None:
            # Empty configuration is allowed
            if len(configuration) == 0:
                return None

            #! If non-empty, must contain an aggregation key
            if "aggregation" not in configuration:
                raise ConfigError(
                    f"The metric configuration for {key} must contain an `aggregation` key. Currently: {configuration}"
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

            except Exception as e:
                raise ConfigError(
                    f"The aggregation configuration for metric {key} is invalid. Currently: {configuration}. While trying to parse, the following exception was encountered: {e}"
                )

        # If not yet initialized, initialize to an empty dictionary
        if value is None:
            return dict()

        if not (hasattr(value, "get") and hasattr(value, "items")):
            raise ConfigError(
                f"Metrics configuration must implement the `get` and `items` attributes like a `dict`. Current type: {type(value)}"
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
                except Exception as e:
                    raise ConfigError(
                        f"The keys in metrics must of type `str`. Currently: {type(metric_key)}. While trying to convert, the following exception was encountered: {e}"
                    )

            if not (hasattr(value, "get") and hasattr(value, "items")):
                raise ConfigError(
                    f"Configuration for metric {metric_key} must implement the `get` and `items` attributes like a `dict`. Current type: {type(value)}"
                )

            # Validate the key of the metric config ============================
            try:
                get_metric(metric_key)
            except Exception as e:
                raise ConfigError(
                    f"The following metric is an invalid metric syntax string: `{metric_key}`. While trying to parse, the following exception was encountered: {e}"
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
                        message=f"There is an experiment group with multiple experiments, but no aggregation method is provided for metric `{metric_key}`.",
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
                    key=metric_key, configuration=metric_config
                )

                updated_metrics_config[metric_key] = metric_config

        return updated_metrics_config

    @metrics.setter
    def metrics(self, value: dict[str, dict[str, typing.Any]]) -> None:
        value = self._validate_metrics(value=value)

        self._metrics = value

    def to_dict(self) -> dict[str, typing.Any]:
        state_dict = {
            "seed": self.seed,
            "num_samples": self.num_samples,
            "ci_probability": self.ci_probability,
            "experiments": dict(self.experiments),
            "metrics": dict(self.metrics),
        }

        return state_dict

    @classmethod
    def from_dict(cls, config_dict: dict[str, typing.Any]) -> typing.Self:
        # Currently, there are no required parameters
        # But might change this in the future
        required_keys = dict()
        optional_keys = Config.__init__.__annotations__.keys()

        missing_required_keys = required_keys - config_dict.keys()
        if len(missing_required_keys) > 0:
            raise ConfigError(
                f"Missing the following required keys: {missing_required_keys}"
            )

        missing_optional_keys = optional_keys - config_dict.keys()
        if len(missing_optional_keys) > 0:
            warnings.warn(
                message=f"Missing the following optional keys: {missing_optional_keys}",
                category=ConfigWarning,
            )

        parsed_config_dict = dict()

        for parameter in [
            "seed",
            "num_samples",
            "ci_probability",
            "experiments",
            "metrics",
        ]:
            parameter_value = config_dict.get(parameter, None)

            if parameter_value is not None:
                parsed_config_dict[parameter] = parameter_value

        unused_keys = set(config_dict.keys() - parsed_config_dict.keys())
        if len(unused_keys) > 0:
            warnings.warn(
                message=f"The following keys were ignored: {unused_keys}",
                category=ConfigWarning,
            )

        instance = cls(**parsed_config_dict)

        return instance

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(pickle.dumps(self.to_dict()))

        hash = hasher.hexdigest()

        return hash

    def __hash__(self) -> int:
        return self.fingerprint.__hash__()
