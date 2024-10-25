import typing
import warnings
from collections import OrderedDict
import hashlib
import pickle

import numpy as np

from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.experiment_aggregation import get_experiment_aggregator
from bayes_conf_mat.io import get_io
from bayes_conf_mat.stats import _DIRICHLET_PRIOR_STRATEGIES


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
        experiments: typing.Optional[
            typing.Mapping[str, typing.Mapping[str, typing.Mapping[str, typing.Any]]]
        ] = None,
        metrics: typing.Optional[
            typing.Mapping[str, typing.Mapping[str, typing.Any]]
        ] = None,
    ):
        # TODO: use more sensible initial values for these parameters, e.g., empty mappings for experiments and metrics
        # Generate the slots for these values
        self._seed = None
        self._num_samples = None
        self._ci_probability = None
        self._experiments = None
        self._metrics = None

        # Now actually set these values, with baked in validation
        self.seed: int = seed
        self.num_samples: int = num_samples
        self.ci_probability: float = ci_probability
        self.experiments: dict[str, dict[str, dict[str, typing.Any]]] = experiments
        self.metrics: dict[str, dict[str, typing.Any]] = metrics

    def _validate_type(self, parameter: str, value: typing.Any) -> None:
        # Get the type we're expecting to see for this parameter
        expected_type: typing.Type = Config.__init__.__annotations__[parameter]

        # Check if optional type
        allows_none: bool = isinstance(expected_type, type(typing.Optional[float]))

        # If Optional, use non-optional type as expected type
        if allows_none:
            expected_type = expected_type.__args__[0]

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

    def _validate_seed(self, value: int) -> None:
        if value < 0:
            raise ConfigError(
                f"Parameter `seed` must be a positive int. Currently: {self.seed}"
            )

        return value

    @seed.setter
    def seed(self, value: int):
        value = self._validate_type(parameter="seed", value=value)
        value = self._validate_seed(value=value)

        self._seed = value

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
    def num_samples(self, value: int):
        value = self._validate_type(parameter="num_samples", value=value)
        value = self._validate_num_samples(value=value)

        self._num_samples = value

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

        if value < 0.0 or value > 1.0:
            raise ConfigError(
                f"Parameter `ci_probability` must be within [0.0, 1.0]. Currently: {value}"
            )

        return value

    @ci_probability.setter
    def ci_probability(self, value: float):
        value = self._validate_type(parameter="ci_probability", value=value)
        value = self._validate_ci_probability(value=value)

        self._ci_probability = value

    @property
    def experiments(self):
        return self._experiments

    def _validate_experiments(
        self,
        value: typing.Mapping[
            str, typing.Mapping[str, typing.Mapping[str, typing.Any]]
        ],
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

                # ==============================================================
                # Validate location ============================================
                # ==============================================================
                # TODO: check if location is valid/exists??
                # if (
                #    "location" not in experiment_config
                #    and experiment_config.get("format", "") != "in_memory"
                # ):
                #    raise ConfigError(
                #        f"Experiment `{experiment_group_name}/{experiment_name}` must contain a `location` key. Currently: {experiment_config}."
                #    )

                # elif isinstance(
                #    experiment_config.get("location", ""),
                #    (str | bytes | os.PathLike)
                #    ):
                #    updated_experiment_config["location"] = experiment_config[
                #        "location"
                #    ]

                # else:
                #    raise ConfigError(
                #        f"Experiment `{experiment_group_name}/{experiment_name}` location is of invalid type. Must be one of {{str, bytes, os.PathLike}}. Currently: {type(experiment_config['location'])}."
                #    )

                # ==============================================================
                # Validate format ==============================================
                # ==============================================================
                if ("format" not in experiment_config) and (
                    "format" not in default_config
                ):
                    raise ConfigError(
                        f"Either experiment `{experiment_group_name}/{experiment_name}` or `{experiment_group_name}/__default__` must contain a `format` key. Currently: {updated_experiment_config}."
                    )
                else:
                    updated_experiment_config["format"] = experiment_config["format"]

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
                    if k
                    not in ["location", "format", "prevalence_prior", "confusion_prior"]
                }

                for k, v in io_kwargs.items():
                    updated_experiment_config[k] = v

                updated_io_kwargs = {
                    k: v
                    for k, v in updated_experiment_config.items()
                    if k not in ["format", "prevalence_prior", "confusion_prior"]
                }

                try:
                    get_io(
                        format=updated_experiment_config["format"],
                        # location=updated_experiment_config["location"],
                        **updated_io_kwargs,
                    )
                except Exception as e:
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}`'s configuration is invalid. Currently: {updated_experiment_config}. While trying to fetch the specific IO method, encountered the following exception: {e}"
                    )

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
        value: typing.Mapping[
            str, typing.Mapping[str, typing.Mapping[str, typing.Any]]
        ],
    ):
        value = self._validate_experiments(value)

        self._experiments = value

    @property
    def num_experiments(self):
        return sum(map(len, self.experiments.values()))

    @property
    def num_experiment_groups(self):
        return len(self.experiments)

    @property
    def metrics(self) -> typing.Any:
        return self._metrics

    def _validate_metrics(
        self, value: typing.Mapping[str, typing.Mapping[str, typing.Any]]
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
                kwargs = dict({k for k in configuration if k != "aggregation"})
                get_experiment_aggregator(
                    configuration["aggregation"],
                    rng=0,
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

        default_config = value.get("__default__", None)

        if default_config is not None:
            validate_metric_configuration("__default__", default_config)

        updated_metrics_config = OrderedDict()
        for metric_key, metric_config in value.items():
            # Do not validate the __default__ metric
            if metric_key == "__default__":
                continue

            # Validate type ====================================================
            if not isinstance(metric_key, str):
                try:
                    k = str(metric_key)
                except Exception as e:
                    raise ConfigError(
                        f"The keys in metrics must of type `str`. Currently: {type(k)}. While trying to convert, the following exception was encountered: {e}"
                    )

            if not (hasattr(value, "get") and hasattr(value, "items")):
                raise ConfigError(
                    f"Configuration for metric {k} must implement the `get` and `items` attributes like a `dict`. Current type: {type(value)}"
                )

            # Validate the key of the metric config ============================
            try:
                get_metric(metric_key)
            except Exception as e:
                raise ConfigError(
                    f"The following metric not a invalid metric syntax string: `{metric_key}`. While trying to parse, the following exception was encountered: {e}"
                )

            # Validate the metric config =======================================
            if (
                metric_config is None or len(metric_config) == 0
            ) and default_config is None:
                # If no metric aggregation config has been passed, make sure
                # there are not more than 1 experiment groups
                if max(map(len, self.experiments.values())) > 1:
                    # Check for when requesting to aggregate
                    # Allow for studies where the user does not want to aggregate
                    warnings.warn(
                        message=f"There is an experiment group with multiple experiments, but no aggregation method is provided for metric `{metric_key}`.",
                        category=ConfigWarning,
                    )

                updated_metrics_config[metric_key] = None

            elif (
                metric_config is None or len(metric_config) == 0
            ) and default_config is not None:
                updated_metrics_config[metric_key] = default_config

            else:
                # Otherwise, validate each metric aggregation configuration
                validate_metric_configuration(metric_key, metric_config)

                updated_metrics_config[metric_key] = metric_config

        return updated_metrics_config

    @metrics.setter
    def metrics(
        self, value: typing.Mapping[str, typing.Mapping[str, typing.Any]]
    ) -> dict[str, dict[str, typing.Any]]:
        value = self._validate_metrics(value=value)

        self._metrics = value

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        state_dict = {
            "seed": self.seed,
            "num_samples": self.num_samples,
            "ci_probability": self.ci_probability,
            "experiments": self.experiments,
            # "metrics": self.metrics,
        }

        return state_dict

    @classmethod
    def from_dict(cls, config_dict: typing.Dict[str, typing.Any]) -> typing.Self:
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

        parsed_config_dict = dict(
            seed=config_dict.get("seed", None),
            num_samples=config_dict.get("num_samples", None),
            ci_probability=config_dict.get("ci_probability", None),
            experiments=config_dict.get("experiments", None),
            metrics=config_dict.get("metrics", None),
        )

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
