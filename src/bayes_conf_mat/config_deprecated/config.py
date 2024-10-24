import typing
from warnings import warn
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

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


@dataclass(kw_only=True)
class Config:
    name: str
    seed: int
    num_samples: int | None = None
    ci_probability: float | None = None

    experiments: typing.Optional[OrderedDict] = None
    metrics: typing.Optional[OrderedDict] = None

    def __post_init__(self):
        self._validate_types()

        # Additional validation
        validation_methods = [
            method
            for method in dir(self)
            if "validate" in method and method != "_validate_types"
        ]

        for method in validation_methods:
            getattr(self, method)()

    def _validate_types(self):
        for param, expected_param_type in Config.__annotations__.items():
            param_val = getattr(self, param)

            if param_val is None:
                continue
            elif isinstance(param_val, expected_param_type):
                continue
            else:
                try:
                    super().__setattr__(param, expected_param_type(param_val))
                except Exception as e:
                    raise ConfigError(
                        f"Parameter `{param}` must be an instance of `{expected_param_type}`, but got `{type(param_val)}`. While trying to convert, the following exception was encountered: {e}"
                    )

    def _validate_name(self) -> None:
        pass

    def _validate_seed(self) -> None:
        if self.seed < 0:
            raise ConfigError(
                f"Parameter `seed` must be a positive int. Currently: {self.seed}"
            )

    def _validate_num_samples(self) -> None:
        if self.num_samples is None:
            warn(
                message="Parameter `num_samples` is `None`. Setting to default value of 10000. This value is arbitrary, however, and should be carefully considered.",
                category=ConfigWarning,
            )

            self.num_samples = 10000

        if self.num_samples <= 0:
            raise ConfigError(
                f"Parameter `num_samples` must be greater than 0. Currently: {self.num_samples}"
            )

        if self.num_samples < 10000:
            warn(
                message=f"Parameter `num_samples` should be large to reduce variability. Consider increasing. Currently: {self.num_samples}",
                category=ConfigWarning,
            )

    def _validate_ci_probability(self) -> None:
        if self.ci_probability is None:
            warn(
                message="Parameter `ci_probability` is `None`. Setting to default value of 0.95. This value is arbitrary, however, and should be carefully considered.",
                category=ConfigWarning,
            )

            self.ci_probability = 0.95

        if self.ci_probability < 0.0 or self.ci_probability > 1.0:
            raise ConfigError(
                f"Parameter `ci_probability` must be within [0.0, 1.0]. Currently: {self.ci_probability}"
            )

    def _validate_metrics(self) -> None:
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
                    f"The metric configuration for {key} is invalid. Currently: {configuration}. While trying to convert, the following exception was encountered: {e}"
                )

        if self.experiments is None:
            return None

        if not isinstance(self.metrics, Mapping):
            raise ConfigError(
                f"Metrics must be a `collections.abc.Mapping` instance. Currently: {type(self.metrics)}"
            )

        for k, v in self.metrics.items():
            if not isinstance(k, str):
                raise ConfigError(
                    f"The keys in metrics must of type `str`. Currently: {type(k)}"
                )

            if not isinstance(v, Mapping):
                raise ConfigError(
                    f"The values in metrics must be a `collections.abc.Mapping` instance. Currently: {type(v)}"
                )

        default_config = self.metrics.get("__default__", None)

        if default_config is not None:
            validate_metric_configuration("__default__", default_config)

        updated_metrics_config = OrderedDict()
        for metric_key, metric_config in self.metrics.items():
            # Do not validate the __default__ metric
            if metric_key == "__default__":
                continue

            # Validate the key of the metric config
            try:
                get_metric(metric_key)
            except Exception as e:
                raise ConfigError(
                    f"All metrics listed in `metrics` must be valid metric syntax strings. The following is invalid: `{metric_key}`. While trying to convert, the following exception was encountered: {e}"
                )

            # Validate the metric config
            if len(metric_config) == 0 and default_config is None:
                if len(self.experiments) > 1:
                    # Check for when requesting to aggregate
                    # Allow for studies where the user does not want to agrgegate
                    warn(
                        message="There are multiple experiments in `experiments`, but no aggregation method is provided for metric `{metric}`.",
                        category=ConfigWarning,
                    )

            elif len(metric_config) == 0:
                updated_metrics_config[metric_key] = default_config

            else:
                validate_metric_configuration(metric_key, metric_config)

                updated_metrics_config[metric_key] = metric_config

            super().__setattr__("metrics", updated_metrics_config)

    def _validate_experiments(self) -> None:
        if self.experiments is None:
            return None

        updated_experiments_config = dict()
        for experiment_group_name, experiment_group in self.experiments.items():
            if not isinstance(experiment_group_name, str):
                raise ConfigError(
                    f"Experiment group `{experiment_group_name}` must be an instance of `str`, but got `{type(experiment_group_name)}`."
                )

            if not isinstance(experiment_group, Mapping):
                raise ConfigError(
                    f"Experiment group configuration must be a `collections.abc.Mapping` instance. Currently: {type(experiment_group)}"
                )

            default_config = experiment_group.get("__default__", {})

            updated_experiment_group_config = dict()
            for experiment_name, experiment_config in experiment_group.items():
                if experiment_name == "__default__":
                    continue

                updated_experiment_config = dict()

                # Put everything from the '__default__' config into the experiment configs
                if len(default_config) > 0:
                    for k, v in default_config.items():
                        updated_experiment_config[k] = v

                # Check the experiment key
                if not isinstance(experiment_name, str):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` must be an instance of `str`, but got `{type(experiment_name)}`."
                    )

                # Validate location and format
                if (
                    "location" not in experiment_config
                    and experiment_config.get("format", None) != "in_memory"
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` config must contain a `location`. Currently: {experiment_config}."
                    )
                else:
                    updated_experiment_config["location"] = experiment_config[
                        "location"
                    ]

                if "format" not in experiment_config:
                    if "format" not in default_config:
                        raise ConfigError(
                            f"Experiment `{experiment_group_name}/{experiment_name}`'s configuration must contain a `format` argument. Currently: {updated_experiment_config}."
                        )
                else:
                    updated_experiment_config["format"] = experiment_config["format"]

                # ==============================================================
                # Validate form and type of the prevalence_prior ===============
                # ==============================================================
                if "prevalence_prior" not in experiment_config:
                    if "prevalence_prior" not in default_config:
                        warn(
                            f"Experiment `{experiment_group_name}/{experiment_name}`'s configuration does not contain a `prevalence_prior`. Defaulting to Haldane's 0 prior.",
                            category=ConfigWarning,
                        )

                        updated_experiment_config["prevalence_prior"] = 0.0

                else:
                    updated_experiment_config["prevalence_prior"] = experiment_config[
                        "prevalence_prior"
                    ]

                prevalence_prior = updated_experiment_config["prevalence_prior"]
                if (
                    isinstance(prevalence_prior, str)
                    and prevalence_prior not in _DIRICHLET_PRIOR_STRATEGIES
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` confusion prior is invalid. Currently: {prevalence_prior}. Must be one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                    )

                elif isinstance(prevalence_prior, list):
                    try:
                        prevalence_prior = np.array(prevalence_prior)
                    except Exception as e:
                        raise ConfigError(
                            f"Experiment `{experiment_group_name}/{experiment_name}` confusion prior is invalid. Currently: {prevalence_prior}. While trying to convert, the following exception was encountered: {e}"
                        )

                elif not (
                    isinstance(prevalence_prior, int)
                    or isinstance(prevalence_prior, float)
                    or isinstance(prevalence_prior, np.ndarray)
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` prevalence prior is of an invalid type. Currently: {type(prevalence_prior)}. Should be one of {{float, List[float], jtyping.Float[np.ndarray, 'num_classes']}}"
                    )

                # ==============================================================
                # Validate form and type of the confusion_prior ================
                # ==============================================================
                if "confusion_prior" not in experiment_config:
                    if "confusion_prior" not in default_config:
                        warn(
                            f"Experiment `{experiment_group_name}/{experiment_name}`'s configuration does not contain a `confusion_prior`. Defaulting to Haldane's 0 prior.",
                            category=ConfigWarning,
                        )

                        updated_experiment_config["confusion_prior"] = 0.0

                else:
                    updated_experiment_config["confusion_prior"] = experiment_config[
                        "confusion_prior"
                    ]

                confusion_prior = updated_experiment_config["confusion_prior"]
                if (
                    isinstance(confusion_prior, str)
                    and confusion_prior not in _DIRICHLET_PRIOR_STRATEGIES
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` confusion prior is invalid. Currently: {confusion_prior}. Must be one of: {set(_DIRICHLET_PRIOR_STRATEGIES.keys())}"
                    )

                elif isinstance(confusion_prior, list):
                    try:
                        confusion_prior = np.array(confusion_prior)
                    except Exception as e:
                        raise ConfigError(
                            f"Experiment `{experiment_group_name}/{experiment_name}` confusion prior is invalid. Currently: {confusion_prior}. While trying to convert, the following exception was encountered: {e}"
                        )

                elif not (
                    isinstance(confusion_prior, int)
                    or isinstance(confusion_prior, float)
                    or isinstance(confusion_prior, np.ndarray)
                ):
                    raise ConfigError(
                        f"Experiment `{experiment_group_name}/{experiment_name}` confusion prior is of an invalid type. Currently: {type(confusion_prior)}. Should be one of {{float, List[List[float]], jtyping.Float[np.ndarray, 'num_classes num_classes']}}"
                    )

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
                    if k
                    not in ["location", "format", "prevalence_prior", "confusion_prior"]
                }

                try:
                    get_io(
                        format=updated_experiment_config["format"],
                        location=updated_experiment_config["location"],
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

        super().__setattr__("experiments", updated_experiments_config)

    @property
    def num_experiments(self):
        return sum(map(len, self.experiments.values()))

    @property
    def num_experiment_groups(self):
        return len(self.experiments)

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: typing.Dict[str, typing.Any]) -> typing.Self:
        required_keys = {"name", "seed"}
        optional_keys = {"num_samples", "ci_probability", "experiments", "metrics"}

        missing_required_keys = required_keys - config_dict.keys()
        if len(missing_required_keys) > 0:
            raise ConfigError(
                f"Missing the following required keys: {missing_required_keys}"
            )

        missing_optional_keys = optional_keys - config_dict.keys()
        if len(missing_optional_keys) > 0:
            warn(
                message=f"Missing the following optional keys: {missing_optional_keys}",
                category=ConfigWarning,
            )

        parsed_config_dict = dict(
            name=config_dict["name"],
            seed=config_dict["seed"],
            num_samples=config_dict.get("num_samples", None),
            ci_probability=config_dict.get("ci_probability", None),
            experiments=config_dict.get("experiments", None),
            metrics=config_dict.get("metrics", None),
        )

        unused_keys = set(config_dict.keys() - parsed_config_dict.keys())
        if len(unused_keys) > 0:
            warn(
                message=f"The following keys were ignored: {unused_keys}",
                category=ConfigWarning,
            )

        instance = cls(**parsed_config_dict)

        return instance
