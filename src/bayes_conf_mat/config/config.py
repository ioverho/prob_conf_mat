import math
from copy import deepcopy
from warnings import warn
from itertools import combinations
import typing
from pathlib import Path

import strictyaml

from bayes_conf_mat.config.schema import schema
from bayes_conf_mat.config.frozen_attr_dict import FrozenAttrDict


ALLOWED_FILE_EXTENSIONS = {".yaml", ".yml"}


class ConfigWarning(Warning):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ConfigError(Exception):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def validate_config(
    config_location: str, encoding: str = "utf-8", return_yaml_config: bool = False
) -> FrozenAttrDict:
    config_location = Path(config_location).resolve()

    if config_location.suffix not in ALLOWED_FILE_EXTENSIONS:
        raise ConfigError(
            f"Config file at `{config_location}` must be one of {ALLOWED_FILE_EXTENSIONS}."
        )
    elif not config_location.exists():
        raise ConfigError(f"File path `{config_location}` does not exist.")
    elif not config_location.is_file():
        raise ConfigError(f"File path `{config_location}` is not a file.")

    parsed_yaml = config_location.read_text(encoding=encoding)

    yaml_config = strictyaml.load(parsed_yaml, schema=schema)

    config = FrozenAttrDict(yaml_config.data)

    config = validate_experiments(config=config)

    if "analysis" in config:
        validate_analysis(config=config)

    if return_yaml_config:
        return config, yaml_config
    else:
        return config


def validate_experiments(config: FrozenAttrDict) -> FrozenAttrDict:
    # ======================================================================
    # Validate the experiment aggregation methods
    # ======================================================================
    default_experiment_aggregation = config.metrics.get("__default__", None)
    if default_experiment_aggregation is not None:
        default_experiment_aggregation = deepcopy(default_experiment_aggregation)

        updated_experiment_aggregation = {
            k: v if v is not None else default_experiment_aggregation
            for k, v in config.metrics.items()
            if k != "__default__"
        }

        config._attrs["metrics"] = FrozenAttrDict(updated_experiment_aggregation)

    # * Experiment groups with a single experiment won't be aggregated
    # Warn the user that the aggregation will be ignored for single experiment experiment-groups
    if any(map(lambda x: len(x) == 1, config.experiments.values())):
        warn(
            "An experiment group with only 1 experiment exists. Experiment aggregations will not be applied to that experiment group.",
            category=ConfigWarning,
        )

    # RULE: if a metric group exists with more than 1 experiment, must define an experiment aggregation strategy for all metrics, or include a `__default__` metric.
    if any(map(lambda x: len(x) >= 1, config.experiments.values())):
        missing_metric_configs = list()
        for metric_name, aggregation_config in config.metrics.items():
            if (
                aggregation_config is None
                or aggregation_config.get("aggregation", None) is None
            ):
                missing_metric_configs.append(metric_name)

        if len(missing_metric_configs) > 0:
            raise ConfigError(
                f"An experiment group with more than 1 experiment exists, but some metrics are missing an aggregation configuration. Either add an `aggregation` tag, or define a `__default__` metric. Violating metrics: {missing_metric_configs}"
            )

    return config


def validate_analysis(config: FrozenAttrDict) -> FrozenAttrDict:
    # ==================================================================
    # Validate the analysis configuration
    # ==================================================================
    # RULE: if listwise is True, config must have at least 2 experiment groups
    if "pairwise_compare" in config.analysis and len(config.experiments) == 1:
        raise ConfigError(
            "If `pairwise_compare` is True, config must have at least 2 experiment groups."
        )

    if config.analysis.pairwise_compare == "all":
        num_combinations = math.comb(len(config.experiments), 2)

        # * Warn the user if there are too many pairwise comparisons
        if num_combinations > 10:
            warn(
                f"The number of pairwise comparison combinations is {num_combinations}. Considering reducing the number of pairwise comparisons, or experiment groups.",
                category=ConfigWarning,
            )

        combs = combinations(config.experiments.keys(), r=2)
        all_possible_experiment_group_combinations = list(combs)

        config.analysis._attrs[
            "pairwise_compare"
        ] = all_possible_experiment_group_combinations
    else:
        unknown_experiment_groups = set()
        for compare_a, compare_b in config.analysis.pairwise_compare:
            if compare_a not in config.experiments:
                unknown_experiment_groups.add(compare_a)

            if compare_b not in config.experiments:
                unknown_experiment_groups.add(compare_b)

        # RULE: all experiment groups included in the pairwise comparison list must be list in the experiments list as well
        if len(unknown_experiment_groups) > 0:
            raise ConfigError(
                f"Found unknown experiment group in `analysis.pairwise_compare`: {unknown_experiment_groups}"
            )

    if "min_sig_diff" in config.analysis:
        included_metrics = set(config.metrics.keys())
        included_msd_metrics = set(config.analysis.min_sig_diff.keys())

        # RULE: all metrics with a min_sig_diff must be included in the metrics section
        if len(included_msd_metrics - included_metrics) > 0:
            raise ConfigError(
                f"All metrics with a min_sig_diff must be included in the metrics section. Current offenders: {included_msd_metrics}"
            )

        for metric in included_metrics - included_msd_metrics:
            config.analysis.min_sig_diff._attrs[metric] = None

        # * Warn the user of any metrics with a default min_sig_diff value
        metrics_without_a_msd = [
            metric
            for metric, min_sig_diff_value in config.analysis.min_sig_diff.items()
            if min_sig_diff_value is None
        ]
        if len(metrics_without_a_msd) > 0:
            warn(
                f"Some metrics have no listed `min_sig_diff`, and will use 0.1 * stdev as default: {metrics_without_a_msd}",
                category=ConfigWarning,
            )
    elif "pairwise_compare" in config.analysis:
        config.analysis._attrs["min_sig_diff"] = FrozenAttrDict(
            {metric: None for metric in config.metrics.keys()}
        )

        warn(
            f"Some metrics have no listed `min_sig_diff`, and will use 0.1 * stdev as default: {config.analysis.min_sig_diff.keys()}",
            category=ConfigWarning,
        )

    if "listwise_compare" in config.analysis:
        config.analysis._attrs["listwise_compare"] = True

        # RULE: if listwise is True, config must have at least 2 experiment groups
        if len(config.experiments) == 1:
            raise ConfigError(
                "If `listwise` is True, config must have at least 2 experiment groups."
            )

        # * Warn the user of the equivalence of listwise and pairwise comparison when num_experiments == 2
        elif len(config.experiments) == 2:
            warn(
                "Listwise comparison is equivalent to pairwise comparison if there are only two experiment groups.",
                category=ConfigWarning,
            )

    return config


class Config:
    def __init__(self, config_location: str, encoding: str = "utf-8") -> None:
        self.encoding = encoding
        self.config_location = Path(config_location).resolve()

        if self.config_location.suffix not in ALLOWED_FILE_EXTENSIONS:
            raise ConfigError(
                f"Config file at `{self.config_location}` must be one of {ALLOWED_FILE_EXTENSIONS}."
            )
        elif not self.config_location.exists():
            raise ConfigError(f"File path `{self.config_location}` does not exist.")
        elif not self.config_location.is_file():
            raise ConfigError(f"File path `{self.config_location}` is not a file.")

        self.parsed_yaml = self.schema_validation()
        self._yaml_config = self.parsed_yaml.as_yaml()
        self.config = FrozenAttrDict(self.parsed_yaml.data)

        self.extended_validation()

    def schema_validation(self):
        return strictyaml.load(
            self.config_location.read_text(encoding=self.encoding), schema=schema
        )

    def extended_validation(self):
        self.validate_experiments()

        if "analysis" in self.config:
            self.validate_analysis()

    def __str__(self):
        return self._yaml_config


class DeprecatedConfig:
    def __init__(self, config_location: str, encoding: str = "utf-8"):
        # First validation =====================================================
        # Checks for adherence to the schema
        # First check if parses as type-safe YAML document
        config_location = Path(config_location)
        self.yaml_config = strictyaml.load(config_location.read_text(), schema=schema)

        # Second validation ====================================================
        self._second_parsing()

        # Add some misc. info to the config ====================================
        self.yaml_config["__misc__"] = {
            "original_location": str(config_location.absolute()),
            "encoding": encoding,
        }

        # Adds a whitespace before the `__misc__` key`
        self.yaml_config.as_marked_up().yaml_set_comment_before_after_key(
            "__misc__", "\n"
        )

    def _second_parsing(self):
        largest_experiment_group_size = -float("inf")
        for experiment_group in self.yaml_config["experiments"].values():
            num_experiments = len(experiment_group)
            if num_experiments > largest_experiment_group_size:
                largest_experiment_group_size = num_experiments

            # RULE: there must be at least one experiment in each experiment group.
            if num_experiments < 1:
                raise ConfigError(
                    "There must be at least one experiment in the experiment group."
                )

        if largest_experiment_group_size > 1:
            no_metric_kwargs = list()
            for metric in self.yaml_config["metrics"].keys():
                if len(self.yaml_config["metrics"][metric]) == 0:
                    no_metric_kwargs.append(metric)

            if "__default__" not in self.yaml_config["metrics"]:
                # RULE: if there is more than 1 experiment, each metric must provide experiment aggregation
                # or user must provide a `__default__` fallback
                raise ConfigError(
                    "If there is more than 1 experiment, the user must either must provide a metric aggregation "
                    + " for each metric, or the user must provide a `__default__` metric. Currently fails for:"
                    + f" {no_metric_kwargs}"
                )
            else:
                # Put the default metric into the other metrics' config
                default_metric_kwargs = deepcopy(
                    self.yaml_config["metrics"]["__default__"]
                )
                del self.yaml_config["metrics"]["__default__"]

                for metric in no_metric_kwargs:
                    self.yaml_config["metrics"][metric] = default_metric_kwargs

        # RULE: the CI probability must be between 0 and 1, exclusive
        # if (
        #    self.yaml_config["ci_probability"].data <= 0.0
        #    or self.yaml_config["ci_probability"].data >= 1.0
        # ):
        #    raise ConfigError("The CI probability must be between 0 and 1, exclusive")

    def __getattribute__(self, __name: str) -> typing.Any:
        """Hacky method for letting this config act as an immutable attrdict/namespace."""
        try:
            return super().__getattribute__(__name)
        except AttributeError as e:
            try:
                attr = self.yaml_config[__name].data
                if isinstance(attr, dict):
                    return FrozenAttrDict(attr)
                else:
                    return attr

            except KeyError:
                raise e

    def to_yaml(self):
        return self.yaml_config.as_yaml()

    def to_dict(self):
        return self.yaml_config.data

    def dump(self, directory_path: str, file_name: typing.Optional[str] = None):
        if file_name is None:
            file_name = "parsed_config"

        with open(
            f"{directory_path}/{file_name}.yaml", "w", encoding=self.__misc__.encoding
        ) as f:
            f.write(self.to_yaml())

    def __str__(self):
        return self.to_yaml()
