import math
from copy import deepcopy
from warnings import warn
from itertools import combinations

from bayes_conf_mat.config.frozen_attr_dict import Config, FrozenAttrDict


class ConfigWarning(Warning):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class ConfigError(Exception):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def validate_experiments(config: Config) -> Config:
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


def validate_analysis(config: Config) -> Config:
    # ==================================================================
    # Validate the analysis configuration
    # ==================================================================
    # RULE: if pairwise is True, config must have at least 2 experiment groups
    if "pairwise_compare" in config.analysis:
        if len(config.experiments) == 1:
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

            config.analysis._attrs["pairwise_compare"] = (
                all_possible_experiment_group_combinations
            )
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
