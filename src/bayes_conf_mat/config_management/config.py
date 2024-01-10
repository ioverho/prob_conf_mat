import typing
from pathlib import Path
from copy import deepcopy
from typing import Any

import strictyaml

from bayes_conf_mat.config_management.schema import schema
from bayes_conf_mat.config_management.frozen_attr_dict import FrozenAttrDict


class ConfigError(Exception):
    def __init__(self, message):
        super().__init__(message)


class Config:
    def __init__(self, config_file_path: str, encoding: str = "utf-8"):
        # First validation =====================================================
        # Checks for adherence to the schema
        # First check if parses as type-safe YAML document
        self.yaml_config = strictyaml.load(
            Path(config_file_path).read_text(), schema=schema
        )

        # Second validation ====================================================
        self._second_parsing()

        # Add some misc. info to the config ====================================
        self.yaml_config["__misc__"] = {"encoding": encoding}

        # Adds a whitespace before the `__misc__` key`
        self.yaml_config.as_marked_up().yaml_set_comment_before_after_key(
            "__misc__", "\n"
        )

    def _second_parsing(self):
        num_experiments = len(self.yaml_config["experiments"])

        # RULE: there must be at least one experiment in the experiment group.
        if num_experiments < 1:
            raise ConfigError(
                "There must be at least one experiment in the experiment group."
            )

        if num_experiments > 1:
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

    def __getattribute__(self, __name: str) -> Any:
        """Hacky method for letting the config act as an immutable attrdict/namespace."""
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
            file_name = self.group_name + "_parsed_config"

        with open(
            f"./{directory_path}/{file_name}.yaml", "w", encoding=self.__misc__.encoding
        ) as f:
            f.write(self.to_yaml())
