from pathlib import Path

import strictyaml

from bayes_conf_mat.config.schema import schema
from bayes_conf_mat.config.frozen_attr_dict import Config
from bayes_conf_mat.config.validation import (
    ConfigError,
    validate_experiments,
    validate_analysis,
)

ALLOWED_FILE_EXTENSIONS = {".yaml", ".yml"}


def load_config_from_text(config_string: str) -> Config:
    yaml_config = strictyaml.load(config_string, schema=schema)

    config = Config(yaml_config)

    config = validate_experiments(config=config)

    if "analysis" in config:
        validate_analysis(config=config)

    return config


def load_config_from_file(config_location: str, encoding: str = "utf-8") -> Config:
    config_location = Path(config_location).resolve()

    if config_location.suffix not in ALLOWED_FILE_EXTENSIONS:
        raise ConfigError(
            f"Config file at `{config_location}` must be one of {ALLOWED_FILE_EXTENSIONS}."
        )
    elif not config_location.exists():
        raise ConfigError(f"File path `{config_location}` does not exist.")
    elif not config_location.is_file():
        raise ConfigError(f"File path `{config_location}` is not a file.")

    parsed_text = config_location.read_text(encoding=encoding)

    config = load_config_from_text(parsed_text)

    return config
