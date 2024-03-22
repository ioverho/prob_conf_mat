import strictyaml

from bayes_conf_mat.stats.dirichlet_distribution import _DIRICHLET_PRIOR_STRATEGIES
from bayes_conf_mat.io.base import IO_REGISTRY
from bayes_conf_mat.experiment_aggregation.base import AGGREGATION_REGISTRY

top_level_validator = {
    "name": strictyaml.Str(),
    "seed": strictyaml.Int(),
    "num_samples": strictyaml.Int(),
    # Default values for prior is now 0.0, aka Haldane's prior
    strictyaml.Optional(
        "prevalence_prior", default=0.0, drop_if_none=False
    ): strictyaml.Enum(_DIRICHLET_PRIOR_STRATEGIES)
    | strictyaml.Float()
    | strictyaml.CommaSeparated(strictyaml.Float()),
    strictyaml.Optional(
        "confusion_prior", default=0.0, drop_if_none=False
    ): strictyaml.Enum(_DIRICHLET_PRIOR_STRATEGIES)
    | strictyaml.Float()
    | strictyaml.Seq(strictyaml.CommaSeparated(strictyaml.Float())),
}

experiments_validator = {
    "experiments": strictyaml.MapPattern(
        key_validator=strictyaml.Str(),
        minimum_keys=1,
        value_validator=strictyaml.MapPattern(
            key_validator=strictyaml.Str(),
            minimum_keys=1,
            value_validator=strictyaml.MapCombined(
                map_validator={
                    "location": strictyaml.Str(),
                    "format": strictyaml.Enum(IO_REGISTRY),
                },
                key_validator=strictyaml.Str(),
                value_validator=strictyaml.Any(),
            ),
        ),
    )
}

metrics_validator = {
    "metrics": strictyaml.MapPattern(
        key_validator=strictyaml.Str(),
        value_validator=strictyaml.EmptyNone()
        | strictyaml.MapCombined(
            map_validator={"aggregation": strictyaml.Enum(AGGREGATION_REGISTRY)},
            key_validator=strictyaml.Str(),
            value_validator=strictyaml.Any(),
        ),
        minimum_keys=1,
    )
}

analysis_validator = {
    strictyaml.Optional("analysis"): strictyaml.MapCombined(
        {
            "ci_probability": strictyaml.Float(),
            strictyaml.Optional(
                "pairwise_compare", drop_if_none=True
            ): strictyaml.UniqueSeq(strictyaml.CommaSeparated(strictyaml.Str()))
            | strictyaml.Regex(r"(all)"),
            strictyaml.Optional(
                "min_sig_diff", drop_if_none=True
            ): strictyaml.MapPattern(
                key_validator=strictyaml.Str(),
                value_validator=strictyaml.Float(),
                minimum_keys=1,
            ),
            strictyaml.Optional(
                "listwise_compare", drop_if_none=True
            ): strictyaml.EmptyNone() | strictyaml.Bool(),
        },
        key_validator=strictyaml.Str(),
        value_validator=strictyaml.Any(),
    )
}

schema = strictyaml.MapCombined(
    top_level_validator
    | experiments_validator
    | metrics_validator
    | analysis_validator,
    key_validator=strictyaml.Str(),
    value_validator=strictyaml.Any(),
)
