import strictyaml

from bayes_conf_mat.math.dirichlet_distribution import _PRIOR_STRATEGIES

top_level_validator = {
    "group_name": strictyaml.Str(),
    "seed": strictyaml.Int(),
    "num_samples": strictyaml.Int(),
    strictyaml.Optional("prior_strategy", default="laplace"): strictyaml.Enum(
        _PRIOR_STRATEGIES
    ),
    strictyaml.Optional("num_proc", default=0): strictyaml.Int(),
}

experiment_validator = {
    "experiments": strictyaml.MapPattern(
        strictyaml.Str(),
        strictyaml.Map(
            {
                "location": strictyaml.Str(),
                "type": strictyaml.Enum(["confusion_matrix", "pred_cond", "cond_pred"]),
                "format": strictyaml.Enum(["txt", "csv", "pickle", "numpy"]),
                strictyaml.Optional("location_2", default=None): strictyaml.Str(),
            }
        ),
    )
}

metrics_validator = {
    # TODO: setup special validation for matching yaml config to metric string
    "metrics": strictyaml.MapPattern(
        strictyaml.Str(),
        strictyaml.MapCombined(
            {
                # TODO: setup special validation for experiment aggregation
                # strictyaml.Optional(
                #    "aggregation", default=None
                # ): strictyaml.MapCombined(),
            },
            strictyaml.Str(),
            strictyaml.Any(),
        )
        | strictyaml.EmptyDict(),
    )
}

misc_validator = {
    strictyaml.Optional("__misc__"): strictyaml.Map(
        {
            "encoding": strictyaml.Str(),
        }
    )
}

schema = strictyaml.Map(
    top_level_validator | experiment_validator | metrics_validator | misc_validator
)
