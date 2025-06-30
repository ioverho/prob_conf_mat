from pathlib import Path
from itertools import product


import prob_conf_mat as pcm
from prob_conf_mat.metrics import METRIC_REGISTRY, AVERAGING_REGISTRY
from prob_conf_mat.io import load_csv


def test_get_metric_values() -> None:
    all_metrics_avgs_combinations = [
        v.aliases[0] for v in METRIC_REGISTRY.values() if v.is_multiclass
    ] + [
        f"{m}@{a}"
        for m, a in product(
            [v.aliases[0] for v in METRIC_REGISTRY.values() if not v.is_multiclass],
            [v.aliases[0] for v in AVERAGING_REGISTRY.values()],
        )
    ]

    study = pcm.Study(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    # Add a bunch of experiments
    conf_mat_paths = Path(
        "./documentation/Getting Started/mnist_digits",
    )
    for file_path in sorted(conf_mat_paths.glob("*.csv")):
        # Split the file name to recover the model and fold
        model, fold = file_path.stem.split("_")

        # Load in the confusion matrix using the utility function
        confusion_matrix = load_csv(location=file_path)

        # Add the experiment to the study
        study.add_experiment(
            experiment_name=f"{model}/fold_{fold}",
            confusion_matrix=confusion_matrix,
            prevalence_prior=0,
            confusion_prior=0,
        )

    for metric_str in all_metrics_avgs_combinations:
        # Add a bucnh of metrics
        study.add_metric(metric=metric_str, aggregation="fe_gaussian")

    for metric_str in all_metrics_avgs_combinations:
        study.get_metric_samples(
            metric=metric_str,
            experiment_name="mlp/aggregated",
            sampling_method="posterior",
        )
