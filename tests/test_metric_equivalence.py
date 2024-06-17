from functools import partial
from pathlib import Path

import numpy as np
import sklearn
import sklearn.metrics

from bayes_conf_mat import Study
from bayes_conf_mat.metrics import (
    MetricCollection,
    METRIC_REGISTRY,
    AveragedMetric,
    RootMetric,
)
from bayes_conf_mat.io.utils.conversion import confusion_matrix_to_pred_cond

METRICS_TO_SKLEARN = {
    "acc": sklearn.metrics.accuracy_score,
    "ba": sklearn.metrics.balanced_accuracy_score,
    "ba+adjusted=True": partial(sklearn.metrics.balanced_accuracy_score, adjusted=True),
    "cohen_kappa": sklearn.metrics.cohen_kappa_score,
    "mcc": sklearn.metrics.matthews_corrcoef,
    "f1": partial(sklearn.metrics.f1_score, average=None),
    "jaccard": partial(sklearn.metrics.jaccard_score, average=None),
    "plr@binary+positive_class=0": lambda x, y: sklearn.metrics.class_likelihood_ratios(
        x,
        y,
        raise_warning=False,
    )[0],
    "nlr@binary+positive_class=0": lambda x, y: sklearn.metrics.class_likelihood_ratios(
        x,
        y,
        raise_warning=False,
    )[1],
    "dor@binary+positive_class=0": lambda x, y: sklearn.metrics.class_likelihood_ratios(
        x,
        y,
        raise_warning=False,
    )[0]
    / sklearn.metrics.class_likelihood_ratios(
        x,
        y,
        raise_warning=False,
    )[1],
    **{
        f"fbeta+beta={beta}": partial(
            sklearn.metrics.fbeta_score, average=None, beta=beta
        )
        for beta in [0.0, 0.5, 1.0, 2.0]
    },
}

TEST_CASES_DIR = Path("./tests/data/confusion_matrices")


def metrics_case(conf_mat_loc: str):
    def get_test_config(conf_mat_loc: str, metrics_string: str):
        config_str = f"""
name: test
seed: 942
num_samples: 1

prevalence_prior: 0.0
confusion_prior: 0.0

ci_probability: 0.95

experiments:
    test:
        test:
            location: {conf_mat_loc}
            format: csv
            type: confusion_matrix

metrics:
    __default__:
        aggregation: beta
        estimation_method: mle
{metrics_string}

"""
        return config_str

    config_string = get_test_config(
        conf_mat_loc=str(conf_mat_loc),
        metrics_string="".join(
            map(lambda x: f"    {x}:\n", list(METRICS_TO_SKLEARN.keys()))
        ),
    )

    study = Study.from_config(config=config_string)

    # Convert the confusion matrix to y_true, y_pred
    conf_mat = study.experiment_groups["test"].experiments["test"].confusion_matrix

    pred_cond = confusion_matrix_to_pred_cond(
        conf_mat,
        pred_first=True,
    )

    y_pred = pred_cond[:, 0]
    y_true = pred_cond[:, 1]

    # Get the metric values from bmc
    metrics_results_dict = next(
        study.sample_metrics(sampling_method="input", aggregated=False)
    )[1]

    # Iterate over the metric results and compare
    for metric, values in metrics_results_dict.items():
        values = np.ravel(np.stack(list(map(lambda x: x[0].values, values))))

        # Some sklearn metrics have a different interface
        if metric[:3] in {"plr", "nlr", "dor"}:
            # Binarize the labels
            y_true_binary = list(map(lambda x: 1 if x == 0 else 0, y_true))
            y_pred_binary = list(map(lambda x: 1 if x == 0 else 0, y_pred))

            # The sklearn function is not consistent here...
            # Sometimes it yields nan, sometimes it yield inf
            sklearn_values = METRICS_TO_SKLEARN[metric](y_true_binary, y_pred_binary)

            values = np.nan_to_num(x=values, nan=np.nan, posinf=np.nan, neginf=np.nan)
            sklearn_values = np.nan_to_num(
                x=sklearn_values, nan=np.nan, posinf=np.nan, neginf=np.nan
            )
        else:
            sklearn_values = METRICS_TO_SKLEARN[metric](y_true, y_pred)

        assert np.allclose(
            values, sklearn_values, equal_nan=True
        ), f"{metric}, {values}, {sklearn_values}, {conf_mat_loc}"


def test_metrics():
    for path in TEST_CASES_DIR.glob("*.csv"):
        metrics_case(path)

    # Track untested metrics
    tested_metrics = MetricCollection(
        list(METRICS_TO_SKLEARN.keys()) + ["prevalence", "model_bias"]
    )

    # Assume that if all tested metrics, their dependencies are correct also
    tested_metric_classes = set()
    for metric_instance in set(tested_metrics.get_compute_order()._metrics.keys()):
        if isinstance(metric_instance, RootMetric):
            continue
        elif isinstance(metric_instance, AveragedMetric):
            metric_class = metric_instance.base_metric.__class__
        else:
            metric_class = metric_instance.__class__

        tested_metric_classes.add(metric_class)

    all_metric_classes = {metric for _, metric in METRIC_REGISTRY.items()}

    untested_metrics = all_metric_classes - tested_metric_classes
    with open("./tests/untested_metrics.txt", "w") as f:
        for metric in sorted(list(map(lambda x: x.__name__, untested_metrics))):
            f.write(f"{metric}\n")
