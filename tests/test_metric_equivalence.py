from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.experiment import Experiment

from functools import partial
from pathlib import Path
from itertools import product

import pytest
import numpy as np
import sklearn
import sklearn.metrics

from prob_conf_mat import Study
from prob_conf_mat.io import (
    load_csv,
    confusion_matrix_to_pred_cond,
)

# ==============================================================================
# Setup all test cases
# ==============================================================================
# Metrics ======================================================================
METRICS_TO_SKLEARN = {
    "acc": sklearn.metrics.accuracy_score,
    "ba": sklearn.metrics.balanced_accuracy_score,
    "ba+adjusted=True": partial(sklearn.metrics.balanced_accuracy_score, adjusted=True),
    "cohen_kappa": sklearn.metrics.cohen_kappa_score,
    "mcc": sklearn.metrics.matthews_corrcoef,
    "f1": partial(sklearn.metrics.f1_score, average=None),
    "jaccard": partial(sklearn.metrics.jaccard_score, average=None),
    "plr@binary+positive_class=0": lambda y_true,
    y_pred: sklearn.metrics.class_likelihood_ratios(
        y_true=y_true,
        y_pred=y_pred,
        raise_warning=False,
    )[0],
    "nlr@binary+positive_class=0": lambda y_true,
    y_pred: sklearn.metrics.class_likelihood_ratios(
        y_true=y_true,
        y_pred=y_pred,
        raise_warning=False,
    )[1],
    "dor@binary+positive_class=0": lambda y_true,
    y_pred: sklearn.metrics.class_likelihood_ratios(
        y_true=y_true,
        y_pred=y_pred,
        raise_warning=False,
    )[0]
    / sklearn.metrics.class_likelihood_ratios(
        y_true=y_true,
        y_pred=y_pred,
        raise_warning=False,
    )[1],
    **{
        f"fbeta+beta={beta}": partial(
            sklearn.metrics.fbeta_score,
            average=None,
            beta=beta,
        )
        for beta in [0.0, 0.5, 1.0, 2.0]
    },
}

# Confusion matrices ===========================================================
TEST_CASES_DIR = Path("./tests/data/confusion_matrices")

# Their combination ============================================================
all_metrics_to_test = METRICS_TO_SKLEARN.keys()

all_confusion_matrices_to_test = list(TEST_CASES_DIR.glob(pattern="*.csv"))

all_args_to_test = list(
    product(
        all_confusion_matrices_to_test,
        all_metrics_to_test,
    ),
)


# ==============================================================================
# The 'assert' code for pytest
# ==============================================================================
@pytest.mark.parametrize(
    argnames="conf_mat_fp,metric",
    argvalues=all_args_to_test,
)
def test_metric_equivalence(conf_mat_fp, metric) -> None:
    def _get_our_value(
        metric: str,
        study: Study,
    ) -> jtyping.Float[np.ndarray, " num_classes"]:
        metric_result = study.get_metric_samples(
            metric=metric,
            experiment_name="test/test",
            sampling_method="input",
        )

        our_value = metric_result.values[0]

        return our_value

    def _get_sklearn_value(
        metric: str,
        study: Study,
    ) -> jtyping.Float[np.ndarray, " *num_classes"]:
        experiment: Experiment = study["test/test"]  # type: ignore
        conf_mat = experiment.confusion_matrix

        pred_cond = confusion_matrix_to_pred_cond(
            confusion_matrix=conf_mat,
            pred_first=True,
        )

        # Need to binarize the cond_pred array, otherwise sklearn complains
        if "binary+positive_class=0" in metric:
            pred_cond = np.where(pred_cond == 0, 1, 0)

        sklearn_func = METRICS_TO_SKLEARN[metric]

        match metric:
            # For some reason cohen's kappa has a different function signature
            case "cohen_kappa":
                sklearn_value = sklearn_func(y1=pred_cond[:, 0], y2=pred_cond[:, 1])
            case "dor@binary+positive_class=0":
                # Catch zero division error, these aren't compared anyways
                try:
                    sklearn_value = sklearn_func(
                        y_pred=pred_cond[:, 0],
                        y_true=pred_cond[:, 1],
                    )
                except ZeroDivisionError:
                    sklearn_value = np.array(np.nan)
            case _:
                sklearn_value = sklearn_func(
                    y_pred=pred_cond[:, 0],
                    y_true=pred_cond[:, 1],
                )

        return sklearn_value

    study = Study(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    study.add_metric(metric=metric)

    conf_mat = load_csv(
        location=conf_mat_fp,
    )

    study.add_experiment(
        experiment_name="test/test",
        confusion_matrix=conf_mat,
        prevalence_prior=0,
        confusion_prior=0,
    )

    our_value = _get_our_value(metric=metric, study=study)

    sklearn_value = _get_sklearn_value(metric=metric, study=study)

    # Only test if either array has finite values
    if np.all(np.isfinite(our_value)) or np.all(np.isfinite(sklearn_value)):
        assert np.allclose(our_value, sklearn_value), (our_value, sklearn_value)
    # If both report non-finite numbers, accept
    else:
        assert True
