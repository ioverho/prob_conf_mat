from functools import partial
from pathlib import Path
from itertools import product

import pytest
import numpy as np
import jaxtyping as jtyping
import sklearn
import sklearn.metrics

from bayes_conf_mat import Study
from bayes_conf_mat.io.utils import confusion_matrix_to_pred_cond

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
            sklearn.metrics.fbeta_score, average=None, beta=beta
        )
        for beta in [0.0, 0.5, 1.0, 2.0]
    },
}

# Confusion matrices ===========================================================
TEST_CASES_DIR = Path("./tests/data/confusion_matrices")

# Their combination ============================================================
all_metrics_to_test = METRICS_TO_SKLEARN.keys()

all_confusion_matrices_to_test = TEST_CASES_DIR.glob("*.csv")

all_test_cases = list(product(all_metrics_to_test, all_confusion_matrices_to_test))


# ==============================================================================
# The 'arrange' code for pytest
# ==============================================================================
@pytest.fixture
def test_case(request):
    def _setup_study(metric: str, conf_mat_fp: str) -> Study:
        study = Study()

        study.add_metric(metric)

        study.add_experiment(
            "test/test",
            confusion_matrix=dict(
                location=conf_mat_fp,
                format="csv",
                type="conf_mat",
            ),
        )

        return study

    def _get_our_value(
        metric: str, study: Study
    ) -> jtyping.Float[np.ndarray, " num_classes"]:
        metric_result = study.get_metric_samples(
            metric, experiment="test", experiment_group="test", sampling_method="input"
        )

        our_value = metric_result.values[0]

        return our_value

    def _get_sklearn_value(
        metric: str, study: Study
    ) -> jtyping.Float[np.ndarray, " *num_classes"]:
        conf_mat = study.experiment_groups["test"].experiments["test"].confusion_matrix

        pred_cond = confusion_matrix_to_pred_cond(
            confusion_matrix=conf_mat, pred_first=True
        )

        # Need to binarize the cond_pred array, otherwise sklearn complains
        if "binary+positive_class=0" in metric:
            pred_cond = np.where(pred_cond == 0, 1, 0)

        sklearn_func = METRICS_TO_SKLEARN[metric]

        if metric == "cohen_kappa":
            sklearn_value = sklearn_func(y1=pred_cond[:, 0], y2=pred_cond[:, 1])
        else:
            sklearn_value = sklearn_func(y_pred=pred_cond[:, 0], y_true=pred_cond[:, 1])

        return sklearn_value

    metric, conf_mat_fp = request.param

    study = _setup_study(metric=metric, conf_mat_fp=conf_mat_fp)

    our_value = _get_our_value(metric=metric, study=study)

    sklearn_value = _get_sklearn_value(metric=metric, study=study)

    return our_value, sklearn_value


# ==============================================================================
# The 'assert' code for pytest
# ==============================================================================
@pytest.mark.parametrize("test_case", all_test_cases, indirect=True, ids=str)
def test_all_cases(test_case):
    our_value, sklearn_value = test_case

    # Only test if either array has finite values
    if np.all(np.isfinite(our_value)) or np.all(np.isfinite(sklearn_value)):
        assert np.allclose(our_value, sklearn_value), (our_value, sklearn_value)
    # If both report non-finite numbers, accept
    else:
        assert True
