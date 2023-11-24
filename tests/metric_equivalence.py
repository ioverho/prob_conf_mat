import unittest
import os
import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import sklearn
import sklearn.metrics

from bayes_conf_mat.confusion_matrix import BayesianConfusionMatrix
from bayes_conf_mat.metrics import (
    compute_all_simple_metrics,
    IMPLEMENTED_COMPLEX_METRICS,
)
from bayes_conf_mat.utils.io import (
    load_integer_csv_into_numpy,
    confusion_matrix_to_pred_target,
)

test_case_tuple = namedtuple(
    "TestCase", ["file_name", "confusion_matrix", "pseudo_samples", "simple_metrics"]
)


class TestMetricEquivalence(unittest.TestCase):
    def setUp(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # Load in all the test cases
            _, _, files = next(os.walk("./tests/confusion_matrices/"))

            self.test_cases = []
            for file_name in files:
                fp = f"./tests/confusion_matrices/{file_name}"
                confusion_matrix = load_integer_csv_into_numpy(fp=fp)

                pseudo_samples = BayesianConfusionMatrix(
                    confusion_matrix=confusion_matrix,
                )._use_input_as_sample()

                self.test_cases.append(
                    test_case_tuple(
                        file_name,
                        confusion_matrix,
                        pseudo_samples,
                        compute_all_simple_metrics(pseudo_samples),
                    ),
                )

    def _throw_a_case_against_metric(
        self,
        test_case,
        our_func,
        sklearn_func,
        **additional_metric_kwargs,
    ):
        our_metric_val = our_func(
            **{
                requirement: test_case.simple_metrics[requirement]
                for requirement in our_func.required_simple_metrics
            },
            **additional_metric_kwargs,
        )

        pred_target = confusion_matrix_to_pred_target(
            test_case.confusion_matrix,
            pred_first=True,
        )

        sklearn_metric_val = sklearn_func(
            pred_target[:, 1],
            pred_target[:, 0],
        )

        is_close = np.allclose(our_metric_val, sklearn_metric_val)

        self.assertTrue(
            is_close,
            msg=f"{our_func.identifier} - {test_case.file_name}: Fail `{our_metric_val[0]}` != `{sklearn_metric_val}`",  # noqa: E501
        )

    def _throw_all_cases_against_metric(
        self,
        our_func,
        sklearn_func,
        **additional_metric_kwargs,
    ):
        for test_case in self.test_cases:
            self._throw_a_case_against_metric(
                test_case=test_case,
                our_func=our_func,
                sklearn_func=sklearn_func,
                **additional_metric_kwargs,
            )

    def _throw_univariate_cases_against_metric(
        self,
        our_func,
        sklearn_func,
        **additional_metric_kwargs,
    ):
        univariate_test_cases = [
            test_case
            for test_case in self.test_cases
            if test_case.confusion_matrix.shape[0] == 2
        ]

        for test_case in univariate_test_cases:
            self._throw_a_case_against_metric(
                test_case=test_case,
                our_func=our_func,
                sklearn_func=sklearn_func,
                **additional_metric_kwargs,
            )

    def test_acc(self):
        sklearn_func = sklearn.metrics.accuracy_score

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["acc"],
            sklearn_func=sklearn_func,
        )

    def test_ba(self):
        sklearn_func = sklearn.metrics.balanced_accuracy_score

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["ba"],
            sklearn_func=sklearn_func,
        )

    def test_adjba(self):
        sklearn_func = sklearn.metrics.balanced_accuracy_score
        sklearn_func = partial(sklearn_func, adjusted=True)

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["adjba"],
            sklearn_func=sklearn_func,
        )

    def test_kappa(self):
        sklearn_func = sklearn.metrics.cohen_kappa_score

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["kappa"],
            sklearn_func=sklearn_func,
        )

    def test_mcc(self):
        sklearn_func = sklearn.metrics.matthews_corrcoef

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["mcc"],
            sklearn_func=sklearn_func,
        )

    def test_fbeta(self):
        sklearn_func = sklearn.metrics.fbeta_score

        for beta in [0.0, 0.5, 1.0, 2.0]:
            sklearn_func = partial(sklearn_func, average=None, beta=beta)

            self._throw_all_cases_against_metric(
                our_func=IMPLEMENTED_COMPLEX_METRICS["fbeta"],
                sklearn_func=sklearn_func,
                beta=beta,
            )

    def test_f1(self):
        sklearn_func = sklearn.metrics.f1_score

        sklearn_func = partial(sklearn_func, average=None)

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["fbeta"],
            sklearn_func=sklearn_func,
        )

    def test_jaccard(self):
        sklearn_func = sklearn.metrics.jaccard_score

        sklearn_func = partial(sklearn_func, average=None)

        self._throw_all_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["jaccard"],
            sklearn_func=sklearn_func,
        )

    def test_plr(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[0]

        self._throw_univariate_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["plr"],
            sklearn_func=sklearn_func,
        )

    def test_nlr(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[1]

        self._throw_univariate_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["nlr"],
            sklearn_func=sklearn_func,
        )

    def test_dor(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[0] / sklearn_func_(x, y)[1]

        self._throw_univariate_cases_against_metric(
            our_func=IMPLEMENTED_COMPLEX_METRICS["dor"],
            sklearn_func=sklearn_func,
        )


if __name__ == "__main__":
    unittest.main()
