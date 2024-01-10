import unittest
import os
import warnings
from functools import partial

import numpy as np
import sklearn
import sklearn.metrics

from bayes_conf_mat.experiment import Experiment
from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.utils.io import (
    load_integer_csv_into_numpy,
    confusion_matrix_to_pred_target,
)


class TestMetricEquivalence(unittest.TestCase):
    def setUp(self):
        # Load in all the test cases
        data_dir = "./data/test_cases"
        _, _, files = next(os.walk(data_dir))

        self.test_cases = []
        for file_name in files:
            fp = f"{data_dir}/{file_name}"
            confusion_matrix = load_integer_csv_into_numpy(fp=fp)

            self.test_cases.append((file_name, confusion_matrix))

    def _throw_a_case_against_metric(
        self,
        test_case,
        metric,
        sklearn_func,
    ):
        file_name, confusion_matrix = test_case

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            experiment = Experiment(confusion_matrix=confusion_matrix)
            experiment.add_metric(metric)

            our_metric_val = experiment._compute_metrics(sample_method="input")[
                metric.name
            ]

            pred_target = confusion_matrix_to_pred_target(
                confusion_matrix,
                pred_first=True,
            )

            sklearn_metric_val = sklearn_func(
                pred_target[:, 1],
                pred_target[:, 0],
            )

        is_close = np.allclose(our_metric_val, sklearn_metric_val)

        self.assertTrue(
            is_close,
            msg=f"{metric.name} - {file_name}: Fail `{our_metric_val[0]}` != `{sklearn_metric_val}`",  # noqa: E501
        )

    def _throw_all_cases_against_metric(
        self,
        metric,
        sklearn_func,
    ):
        for test_case in self.test_cases:
            self._throw_a_case_against_metric(
                test_case=test_case,
                metric=metric,
                sklearn_func=sklearn_func,
            )

    def _throw_univariate_cases_against_metric(
        self,
        metric,
        sklearn_func,
    ):
        univariate_test_cases = [
            test_case for test_case in self.test_cases if test_case[1].shape[0] == 2
        ]

        for test_case in univariate_test_cases:
            self._throw_a_case_against_metric(
                test_case=test_case,
                metric=metric,
                sklearn_func=sklearn_func,
            )

    def test_acc(self):
        sklearn_func = sklearn.metrics.accuracy_score

        self._throw_all_cases_against_metric(
            metric=get_metric("acc"),
            sklearn_func=sklearn_func,
        )

    def test_ba(self):
        sklearn_func = sklearn.metrics.balanced_accuracy_score

        self._throw_all_cases_against_metric(
            metric=get_metric("ba"),
            sklearn_func=sklearn_func,
        )

    def test_adjba(self):
        sklearn_func = sklearn.metrics.balanced_accuracy_score
        sklearn_func = partial(sklearn_func, adjusted=True)

        self._throw_all_cases_against_metric(
            metric=get_metric("ba+adjusted=True"),
            sklearn_func=sklearn_func,
        )

    def test_kappa(self):
        sklearn_func = sklearn.metrics.cohen_kappa_score

        self._throw_all_cases_against_metric(
            metric=get_metric("cohen_kappa"),
            sklearn_func=sklearn_func,
        )

    def test_mcc(self):
        sklearn_func = sklearn.metrics.matthews_corrcoef

        self._throw_all_cases_against_metric(
            metric=get_metric("mcc"),
            sklearn_func=sklearn_func,
        )

    def test_fbeta(self):
        sklearn_func = sklearn.metrics.fbeta_score

        for beta in [0.0, 0.5, 1.0, 2.0]:
            sklearn_func = partial(sklearn_func, average=None, beta=beta)

            self._throw_all_cases_against_metric(
                metric=get_metric(f"fbeta+beta={beta}"),
                sklearn_func=sklearn_func,
            )

    def test_f1(self):
        sklearn_func = sklearn.metrics.f1_score

        sklearn_func = partial(sklearn_func, average=None)

        self._throw_all_cases_against_metric(
            metric=get_metric("f1"),
            sklearn_func=sklearn_func,
        )

    def test_jaccard(self):
        sklearn_func = sklearn.metrics.jaccard_score

        sklearn_func = partial(sklearn_func, average=None)

        self._throw_all_cases_against_metric(
            metric=get_metric("jaccard"),
            sklearn_func=sklearn_func,
        )

    def test_plr(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[0]

        self._throw_univariate_cases_against_metric(
            metric=get_metric("plr@binary+positive_class=1"),
            sklearn_func=sklearn_func,
        )

    def test_nlr(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[1]

        self._throw_univariate_cases_against_metric(
            metric=get_metric("nlr@binary+positive_class=1"),
            sklearn_func=sklearn_func,
        )

    def test_dor(self):
        sklearn_func_ = sklearn.metrics.class_likelihood_ratios

        def sklearn_func(x, y):
            return sklearn_func_(x, y)[0] / sklearn_func_(x, y)[1]

        self._throw_univariate_cases_against_metric(
            metric=get_metric("dor@binary+positive_class=1"),
            sklearn_func=sklearn_func,
        )


if __name__ == "__main__":
    unittest.main()
