import warnings

import numpy as np
import pytest

from prob_conf_mat.config import ConfigWarning
from prob_conf_mat.study import Study
from prob_conf_mat.experiment_group import ExperimentGroup
from prob_conf_mat.experiment import Experiment, ExperimentResult
from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregationResult


class TestStudy:
    def test_getitem(self):
        # First define a common setup
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )
        study.add_experiment(
            "test/foo",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Should fetch at the right level
        assert isinstance(study["test"], ExperimentGroup)
        assert isinstance(study["test/test_a"], Experiment)
        assert isinstance(study["test/foo"], Experiment)

        # Should raise an error when trying to fetch non-existent experiment group
        with pytest.raises(
            KeyError,
            match=r"No experiment group with name .* is currently present",
        ):
            study["foo"]

        with pytest.raises(
            KeyError,
            match=r"No experiment with name .* is currently present",
        ):
            study["test/bar"]

        # Should only accept strings
        with pytest.raises(
            TypeError,
        ):
            study[0]  # type: ignore

        # Should only accept strings
        with pytest.raises(
            ValueError,
        ):
            study["foo/bar/baz"]

        # Test unknown experiment group
        with pytest.raises(
            KeyError,
        ):
            study["foo/bar"]

    def test_init(self):
        # First experiment, then metric
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        # First metric, then experiment
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Experiment in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            experiments={
                "test": {
                    "test": {
                        "confusion_matrix": [[1, 0], [0, 1]],
                        "prevalence_prior": 0,
                        "confusion_prior": 0,
                    },
                },
            },
        )

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        # Metric in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            metrics={
                "acc": {"aggregation": "fe_gaussian"},
            },
        )

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        # Experiment and metric in __init__
        study = Study(
            seed=0,
            num_samples=10000,
            ci_probability=0.95,
            experiments={
                "test": {
                    "test": {
                        "confusion_matrix": [[1, 0], [0, 1]],
                        "prevalence_prior": 0,
                        "confusion_prior": 0,
                    },
                },
            },
            metrics={
                "acc": {"aggregation": "fe_gaussian"},
            },
        )

    def test_metric_label_validation(self):
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="acc", aggregation="normal")
        study.add_metric(metric="f1", aggregation="normal")

        # Test valid multiclass label
        study._validate_metric_class_label_combination(
            metric="acc",
            class_label=0,
        )

        study._validate_metric_class_label_combination(
            metric="acc",
            class_label=None,
        )

        # Test accessing invalid class
        with pytest.warns(match="Metric is multiclass, ignoring class label."):
            study._validate_metric_class_label_combination(
                metric="acc",
                class_label=1,
            )

        # Test valid binary label
        study._validate_metric_class_label_combination(
            metric="f1",
            class_label=0,
        )

        study._validate_metric_class_label_combination(
            metric="f1",
            class_label=1,
        )

        # Test accessing invalid class
        with pytest.raises(
            ValueError,
            match="is not multiclass. You must provide a class label.",
        ):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=None,
            )

        with pytest.raises(ValueError, match="Class label must be in range"):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=-1,
            )

        with pytest.raises(ValueError, match="Class label must be in range"):
            study._validate_metric_class_label_combination(
                metric="f1",
                class_label=3,
            )

        # Test unknown metric
        with pytest.raises(KeyError, match="Could not find metric"):
            study._validate_metric_class_label_combination(
                metric="foobarbaz",
                class_label=None,
            )

    @pytest.mark.parametrize(
        argnames=("pass_conf_mat", "pass_y_true", "pass_y_pred"),
        argvalues=[
            (cm, yt, yp)
            for cm in (True, False)
            for yt in (True, False)
            for yp in (True, False)
        ],
    )
    def test_add_experiment_input_combinations(
        self,
        pass_conf_mat: bool,
        pass_y_true: bool,
        pass_y_pred: bool,
    ):
        conf_mat = [[2, 0], [0, 2]]
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]
        # Rows are ground truth, columns are predictions
        conf_mat_from_y = [[1, 1], [0, 2]]

        kwargs = dict(
            confusion_matrix=conf_mat if pass_conf_mat else None,
            y_true=y_true if pass_y_true else None,
            y_pred=y_pred if pass_y_pred else None,
            prevalence_prior=0,
            confusion_prior=0,
        )

        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        if pass_conf_mat and (pass_y_true or pass_y_pred):
            # Confusion matrix takes precedence, with a warning
            with pytest.warns(
                ConfigWarning, match="Both `confusion_matrix` and `y_true`"
            ):
                study.add_experiment("test/test_a", **kwargs)
            expected = conf_mat

        elif pass_conf_mat:
            # Only the confusion matrix: no warning
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=ConfigWarning)
                study.add_experiment("test/test_a", **kwargs)
            expected = conf_mat

        elif pass_y_true and pass_y_pred:
            # Only y_true/y_pred: confusion matrix is computed from them
            study.add_experiment("test/test_a", **kwargs)
            expected = conf_mat_from_y

        else:
            # Missing the confusion matrix and at least one of y_true/y_pred
            with pytest.raises(ValueError, match=r"If `confusion_matrix=None`"):
                study.add_experiment("test/test_a", **kwargs)
            return

        assert np.array_equal(
            study["test/test_a"].confusion_matrix,  # type: ignore
            expected,
        )

    def test_pairwise_comparison(self):
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_metric(metric="acc", aggregation="normal")

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        with pytest.raises(ValueError):
            study.report_pairwise_comparison(metric="acc", experiment_a="test/test_a", experiment_b="test/test_a",)

    def test_get_samples_api(self):
        # Define some studies
        study_a = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study_a.add_experiment(
            "test/test_a",
            confusion_matrix=[[8, 2], [1, 9]],
            prevalence_prior=0,
            confusion_prior=0,
        )
        study_a.add_experiment(
            "test/test_b",
            confusion_matrix=[[7, 3], [2, 8]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study_a.add_metric(metric="acc", aggregation="fe_gaussian")

        study_b = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study_b.add_experiment(
            "test/test_a",
            confusion_matrix=[[8, 2], [1, 9]],
            prevalence_prior=0,
            confusion_prior=0,
        )
        study_b.add_experiment(
            "test/test_b",
            confusion_matrix=[[7, 3], [2, 8]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study_b.add_metric(metric="acc", aggregation="fe_gaussian")

        # Check RNG equivalence
        result_a = study_a.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        result_b = study_b.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        # Identically seeded studies produce identical samples
        assert np.array_equal(result_a.values, result_b.values)

        # Test get_metric_samples accepts both metric string and instance
        result_from_str = study_a.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        result_from_instance = study_a.get_metric_samples(
            metric=study_a._metrics_store["acc"],
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        # Both spellings hit the same cache entry
        assert result_from_instance is result_from_str

        for sampling_method in ["posterior", "prior", "random", "input"]:
            result = study_a.get_metric_samples(
                metric="acc",
                experiment_name="test/test_a",
                sampling_method=sampling_method,
            )

            assert isinstance(result, ExperimentResult)
            assert isinstance(result.values, np.ndarray
            )

        # Test return type
        result = study_a.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        assert isinstance(result, ExperimentResult)
        assert result.metric.name == "acc"
        assert result.values.shape[0] == study_a.num_samples

        result = study_a.get_metric_samples(
            metric="acc",
            experiment_name="test/aggregated",
            sampling_method="posterior",
        )

        assert isinstance(result, ExperimentAggregationResult)


# Groups are deliberately not inserted in sorted order, and include names with
# regex metacharacters, mixed case and a bare (group-less) registration
FILTER_EXPERIMENT_NAMES = [
    "transformer/bert",
    "transformer/roberta",
    "transformer/model.v2",
    "transformer/modelXv2",
    "transformer/run(1)",
    "cnn/resnet",
    "cnn/bert",
    "bert",  # Registered without a group, becomes 'bert/bert'
    "bert/large",
    "Baseline/majority",
    "seeds/run_1",
    "seeds/run_2",
    "seeds/run_a",
]

FILTER_ALL_EXPERIMENTS = sorted(
    name if "/" in name else f"{name}/{name}" for name in FILTER_EXPERIMENT_NAMES
)

FILTER_GROUPS = ["transformer", "cnn", "bert", "Baseline", "seeds"]


@pytest.fixture
def filter_study() -> Study:
    study = Study(seed=0, num_samples=10000, ci_probability=0.95)

    for name in FILTER_EXPERIMENT_NAMES:
        study.add_experiment(
            name,
            confusion_matrix=[[1, 0], [0, 1]],
            prevalence_prior=0,
            confusion_prior=0,
        )

    return study


class TestFilterExperiments:
    def test_fixture_sanity(self, filter_study: Study):
        assert filter_study.all_experiments() == FILTER_ALL_EXPERIMENTS

    @pytest.mark.parametrize(argnames="pattern", argvalues=["*", "*/*"])
    def test_match_all(self, filter_study: Study, pattern: str):
        assert filter_study.filter_experiments(pattern) == FILTER_ALL_EXPERIMENTS

    def test_default_pattern_matches_all(self, filter_study: Study):
        assert filter_study.filter_experiments() == FILTER_ALL_EXPERIMENTS

    @pytest.mark.parametrize(
        argnames=("pattern", "expected"),
        argvalues=[
            # Whole group
            ("cnn/*", ["cnn/bert", "cnn/resnet"]),
            ("bert/*", ["bert/bert", "bert/large"]),
            # Same experiment name across groups
            ("*/bert", ["bert/bert", "cnn/bert", "transformer/bert"]),
            # Exact name
            ("transformer/roberta", ["transformer/roberta"]),
            # Wildcards in both parts
            (
                "*/run*",
                [
                    "seeds/run_1",
                    "seeds/run_2",
                    "seeds/run_a",
                    "transformer/run(1)",
                ],
            ),
            ("c*/*e*", ["cnn/bert", "cnn/resnet"]),
        ],
    )
    def test_glob_patterns(
        self, filter_study: Study, pattern: str, expected: list[str]
    ):
        assert filter_study.filter_experiments(pattern) == expected

    @pytest.mark.parametrize(
        argnames=("pattern", "expected"),
        argvalues=[
            ("seeds/run_?", ["seeds/run_1", "seeds/run_2", "seeds/run_a"]),
            ("seeds/run_[0-9]", ["seeds/run_1", "seeds/run_2"]),
            ("seeds/run_[12]", ["seeds/run_1", "seeds/run_2"]),
            ("seeds/run_[!0-9]", ["seeds/run_a"]),
            ("*/model?v2", ["transformer/model.v2", "transformer/modelXv2"]),
        ],
    )
    def test_single_character_wildcards(
        self, filter_study: Study, pattern: str, expected: list[str]
    ):
        assert filter_study.filter_experiments(pattern) == expected

    def test_bare_name_matches_name_slash_name(self, filter_study: Study):
        # A bare name follows the registration convention: 'bert' -> 'bert/bert'.
        # It must *not* select the whole 'bert' group
        assert filter_study.filter_experiments("bert") == ["bert/bert"]

        # 'cnn' has no 'cnn/cnn' experiment
        assert filter_study.filter_experiments("cnn") == []

    @pytest.mark.parametrize(
        argnames=("pattern", "expected"),
        argvalues=[
            # '.' is literal, so 'modelXv2' must not match
            ("transformer/model.v2", ["transformer/model.v2"]),
            ("*/model.v2", ["transformer/model.v2"]),
            # Parentheses are literal, and must not raise a regex error
            ("*/run(1)", ["transformer/run(1)"]),
            # Regex syntax has no special meaning
            ("*/bert|resnet", []),
            ("*/ber.+", []),
            ("^cnn/*", []),
        ],
    )
    def test_regex_metacharacters_are_literal(
        self, filter_study: Study, pattern: str, expected: list[str]
    ):
        assert filter_study.filter_experiments(pattern) == expected

    @pytest.mark.parametrize(
        argnames="pattern",
        argvalues=[
            # Partial matches must not return fragments or substrings
            "*/res",
            "*/ber",
            "ert",
            "nsformer/*",
            "grp/x",
        ],
    )
    def test_only_whole_names_match(self, filter_study: Study, pattern: str):
        assert filter_study.filter_experiments(pattern) == []

    def test_wildcards_do_not_cross_divider(self, filter_study: Study):
        # As a single regex, 'trans*bert' would match 'transformer/bert'
        assert filter_study.filter_experiments("trans*bert") == []
        assert filter_study.filter_experiments("transformer*/bert") == [
            "transformer/bert"
        ]

    def test_case_sensitive(self, filter_study: Study):
        assert filter_study.filter_experiments("Baseline/*") == ["Baseline/majority"]
        assert filter_study.filter_experiments("baseline/*") == []
        assert filter_study.filter_experiments("*/BERT") == []

    def test_no_matches_returns_empty_list(self, filter_study: Study):
        assert filter_study.filter_experiments("nonexistent/*") == []

    def test_results_are_sorted(self, filter_study: Study):
        result = filter_study.filter_experiments("*")

        assert result == sorted(result)
        # 'Baseline' sorts before lowercase groups, despite being inserted later
        assert result[0] == "Baseline/majority"

    @pytest.mark.parametrize(
        argnames="pattern",
        argvalues=["a/b/c", "*/*/*", "transformer/bert/"],
    )
    def test_too_many_dividers_raises(self, filter_study: Study, pattern: str):
        with pytest.raises(ValueError, match="Must have at most 1 '/' character"):
            filter_study.filter_experiments(pattern)

    @pytest.mark.parametrize(argnames="pattern", argvalues=["agg", "aggregated"])
    def test_aggregated_keyword(self, filter_study: Study, pattern: str):
        result = filter_study.filter_experiments(pattern)

        assert sorted(result) == sorted(
            f"{group}/aggregated" for group in FILTER_GROUPS
        )

    @pytest.mark.parametrize(argnames="pattern", argvalues=["*", "*/*", "agg"])
    def test_empty_study(self, pattern: str):
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        assert study.filter_experiments(pattern) == []

    def test_does_not_mutate_study(self, filter_study: Study):
        before = filter_study.all_experiments()

        filter_study.filter_experiments("*/bert")
        filter_study.filter_experiments("agg")

        assert filter_study.all_experiments() == before
