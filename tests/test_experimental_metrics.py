import warnings

import pytest

from prob_conf_mat.config import ConfigError
from prob_conf_mat.metrics import (
    _ROOT_METRICS,  # pyright: ignore[reportPrivateUsage]
    AVERAGING_REGISTRY,
    METRIC_REGISTRY,
    get_metric,
)
from prob_conf_mat.metrics.abc import Averaging
from prob_conf_mat.metrics.experimental import (
    ExperimentalMetricsError,
    enable_experimental_metrics,
    is_experimental,
    is_experimental_metrics_enabled,
    mark_experimental,
)
from prob_conf_mat.metrics.experimental.interface import ExperimentalWarning
from prob_conf_mat.metrics.experimental.ordinal_regression import MeanAbsoluteError
from prob_conf_mat.study import Study

EXPERIMENTAL_METRIC_ALIASES = sorted(
    alias for alias, cls in METRIC_REGISTRY.items() if is_experimental(cls)
)
STABLE_METRIC_ALIASES = sorted(
    alias for alias, cls in METRIC_REGISTRY.items() if not is_experimental(cls)
)

_TEST_AVERAGING_ALIAS = "_test_experimental_averaging"


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture(autouse=True)
def _reset_experimental_state():
    """Guarantees every test starts and ends with experimental metrics disabled."""
    enable_experimental_metrics(False)
    yield
    enable_experimental_metrics(False)


@pytest.fixture
def experimental_enabled():
    """Enables experimental metrics, swallowing the expected opt-in warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ExperimentalWarning)
        enable_experimental_metrics()
    yield


@pytest.fixture
def experimental_averaging():
    """Registers a throwaway experimental averaging method, and removes it afterwards.

    No experimental averaging methods currently ship with the library, so this is the only
    way to exercise the averaging branch of the experimental gate.
    """

    @mark_experimental(since="test")
    class _ExperimentalAveraging(Averaging):
        full_name = "Experimental Test Averaging"
        dependencies = ()
        sklearn_equivalent = None
        aliases = (_TEST_AVERAGING_ALIAS,)

        def compute_average(self, metric_values):
            return metric_values.mean(axis=1)

    yield _ExperimentalAveraging

    AVERAGING_REGISTRY.pop(_TEST_AVERAGING_ALIAS, None)


def _make_study(**kwargs) -> Study:
    return Study(seed=0, num_samples=10000, ci_probability=0.95, **kwargs)


# ==============================================================================
# The global switch
# ==============================================================================
class TestEnableExperimentalMetrics:
    def test_disabled_by_default(self):
        assert is_experimental_metrics_enabled() is False

    def test_enable(self):
        with pytest.warns(ExperimentalWarning, match="Experimental metrics enabled"):
            enable_experimental_metrics()

        assert is_experimental_metrics_enabled() is True

    def test_enable_default_argument_is_true(self):
        with pytest.warns(ExperimentalWarning):
            enable_experimental_metrics()

        assert is_experimental_metrics_enabled()

    def test_disable(self, experimental_enabled):
        assert is_experimental_metrics_enabled()

        enable_experimental_metrics(False)

        assert is_experimental_metrics_enabled() is False

    def test_disable_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ExperimentalWarning)
            enable_experimental_metrics(False)

    def test_warns_every_time_it_is_enabled(self):
        with pytest.warns(ExperimentalWarning):
            enable_experimental_metrics()

        with pytest.warns(ExperimentalWarning):
            enable_experimental_metrics()

    def test_warning_points_at_caller(self):
        with pytest.warns(ExperimentalWarning) as record:
            enable_experimental_metrics()

        # stacklevel=2 should attribute the warning to this file, not interface.py
        assert record[0].filename == __file__

    def test_error_is_value_error(self):
        # Existing code catching ValueError from get_metric should keep working
        assert issubclass(ExperimentalMetricsError, ValueError)


# ==============================================================================
# Marking metrics as experimental
# ==============================================================================
class TestMarkExperimental:
    def test_mark_experimental_sets_attributes(self):
        @mark_experimental(since="v9.9.9", reason="testing")
        class Foo:
            pass

        assert is_experimental(Foo)
        assert Foo._experimental is True
        assert Foo._experimental_since == "v9.9.9"
        assert Foo._experimental_reason == "testing"

    def test_mark_experimental_default_reason(self):
        @mark_experimental(since="v9.9.9")
        class Foo:
            pass

        assert Foo._experimental_reason == ""

    def test_mark_experimental_requires_keyword_since(self):
        with pytest.raises(TypeError):
            mark_experimental("v9.9.9")  # type: ignore

    def test_there_are_experimental_metrics(self):
        # Guards the parametrized tests below against silently testing nothing
        assert len(EXPERIMENTAL_METRIC_ALIASES) > 0

    @pytest.mark.parametrize(
        "alias",
        [
            "mae",
            "mse",
            "rmse",
            "offby1",
            "pwk",
            "lwk",
            "qwk",
            "precision_gain",
            "recall_gain",
            "f1_gain",
            "fbeta_gain",
        ],
    )
    def test_known_experimental_metrics_are_marked(self, alias):
        assert is_experimental(METRIC_REGISTRY[alias])

    @pytest.mark.parametrize("alias", ["acc", "f1", "precision", "recall"])
    def test_stable_metrics_are_not_marked(self, alias):
        assert not is_experimental(METRIC_REGISTRY[alias])

    @pytest.mark.parametrize("alias", sorted(AVERAGING_REGISTRY.keys()))
    def test_shipped_averaging_methods_are_not_marked(self, alias):
        assert not is_experimental(AVERAGING_REGISTRY[alias])

    def test_subclass_of_experimental_metric_is_experimental(self):
        # e.g. LinearWeightedKappa subclasses PolynomialWeightedKappa
        assert is_experimental(METRIC_REGISTRY["lwk"])
        assert is_experimental(METRIC_REGISTRY["qwk"])

    @pytest.mark.parametrize("alias", STABLE_METRIC_ALIASES)
    def test_stable_metrics_do_not_depend_on_experimental_metrics(
        self, alias, experimental_enabled
    ):
        for dependency in METRIC_REGISTRY[alias].dependencies:
            if dependency in _ROOT_METRICS:
                continue

            assert not is_experimental(get_metric(dependency)), (
                f"Stable metric `{alias}` depends on experimental metric `{dependency}`"
            )


# ==============================================================================
# The gate in get_metric
# ==============================================================================
class TestGetMetricGate:
    @pytest.mark.parametrize("alias", EXPERIMENTAL_METRIC_ALIASES)
    def test_experimental_metric_raises_when_disabled(self, alias):
        with pytest.raises(ExperimentalMetricsError):
            get_metric(alias)

    @pytest.mark.parametrize("alias", EXPERIMENTAL_METRIC_ALIASES)
    def test_experimental_metric_allowed_when_enabled(
        self, alias, experimental_enabled
    ):
        metric = get_metric(alias)

        assert isinstance(metric, METRIC_REGISTRY[alias])
        assert is_experimental(metric)

    def test_experimental_metric_with_kwargs(self, experimental_enabled):
        metric = get_metric("fbeta_gain+beta=2.0")

        assert metric.name == "fbeta_gain+beta=2.0"
        assert is_experimental(metric)

    def test_experimental_metric_with_kwargs_raises_when_disabled(self):
        with pytest.raises(ExperimentalMetricsError):
            get_metric("pwk+power=2.0")

    @pytest.mark.parametrize("alias", ["acc", "f1", "f1@macro", "f1@weighted"])
    def test_stable_metrics_unaffected_when_disabled(self, alias):
        metric = get_metric(alias)

        assert not is_experimental(metric)

    def test_root_metrics_unaffected_when_disabled(self):
        for root_metric in _ROOT_METRICS:
            get_metric(root_metric)

    def test_unknown_metric_still_raises_registry_error(self):
        # An unknown name should not be reported as experimental
        with pytest.raises(ValueError, match="Metric alias must be registered") as e:
            get_metric("not_a_real_metric")

        assert not isinstance(e.value, ExperimentalMetricsError)

    def test_gate_is_reevaluated_on_every_call(self):
        # Regression test: if get_metric were cached, toggling would have no effect
        with pytest.raises(ExperimentalMetricsError):
            get_metric("mae")

        with pytest.warns(ExperimentalWarning):
            enable_experimental_metrics()
        get_metric("mae")

        enable_experimental_metrics(False)
        with pytest.raises(ExperimentalMetricsError):
            get_metric("mae")


class TestGetMetricGateAveraged:
    def test_experimental_metric_stable_averaging_raises_when_disabled(self):
        with pytest.raises(ExperimentalMetricsError):
            get_metric("precision_gain@macro")

    def test_experimental_metric_stable_averaging_allowed_when_enabled(
        self, experimental_enabled
    ):
        metric = get_metric("precision_gain@macro")

        assert is_experimental(metric)
        assert is_experimental(metric.base_metric)
        assert not is_experimental(metric.averaging)

    def test_stable_metric_experimental_averaging_raises_when_disabled(
        self, experimental_averaging
    ):
        with pytest.raises(ExperimentalMetricsError):
            get_metric(f"f1@{_TEST_AVERAGING_ALIAS}")

    def test_stable_metric_experimental_averaging_allowed_when_enabled(
        self, experimental_averaging, experimental_enabled
    ):
        metric = get_metric(f"f1@{_TEST_AVERAGING_ALIAS}")

        assert is_experimental(metric)
        assert not is_experimental(metric.base_metric)
        assert is_experimental(metric.averaging)

    def test_experimental_metric_experimental_averaging(
        self, experimental_averaging, experimental_enabled
    ):
        metric = get_metric(f"precision_gain@{_TEST_AVERAGING_ALIAS}")

        assert is_experimental(metric)

    def test_stable_composition_is_not_experimental(self):
        metric = get_metric("f1@macro")

        assert not is_experimental(metric)


# ==============================================================================
# The gate through the Study API
# ==============================================================================
class TestStudyGate:
    def test_add_metric_raises_when_disabled(self):
        study = _make_study()

        with pytest.raises(ConfigError) as e:
            study.add_metric(metric="mae")

        assert isinstance(e.value.__cause__, ExperimentalMetricsError)

    def test_add_metric_failure_does_not_add_to_metric_store(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_metric(metric="mae")

        assert len(study._metrics_store) == 0

    def test_add_metric_failure_does_not_add_to_config(self):
        study = _make_study()

        with pytest.raises(ConfigError):
            study.add_metric(metric="mae")

        assert "mae" not in study.metrics

    def test_add_averaged_metric_raises_when_disabled(self):
        study = _make_study()

        with pytest.raises(ConfigError) as e:
            study.add_metric(metric="precision_gain@macro")

        assert isinstance(e.value.__cause__, ExperimentalMetricsError)

    def test_add_metric_instance_raises_when_disabled(self):
        # Instances bypass get_metric on construction, but the config still
        # validates the metric's name string
        study = _make_study()

        with pytest.raises(ConfigError) as e:
            study.add_metric(metric=MeanAbsoluteError())

        assert isinstance(e.value.__cause__, ExperimentalMetricsError)

    def test_metrics_in_init_raise_when_disabled(self):
        with pytest.raises(ConfigError) as e:
            _make_study(metrics={"mae": {}})

        assert isinstance(e.value.__cause__, ExperimentalMetricsError)

    def test_from_dict_raises_when_disabled(self):
        config = {
            "seed": 0,
            "num_samples": 10000,
            "ci_probability": 0.95,
            "metrics": {"mae": {}},
        }

        with pytest.raises(ConfigError) as e:
            Study.from_dict(config)

        assert isinstance(e.value.__cause__, ExperimentalMetricsError)

    def test_add_metric_allowed_when_enabled(self, experimental_enabled):
        study = _make_study()

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[2, 1, 0], [0, 3, 1], [1, 0, 2]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="mae")
        study.add_metric(metric="precision_gain@macro")
        study.add_metric(metric=MeanAbsoluteError())

        assert "mae" in study.metrics
        assert "precision_gain@macro" in study.metrics

    def test_metrics_in_init_allowed_when_enabled(self, experimental_enabled):
        study = _make_study(metrics={"mae": {}, "acc": {}})

        assert "mae" in study.metrics
        assert "acc" in study.metrics

    def test_round_trip_when_enabled(self, experimental_enabled):
        study = _make_study(metrics={"mae": {}})

        restored = Study.from_dict(study.to_dict())

        assert "mae" in restored.metrics

    def test_stable_metrics_unaffected_when_disabled(self):
        study = _make_study()

        study.add_metric(metric="acc")
        study.add_metric(metric="f1@macro")

        assert "acc" in study.metrics
        assert "f1@macro" in study.metrics
