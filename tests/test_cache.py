import numpy as np
import pytest

from prob_conf_mat.experiment import ExperimentResult
from prob_conf_mat.study import Study
from prob_conf_mat.utils.cache import InMemoryCache, NotInCache

FINGERPRINT_A = "fingerprint_a"
FINGERPRINT_B = "fingerprint_b"


class TestInMemoryCache:
    def test_roundtrip(self):
        cache = InMemoryCache()

        # Single key
        cache.cache(fingerprint=FINGERPRINT_A, keys=["foo"], value="bar")
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["foo"]) == "bar"

        # Nested keys
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b", "c"], value=42)
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["a", "b", "c"]) == 42

        # Values need not be hashable
        value = np.arange(10)
        cache.cache(fingerprint=FINGERPRINT_A, keys=["arr"], value=value)
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["arr"]) is value

    def test_overwrite(self):
        cache = InMemoryCache()

        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="old")
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="new")

        assert cache.load(fingerprint=FINGERPRINT_A, keys=["a", "b"]) == "new"

    def test_load_missing_returns_default(self):
        cache = InMemoryCache()
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")

        # Missing top-level key
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["foo"]) is NotInCache

        # Missing nested key
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["a", "c"]) is NotInCache

        # A custom default is returned instead of the sentinel
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["foo"], default=0) == 0

    def test_isin(self):
        cache = InMemoryCache()

        assert not cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "b"])

        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")

        assert cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "b"])
        assert not cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "c"])

    def test_fingerprint_mismatch(self):
        cache = InMemoryCache()
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")

        # Loading with a different fingerprint misses, even for known keys
        assert cache.load(fingerprint=FINGERPRINT_B, keys=["a", "b"]) is NotInCache
        assert not cache.isin(fingerprint=FINGERPRINT_B, keys=["a", "b"])

    def test_new_fingerprint_invalidates_cache(self):
        cache = InMemoryCache()
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")

        # Caching under a new fingerprint destroys the old state
        cache.cache(fingerprint=FINGERPRINT_B, keys=["c"], value="qux")

        assert cache.fingerprint == FINGERPRINT_B
        assert cache.load(fingerprint=FINGERPRINT_B, keys=["c"]) == "qux"
        assert not cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "b"])
        assert not cache.isin(fingerprint=FINGERPRINT_B, keys=["a", "b"])

    def test_clean(self):
        cache = InMemoryCache()
        cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")

        cache.clean()

        # The contents are gone, even under the current fingerprint
        assert not cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "b"])
        assert cache.load(fingerprint=FINGERPRINT_A, keys=["a", "b"]) is NotInCache

        # The aliases behave identically
        for method in (cache.clear, cache.empty):
            cache.cache(fingerprint=FINGERPRINT_A, keys=["a", "b"], value="baz")
            method()
            assert not cache.isin(fingerprint=FINGERPRINT_A, keys=["a", "b"])


class TestGetMetricSamples:
    @staticmethod
    def make_study() -> Study:
        study = Study(seed=0, num_samples=10000, ci_probability=0.95)

        study.add_experiment(
            "test/test_a",
            confusion_matrix=[[8, 2], [1, 9]],
            prevalence_prior=0,
            confusion_prior=0,
        )
        study.add_experiment(
            "test/test_b",
            confusion_matrix=[[7, 3], [2, 8]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        study.add_metric(metric="acc", aggregation="fe_gaussian")

        return study


    def test_result_is_cached(self):
        study = self.make_study()

        keys = ["acc", "test", "test_a", "posterior"]
        assert not study.cache.isin(fingerprint=study.fingerprint, keys=keys)

        first = study.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        assert study.cache.isin(fingerprint=study.fingerprint, keys=keys)

        second = study.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        # The second request is served from the cache
        assert second is first

    def test_config_change_invalidates_cache(self):
        study = self.make_study()

        first = study.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        old_fingerprint = study.fingerprint

        # Adding an experiment changes the config, and thus the fingerprint
        study.add_experiment(
            "test/test_c",
            confusion_matrix=[[6, 4], [3, 7]],
            prevalence_prior=0,
            confusion_prior=0,
        )

        assert study.fingerprint != old_fingerprint

        keys = ["acc", "test", "test_a", "posterior"]
        assert not study.cache.isin(fingerprint=study.fingerprint, keys=keys)

        # The stale result is recomputed, not served from the cache
        second = study.get_metric_samples(
            metric="acc",
            experiment_name="test/test_a",
            sampling_method="posterior",
        )

        assert second is not first
        assert isinstance(second, ExperimentResult)

    def test_unregistered_metric_raises(self):
        study = self.make_study()

        with pytest.raises(ValueError, match="has not been registered"):
            study.get_metric_samples(
                metric="f1",
                experiment_name="test/test_a",
                sampling_method="posterior",
            )

    def test_unknown_experiment_raises(self):
        study = self.make_study()

        with pytest.raises(ValueError, match=r"does not \(yet\) exist"):
            study.get_metric_samples(
                metric="acc",
                experiment_name="test/foobar",
                sampling_method="posterior",
            )

        # Unknown experiment group, requesting the aggregated result
        with pytest.raises(ValueError, match=r"does not \(yet\) exist"):
            study.get_metric_samples(
                metric="acc",
                experiment_name="foobar/aggregated",
                sampling_method="posterior",
            )

    def test_invalid_sampling_method_raises(self):
        study = self.make_study()

        with pytest.raises(ValueError, match="Must be one of"):
            study.get_metric_samples(
                metric="acc",
                experiment_name="test/test_a",
                sampling_method="foobar",
            )
