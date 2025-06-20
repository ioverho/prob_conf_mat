import jaxtyping as jtyping
import numpy as np
import pytest
import scipy.stats

from prob_conf_mat.utils.rng import RNG
from prob_conf_mat.experiment_aggregation import get_experiment_aggregator
from prob_conf_mat.stats.truncated_sampling import truncated_sample

SEED = 0
DIMS = 8
NUM_EXPERIMENTS = 1
NUM_SAMPLES = 10000

NP_RNG = np.random.default_rng(seed=SEED)

NON_SINGLETON_AGG_METHODS = [
    "beta",
    "fe_gaussian",
    "gamma",
    "histogram",
    "re_gaussian",
]


def unbounded_samples(
    rng,
    num_experiments: int,
    num_samples: int,
) -> jtyping.Float[
    np.ndarray,
    "num_samples num_experiments dims",
]:
    samples = rng.normal(
        loc=0.5,
        scale=0.1,
        size=(num_samples, num_experiments),
    )

    return samples


def bounded_samples(
    rng,
    num_experiments: int,
    num_samples: int,
    bounds: tuple[float, float],
) -> jtyping.Float[
    np.ndarray,
    "num_samples num_experiments dims",
]:
    samples = np.stack(
        arrays=[
            truncated_sample(
                sampling_distribution=scipy.stats.norm(
                    loc=0.5,
                    scale=0.1,
                ),
                bounds=bounds,
                rng=rng,
                num_samples=num_samples,
            )
            for _ in range(num_experiments)
        ],
        axis=1,
    )

    return samples


class TestSingletonAggregator:
    aggregator = get_experiment_aggregator(
        aggregation="singleton",
        rng=RNG(seed=SEED),
    )

    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            unbounded_samples(rng=NP_RNG, num_experiments=1, num_samples=10000),
        ],
    )
    def test_valid(
        self,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        self.aggregator.aggregate(
            experiment_samples=samples,
            bounds=(-float("inf"), float("inf")),
        )

    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            unbounded_samples(rng=NP_RNG, num_experiments=2, num_samples=10000),
        ],
    )
    def test_too_many_experiments(
        self,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError,
            match="Parameter `num_experiments` > 1.",
        ):
            self.aggregator.aggregate(
                experiment_samples=samples,
                bounds=(-float("inf"), float("inf")),
            )

    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            unbounded_samples(rng=NP_RNG, num_experiments=1, num_samples=10000),
        ],
    )
    def test_values(
        self,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        output_samples = self.aggregator.aggregate(
            experiment_samples=samples,
            bounds=(-float("inf"), float("inf")),
        )

        np.allclose(samples, output_samples)


class TestNonSingletonAggregator:
    @pytest.mark.parametrize("aggregation", NON_SINGLETON_AGG_METHODS)
    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            bounded_samples(
                rng=NP_RNG,
                num_experiments=2,
                num_samples=10000,
                bounds=(0, 1),
            ),
        ],
    )
    def test_valid(
        self,
        aggregation: str,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        aggregator = get_experiment_aggregator(
            aggregation=aggregation,
            rng=RNG(seed=SEED),
        )

        aggregator.aggregate(
            experiment_samples=samples,
            bounds=(0, 1),
        )

    @pytest.mark.parametrize("aggregation", NON_SINGLETON_AGG_METHODS)
    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            unbounded_samples(
                rng=NP_RNG,
                num_experiments=2,
                num_samples=10000,
            ),
        ],
    )
    def test_infinite_bounds(
        self,
        aggregation: str,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        aggregator = get_experiment_aggregator(
            aggregation=aggregation,
            rng=RNG(seed=SEED),
        )

        match aggregation:
            case "beta":
                with pytest.raises(NotImplementedError):
                    aggregator.aggregate(
                        experiment_samples=samples,
                        bounds=(-float("inf"), float("inf")),
                    )

            case "gamma":
                with pytest.raises(ValueError):
                    aggregator.aggregate(
                        experiment_samples=samples,
                        bounds=(-float("inf"), float("inf")),
                    )

            case _:
                aggregator.aggregate(
                    experiment_samples=samples,
                    bounds=(-float("inf"), float("inf")),
                )

    @pytest.mark.parametrize("aggregation", NON_SINGLETON_AGG_METHODS)
    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            bounded_samples(
                rng=NP_RNG,
                num_experiments=2,
                num_samples=10000,
                bounds=(0, 1),
            ),
        ],
    )
    def test_bounds(
        self,
        aggregation: str,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        aggregator = get_experiment_aggregator(
            aggregation=aggregation,
            rng=RNG(seed=SEED),
        )

        out_samples = aggregator.aggregate(
            experiment_samples=samples,
            bounds=(0, 1),
        )

        assert np.max(out_samples) <= 1.0
        assert np.min(out_samples) >= 0.0

    @pytest.mark.parametrize("aggregation", NON_SINGLETON_AGG_METHODS)
    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            bounded_samples(
                rng=NP_RNG,
                num_experiments=2,
                num_samples=10000,
                bounds=(0.4, 0.75),
            ),
        ],
    )
    def test_uneven_bounds(
        self,
        aggregation: str,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        aggregator = get_experiment_aggregator(
            aggregation=aggregation,
            rng=RNG(seed=SEED),
        )

        out_samples = aggregator.aggregate(
            experiment_samples=samples,
            bounds=(0.40, 0.75),
        )

        assert np.max(out_samples) <= 0.75
        assert np.min(out_samples) >= 0.40

    @pytest.mark.parametrize("aggregation", NON_SINGLETON_AGG_METHODS)
    @pytest.mark.parametrize(
        argnames="samples",
        argvalues=[
            bounded_samples(
                rng=NP_RNG,
                num_experiments=2,
                num_samples=10000,
                bounds=(0, 1),
            ),
        ],
    )
    def test_variance(
        self,
        aggregation: str,
        samples: jtyping.Float[np.ndarray, "num_samples num_experiments"],
    ) -> None:
        aggregator = get_experiment_aggregator(
            aggregation=aggregation,
            rng=RNG(seed=SEED),
        )

        out_samples = aggregator.aggregate(
            experiment_samples=samples,
            bounds=(0, 1),
        )

        assert np.all(np.var(out_samples) <= np.var(samples, axis=0)), (
            np.var(out_samples),
            np.var(samples, axis=0),
        )
