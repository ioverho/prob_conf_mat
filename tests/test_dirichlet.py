import pytest
import numpy as np


from prob_conf_mat.stats.dirichlet_distribution import dirichlet_prior, dirichlet_sample

ALL_SEEDS = np.random.default_rng(seed=0).integers(low=0, high=2**31 - 1, size=(10,))
ALL_DIMS = [2, 4, 8, 16]


class TestDirichletPrior:
    def test_numeric_strategy(self) -> None:
        # Test float
        dirichlet_prior(strategy=1.0, shape=(2, 2))

        # Test int
        dirichlet_prior(strategy=1, shape=(2, 2))

        # Test bounds
        dirichlet_prior(strategy=0.0, shape=(2, 2))

        dirichlet_prior(strategy=np.nan, shape=(2, 2))

        with pytest.raises(ValueError):
            dirichlet_prior(strategy=-1, shape=(2, 2))

    def test_str(self) -> None:
        # Test valid
        dirichlet_prior(strategy="haldane", shape=(2, 2))

        # Test invalid
        with pytest.raises(ValueError):
            dirichlet_prior(strategy="foobarbaz", shape=(2, 2))

    def test_arraylike(self) -> None:
        # Test valid
        dirichlet_prior(strategy=[[0, 0], [0, 0]], shape=(2, 2))

        dirichlet_prior(strategy=np.zeros((2, 2)), shape=(2, 2))

        # Test invalid shape
        with pytest.raises(ValueError):
            dirichlet_prior(strategy=[[0, 0], [0, 0], [0, 0]], shape=(2, 2))

        # Test invalid object
        class Dummy: ...

        with pytest.raises(ValueError):
            dirichlet_prior(strategy=Dummy, shape=(2, 2))  # type: ignore


class TestDirichletDistribution:
    @pytest.mark.parametrize("seed", argvalues=ALL_SEEDS)
    @pytest.mark.parametrize("dim", argvalues=ALL_DIMS)
    def test_distribution_equivalence_1d(self, seed: int, dim: int):
        rng = np.random.default_rng(seed=seed)

        alphas = rng.integers(low=1, high=10, size=(dim,))

        rng = np.random.default_rng(seed=seed)

        np_dirichlet_samples = rng.dirichlet(alpha=alphas, size=10000)

        rng = np.random.default_rng(seed=seed)

        our_dirichlet_samples = dirichlet_sample(
            rng=rng,  # type: ignore
            alphas=alphas,
            num_samples=10000,
        )

        assert np.allclose(
            np_dirichlet_samples,
            our_dirichlet_samples,
        ), "Distributions do not match"

    @pytest.mark.parametrize("seed", argvalues=ALL_SEEDS)
    @pytest.mark.parametrize("dim", argvalues=ALL_DIMS)
    def test_distribution_equivalence_2d(self, seed: int, dim: int) -> None:
        num_samples = 10000

        rng = np.random.default_rng(seed=seed)

        alphas = rng.integers(low=1, high=10, size=(dim, dim))

        rng = np.random.default_rng(seed=seed)

        np_dirichlet_samples = np.stack(
            arrays=[
                rng.dirichlet(alpha=alphas_, size=num_samples) for alphas_ in alphas
            ],
            axis=1,
        )

        rng = np.random.default_rng(seed=seed)

        our_dirichlet_samples = dirichlet_sample(
            rng=rng,  # type: ignore
            alphas=alphas,
            num_samples=num_samples,
        )

        #! Means of the different distributions must be within 1% of each other
        mean_geom_mean_relative_error = np.abs(
            np.exp(
                np.log(np.mean(our_dirichlet_samples.reshape(num_samples, -1), axis=0))
                - np.log(
                    np.mean(np_dirichlet_samples.reshape(num_samples, -1), axis=0),
                ),
            )
            - 1,
        )

        assert np.mean(mean_geom_mean_relative_error) < 0.01, (
            f"Means of distributions do not match: {np.mean(mean_geom_mean_relative_error)}"
        )

        #! Variances of the different distributions must be within 5% of each other
        var_geom_mean_relative_error = np.abs(
            np.exp(
                np.log(np.var(our_dirichlet_samples.reshape(num_samples, -1), axis=0))
                - np.log(
                    np.var(np_dirichlet_samples.reshape(num_samples, -1), axis=0),
                ),
            )
            - 1,
        )

        assert np.mean(var_geom_mean_relative_error) < 0.05, (
            f"Variances of distributions do not match: {np.mean(var_geom_mean_relative_error)}"
        )
