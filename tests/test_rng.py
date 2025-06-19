from __future__ import annotations
import typing

import pytest

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import numpy as np
import scipy.stats as stats
import itertools

from prob_conf_mat.utils import RNG

ALL_SEEDS = np.random.default_rng(seed=0).integers(low=0, high=2**31 - 1, size=(10,))


def generate_tree(meta_seed: int, root_seed: int, max_children: int) -> list[RNG]:
    meta_rng = np.random.default_rng(meta_seed)

    tree = [RNG(seed=root_seed)]

    i = 0
    while (i < len(tree) - 1) or (i == 0):
        node = tree[i]

        if len(node.position) == 0:
            child_nodes = node.spawn(
                n_children=int(meta_rng.integers(low=2, high=max_children)),
            )

            tree.extend(child_nodes)

        elif len(node.position) < 4:
            child_nodes = node.spawn(
                n_children=int(meta_rng.integers(low=0, high=max_children)),
            )

            tree.extend(child_nodes)

        i += 1

    return tree


def p_independent(
    node_a: RNG,
    node_b: RNG,
    sample_size: int = 10000,
    num_bins: int = 10,
) -> float:
    node_a_samples = node_a.random(size=(sample_size,))
    node_b_samples = node_b.random(size=(sample_size,))

    bin_counts, _, _ = np.histogram2d(
        x=node_a_samples,
        y=node_b_samples,
        bins=num_bins,
        range=((0, 1), (0, 1)),
    )

    p = stats.power_divergence(f_obs=bin_counts, axis=None, lambda_=0).pvalue  # type: ignore

    return p


def num_rejected_hypotheses(
    p_vals: jtyping.Float[np.ndarray, " num_tests"],
    alpha: float,
) -> int:
    """Uses Holm–Bonferroni's step-down procedure to count the number of rejected null-hypotheses.

    This should correct for the family-wise error rate.

    Args:
        p_vals (jtyping.Float[np.ndarray, "num_tests"]): the p values of all conducted tests
        alpha (float): the significance level

    Returns:
        int: the number of hypotheses that should be rejected according to the provided significance level
    """  # noqa: E501
    fail_to_reject_null = np.sort(p_vals) > alpha / (
        p_vals.shape[0] - np.arange(p_vals.shape[0])
    )

    n_rejected = (
        np.where(fail_to_reject_null)[0][0] if np.any(fail_to_reject_null) else 0
    )

    return n_rejected


# Tests ========================================================================
class TestRNG:
    @pytest.mark.parametrize(argnames="seed", argvalues=ALL_SEEDS)
    def test_equivalence_to_numpy(self, seed: int):
        # Test root node
        our_rng = RNG(seed=seed)
        np_rng = np.random.default_rng(seed)

        our_value = our_rng.random(size=(1000,))
        np_value = np_rng.random(size=(1000,))

        assert np.allclose(our_value, np_value)

        # Test child node
        our_child_rng = our_rng.spawn(1)[0]
        np_child_rng = np.random.default_rng(seed).spawn(1)[0]

        our_value = our_child_rng.random(size=(1000,))
        np_value = np_child_rng.random(size=(1000,))

        assert np.allclose(our_value, np_value)

    @pytest.mark.parametrize(argnames="seed", argvalues=ALL_SEEDS)
    def test_independence(self, seed: int):
        tree = generate_tree(meta_seed=seed, root_seed=0, max_children=4)

        all_pvals = []
        for node_a, node_b in itertools.combinations(tree, 2):
            all_pvals.append(
                p_independent(node_a, node_b, sample_size=10000, num_bins=100),
            )

        all_pvals = np.stack(all_pvals)

        n_rejected = num_rejected_hypotheses(p_vals=all_pvals, alpha=0.1)

        assert n_rejected == 0, n_rejected

    @pytest.mark.parametrize(
        argnames="seed_a,seed_b",
        argvalues=itertools.combinations(ALL_SEEDS, 2),
    )
    def test_changing_seed(self, seed_a: int, seed_b: int):
        tree_generator_seed = abs(seed_a - seed_b)

        # Sample a tree
        tree_a = generate_tree(
            meta_seed=tree_generator_seed,
            root_seed=seed_a,
            max_children=4,
        )

        samples_a = []
        for node in tree_a:
            samples_a.append(node.random())

        samples_a = np.stack(samples_a)

        # Sample a second, independent tree
        tree_b = generate_tree(
            meta_seed=tree_generator_seed,
            root_seed=seed_b,
            max_children=4,
        )

        samples_b = []
        for node in tree_b:
            samples_b.append(node.random())

        samples_b = np.stack(samples_b)

        #! Samples from different trees with different seeds should be independent
        assert not np.allclose(samples_a, samples_b)

        # Set the seed on the root node to the same seed as tree_b
        tree_a[0].seed = seed_b

        samples_a_reset = []
        for node in tree_a:
            samples_a_reset.append(node.random())

        samples_a_reset = np.stack(samples_a_reset)

        #! Samples from different trees with the same seed should be the same
        #! Setting the seed on the root node should reset the tree's state
        assert np.allclose(samples_a_reset, samples_b)

        # Set the seed back to the initial seed of tree_a
        tree_a[0].seed = seed_a

        samples_a_reset = []
        for node in tree_a:
            samples_a_reset.append(node.random())

        samples_a_reset = np.stack(samples_a_reset)

        #! Samples from different trees with the same seed should be the same
        #! Setting the seed on the root node should reset the tree's state
        assert np.allclose(samples_a_reset, samples_a)
