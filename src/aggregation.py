import numpy as np
import jaxtyping as jtyping


def numpy_batched_arithmetic_mean(
    array: jtyping.Float[jtyping.Array, " num_samples ... final_dim"],
    keepdims: bool = True,
):
    return np.mean(array, axis=-1, keepdims=keepdims)


def numpy_batched_convex_combination(
    array: jtyping.Float[jtyping.Array, " num_samples ... final_dim"],
    convex_weights: jtyping.Float[jtyping.Array, " num_samples final_dim"],
    keepdims: bool = True,
):
    return np.sum(convex_weights * array, axis=-1, keepdims=keepdims)


def numpy_batched_harmonic_mean(
    array: jtyping.Float[jtyping.Array, " num_samples ... final_dim"],
    keepdims: bool = True,
):
    return np.power(np.mean(np.power(array, -1), axis=-1, keepdims=keepdims), -1)


def numpy_batched_geometric_mean(
    array: jtyping.Float[jtyping.Array, " num_samples ... final_dim"],
    keepdims: bool = True,
):
    return np.exp(np.mean(np.log(array), dim=-1, keepdims=keepdims))
