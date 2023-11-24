import numpy as np
import jaxtyping as jtyping


def numpy_batched_arithmetic_mean(
    array: jtyping.Float[np.ndarray, " num_samples ... final_dim"],
    axis: int = -1,
    keepdims: bool = True,
):
    return np.mean(array, axis=axis, keepdims=keepdims)


def numpy_batched_convex_combination(
    array: jtyping.Float[np.ndarray, " num_samples ... final_dim"],
    convex_weights: jtyping.Float[np.ndarray, " num_samples final_dim"],
    axis: int = -1,
    keepdims: bool = True,
):
    return np.sum(convex_weights * array, axis=axis, keepdims=keepdims)


def numpy_batched_harmonic_mean(
    array: jtyping.Float[np.ndarray, " num_samples ... final_dim"],
    axis: int = -1,
    keepdims: bool = True,
):
    return np.power(np.mean(np.power(array, -1), axis=axis, keepdims=keepdims), -1)


def numpy_batched_geometric_mean(
    array: jtyping.Float[np.ndarray, " num_samples ... final_dim"],
    axis: int = -1,
    keepdims: bool = True,
):
    return np.exp(np.mean(np.log(array), axis=axis, keepdims=keepdims))
