from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import numpy as np


def numpy_batched_arithmetic_mean(
    array: jtyping.Float[np.ndarray, "... axis ..."],
    axis: int = -1,
    *,
    keepdims: bool = True,
) -> jtyping.Float[np.ndarray, "... 1 ..."]:
    """Computes the [arithmetic mean](https://en.wikipedia.org/wiki/Arithmetic_mean) over an axis."""  # noqa: E501
    return np.mean(array, axis=axis, keepdims=keepdims)


def numpy_batched_convex_combination(
    array: jtyping.Float[np.ndarray, "... axis ..."],
    axis: int = -1,
    *,
    convex_weights: jtyping.Float[np.ndarray, " num_samples final_dim"],
    keepdims: bool = True,
) -> jtyping.Float[np.ndarray, "... 1 ..."]:
    """Computes a [weighted mean](https://en.wikipedia.org/wiki/Weighted_arithmetic_mean) over an axis."""  # noqa: E501
    return np.sum(convex_weights * array, axis=axis, keepdims=keepdims)


def numpy_batched_harmonic_mean(
    array: jtyping.Float[np.ndarray, "... axis ..."],
    axis: int = -1,
    *,
    keepdims: bool = True,
) -> jtyping.Float[np.ndarray, "... 1 ..."]:
    """Computes the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean) over an axis."""  # noqa: E501
    return np.power(np.mean(np.power(array, -1), axis=axis, keepdims=keepdims), -1)


def numpy_batched_geometric_mean(
    array: jtyping.Float[np.ndarray, "... axis ..."],
    axis: int = -1,
    *,
    keepdims: bool = True,
) -> jtyping.Float[np.ndarray, "... 1 ..."]:
    """Computes the [weighted mean](https://en.wikipedia.org/wiki/Geometric_mean) over an axis."""  # noqa: E501
    return np.exp(np.mean(np.log(array), axis=axis, keepdims=keepdims))
