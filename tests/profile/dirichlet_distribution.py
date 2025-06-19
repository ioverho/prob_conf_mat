from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

from timeit import timeit

import numpy as np
from opt_einsum import contract
from tqdm import tqdm

from prob_conf_mat.utils.rng import RNG


def normalize_array_baseline(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    return arr / np.sum(arr, axis=-1, keepdims=True)


def normalize_array_1(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    np.divide(arr, np.sum(arr, axis=-1, keepdims=True), out=arr)
    return arr


def normalize_array_2(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    arr /= np.sum(a=arr, axis=-1, keepdims=True)
    return arr


def normalize_array_3(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = np.einsum("...i->...", arr)
    arr /= sum_arr[..., np.newaxis]
    return arr


def normalize_array_4(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    np.divide(arr, np.einsum("...i->...", arr)[..., np.newaxis], out=arr)
    return arr


def normalize_array_5(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = contract("...i->...", arr)
    arr /= sum_arr[..., np.newaxis]
    return arr


def normalize_array_6(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    np.divide(arr, contract("...i->...", arr)[..., np.newaxis], out=arr)
    return arr


def normalize_array_7(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = np.empty((arr.shape[0], arr.shape[1]))
    contract("...i->...", arr, out=sum_arr)
    arr /= sum_arr[..., np.newaxis]
    return arr


def normalize_array_8(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = contract("...i->...", arr, optimize="greedy")
    arr /= sum_arr[..., np.newaxis]
    return arr


def normalize_array_9(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = contract("...i->...", arr, optimize="optimal")
    arr /= sum_arr[..., np.newaxis]
    return arr


def normalize_array_10(  # noqa
    arr: jtyping.Float[np.ndarray, "... num_classes"],
) -> jtyping.Float[np.ndarray, "... num_classes"]:
    sum_arr = np.empty((arr.shape[0], arr.shape[1]))
    sum_arr = np.einsum("...i->...", arr)
    arr /= sum_arr[..., np.newaxis]
    return arr


ALL_NORMALIZATION_FUNCS = [
    # normalize_array_1,
    # normalize_array_2,
    normalize_array_3,
    normalize_array_4,
    normalize_array_5,
    # normalize_array_6,
    normalize_array_7,
    # normalize_array_8,
    # normalize_array_9,
    normalize_array_10,
]

if __name__ == "__main__":
    repeats = 10
    iterations = 1000
    num_samples = 10000
    alphas = np.ones((10, 10), dtype=np.float64)

    rng = RNG(seed=0)

    # Broadcast alphas to the desired shape
    alphas = np.broadcast_to(alphas, (num_samples, *alphas.shape))

    # Generate independent gamma-distributed samples
    arr = rng.standard_gamma(alphas)

    all_pcall = []
    for i in tqdm(range(repeats + 1)):
        total_baseline_time = timeit(
            "normalize_array_baseline(arr=arr)",
            number=iterations,
            globals=globals(),
        )

        baseline_normalized_arr = normalize_array_baseline(arr=arr)

        all_times = [total_baseline_time]
        for func in tqdm(ALL_NORMALIZATION_FUNCS):
            total_time = timeit(
                f"{func.__name__}(arr=arr)",
                number=iterations,
                globals=globals(),
            )

            normalized_arr = func(arr=arr)

            assert np.allclose(baseline_normalized_arr, normalized_arr)

            all_times.append(total_time)

        if i != 0:
            all_pcall.append(all_times)

    all_pcall = np.array(all_pcall)
    print(np.median(all_pcall, axis=0))

    for i, func in enumerate([normalize_array_baseline] + ALL_NORMALIZATION_FUNCS):
        print(
            f"{func.__name__:<30} |"
            f" {np.median(all_pcall[:, i]):.2f}"
            f" {np.median(all_pcall[:, i] / iterations):.2e}",
            f" {np.std(all_pcall[:, i]):.2f}",
            f" {np.std(all_pcall[:, i] / iterations):.2e}",
        )
