from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import numpy as np


def histogram_mode_estimator(
    samples: jtyping.Float[np.ndarray, " num_samples"],
    bounds: tuple[float, float] | None = None,
) -> float:
    """"Tries to estimate the mode of a distribution from its samples."""
    bin_counts, bin_edges = np.histogram(samples, bins="auto", range=bounds)
    modal_bin = np.argmax(bin_counts)

    mode = (bin_edges[modal_bin] + bin_edges[modal_bin + 1]) / 2

    return mode


# TODO: remove or integrate KDE mode estimation
# def kde_mode_estimator(
#     samples: jtyping.Float[np.ndarray, " num_samples"],
#     range: typing.Optional[typing.Tuple[float, float]] = None,
# )-> float:
#     from lightkde import kde_1d

#     kernel_densities, evaluation_points = kde_1d(
#         samples,
#         x_min=range[0] if range is not None else None,
#         x_max=range[1] if range is not None else None,
#     )

#     modal_point = np.argmax(kernel_densities)

#     mode = evaluation_points[modal_point]

#     return mode
