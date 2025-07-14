import dataclasses

import numpy as np
import jaxtyping as jtyping
import scipy


@dataclasses.dataclass(frozen=True, kw_only=True)
class KDEResult:
    """_summary_."""

    kernel: scipy.stats._kde.gaussian_kde
    x_vals: jtyping.Float[np.ndarray, " num_kernel_samples"]
    y_vals: jtyping.Float[np.ndarray, " num_kernel_samples"]
    bounds: tuple[float, float]


def compute_kde(
    samples: jtyping.Float[np.ndarray, " num_samples"],
    bw_method: str | None = None,
    bw_adjust: float = 1,
    cut: float = 3,
    clip: tuple[float, float] | None = None,
    grid_samples: int = 200,
) -> KDEResult:
    """Computes a Gaussian KDE.

    Tries to match the interface used by [Seaborn's KDE estimator](https://github.com/mwaskom/seaborn/blob/master/seaborn/_statistics.py#L42).

    Args:
        samples (Float[ndarray, 'num_samples']): the experiment samples
        bw_method (str | None, optional): the bandwidth selection method to use.
            Defaults to Scipy's default: 'scott'.
        bw_adjust (float, optional):
            Factor that multiplicatively scales the bandwidth chosen.
            Increasing will make the curve smoother.
            Defaults to 1.
        cut (float, optional): Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints.
            When set to 0, truncate the curve at the data limits.
            Defaults to 3.
        clip (tuple[float, float] | None, optional): the bounds outside of which no density values
            will be computed.
            Defaults to None.
        grid_samples (int, optional): the number of KDE points to evaluate at.
            Defaults to 200.

    Returns:
        KDEResult: the output of the KDE
    """
    # Check the space we're allowed to plot in
    empirical_min = np.min(samples)
    empirical_max = np.max(samples)

    min_grid_val = max(
        empirical_min - cut * bw_adjust,
        clip[0] if clip is not None else -np.inf,
    )

    max_grid_val = min(
        empirical_max + cut * bw_adjust,
        clip[1] if clip is not None else np.inf,
    )

    grid_x_vals = np.linspace(min_grid_val, max_grid_val, num=grid_samples)

    # Compute the kernel
    kernel = scipy.stats.gaussian_kde(dataset=samples, bw_method=bw_method)
    kernel.set_bandwidth(bw_method=kernel.factor * bw_adjust)

    # Compute the kernel output
    grid_y_vals = kernel(grid_x_vals)

    result = KDEResult(
        kernel=kernel,
        x_vals=grid_x_vals,
        y_vals=grid_y_vals,
        bounds=(min_grid_val, max_grid_val),
    )

    return result
