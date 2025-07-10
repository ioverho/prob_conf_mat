from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
import typing

from prob_conf_mat.stats.mode_estimation import histogram_mode_estimator
from prob_conf_mat.stats.hdi_estimation import hdi_estimator


@dataclass(frozen=True)
class PosteriorSummary:
    """Summary statistics of some probability distribution."""

    median: float
    mode: float
    ci_probability: float
    hdi: tuple[float, float]
    skew: float
    kurtosis: float

    @property
    def metric_uncertainty(self) -> float:
        """The metric uncertainty (MU), defined as the size of the HDI.

        Returns:
            float: the MU
        """
        return self.hdi[1] - self.hdi[0]

    @property
    def headers(self) -> list[str]:
        """The column headers."""
        return [
            "Median",
            "Mode",
            f"{self.ci_probability * 100:.1f}% HDI",
            "MU",
            "Skew",
            "Kurt",
        ]

    def as_dict(self) -> dict[str, float | tuple[float, float]]:
        """Returns the dict representation of the statistics.

        Useful for coverting to a table.

        Returns:
            dict[str, float | tuple[float, float]]
        """
        d = {
            "Median": self.median,
            "Mode": self.mode,
            f"{self.ci_probability * 100:.1f}% HDI": self.hdi,
            "MU": self.metric_uncertainty,
            "Skew": self.skew,
            "Kurt": self.kurtosis,
        }

        return d


def summarize_posterior(
    posterior_samples: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
) -> PosteriorSummary:
    """Summarizes a distribution, assumed to be a posterior, based on samples from it.

    Args:
        posterior_samples (jtyping.Float[np.ndarray, " num_samples"]): samples from the posterior.
        ci_probability (float): the probability under the HDI.

    Returns:
        PosteriorSummary: the summary statistics.
    """
    summary = PosteriorSummary(
        median=typing.cast("float", np.median(posterior_samples)),
        mode=typing.cast("float", histogram_mode_estimator(samples=posterior_samples)),
        ci_probability=ci_probability,
        hdi=typing.cast(
            "tuple[float, float]",
            hdi_estimator(posterior_samples, prob=ci_probability),
        ),
        skew=stats.skew(posterior_samples),
        kurtosis=stats.kurtosis(posterior_samples),
    )

    return summary
