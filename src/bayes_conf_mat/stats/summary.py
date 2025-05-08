import numpy as np
import jaxtyping as jtyping
import scipy.stats as stats
from dataclasses import dataclass
import typing

from bayes_conf_mat.stats.mode_estimation import histogram_mode_estimator
from bayes_conf_mat.stats.hdi_estimation import hdi_estimator


@dataclass(frozen=True)
class PosteriorSummary:
    median: float
    mode: float
    ci_probability: float
    hdi: tuple[float, float]
    skew: float
    kurtosis: float

    @property
    def metric_uncertainty(self) -> float:
        return self.hdi[1] - self.hdi[0]

    @property
    def headers(self) -> list[str]:
        return [
            "Median",
            "Mode",
            f"{self.ci_probability*100:.1f}% HDI",
            "MU",
            "Skew",
            "Kurt",
        ]

    def as_dict(self) -> dict[str, typing.Any]:
        d = {
            "Median": self.median,
            "Mode": self.mode,
            f"{self.ci_probability*100:.1f}% HDI": self.hdi,
            "MU": self.metric_uncertainty,
            "Skew": self.skew,
            "Kurt": self.kurtosis,
        }

        return d


def summarize_posterior(
    posterior_samples: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
):
    summary = PosteriorSummary(
        median=typing.cast(float, np.median(posterior_samples)),
        mode=typing.cast(float, histogram_mode_estimator(posterior_samples)),
        ci_probability=ci_probability,
        hdi=typing.cast(tuple[float, float], hdi_estimator(posterior_samples, prob=ci_probability)),
        skew=stats.skew(posterior_samples),
        kurtosis=stats.kurtosis(posterior_samples),
    )

    return summary
