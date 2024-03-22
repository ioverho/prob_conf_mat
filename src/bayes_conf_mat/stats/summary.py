import numpy as np
import jaxtyping as jtyping
import scipy.stats as stats
from dataclasses import dataclass
import typing

from bayes_conf_mat.stats.mode_estimation import histogram_mode_estimator
from bayes_conf_mat.stats.hdi_estimation import hdi_estimator


@dataclass(frozen=True)
class PosteriorSummary:
    median: jtyping.Float[np.ndarray, " num_samples"]
    mode: jtyping.Float[np.ndarray, " num_samples"]
    ci_probability: float
    hdi: typing.Tuple[float, float]
    skew: float
    kurtosis: float

    def as_dict(self):
        d = {
            "Median": self.median,
            "Mode": self.mode,
            f"{self.ci_probability*100:.1f}% HDI": self.hdi,
            "Skew": self.skew,
            "Kurt": self.kurtosis,
        }

        return d


def summarize_posterior(
    posterior_samples: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
):
    summary = PosteriorSummary(
        median=np.median(posterior_samples),
        mode=histogram_mode_estimator(posterior_samples),
        ci_probability=ci_probability,
        hdi=hdi_estimator(posterior_samples, prob=ci_probability),
        skew=stats.skew(posterior_samples),
        kurtosis=stats.kurtosis(posterior_samples),
    )

    return summary
