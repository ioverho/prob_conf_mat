from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import jaxtyping as jtyping

    from prob_conf_mat.utils.typing import MetricLike

from dataclasses import dataclass

import numpy as np

from prob_conf_mat.utils.formatting import fmt
from prob_conf_mat.stats import (
    summarize_posterior,
    PosteriorSummary,
    wilson_score_interval,
)

DELTA = "Δ"


@dataclass(frozen=True)
class PairwiseComparisonResult:
    lhs_name: str | None
    rhs_name: str | None
    metric: MetricLike

    observed_diff: float | None
    diff_dist: jtyping.Float[np.ndarray, " num_samples"]
    diff_dist_summary: PosteriorSummary

    direction: str
    p_direction: float
    p_direction_interpretation: str
    p_direction_score_interval_width: float

    min_sig_diff: float
    p_sig_neg: float
    p_rope: float
    p_sig_pos: float

    p_bi_sig: float
    p_bi_sig_interpretation: str
    p_bi_sig_score_interval_width: float

    p_rope_random: float
    bf_rope: float

    p_uni_sig: float
    # p_uni_sig_interpretation: str
    p_uni_sig_score_interval_width: float

    def template_sentence(self, precision: int = 4):
        # Build the template sentence
        template_sentence = ""
        template_sentence += f"Experiment {self.lhs_name}'s {self.metric.name} being"
        template_sentence += (
            " greater" if self.diff_dist_summary.median > 0 else " lesser"
        )
        template_sentence += f" than {self.rhs_name} could be considered"
        template_sentence += f" '{self.p_direction_interpretation}'* "

        # Existence statistics
        template_sentence += f"(Median {DELTA}={fmt(float(self.diff_dist_summary.median), precision=precision)}, "
        template_sentence += f"{fmt(self.diff_dist_summary.ci_probability, precision=precision, mode='%')} HDI="
        template_sentence += (
            f"[{fmt(self.diff_dist_summary.hdi[0], precision=4, mode='f')}, "
        )
        template_sentence += (
            f"{fmt(self.diff_dist_summary.hdi[1], precision=4, mode='f')}], "
        )
        template_sentence += (
            f"p_direction={fmt(self.p_direction, precision=4, mode='%')}).\n\n"
        )

        # Bidirectional significance
        template_sentence += (
            f"There is a {fmt(self.p_bi_sig, precision=precision, mode='%')}"
        )
        template_sentence += (
            " probability that this difference is bidirectionally significant"
        )
        template_sentence += (
            f" (ROPE=[{fmt(-self.min_sig_diff, precision=precision, mode='f')}, "
        )
        template_sentence += (
            f"{fmt(self.min_sig_diff, precision=precision, mode='f')}], "
        )
        template_sentence += (
            f"p_ROPE={fmt(self.p_rope, precision=precision, mode='%')}).\n\n"
        )
        template_sentence += (
            f"Bidirectional significance could be considered "
            f"'{self.p_bi_sig_interpretation}'*.\n\n"
        )

        # Unidirectional significance
        template_sentence += (
            f"There is a {fmt(self.p_uni_sig, precision=precision, mode='%')}"
        )
        template_sentence += (
            f" probability that this difference is significantly {self.direction}"
        )
        template_sentence += (
            f" (p_pos={fmt(self.p_sig_pos, precision=precision, mode='%')},"
        )
        template_sentence += (
            f" p_neg={fmt(self.p_sig_neg, precision=precision, mode='%')}).\n\n"
        )

        if self.bf_rope is not None:
            # Bidirectional significance to random
            template_sentence += (
                f"Relative to two random models "
                f"(p_ROPE,random={fmt(self.p_rope_random, precision=precision, mode='%')})"
            )

            log_bf_rope = np.log10(self.bf_rope)
            bf_rope_direction = "less" if np.sign(log_bf_rope) > 0.0 else "more"
            bf_rope_magnitude = np.power(10, np.abs(log_bf_rope))

            template_sentence += (
                f" significance is {fmt(bf_rope_magnitude, precision=precision, mode='f')} times "
                f"{bf_rope_direction} likely.\n\n"
            )

        template_sentence += (
            "* These interpretations are based off of loose guidelines, "
            "and should change according to the application."
        )

        return template_sentence


def pd_interpretation_guideline(
    pd: float,
) -> (
    typing.Literal["certain"]
    | typing.Literal["probable"]
    | typing.Literal["likely"]
    | typing.Literal["possible"]
    | typing.Literal["dubious"]
):
    # https://easystats.github.io/bayestestR/articles/guidelines.html#existence
    if pd < 0.0 or pd > 1.0:
        raise ValueError(f"Encountered pd value of {pd}, outside of range (0, 1).")
    if pd > 0.999:
        existence = "certain"
    elif pd > 0.99:
        existence = "probable"
    elif pd > 0.97:
        existence = "likely"
    elif pd > 0.95:
        existence = "possible"
    elif pd <= 0.95:
        existence = "dubious"
    else:
        raise ValueError(
            f"Encountered pd value of {pd}, somehow outside of range (0, 1).",
        )

    return existence


def p_rope_interpretation_guideline(
    p_rope: float,
) -> (
    typing.Literal["certain"]
    | typing.Literal["probable"]
    | typing.Literal["undecided"]
    | typing.Literal["probably negligible"]
    | typing.Literal["negligible"]
):
    # https://easystats.github.io/bayestestR/articles/guidelines.html#significance
    if p_rope < 0.0 or p_rope > 1.0:
        raise ValueError(f"Encountered p_rope value of {p_rope}, outside of range.")
    if p_rope < 0.01:
        significance = "certain"
    elif p_rope < 0.025:
        significance = "probable"
    elif p_rope >= 0.025 and p_rope <= 0.975:
        significance = "undecided"
    elif p_rope > 0.975 and p_rope <= 0.99:
        significance = "probably negligible"
    elif p_rope > 0.99:
        significance = "negligible"
    else:
        raise ValueError(
            f"Encountered p_rope value of {p_rope}, somehow outside of range.",
        )

    return significance


def pairwise_compare(
    metric: MetricLike,
    diff_dist: jtyping.Float[np.ndarray, " num_samples"],
    ci_probability: float,
    min_sig_diff: float | None = None,  # type: ignore
    lhs_name: str | None = None,
    rhs_name: str | None = None,
    random_diff_dist: jtyping.Float[np.ndarray, " num_samples"] | None = None,
    observed_difference: float | None = None,
) -> PairwiseComparisonResult:
    # Find central tendency of diff dit
    diff_dist_summary = summarize_posterior(diff_dist, ci_probability=ci_probability)

    # Probability of existence
    if diff_dist_summary.median > 0:
        pd: float = np.mean(diff_dist > 0)  # type: ignore
    else:
        pd: float = np.mean(diff_dist < 0)  # type: ignore

    pd_interpretation = pd_interpretation_guideline(pd=pd)

    # Define a default ROPE
    if min_sig_diff is None:
        min_sig_diff: float = 0.1 * np.std(diff_dist)  # type: ignore

    # Count the number of instances within each bin
    # Significantly negative, within ROPE, significantly positive
    counts, _ = np.histogram(
        diff_dist,
        bins=[-float("inf"), -min_sig_diff, min_sig_diff + 1e-8, float("inf")],
    )

    p_sig_neg, p_rope, p_sig_pos = counts / diff_dist.shape[0]

    p_bi_sig = 1 - p_rope
    p_rope_interpretation = p_rope_interpretation_guideline(p_rope)

    # Compare p_ROPE to random distributions
    if random_diff_dist is not None and min_sig_diff > 0.0:
        p_bi_sig_random = np.mean(
            (random_diff_dist < -min_sig_diff) | (random_diff_dist > min_sig_diff),
        )
        p_rope_random = 1 - p_bi_sig_random
        bf_rope = p_rope / p_rope_random

    else:
        p_bi_sig_random = None
        p_rope_random = None
        bf_rope = None

    result = PairwiseComparisonResult(
        # Admin
        lhs_name=lhs_name,
        rhs_name=rhs_name,
        metric=metric,
        # The difference distribution
        observed_diff=observed_difference,
        diff_dist=diff_dist,
        diff_dist_summary=diff_dist_summary,
        # Existence
        direction="positive" if diff_dist_summary.median > 0 else "negative",
        p_direction=pd,
        p_direction_interpretation=pd_interpretation,
        p_direction_score_interval_width=wilson_score_interval(
            p=pd,
            n=diff_dist.shape[0],
        ),
        # Significance buckets
        min_sig_diff=min_sig_diff,
        p_sig_neg=p_sig_neg,
        p_rope=p_rope,
        p_sig_pos=p_sig_pos,
        # Bidirectional significance
        p_bi_sig=p_bi_sig,
        p_bi_sig_interpretation=p_rope_interpretation,
        p_bi_sig_score_interval_width=wilson_score_interval(
            p=1 - p_rope,
            n=diff_dist.shape[0],
        ),
        # Significance relative to random
        p_rope_random=p_rope_random,  # type: ignore
        bf_rope=bf_rope,  # type: ignore
        # Unidirectional significance
        p_uni_sig=p_sig_pos if diff_dist_summary.median > 0 else p_sig_neg,
        p_uni_sig_score_interval_width=wilson_score_interval(
            p=p_sig_pos if diff_dist_summary.median > 0 else p_sig_neg,
            n=diff_dist.shape[0],
        ),
    )

    return result
