from dataclasses import dataclass

import numpy as np
import jaxtyping as jtyping

from bayes_conf_mat.experiment import ExperimentResult
from bayes_conf_mat.experiment_manager import ExperimentAggregationResult
from bayes_conf_mat.utils import fmt
from bayes_conf_mat.stats import (
    summarize_posterior,
    PosteriorSummary,
    wilson_score_interval,
)

DELTA = "Δ"


@dataclass(frozen=True)
class PairwiseComparisonResult:
    lhs_name: str
    rhs_name: str
    metric_name: str

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

    p_uni_sig: float
    p_uni_sig_score_interval_width: float

    def template_sentence(self, precision: int = 4):
        # Build the template sentence
        template_sentence = ""
        template_sentence += (
            f"Experiment <{self.lhs_name}>'s <{self.metric_name}> being"
        )
        template_sentence += (
            " greater" if self.diff_dist_summary.median > 0 else " lesser"
        )
        template_sentence += f" than <{self.rhs_name}> is"
        template_sentence += f" '{self.p_direction_interpretation}' "

        # Existence statistics
        template_sentence += f"(Median {DELTA}={fmt(self.diff_dist_summary.median, precision=precision)}, "
        template_sentence += f"{fmt(self.diff_dist_summary.ci_probability, precision=precision, mode="%")} HDI="
        template_sentence += (
            f"[{fmt(self.diff_dist_summary.hdi[0], precision=4, mode="f")}, "
        )
        template_sentence += (
            f"{fmt(self.diff_dist_summary.hdi[1], precision=4, mode="f")}], "
        )
        template_sentence += (
            f"p_direction={fmt(self.p_direction, precision=4, mode="%")})."
        )

        # Bidirectional significance
        template_sentence += (
            f"\nThere is a {fmt(self.p_bi_sig, precision=precision, mode="%")}"
        )
        template_sentence += (
            " probability that this difference is bidirectionally significant"
        )
        template_sentence += (
            f" (ROPE=[{fmt(-self.min_sig_diff, precision=4, mode="f")}, "
        )
        template_sentence += f"{fmt(self.min_sig_diff, precision=4, mode="f")}], "
        template_sentence += (
            f"p_ROPE={fmt(self.p_rope, precision=precision, mode="%")})."
        )
        template_sentence += f"\nBidirectional significance could be considered '{self.p_bi_sig_interpretation}'."

        # Unidirectional significance
        template_sentence += (
            f"\nThere is a {fmt(self.p_uni_sig, precision=precision, mode="%")}"
        )
        template_sentence += (
            f" probability that this difference is significantly {self.direction}"
        )
        template_sentence += (
            f" (p_pos={fmt(self.p_sig_pos, precision=precision, mode="%")},"
        )
        template_sentence += (
            f" p_neg={fmt(self.p_sig_neg, precision=precision, mode="%")})."
        )

        return template_sentence

    def sensitivity_analysis(self):
        return {
            "p_direction": self.p_direction_interval_widt,
            "p_bi_sig": self.p_bi_sig_interval_widt,
            "p_uni_sig": self.p_uni_sig_interval_widt,
        }


def pd_interpretation_guideline(pd: float):
    # https://easystats.github.io/bayestestR/articles/guidelines.html#existence
    if pd < 0.0 or pd > 1.0:
        raise ValueError(f"Found pd value of {pd}, outside of range.")
    elif pd > 0.999:
        existence = "certain"
    elif pd > 0.99:
        existence = "probable"
    elif pd > 0.97:
        existence = "likely"
    elif pd > 0.95:
        existence = "possible"
    elif pd <= 0.95:
        existence = "dubious"

    return existence


def p_rope_interpretation_guideline(p_rope: float):
    # https://easystats.github.io/bayestestR/articles/guidelines.html#significance
    if p_rope < 0.0 or p_rope > 1.0:
        raise ValueError(f"Found p_rope value of {p_rope}, outside of range.")
    elif p_rope < 0.01:
        significance = "certain"
    elif p_rope < 0.025:
        significance = "probable"
    elif p_rope >= 0.025 and p_rope <= 0.975:
        significance = "undecided"
    elif p_rope > 0.975 and p_rope <= 0.99:
        significance = "probably negligible"
    elif p_rope > 0.99:
        significance = "negligible"

    return significance


def compare_posteriors(
    lhs: ExperimentResult | ExperimentAggregationResult,
    rhs: ExperimentResult | ExperimentAggregationResult,
    ci_probability: float,
    min_sig_diff: float,
):
    if lhs.metric.name != rhs.metric.name:
        raise ValueError(
            f"The metric used to compute lhs and rhs are not the same: {lhs.metric.name} != {rhs.metric.name}"
        )

    # Compute the difference distribution
    diff_dist = lhs.values - rhs.values

    # Find central tendency of diff dit
    diff_dist_summary = summarize_posterior(diff_dist, ci_probability=ci_probability)

    # Probability of existence
    if diff_dist_summary.median > 0:
        pd = np.mean(diff_dist > 0)
    else:
        pd = np.mean(diff_dist < 0)

    pd_interpretation = pd_interpretation_guideline(pd)

    # Define a default ROPE
    if min_sig_diff is None:
        min_sig_diff = (
            0.1 * np.sqrt(np.power(lhs.values, 2) + np.power(rhs.values, 2))[0]
        )

    # Count the number of instances within each bin
    # Significantly negative, within ROPE, significantly positive
    counts, _ = np.histogram(
        diff_dist,
        bins=[-float("inf"), -min_sig_diff, min_sig_diff + 1e-8, float("inf")],
    )

    p_sig_neg, p_rope, p_sig_pos = counts / diff_dist.shape[0]

    p_bi_sig = 1 - p_rope
    p_rope_interpretation = p_rope_interpretation_guideline(p_rope)

    result = PairwiseComparisonResult(
        # Admin
        lhs_name=lhs.name,
        rhs_name=rhs.name,
        metric_name=lhs.metric.name,
        # The difference distribution
        diff_dist=diff_dist,
        diff_dist_summary=diff_dist_summary,
        # Existence
        direction="positive" if diff_dist_summary.median > 0 else "negative",
        p_direction=pd,
        p_direction_interpretation=pd_interpretation,
        p_direction_score_interval_width=wilson_score_interval(
            p=pd, n=diff_dist.shape[0]
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
            p=1 - p_rope, n=diff_dist.shape[0]
        ),
        # Unidirectional significance
        p_uni_sig=p_sig_pos if diff_dist_summary.median > 0 else p_sig_neg,
        p_uni_sig_score_interval_width=wilson_score_interval(
            p=p_sig_pos if diff_dist_summary.median > 0 else p_sig_neg,
            n=diff_dist.shape[0],
        ),
    )

    return result
