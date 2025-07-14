# Experiment Aggregation

With machine-learning setups, it can happen that we have several experiments measuring the same effect. For example, when we perform cross-fold validation, we try to measure the generalization gap for the same learning algorithm, just using different subsets of the same dataset. In those cases, we're typically not interested in the individual experiment values. Instead, we want to know the statistics about the average experiment distribution.

In `bayes_conf_mat`, we call this experiment aggregation. Using only samples drawn from the $M$ individual empirical metric distributions, $\mu\sim p(\mu_{m})$, we wish to find the distribution of the aggregate distribution $q(\mu)$. Ideally, the aggregate distribution consolidates the information present in each experiment distribution, and produces a distribution that is more confident about the true metric value than any individual experiment could be.

Specifically for `bayes_conf_mat`, we utilize two frameworks specifically for producing such aggregate distributions.

## Meta-Analysis

The first, and by far most common, approach to combining different probability distributions is through a method called [meta-analysis](https://en.wikipedia.org/wiki/Meta-analysis#Statistical_models_for_aggregate_data). The simplest meta-analysis estimator is the fixed-effects model, which infers the parameters of the aggregate distribution from an inverse variance weighted mean of the individual distributions.

$$
\begin{aligned}
    w_{i}&=\dfrac{\sigma_{i}^{-2}}{\sum_{j}^{M}\sigma_{j}^{-2}} \\
    \tilde{\mu}&=\sum_{i}^{M}w_{i}\mu_{i} \\
    \tilde{\sigma}^2&=\dfrac{1}{\sum_{i}^{M}\sigma_{i}^{-2}} \\
    q(\mu;\tilde{\mu}, \tilde{\sigma})&=\mathcal{N}(\mu;\tilde{\mu}, \tilde{\sigma})
\end{aligned}
$$

There are many more meta-analysis tools, including [Bayesian approaches](https://bookdown.org/MathiasHarrer/Doing_Meta_Analysis_in_R/bayesian-ma.html), but these come with additional complexity and assumptions, with little added benefit for `bayes_conf_mat`.

We implement the following aggregation methods stemming from this aggregation framework:

1. [`FEGaussianAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator): the standard fixed effects model
2. [`REGaussianAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator): a random effects model that tries to correct for inter-experiment heterogeneity

A standard work on meta-analyses (from the perspective of systematic reviews) is the [Cochrane Handbook](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)[^1].

[^1]: [Chandler, J., Cumpston, M., Li, T., Page, M. J., & Welch, V. J. H. W. (2019). Cochrane handbook for systematic reviews of interventions. Hoboken: Wiley, 4.](https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-10)

## Conflation

The second framework for consolidating different experiments is conflation. It's a term coined by Hill in [Hill (2011)](https://arxiv.org/abs/0808.1808)[^2] & [Hill & Miller (2011)](https://arxiv.org/abs/1005.4978)[^3], leverages a very simple method derived from probability theory first principles. Specifically, for a set of $M$ probability distributions over the same space (in our case, the classification evaluation metric $\mu$), then the *conflated* distribution is computed as:

[^2]: [Hill, T. (2011). Conflations of probability distributions. Transactions of the American Mathematical Society, 363(6), 3351-3372.](https://arxiv.org/abs/0808.1808)
[^3]: [Hill, T. P., & Miller, J. (2011). How to combine independent data sets for the same quantity. Chaos: An Interdisciplinary Journal of Nonlinear Science, 21(3).](https://arxiv.org/abs/1005.4978)

$$\begin{aligned}
    q(\mu)&=\&\left(p_{1}(\mu), p_{2}(\mu),\ldots,p_{M}(\mu)\right) \\
    &=\dfrac{\prod_{i=1}^{M} p_{i}(\mu)}{\int_{-\infty}^{\infty}\prod_{j=1}^{M} p_{j}(\mu)d\mu}\end{aligned}$$

In other words, the renormalized product of the individual distributions. For discrete distributions, the integral becomes a sum over the distributions' support.

This method has a nice intuition: the only way a value $\mu$ receives large probability mass/density in the aggregate distribution, is by having mutual agreement among all individual distributions.

In their papers, Hill & Miller[^2][^3] go on to show that conflation uniquely minimizes the loss of Shannon information due to aggregation. What this means, is that the conflated distribution minimizes the additional 'surprise' incurred when replacing each $p_i(\mu)$ with $\&(p_i(\mu))$. They also go on to prove a variety of other useful properties, but most importantly, show that under normality assumptions, the fixed-effect meta-analytical estimator is just a special case of conflation.

We implement the following aggregation methods stemming from this aggregation framework:

1. [`FEGaussianAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator): the conflation of several Gaussian distributions. Equal to the above meta-analysis framework
2. [`BetaAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator): the conflation of several Beta distributions. Good for metrics with values near its maxima (or minima)
3. [`GammaAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator): the conflation of several Gamma distributions. Good for positively unbounded metrics
4. [`HistogramAggregator`](https://ioverho.github.io/prob_conf_mat/Reference/Experiment%20Aggregation/index.html#prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator): the conflation of the discretized histogram approximation of the individual experiment samples

## Parametric Assumptions

So far, we've been careful not to introduce parametric assumptions into the distributions of experiment metrics. Despite this, each metric distribution has its own [distinct shape, even when using uninformative priors](https://ioverho.github.io/prob_conf_mat/How%20To%20Guides/choosing_a_prior.html#effect-of-uninformative-priors-on-metric-samples).

When using the listed experiment aggregation methods, however, most require the user to make some assumptions about the metric distributions. While the conflation operator makes non-parametric experiment aggregation possible, in theory, this still requires having access to the probability density/mass function. By design, we don't have access to the PDF/PMF, and have to estimate this from the samples. Unfortunately, especially as distributions become more narrow, it can happen that large parts of the support receive 0 density. As a result, the products in the conflation operator become 0 as well, resulting in an indeterminate expression.

If all the individual distributions share a support, we can just compute the product over the region where all distributions have non-zero density. There are situations, however, where we would not expect such a region to exist: specifically whenever there is high inter-experiment heterogeneity.

This can be seen in the following two examples. We have two experiments whose empirical distributions do not overlap in the region between the two experimental distributions, precisely where we would expect the aggregate distribution to lie.

<picture>
  <img alt="Experiment aggregation under heterogeneity with a histogram aggregator" src="../assets/figures/examples/aggregation_with_heterogeneity_histogram.svg" width="80%" style="display: block;margin-left: auto;margin-right: auto; max-width: 500;">
</picture>

Using the histogram aggregator, we get an implausible aggregate distribution, despite it maximizing the areas of high density in the individual experiment distributions. Hence, making a parametric assumption (as we do in [the how to guides](https://ioverho.github.io/prob_conf_mat/How%20To%20Guides/configuration.html)) would be prudent here.
