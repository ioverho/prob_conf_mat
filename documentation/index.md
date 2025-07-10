---
hide:
  #- toc
---

<div style="text-align: center;" align="center">
<div style="margin-bottom: 1rem;">
  <a href="." aria-label="Probabilistic Confusion Matrices" data-md-component="logo" title="Probabilistic Confusion Matrices">
    <img id="logo_light" src="assets/logo_rectangle_dark_text.svg" alt="logo" style="margin-left: auto;margin-right: auto;">
    <img id="logo_dark" src="assets/logo_rectangle_light_text.svg" alt="logo" style="margin-left: auto;margin-right: auto;">
  </a>
</div>

<div style="text-align: center;" align="center">

<a href="https://github.com/ioverho/prob_conf_mat/actions/workflows/test.yaml" >
 <img src="https://github.com/ioverho/prob_conf_mat/actions/workflows/test.yaml/badge.svg"/ alt="Tests status">
</a>

<a href="https://codecov.io/github/ioverho/prob_conf_mat" >
 <img src="https://codecov.io/github/ioverho/prob_conf_mat/graph/badge.svg?token=EU85JBF8M2"/ alt="Codecov report">
</a>

<a href="./LICENSE" >
 <img alt="GitHub License" src="https://img.shields.io/github/license/ioverho/prob_conf_mat">
</a>

<a href="https://pypi.org/project/prob-conf-mat/" >
  <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/prob_conf_mat">
</a>

<h1>Probabilistic Confusion Matrices</h1>

</div>
</div>

**`prob_conf_mat`** is a Python package for performing statistical inference with confusion matrices. It quantifies the amount of uncertainty present, aggregates semantically related experiments into experiment groups, and compares experiments against each other for significance.

## Installation

Installation can be done using from [pypi](https://pypi.org/project/prob-conf-mat/) can be done using `pip`:

```bash
pip install prob_conf_mat
```

Or, if you're using [`uv`](https://docs.astral.sh/uv/), simply run:

```bash
uv add prob_conf_mat
```

The project currently depends on the following packages:

<details>
  <summary>Dependency tree</summary>

```txt
bayes-conf-mat
├── jaxtyping v0.3.2
├── matplotlib v3.10.3
├── numpy v2.3.0
├── scipy v1.15.3
├── seaborn v0.13.2
│   └── pandas v2.3.0
└── tabulate v0.9.0

```

</details>

### Development Environment

This project was developed using [`uv`](https://docs.astral.sh/uv/). To install the development environment, simply clone this github repo:

```bash
git clone https://github.com/ioverho/prob_conf_mat.git
```

And then run the `uv sync --dev` command:

```bash
uv sync --dev
```

The development dependencies should automatically install into the `.venv` folder.

## Documentation

For more information about the package, motivation, how-to guides and implementation, please see the [documentation website](./index.md). We try to use [Daniele Procida's structure for Python documentation](https://docs.divio.com/documentation-system/).

The documentation is broadly divided into 4 sections:

1. **Getting Started**: a collection of small tutorials to help new users get started
2. **How To**: more expansive guides on how to achieve specific things
3. **Reference**: in-depth information about how to interface with the library
4. **Explanation**: explanations about *why* things are the way they are

|                 | Learning                                | Coding                                |
| --------------- | --------------------------------------- | ------------------------------------- |
| **Practical**   | [Getting Started](./Getting%20Started/01_estimating_uncertainty.md) | [How-To Guides](./How%20To%20Guides/configuration.md) |
| **Theoretical** | [Explanation](./Explanation/generating_confusion_matrices.md)           | [Reference](./Reference/Study.md)                             |


## Quick Start

In depth tutorials taking you through all basic steps are available on the [documentation site](./Getting%20Started/01_estimating_uncertainty.md). For the impatient, here's a standard use case.

First define a study, and set some sensible hyperparameters for the simulated confusion matrices.

```python
from prob_conf_mat import Study

study = Study(
    seed=0,
    num_samples=10000,
    ci_probability=0.95,
)
```

Then add a experiment and confusion matrix to the study:

```python
study.add_experiment(
  experiment_name="model_1/fold_0",
  confusion_matrix=[
    [13, 0, 0],
    [0, 10, 6],
    [0,  0, 9],
  ],
  confusion_prior=0,
  prevalence_prior=1,
)
```

Finally, add some metrics to the study:

```python
study.add_metric("acc")
```

We are now ready to start generating summary statistics about this experiment. For example:

```python
study.report_metric_summaries(
  metric="acc",
  table_fmt="github"
)
```

| Group   | Experiment   |   Observed |   Median |   Mode |        95.0% HDI |     MU |    Skew |   Kurt |
|---------|--------------|------------|----------|--------|------------------|--------|---------|--------|
| model_1 | fold_0       |     0.8421 |   0.8499 | 0.8673 | [0.7307, 0.9464] | 0.2157 | -0.5627 | 0.2720 |

So while this experiment achieves an accuracy of 84.21%, a more reasonable estimate (given the size of the test set, and) would be 84.99%. There is a 95% probability that the true accuracy lies between 73.07%-94.64%.

Visually that looks something like:

```python
fig = study.plot_metric_summaries(metric="acc")
```

<picture>
  <img alt="Metric distribution" src="assets/figures/readme/uncertainty_fig.svg" width="80%" style="display: block;margin-left: auto;margin-right: auto; max-width: 500;">
</picture>

Now let's add a confusion matrix for the same model, but estimated using a different fold. We want to know what the average performance is for that model across the different folds:

```python
study.add_experiment(
  experiment_name="model_1/fold_1",
  confusion_matrix=[
      [12, 1, 0],
      [1, 8, 7],
      [0, 2, 7],
  ],
  confusion_prior=0,
  prevalence_prior=1,
)
```

We can equip each metric with an inter-experiment aggregation method, and we can then request summary statistics about the aggregate performance of the experiments using `'model_1'`:

```python
study.add_metric(
    metric="acc",
    aggregation="beta",
)

fig = study.plot_forest_plot(metric="acc")
```

<picture>
  <img alt="Forest plot" src="assets/figures/readme/forest_plot.svg" width="80%" style="display: block;margin-left: auto;margin-right: auto; max-width: 500;">
</picture>

Note that estimated aggregate accuracy has much less uncertainty (a smaller HDI/MU).

These experiments seem pretty different. But is this difference significant? Let's assume that for this example a difference needs to be at least `'0.05'` to be considered significant. In that case, we can quickly request the probability of their difference:

```python
fig = study.plot_pairwise_comparison(
    metric="acc",
    experiment_a="model_1/fold_0",
    experiment_b="model_1/fold_1",
    min_sig_diff=0.05,
)
```

<picture>
  <img alt="Comparison plot" src="assets/figures/readme/comparison_plot.svg" width="80%" style="display: block;margin-left: auto;margin-right: auto; max-width: 500;">
</picture>

There's about an 82% probability that the difference is in fact significant. While likely, there isn't quite enough data to be sure.

## Credits

The following are some packages and libraries which served as inspiration for aspects of this project: [arviz](https://python.arviz.org/en/stable/), [bayestestR](https://easystats.github.io/bayestestR/), [BERTopic](https://github.com/MaartenGr/BERTopic), [jaxtyping](https://github.com/patrick-kidger/jaxtyping), [mici](https://github.com/matt-graham/mici), , [python-ci](https://github.com/stinodego/python-ci), [statsmodels](https://www.statsmodels.org/stable/index.md).

A lot of the approaches and methods used in this project come from published works. Some especially important works include:

1. Goutte, C., & Gaussier, E. (2005). [A probabilistic interpretation of precision, recall and F-score, with implication for evaluation](https://link.springer.com/chapter/10.1007/978-3-540-31865-1_25). In European conference on information retrieval (pp. 345-359). Berlin, Heidelberg: Springer Berlin Heidelberg.
2. Tötsch, N., & Hoffmann, D. (2021). [Classifier uncertainty: evidence, potential impact, and probabilistic treatment](https://peerj.com/articles/cs-398/). PeerJ Computer Science, 7, e398.
3. Kruschke, J. K. (2013). [Bayesian estimation supersedes the t test](https://pubmed.ncbi.nlm.nih.gov/22774788/). Journal of Experimental Psychology: General, 142(2), 573.
4. Makowski, D., Ben-Shachar, M. S., Chen, S. A., & Lüdecke, D. (2019). [Indices of effect existence and significance in the Bayesian framework](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.02767/full). Frontiers in psychology, 10, 2767.
5. Hill, T. (2011). [Conflations of probability distributions](https://www.ams.org/journals/tran/2011-363-06/S0002-9947-2011-05340-7/S0002-9947-2011-05340-7.pdf). Transactions of the American Mathematical Society, 363(6), 3351-3372.
6. Chandler, J., Cumpston, M., Li, T., Page, M. J., & Welch, V. J. H. W. (2019). [Cochrane handbook for systematic reviews of interventions](https://www.cochrane.org/authors/handbooks-and-manuals/handbook). Hoboken: Wiley, 4.

## Citation

```bibtex
@software{ioverho_prob_conf_mat,
    author = {Verhoeven, Ivo},
    license = {MIT},
    title = {{prob\_conf\_mat}},
    url = {https://github.com/ioverho/prob_conf_mat}
}
```
