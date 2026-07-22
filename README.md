<div style="text-align: center;" align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://ioverho.github.io/prob_conf_mat/assets/logo_rectangle_light_text.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://ioverho.github.io/prob_conf_mat/assets/logo_rectangle_dark_text.svg">
  <img alt="Logo" src="https://ioverho.github.io/prob_conf_mat/assets/logo_rectangle_dark_text.svg" width="150px">
</picture>

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

</div>

<div style="text-align: center;" align="center">

<a href="https://docs.basedpyright.com" >
  <img alt="basedpyright - checked" src="https://img.shields.io/endpoint?url=https://docs.basedpyright.com/latest/badge.json">
</a>

<a href="https://github.com/astral-sh/ruff" >
  <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
</a>

<a href="https://github.com/j178/prek" >
  <img alt="Ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/j178/prek/master/docs/assets/badge-v0.json">
</a>

</div>

<div style="text-align: center;" align="center">

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
prob-conf-mat
├── jaxtyping
├── matplotlib
├── numpy
├── scipy
└── tabulate
```

Additionally, [`pandas`](https://pandas.pydata.org/) is an optional dependency for some reporting functions.

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

## Quick Start

```python
from prob_conf_mat import Study

study = Study(seed=0, num_samples=10000, ci_probability=0.95)

study.add_experiment(
  experiment_name="baseline",
  confusion_matrix=[[50, 2, 1], [3, 45, 4], [0, 5, 40]]
)

study.add_metric("f1@macro")

print(study.report_metric_summaries(metric="f1@macro", table_fmt="github"))
```

| Group    | Experiment | Observed | Median | Mode   | 95.0% HDI        | MU     | Skew    | Kurt   |
| -------- | ---------- | -------- | ------ | ------ | ---------------- | ------ | ------- | ------ |
| baseline | baseline   | 0.8992   | 0.9001 | 0.9043 | [0.8507, 0.9448] | 0.0941 | -0.4361 | 0.3485 |


Rather than a single point estimate, you get a distribution of plausible macro F1 score, which we summarize using a variety of statistics.

See the [Getting Started tutorials](https://ioverho.github.io/prob_conf_mat/getting_started/) and the worked-example notebooks in [`docs/getting_started/examples`](docs/getting_started/examples/) for more.

## Documentation

For more information about the package, motivation, how-to guides and implementation, please see the [documentation website](https://ioverho.github.io/prob_conf_mat). We try to use [Daniele Procida's structure for Python documentation](https://docs.divio.com/documentation-system/).

The documentation is broadly divided into 4 sections:

1. **Getting Started**: a collection of small tutorials to help new users get started
2. **How To**: more expansive guides on how to achieve specific things
3. **Reference**: in-depth information about how to interface with the library
4. **Explanation**: explanations about _why_ things are the way they are

|                 | Learning                                                                                          | Coding                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Practical**   | [Getting Started](https://ioverho.github.io/prob_conf_mat/getting_started/)                       | [How-To Guides](https://ioverho.github.io/prob_conf_mat/how_to/configuration/) |
| **Theoretical** | [Explanation](https://ioverho.github.io/prob_conf_mat/explanation/generating_confusion_matrices/) | [Reference](https://ioverho.github.io/prob_conf_mat/reference/Study/)          |

## Development

This project was developed using the following (amazing) tools:

1. Package management: [`uv`](https://docs.astral.sh/uv/)
2. Linting: [`ruff`](https://docs.astral.sh/ruff/)
3. Static Type-Checking: [`basedpyright`](https://docs.basedpyright.com)
4. Documentation: [`zensical`](https://zensical.org/docs/get-started/)
5. Pre-commit: [`prek`](https://prek.j178.dev/)

Most of the common development commands are included in `./Makefile`. If `make` is installed, you can immediately run the following commands:

```txt
Usage:
  make <target>

Utility
  help             Display this help
  hello-world      Tests uv and make
  clean            Clean up caches and build artifacts

Environment
  install          Install default dependencies
  install-dev      Install dev dependencies
  upgrade          Upgrade installed dependencies
  export           Export uv to requirements.txt file

Testing, Linting, Typing & Formatting
  test             Runs all tests
  coverage         Checks test coverage
  lint             Run linting
  type             Run static typechecking
  commit           Run pre-commit checks
  commit-log       Run pre-commit checks in verbose mode and log output to external file

Documentation
  docs-build       Update the docs
  docs-serve       Serve documentation site
```

## Credits

The following are some packages and libraries which served as inspiration for aspects of this project: [arviz](https://python.arviz.org/en/stable/), [bayestestR](https://easystats.github.io/bayestestR/), [BERTopic](https://github.com/MaartenGr/BERTopic), [jaxtyping](https://github.com/patrick-kidger/jaxtyping), [mici](https://github.com/matt-graham/mici), , [python-ci](https://github.com/stinodego/python-ci), [statsmodels](https://www.statsmodels.org/stable/index.html).

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
