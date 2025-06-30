<div style="text-align: center;" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documentation/_static/logo_rectangle_light_text.svg">
  <source media="(prefers-color-scheme: light)" srcset="documentation/_static/logo_rectangle.svg">
  <img alt="Logo" src="documentation/_static/logo_rectangle.svg" width="150px">
</picture>
<div style="text-align: center;" align="center">

<a href="https://github.com/ioverho/prob_conf_mat/actions/workflows/test.yaml" >
 <img src="https://github.com/ioverho/prob_conf_mat/actions/workflows/test.yaml/badge.svg"/ alt="Tests status">
</a>

<a href="https://codecov.io/github/ioverho/prob_conf_mat" >
 <img src="https://codecov.io/github/ioverho/prob_conf_mat/graph/badge.svg?token=EU85JBF8M2"/ alt="Codecov report">
</a>

<a href="./LICENSE" >
 <img src="https://img.shields.io/badge/License-MIT-yellow.svg)"/ alt="License">
</a>

<h1>Probabilistic Confusion Matrices</h1>

</div>
</div>

**`prob_conf_mat`** is a Python package for performing statistical inference with confusion matrices. It quantifies the amount of uncertainty present, aggregates semantically related experiments into experiment groups, and compares experiments against each other for significance.

## Installation

<!-- Add a link to pypi repository -->
Installation can be done using from [pypi]() can be done using `pip`:

```bash
pip install prob_conf_mat
```

The project currently depends on the following packages:

<details>
  <summary>Dependency tree</summary>

```txt
bayes-conf-mat v0.1.0
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

## Quick Start

```python
from prob_conf_mat import Study

study = Study(
    seed=0,
    num_samples=10000,
    ci_probability=0.95,
)
```

<!-- Add experiment -->
<!-- Add metric -->
<!-- Request summary -->

| Group   | Experiment   |   Observed |   Median |   Mode |        95.0% HDI |     MU |    Skew |   Kurt |
|---------|--------------|------------|----------|--------|------------------|--------|---------|--------|
| test    | test         |     1.0000 |   0.8820 | 0.9106 | [0.7428, 0.9710] | 0.2283 | -0.9474 | 1.2644 |

<!-- Plot something -->

## Documentation

<!-- Link to the documentation here -->
<!-- Include table with some quick start tutorials -->
For more information about the package, motivation, how-to guides and implementation, please see the [documentation website](). We try to use [Daniele Procida's structure for Python documentation](https://docs.divio.com/documentation-system/).

The documentation is broadly divided into 4 sections:

1. **Getting Started**: a collection of small tutorials to help new users get started
2. **How To**: more expansive guides on how to achieve specific things
3. **Reference**: in-depth information about how to interface with the library
4. **Explanation**: explanations about *why* things are the way they are

|                 | Learning        | Coding        |
| --------------- | --------------- | ------------- |
| **Practical**   | Getting Started | How-To Guides |
| **Theoretical** | Explanation     | Reference     |

## Development

This project was developed using the following (amazing) tools:

1. Package management: [`uv`](https://docs.astral.sh/uv/)
2. Linting: [`ruff`](https://docs.astral.sh/ruff/)
3. Static Type-Checking: [`pyright`](https://microsoft.github.io/pyright/)
4. Documentation: [`mkdocs`](https://www.mkdocs.org/)
5. CI: [`pre-commit`](https://pre-commit.com/)

Most of the common development commands are included in `./Makefile`. If `make` is installed, you can immediately run the following commands:

```txt
Usage:
  make <target>

Utility
  help             Display this help
  hello-world      Tests uv and make

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

Documentation
  mkdocs           Update the docs
  mkdocs-serve     Serve documentation site
```
