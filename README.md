<div style="text-align: center;" align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="documentation/_static/logo_rectangular.svg">
  <source media="(prefers-color-scheme: light)" srcset="documentation/_static/logo_rectangular.svg">
  <img alt="Logo" src="documentation/_static/logo_rectangular.svg" width="150px">
</picture>
<div style="text-align: center;" align="center">

# Probabilistic Confusion Matrices

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
git clone https://github.com/ioverho/bayes_conf_mat.git
```

And then run the `uv sync --dev` command:

```bash
uv sync --dev
```

The development dependencies should automatically install into the `.venv` folder.

## Documentation

<!-- Link to the documentation here -->
<!-- Include table with some quick start tutorials -->

## Quick Start

```python
import prob_conf_mat as pcm

study = pcm.Study(
    seed=0,
    num_samples=10000,
    ci_probability=0.95,
)
```

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
