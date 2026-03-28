# Scientific Python - Repo Review

Found on [learn.scientific-python.org](https://learn.scientific-python.org/development/guides/repo-review/?repo=ioverho%2Fprob_conf_mat&ref=HEAD&refType=branch).

## Validate-PyProject

- [ ] **VPP001**: Validate pyproject.toml

  Invalid pyproject.toml! Error: tool.uv.exclude-newer cannot be validated by any definition:

## General

- Detected build backend: hatchling.build
- Detected license(s): MIT License
- Python requires: >=3.11

- [x] **PY001**: Has a pyproject.toml
- [x] **PY002**: Has a README.(md|rst) file
- [x] **PY003**: Has a LICENSE\* file
- [x] **PY004**: Has docs folder
- [x] **PY005**: Has tests folder
- [x] **PY006**: Has pre-commit config
- [ ] **PY007**: Supports an easy task runner (nox, tox, pixi, etc.)

  Projects must have a noxfile.py, tox.ini, or tool.hatch.envs/tool.spin/tool.tox in pyproject.toml to encourage new contributors.

## PyProject

- [x] **PP002**: Has a proper build-system table
- [x] **PP003**: Does not list wheel as a build-dep
- [x] **PP004**: Does not upper cap Python requires
- [x] **PP005**: Using SPDX project.license should not use deprecated trove classifiers [skipped]
- [x] **PP006**: The dev dependency group should be defined
- [ ] **PP301**: Has pytest in pyproject

  Must have a [tool.pytest] (pytest 9+) or [tool.pytest.ini_options] (pytest 6+) configuration section in pyproject.toml. If you must have it in ini format, ignore this check. pytest.toml and .pytest.toml files (pytest 9+) are also supported.

- [ ] **PP302**: Sets a minimum pytest to at least 6 or 9 [skipped]
- [ ] **PP303**: Sets the test paths [skipped]
- [ ] **PP304**: Sets the log level in pytest [skipped]
- [ ] **PP305**: Specifies strict xfail [skipped]
- [ ] **PP306**: Specifies strict config [skipped]
- [ ] **PP307**: Specifies strict markers [skipped]
- [ ] **PP308**: Specifies useful pytest summary [skipped]
- [ ] **PP309**: Filter warnings specified [skipped]

## GitHub Actions

- [x] **GH100**: Has GitHub Actions config
- [x] **GH101**: Has nice names
- [ ] **GH102**: Auto-cancel on repeated PRs

  At least one workflow should auto-cancel.

  ```yaml
  concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: true
  ```

- [x] **GH103**: At least one workflow with manual dispatch trigger
- [x] **GH104**: Use unique names for upload-artifact
- [ ] **GH200**: Maintained by Dependabot

  All projects should have a .github/dependabot.yml file to support at least GitHub Actions regular updates. Something like this:

  ```yaml
  version: 2
  updates:
  # Maintain dependencies for GitHub Actions
      - package-ecosystem: "github-actions"
      directory: "/"
      schedule:
          interval: "weekly"
  ```

- [x] **GH210**: Maintains the GitHub action versions with Dependabot [skipped]
- [x] **GH211**: Do not pin core actions as major versions [skipped]
- [x] **GH212**:: Require GHA update grouping [skipped]

## Pre-commit

- [x] **PC100**: Has pre-commit-hooks
- [x] **PC110**: Uses black or ruff-format
- [x] **PC111**: Uses blacken-docs [skipped]
- [ ] **PC140**: Uses a type checker

  Must have <https://github.com/pre-commit/mirrors-mypy> in .pre-commit-config.yaml

  **Response**: we use [basedpyright](https://docs.basedpyright.com/latest/), which has been added to the pre-commit file

- [ ] **PC160**: Uses a spell checker

  Must have one of <https://github.com/codespell-project/codespell>, <https://github.com/crate-ci/typos> in .pre-commit-config.yaml

  **Response**: common spellcheckers have low signal-to-noise ratio, and should not be run for every commit. Instead, we use this infrequently for code and constantly for documentation

- [ ] **PC170**: Uses PyGrep hooks (only needed if rST present)

  Must have <https://github.com/pre-commit/pygrep-hooks> in .pre-commit-config.yaml

  **Response**: no rST is present, and other hooks are covered by ruff/basedpyright

- [x] **PC180**: Uses a markdown formatter
- [x] **PC190**: Uses Ruff
- [x] **PC191**: Ruff show fixes if fixes enabled
- [x] **PC192**: Ruff uses `ruff-check` instead of `ruff` (legacy)
- [ ] **PC901**: Custom pre-commit CI update message

  Should have something like this in .pre-commit-config.yaml:

  ```yaml
  ci:
    autoupdate_commit_msg: "chore(deps): update pre-commit hooks"
  ```

- [ ] **PC902**: Custom pre-commit CI autofix message

  Should have something like this in .pre-commit-config.yaml:

  ```yaml
  ci:
    autofix_commit_msg: "style: pre-commit fixes"
  ```

- [ ] **PC903**: Specified pre-commit CI schedule

  Should set some schedule: weekly (default), monthly, or quarterly.

  ```yaml
  ci:
    autoupdate_schedule: "monthly"
  ```

## MyPy

Omitted, we use [basedpyright](https://docs.basedpyright.com/latest/)

## Ruff

- [x] **RF001**: Has Ruff config
- [x] **RF002**: Target version must be set
- [x] **RF003**: src directory doesn't need to be specified anymore (0.6+)
- [x] **RF101**: Bugbear must be selected
- [x] **RF102**: isort must be selected
- [x] **RF103**: pyupgrade must be selected
- [x] **RF201**: Avoid using deprecated config settings
- [x] **RF202**: Use (new) lint config section

## ReadTheDocs

## Setuptools Config

## Nox
