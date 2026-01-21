[:lucide-ship: Releases](https://github.com/ioverho/prob_conf_mat/releases)

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- additional examples

### Changed

- better io utilities interface

## [v0.3.0](https://github.com/ioverho/prob_conf_mat/releases/tag/v0.2.0) - 2026-01-21

Full Changelog: [`0.2.0...0.3.0`](https://github.com/ioverho/prob_conf_mat/compare/0.2.0...0.3.0)

### Added

- Custom background colours in plots

### Changed

- added default values to the `Study.get_*` methods
- moved docs from [mkdocs](https://www.mkdocs.org/) to [zensical](https://zensical.org)
- consistent colour palette for different plots

### Fixed

- fixed experiment aggregation ignoring all classes beyond 1
- fixed error when generating observed difference in pairwise comparison (dimensionality was non-zero)

## [v0.2.0](https://github.com/ioverho/prob_conf_mat/releases/tag/v0.2.0) - 2025-12-12

Full Changelog: [`0.1.0...0.2.0`](https://github.com/ioverho/prob_conf_mat/compare/0.1.0...0.2.0)

### Changed

- Changed definition of BF-RoPE to an odds ratio instead of a probability ratio. This better matches the definitions provided in Morey & Rouder [1] and Makowski et al. [2].

### References

[1] Richard D. Morey and Jeffrey N. Rouder. “Bayes Factor Approaches for Testing Interval Null Hypotheses.” In: Psychological Methods 16.4 (Dec. 2011), pp. 406–419. ISSN: 1939-1463, 1082-989X. DOI: 10.1037/a0024377. URL: https://doi.apa.org/doi/10.1037/a0024377.

[2] Dominique Makowski et al. “Indices of Effect Existence and Significance in the Bayesian Framework”. In: Frontiers in Psychology 10 (Dec. 10, 2019). ISSN: 1664-1078. DOI: 10.3389/fpsyg.2019.02767. URL: https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.02767/full.

## [v0.1.0](https://github.com/ioverho/prob_conf_mat/releases/tag/v0.1.0) - 2025-08-06

This release should serve as the first feature-complete version of this library. Before release 1.0.0, I'd like to focus on tweaks and additional documentation, not new (major) new features.

### Added

- additional documentation
- allow float confusion matrices

### Removed

- type checking blocks removed from codecov
