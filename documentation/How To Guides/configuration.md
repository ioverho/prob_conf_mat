# Configuration

We use [strictyaml](https://hitchdev.com/strictyaml/) to convert a YAML configuration into Python in a type-safe manner.

This document outlines the schema used to validate the YAML configuration file. Each of the headings below specifies a top-level tag in the document. Subheadings denote important child tags requiring additional explanation. Each section comes with its own examples.

When parsing the configuration file, it is tested against the schema described below. If it fails, it should provide helpful fixes.

[//]: # (#TODO: link folder containing example configuration files)

Regardless, this might be overwhelming to many. Folder [examples] contains some example configuration files that should allow you to get started quickly.

## `name`

*Type*: `scalar:str`

The name of the analysis, only used when saving a report to disk.

*Example*:

```yaml
name: "example"
```

## 2. `seed`

[^ Jump to top ^](#configuration)

*Type*: `scalar:int`

The integer value used to initialise the random number generator.

*Example*:

```yaml
seed: 42
```

## 3. `num_samples`

[^ Jump to top ^](#configuration)

*Type*: `scalar:int`, strictly positive

The number of synthetic confusion matrices to sample. This will determine the variance of the posteriors. At least 10 000 is recommended, but more is better. Keep in mind that the memory requirement scales linearly with the number of samples; $\mathcal{O}\left(\mathtt{num\_samples}\cdot\mathtt{num\_classes}^2\right)$.

*Example*:

```yaml
num_samples: 10000
```

## 4. `prevalence_prior`

[^ Jump to top ^](#configuration)

*Type*: `scalar:str` | `scalar:float`, positive | `sequence:float`, positive

The prior used for the prevalence distribution, i.e., the proportional occurrence of a condition. The type must be one of:
    - a `scalar:str`, corresponding to an [implemented prior strategy](../../src/prob_conf_mat/stats/dirichlet_distribution.py)
    - a `scalar:float`, resulting in a constant vector of that value
    - a `sequence:float` (e.g., a YAML list of floats), in which case the full prior should be specified. The list *must* contain the same number of columns as there are classes in the confusion matrix

*Example*:

```yaml
prevalence_prior: "laplace"
```

```yaml
prevalence_prior:
    - 1.0
    - 2.5
    - 1.0
```

## 5. `confusion_prior`

[^ Jump to top ^](#configuration)

*Type*: `scalar:str` | `scalar:float`, positive | `sequence:sequence:float`, positive

The prior used for the confusion distribution, i.e., the proportional occurrence of a prediction given a condition. The type must be one of:
    - a `scalar:str`, corresponding to an [implemented prior strategy](../src/prob_conf_mat/stats/dirichlet_distribution.py)
    - a `scalar:float`, resulting in a constant matrix of that value
    - a `sequence:sequence:float` (e.g. a YAML list of lists), in which case the full prior should be specified. The list *must* contain the same number of columns *and* columns as there are classes in the confusion matrix

*Example*:

```yaml
prevalence_prior: "laplace"
```

```yaml
prevalence_prior:
    -
        - 0.5
        - 0.5
    -
        - 1.0
        - 1.0
```

## 6. `experiments`

[^ Jump to top ^](#configuration)

*Type*: `mapping`

The collection of experiment groups, structured as a mapping. Each key must provide a *unique* experiment group name. The content of the experiment group mapping is described in [`$experiment group$`](#61-experiment-group).

*Example*:

```yaml
experiments:
    Experiment Group 1: ...

    Experiment Group 2: ...
```

### 6.1. `$experiment group$`
*Type*: `mapping:mapping:scalar`

[//]: # (#TODO: link to documentation on IO methods for more information)

The collection of experiments belonging to an experiment group. It should also detail how to fetch each experiment's data, structured as a nested mapping. Each top-level mapping key must provide a *unique* experiment name. The bottom-level mapping is directly passed to [an IO class](). As such, it should contain:

1. `location`, of type `scalar:str`. The file path to the experiment's data.
2. `format`, of type `scalar:str`. The format of the experiment's data. Must be one of the registered formats.
3. Any keyword arguments for the IO method.

*Example*:

```yaml
Experiment Group 1:
    Experiment 1:
        location: "/location/to/some/confusion/matrix.format"
        format: "format"
        format_kwarg: true
```

## 7. `metrics`

[^ Jump to top ^](#configuration)

*Type*: `mapping`

Foo bar...
