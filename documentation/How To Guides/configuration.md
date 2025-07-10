---
title: "Configure a Study"
---

# Configuration

All interaction with experiments should ideally go through the [`Study`](../Reference/Study.md#prob_conf_mat.study.Study) class. It controls all sampling and experiment aggregation, and can produce reports about samples.

Behind the scenes, the configuration of a study is controlled by its parent class: [`Config`](../Reference/Study.md#prob_conf_mat.config.Config). It has 5 properties which can be set by the users, each of which is validated upon instantiation, and whenever it is set.

## `seed`

Type: `int | None`

This is the random seed used to initialize the RNG. If passing `None`, it defaults to the current time, in fractional seconds. For the purpose of reproducibility, however, it is recommended that you set this value yourself.

It must be a positive integer.

## `num_samples`

Type: `int | None`

This is the number of synthetic confusion matrices to sample for each experiment. A higher value is better, since it reduces the amount of variation between different runs of the same study, but also more computationally expensive. Keep in mind that confusion matrices are square by definition, such that the computational and storage costs scales as,

$$
\mathcal{O}(\mathtt{num\_samples}\cdot \mathtt{num\_classes}^2)
$$

Defaults to 10000, the minimum recommended value,

It must be a *strictly* positive integer, so 0 is not allowed. If you want to use the input confusion matrix, you can 'sample' it using the `Study.get_metric_samples` method, with `sampling_method="input"`.

## `ci_probability`

Type: `float | None`

This is the size of the credibility intervals to compute. Defaults to 0.95, which is an arbitrary value, and should be carefully considered. For more information about the size of the credible interval, see this [bayestestR article](https://easystats.github.io/bayestestR/articles/credible_interval).

It must be a float in the range $(0, 1)$. The extremes are not permitted, as these can lead to infinitesimal or infinite credible intervals, respectively.

## `experiments`

Type: `dict[str, dict[str, dict[str, Any]]]`

This represents the experiments and experiment groups that comprise the study. These must be passed as a double nested `dict` that contains (1) the experiment group name, (2) the experiment name, (3) and finally any IO or prior hyperparameters. Defaults to an empty dict, in which case there are no experiments in the study.

Given that this is a fairly convoluted data structure, the [`Study.add_experiment`](../Reference/Study.md#prob_conf_mat.study.Study.add_experiment) is provided, and should be much easier to use.

An example of a valid `experiments` value is:

```python
{
    "EXPERIMENT_GROUP_NAME": {
        "EXPERIMENT_NAME": {
            # Int[ArrayLike, "num_classes num_classes"]
            "confusion_matrix": [[...], ...],
            "confusion_prior": 0,
            "prevalence_prior": 1,
        }
    }
}
```

The innermost `dict` is additionally checked to make sure it yields valid `Experiment` instances:

- there must be a valid confusion matrix under the `'confusion_matrix'` key
- the values under `'confusion_prior'` and `'prevalence_prior'` must be a valid Dirichlet prior strategy (see the [`dirichlet_prior`](../Reference/Statistics#prob_conf_mat.stats.dirichlet_distribution.dirichlet_prior) method)

## `metrics`

Type: `dict[str, dict[str, Any]]`

Finally, the metrics are represented as a nested `dict` that contains (1) the metric as [metric syntax strings](../Reference/Metrics/index#interface), (2) and any metric aggregation parameters. Defaults to an empty dict, in which case no metrics are added to the study.

Similar to the `experiments` configuration, since this is a convoluted data structure, a more convenient [`Study.add_metric`](../Reference/Study.md#prob_conf_mat.study.Study.add_metric) method is provided.

An example of a valid `metrics` value is:

```python
{
    "METRIC_SYNTAX_STRING": {
        "aggregation": "AGGREGATION_NAME",
    }
}
```

The innermost `dict` contains the necessary parameters to instantiate the [`ExperimentAggregation`](../Reference/Experiment Aggregation/index) method. It can be empty, as long as there is only one experiment per experiment group. Otherwise, it must contain an `'aggregation'` key and value.

## Accessing the configuration

The complete configuration can be parsed and written to a `dict` containing only built-in types. This is useful for serialization using JSON, YAML or TOML, for example. To access this config `dict`, simply use the [`Study.to_dict`](../Reference/Study.md#prob_conf_mat.study.Study.to_dict) method.

See for example, the `dict` produced in the [Interfacing with the Filesystem](../Getting Started/04_interfacing_with_the_filesystem) notebook.

```python
>>> study.to_dict()
{ 'seed': 0,
  'num_samples': 10000,
  'ci_probability': 0.95,
  'experiments': { 'mlp': { 'fold_0': {...},
                            'fold_1': {...},
                            'fold_2': {...},
                            'fold_3': {...},
                            'fold_4': {...}},
                   'svm': { 'fold_0': {...},
                            'fold_1': {...},
                            'fold_2': {...},
                            'fold_3': {...},
                            'fold_4': {...}}},
  'metrics': { 'acc': {'aggregation': 'fe_gaussian'},
               'f1': {'aggregation': 'fe_gaussian'},
               'f1@weighted': {'aggregation': 'fe_gaussian'},
               'f1@macro': {'aggregation': 'fe_gaussian'},
               'mcc': {'aggregation': 'beta'}}}
```

Similarly, the study can be instantiated using the [`Study.from_dict`](../Reference/Study.md#prob_conf_mat.study.Study.from_dict) method. Keep in mind that this is a [classmethod](https://docs.python.org/3.11/library/functions#classmethod).

```python
study = Study.from_dict(config)
```

While this creates a study instance with the same configuration, its output will *generally* not be identical, since this depends on the order of operations. However, the performing the same operations in the same order with identical study instances should yield the same output.
