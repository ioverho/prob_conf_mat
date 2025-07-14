# Experiment Aggregators

An experiment aggregation method consolidates information from the empirical metric distributions of individual experiments, and creates an aggregate distribution that summarizes the average performance for all experiments in the same experiment group.

The configuration oft the experiment aggregator must be specified along with the metric, preferably using the [`Study.add_metric`](https://ioverho.github.io/prob_conf_mat/Reference/Study.html#prob_conf_mat.study.Study.add_metric) method. The `aggregation` key must correspond to one of the aliases listed in the table below.

To add several experiments to the same [ExperimentGroup](//ioverho.github.io/prob_conf_mat/Reference/ExperimentGroup.html), use the [`Study.add_experiment`](https://ioverho.github.io/prob_conf_mat/Reference/Study.html#prob_conf_mat.study.Study.add_experiment) method, and pass the experiment name as `'${GROUP_NAME}/${EXPERIMENT_NAME}'`, where `${GROUP_NAME}` is the name of the ExperimentGroup, and `${EXPERIMENT_NAME}` is the name of the Experiment.

## Aliases

The following aliases are available:

| Alias              | Method                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------|
| 'beta'             | [BetaAggregator](#prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator)             |
| 'beta_conflation'  | [BetaAggregator](#prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator)             |
| 'fe'               | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fe_gaussian'      | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fe_normal'        | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fixed_effect'     | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'gamma'            | [GammaAggregator](#prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator)           |
| 'gamma_conflation' | [GammaAggregator](#prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator)           |
| 'gaussian'         | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'hist'             | [HistogramAggregator](#prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator)   |
| 'histogram'        | [HistogramAggregator](#prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator)   |
| 'identity'         | [SingletonAggregator](#prob_conf_mat.experiment_aggregation.aggregators.SingletonAggregator)   |
| 'normal'           | [FEGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'random_effect'    | [REGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're'               | [REGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're_gaussian'      | [REGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're_normal'        | [REGaussianAggregator](#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 'singleton'        | [SingletonAggregator](#prob_conf_mat.experiment_aggregation.aggregators.SingletonAggregator)   |

## Abstract Base Classes

::: prob_conf_mat.experiment_aggregation.abc.ExperimentAggregator
    options:
        heading_level: 3

## Methods

::: prob_conf_mat.experiment_aggregation.aggregators.SingletonAggregator
    options:
        heading_level: 3

::: prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator
    options:
        heading_level: 3

::: prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator
    options:
        heading_level: 3

::: prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator
    options:
        heading_level: 3

::: prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator
    options:
        heading_level: 3

::: prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator
    options:
        heading_level: 3
