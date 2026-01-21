# Overview

An experiment aggregation method consolidates information from the empirical metric distributions of individual experiments, and creates an aggregate distribution that summarizes the average performance for all experiments in the same experiment group.

The configuration oft the experiment aggregator must be specified along with the metric, preferably using the [`Study.add_metric`](../Study#prob_conf_mat.study.Study.add_metric) method. The `aggregation` key must correspond to one of the aliases listed in the table below.

To add several experiments to the same [ExperimentGroup](../../reference/ExperimentGroup), use the [`Study.add_experiment`](../Study#prob_conf_mat.study.Study.add_experiment) method, and pass the experiment name as `'${GROUP_NAME}/${EXPERIMENT_NAME}'`, where `${GROUP_NAME}` is the name of the ExperimentGroup, and `${EXPERIMENT_NAME}` is the name of the Experiment.

The following aliases are available:

| Alias              | Method                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------|
| 'beta'             | [BetaAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator)             |
| 'beta_conflation'  | [BetaAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.BetaAggregator)             |
| 'fe'               | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fe_gaussian'      | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fe_normal'        | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'fixed_effect'     | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'gamma'            | [GammaAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator)           |
| 'gamma_conflation' | [GammaAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.GammaAggregator)           |
| 'gaussian'         | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'hist'             | [HistogramAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator)   |
| 'histogram'        | [HistogramAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.HistogramAggregator)   |
| 'identity'         | [SingletonAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.SingletonAggregator)   |
| 'normal'           | [FEGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.FEGaussianAggregator) |
| 'random_effect'    | [REGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're'               | [REGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're_gaussian'      | [REGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 're_normal'        | [REGaussianAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.REGaussianAggregator) |
| 'singleton'        | [SingletonAggregator](./experiment_aggregators/#prob_conf_mat.experiment_aggregation.aggregators.SingletonAggregator)   |
