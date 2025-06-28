# Heterogeneity

With experiment aggregation, `prob_conf_mat` tries to estimate the distribution of the average experiment in an ExperimentGroup. Implicitly, this assumes that all experiments come from the same distribution, and that all between experiment variance can be explained by random noise.

This is not always the case. For example, if the experiments represent the same model tested on different benchmarks (or different models tested on the same benchmark). In these cases, inter-experiment heterogeneity can exist.

Heterogeneity can lead to large inter- (or between) experiment variance, which in turn can make estimating an aggregate difficult. The methods in this module try to estimate the degree of heterogeneity present, so users are better informed as to the quality of the experiment aggregation.

See [the guide on experiment aggregation for more details](/ioverho/prob_conf_mat/Explanation/experiment_aggregation.html).

---

::: prob_conf_mat.experiment_aggregation.heterogeneity.HeterogeneityResult
::: prob_conf_mat.experiment_aggregation.heterogeneity.heterogeneity_dl
::: prob_conf_mat.experiment_aggregation.heterogeneity.heterogeneity_pm
::: prob_conf_mat.experiment_aggregation.heterogeneity.estimate_i2
::: prob_conf_mat.experiment_aggregation.heterogeneity.interpret_i2
