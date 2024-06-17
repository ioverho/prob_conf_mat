# Metrics and Averaging

Metrics, within the scope of this project, summarize a model's performance on some test set. It does so by comparing the model's class predictions against a paired set of condition labels (i.e. the ground truth class). The value a metric function spits out, should tell you something about the model's classification performance, whether it's good, bad or something in between.

Metrics can be either:

1. **multiclass**, in which case they spit out a single value that combines all classes in one go
2. **binary**, in which case they compute a value for each class individually

Usually, the former is a better indication for the overall performance of the model, whereas the latter provides more (usually supporting) fine-grained detail. To convert a binary metric into a multiclass metric, it can be composed with an averaging method. The averaging method takes in the $k$ dimensional array of metric values (where $k$ the number of classes), and yields a scalar value that combines all the per-class values.

## Interface

Usually, you will not be interacting with the metrics themselves. Instead, this library provides users with high-level methods for defining metrics and collections of metrics. The easiest method for constructing metrics is by passing a **metric syntax** string.

A valid metric syntax string consists of (in order):

1. [Required] A registered metric alias ([see below](#metrics))
2. [Required] Any keyword arguments that need to be passed to the metric function
3. [Optional] An `@` symbol
4. [Optional] A registered averaging method alias ([see below](#averaging))
5. [Optional] Any keyword arguments that need to be passed to the averaging function

No spaces should be used. Instead, keywords arguments start with a `+` prepended to the key, followed by a `=` and the value. All together:

```text
<metric-alias>+<arg-key>=<arg-val>@<avg-method-alias>+<arg-key>=<arg-value>
```

Only numeric (float, int) or string arguments are accepted. The strings "None", "True" and "False" are converted to their Pythonic counterpart. The order of the keyword arguments does not matter, as long as they appear in the correct block.

### Examples

1. `f1`: the F1 score
2. `mcc`: the MCC score
3. `ppv`: the Positive Predictive Value
4. `precision`: also Positive Predictive Value, as its a registered alias ([see below](#metrics))
5. `fbeta+beta=3.0`: the F3 score
6. `f1@macro`: the macro averaged F1 score
7. `ba+adjusted=True@binary+positive_class=2`: the chance-correct balanced accuracy score, but only for class 2 (starting at 0)
8. `p4@geometric`: the geometric mean of the P4 scores
9. `mcc@harmonic`: the MCC score, since it's already a multiclass metric, the averaging is ignored

## Metrics

The following lists all implemented metrics, by alias

| Alias                              | Metric                                                                                                        | Multiclass   | sklearn                 |
|------------------------------------|---------------------------------------------------------------------------------------------------------------|--------------|-------------------------|
| 'acc'                              | [`Accuracy`](Metrics.md#bayes_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'accuracy'                         | [`Accuracy`](Metrics.md#bayes_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'ba'                               | [`BalancedAccuracy`](Metrics.md#bayes_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'balanced_accuracy'                | [`BalancedAccuracy`](Metrics.md#bayes_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'bookmaker_informedness'           | [`Informedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'cohen_kappa'                      | [`CohensKappa`](Metrics.md#bayes_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'critical_success_index'           | [`JaccardIndex`](Metrics.md#bayes_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'delta_p'                          | [`Markedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'diag_mass'                        | [`DiagMass`](Metrics.md#bayes_conf_mat.metrics._metrics.DiagMass)                                             | False        |                         |
| 'diagnostic_odds_ratio'            | [`DiagnosticOddsRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'dor'                              | [`DiagnosticOddsRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'f1'                               | [`F1`](Metrics.md#bayes_conf_mat.metrics._metrics.F1)                                                         | False        | f1_score                |
| 'fall-out'                         | [`FalsePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fall_out'                         | [`FalsePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'false_discovery_rate'             | [`FalseDiscoveryRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'false_negative_rate'              | [`FalseNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'false_omission_rate'              | [`FalseOmissionRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'false_positive_rate'              | [`FalsePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fbeta'                            | [`FBeta`](Metrics.md#bayes_conf_mat.metrics._metrics.FBeta)                                                   | False        | fbeta_score             |
| 'fdr'                              | [`FalseDiscoveryRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'fnr'                              | [`FalseNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'for'                              | [`FalseOmissionRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'fpr'                              | [`FalsePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'hit_rate'                         | [`TruePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'informedness'                     | [`Informedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'jaccard'                          | [`JaccardIndex`](Metrics.md#bayes_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'jaccard_index'                    | [`JaccardIndex`](Metrics.md#bayes_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'kappa'                            | [`CohensKappa`](Metrics.md#bayes_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'markedness'                       | [`Markedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'matthews_corrcoef'                | [`MatthewsCorrelationCoefficient`](Metrics.md#bayes_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'matthews_correlation_coefficient' | [`MatthewsCorrelationCoefficient`](Metrics.md#bayes_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'mcc'                              | [`MatthewsCorrelationCoefficient`](Metrics.md#bayes_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'miss_rate'                        | [`FalseNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'model_bias'                       | [`ModelBias`](Metrics.md#bayes_conf_mat.metrics._metrics.ModelBias)                                           | False        |                         |
| 'negative_likelihood_ratio'        | [`NegativeLikelihoodRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'negative_predictive_value'        | [`NegativePredictiveValue`](Metrics.md#bayes_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'nlr'                              | [`NegativeLikelihoodRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'npv'                              | [`NegativePredictiveValue`](Metrics.md#bayes_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'p4'                               | [`P4`](Metrics.md#bayes_conf_mat.metrics._metrics.P4)                                                         | False        |                         |
| 'phi'                              | [`MatthewsCorrelationCoefficient`](Metrics.md#bayes_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'phi_coefficient'                  | [`MatthewsCorrelationCoefficient`](Metrics.md#bayes_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'plr'                              | [`PositiveLikelihoodRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_likelihood_ratio'        | [`PositiveLikelihoodRatio`](Metrics.md#bayes_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_predictive_value'        | [`PositivePredictiveValue`](Metrics.md#bayes_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'ppv'                              | [`PositivePredictiveValue`](Metrics.md#bayes_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'precision'                        | [`PositivePredictiveValue`](Metrics.md#bayes_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'prevalence'                       | [`Prevalence`](Metrics.md#bayes_conf_mat.metrics._metrics.Prevalence)                                         | False        |                         |
| 'recall'                           | [`TruePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'selectivity'                      | [`TrueNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'sensitivity'                      | [`TruePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'specificity'                      | [`TrueNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'threat_score'                     | [`JaccardIndex`](Metrics.md#bayes_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'tnr'                              | [`TrueNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'tpr'                              | [`TruePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'true_negative_rate'               | [`TrueNegativeRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'true_positive_rate'               | [`TruePositiveRate`](Metrics.md#bayes_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'youden_j'                         | [`Informedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'youdenj'                          | [`Informedness`](Metrics.md#bayes_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |

## Averaging

The following lists all implemented metric averaging methods, by alias

| Alias              | Metric                                                                                     | sklearn   |
|--------------------|--------------------------------------------------------------------------------------------|-----------|
| 'binary'           | [`SelectPositiveClass`](Averaging.md#bayes_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'geom'             | [`GeometricMean`](Averaging.md#bayes_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'geometric'        | [`GeometricMean`](Averaging.md#bayes_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'harm'             | [`HarmonicMean`](Averaging.md#bayes_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'harmonic'         | [`HarmonicMean`](Averaging.md#bayes_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'macro'            | [`MacroAverage`](Averaging.md#bayes_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'macro_average'    | [`MacroAverage`](Averaging.md#bayes_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'mean'             | [`MacroAverage`](Averaging.md#bayes_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'select'           | [`SelectPositiveClass`](Averaging.md#bayes_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'select_positive'  | [`SelectPositiveClass`](Averaging.md#bayes_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'weighted'         | [`WeightedAverage`](Averaging.md#bayes_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'weighted_average' | [`WeightedAverage`](Averaging.md#bayes_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
