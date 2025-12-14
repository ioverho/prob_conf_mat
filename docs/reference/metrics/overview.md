# Overview

The following lists all implemented metrics, by alias. These can be used when composing metrics using [metric syntax strings](../how_to/metric_syntax/).


| Alias                              | Metric                                                                                                       | Multiclass   | sklearn                 |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------|-------------------------|
| 'acc'                              | [`Accuracy`](./metrics#prob_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'accuracy'                         | [`Accuracy`](./metrics#prob_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'ba'                               | [`BalancedAccuracy`](./metrics#prob_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'balanced_accuracy'                | [`BalancedAccuracy`](./metrics#prob_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'bm'                               | [`Informedness`](./metrics#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'bookmaker_informedness'           | [`Informedness`](./metrics#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'cohen_kappa'                      | [`CohensKappa`](./metrics#prob_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'critical_success_index'           | [`JaccardIndex`](./metrics#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'delta_p'                          | [`Markedness`](./metrics#prob_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'diag_mass'                        | [`DiagMass`](./metrics#prob_conf_mat.metrics._metrics.DiagMass)                                             | False        |                         |
| 'diagnostic_odds_ratio'            | [`DiagnosticOddsRatio`](./metrics#prob_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'dor'                              | [`DiagnosticOddsRatio`](./metrics#prob_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'f1'                               | [`F1`](./metrics#prob_conf_mat.metrics._metrics.F1)                                                         | False        | f1_score                |
| 'fall-out'                         | [`FalsePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fall_out'                         | [`FalsePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'false_discovery_rate'             | [`FalseDiscoveryRate`](./metrics#prob_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'false_negative_rate'              | [`FalseNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'false_omission_rate'              | [`FalseOmissionRate`](./metrics#prob_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'false_positive_rate'              | [`FalsePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fbeta'                            | [`FBeta`](./metrics#prob_conf_mat.metrics._metrics.FBeta)                                                   | False        | fbeta_score             |
| 'fdr'                              | [`FalseDiscoveryRate`](./metrics#prob_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'fnr'                              | [`FalseNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'for'                              | [`FalseOmissionRate`](./metrics#prob_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'fpr'                              | [`FalsePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'hit_rate'                         | [`TruePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'informedness'                     | [`Informedness`](./metrics#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'jaccard'                          | [`JaccardIndex`](./metrics#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'jaccard_index'                    | [`JaccardIndex`](./metrics#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'kappa'                            | [`CohensKappa`](./metrics#prob_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'ldor'                             | [`LogDiagnosticOddsRatio`](./metrics#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'lnlr'                             | [`LogNegativeLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_diagnostic_odds_ratio'        | [`LogDiagnosticOddsRatio`](./metrics#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'log_dor'                          | [`LogDiagnosticOddsRatio`](./metrics#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'log_negative_likelihood_ratio'    | [`LogNegativeLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_nlr'                          | [`LogNegativeLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_plr'                          | [`LogPositiveLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_positive_likelihood_ratio'    | [`LogPositiveLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'lplr'                             | [`LogPositiveLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'markedness'                       | [`Markedness`](./metrics#prob_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'matthews_corrcoef'                | [`MatthewsCorrelationCoefficient`](./metrics#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'matthews_correlation_coefficient' | [`MatthewsCorrelationCoefficient`](./metrics#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'mcc'                              | [`MatthewsCorrelationCoefficient`](./metrics#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'miss_rate'                        | [`FalseNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'model_bias'                       | [`ModelBias`](./metrics#prob_conf_mat.metrics._metrics.ModelBias)                                           | False        |                         |
| 'negative_likelihood_ratio'        | [`NegativeLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'negative_predictive_value'        | [`NegativePredictiveValue`](./metrics#prob_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'nlr'                              | [`NegativeLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'npv'                              | [`NegativePredictiveValue`](./metrics#prob_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'p4'                               | [`P4`](./metrics#prob_conf_mat.metrics._metrics.P4)                                                         | False        |                         |
| 'phi'                              | [`MatthewsCorrelationCoefficient`](./metrics#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'phi_coefficient'                  | [`MatthewsCorrelationCoefficient`](./metrics#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'plr'                              | [`PositiveLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_likelihood_ratio'        | [`PositiveLikelihoodRatio`](./metrics#prob_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_predictive_value'        | [`PositivePredictiveValue`](./metrics#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'ppv'                              | [`PositivePredictiveValue`](./metrics#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'precision'                        | [`PositivePredictiveValue`](./metrics#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'prev_thresh'                      | [`PrevalenceThreshold`](./metrics#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'prevalence'                       | [`Prevalence`](./metrics#prob_conf_mat.metrics._metrics.Prevalence)                                         | False        |                         |
| 'prevalence_threshold'             | [`PrevalenceThreshold`](./metrics#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'pt'                               | [`PrevalenceThreshold`](./metrics#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'recall'                           | [`TruePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'selectivity'                      | [`TrueNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'sensitivity'                      | [`TruePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'specificity'                      | [`TrueNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'threat_score'                     | [`JaccardIndex`](./metrics#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'tnr'                              | [`TrueNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'tpr'                              | [`TruePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'true_negative_rate'               | [`TrueNegativeRate`](./metrics#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'true_positive_rate'               | [`TruePositiveRate`](./metrics#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'youden_j'                         | [`Informedness`](./metrics#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'youdenj'                          | [`Informedness`](./metrics#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
