## Aliases

The following lists all implemented metrics, by alias. These can be used when composing metrics using [metric syntax strings](http://ioverho.github.io/prob_conf_mat/How%20To%20Guides/metric_syntax.html).

| Alias                              | Metric                                                                                                       | Multiclass   | sklearn                 |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|--------------|-------------------------|
| 'acc'                              | [`Accuracy`](Metrics.md#prob_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'accuracy'                         | [`Accuracy`](Metrics.md#prob_conf_mat.metrics._metrics.Accuracy)                                             | True         | accuracy_score          |
| 'ba'                               | [`BalancedAccuracy`](Metrics.md#prob_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'balanced_accuracy'                | [`BalancedAccuracy`](Metrics.md#prob_conf_mat.metrics._metrics.BalancedAccuracy)                             | True         | balanced_accuracy_score |
| 'bm'                               | [`Informedness`](Metrics.md#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'bookmaker_informedness'           | [`Informedness`](Metrics.md#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'cohen_kappa'                      | [`CohensKappa`](Metrics.md#prob_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'critical_success_index'           | [`JaccardIndex`](Metrics.md#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'delta_p'                          | [`Markedness`](Metrics.md#prob_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'diag_mass'                        | [`DiagMass`](Metrics.md#prob_conf_mat.metrics._metrics.DiagMass)                                             | False        |                         |
| 'diagnostic_odds_ratio'            | [`DiagnosticOddsRatio`](Metrics.md#prob_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'dor'                              | [`DiagnosticOddsRatio`](Metrics.md#prob_conf_mat.metrics._metrics.DiagnosticOddsRatio)                       | False        |                         |
| 'f1'                               | [`F1`](Metrics.md#prob_conf_mat.metrics._metrics.F1)                                                         | False        | f1_score                |
| 'fall-out'                         | [`FalsePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fall_out'                         | [`FalsePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'false_discovery_rate'             | [`FalseDiscoveryRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'false_negative_rate'              | [`FalseNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'false_omission_rate'              | [`FalseOmissionRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'false_positive_rate'              | [`FalsePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'fbeta'                            | [`FBeta`](Metrics.md#prob_conf_mat.metrics._metrics.FBeta)                                                   | False        | fbeta_score             |
| 'fdr'                              | [`FalseDiscoveryRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseDiscoveryRate)                         | False        |                         |
| 'fnr'                              | [`FalseNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'for'                              | [`FalseOmissionRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseOmissionRate)                           | False        |                         |
| 'fpr'                              | [`FalsePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalsePositiveRate)                           | False        |                         |
| 'hit_rate'                         | [`TruePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'informedness'                     | [`Informedness`](Metrics.md#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'jaccard'                          | [`JaccardIndex`](Metrics.md#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'jaccard_index'                    | [`JaccardIndex`](Metrics.md#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'kappa'                            | [`CohensKappa`](Metrics.md#prob_conf_mat.metrics._metrics.CohensKappa)                                       | True         | cohen_kappa_score       |
| 'ldor'                             | [`LogDiagnosticOddsRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'lnlr'                             | [`LogNegativeLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_diagnostic_odds_ratio'        | [`LogDiagnosticOddsRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'log_dor'                          | [`LogDiagnosticOddsRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio)                 | False        |                         |
| 'log_negative_likelihood_ratio'    | [`LogNegativeLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_nlr'                          | [`LogNegativeLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_plr'                          | [`LogPositiveLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'log_positive_likelihood_ratio'    | [`LogPositiveLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'lplr'                             | [`LogPositiveLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio)         | False        | class_likelihood_ratios |
| 'markedness'                       | [`Markedness`](Metrics.md#prob_conf_mat.metrics._metrics.Markedness)                                         | False        |                         |
| 'matthews_corrcoef'                | [`MatthewsCorrelationCoefficient`](Metrics.md#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'matthews_correlation_coefficient' | [`MatthewsCorrelationCoefficient`](Metrics.md#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'mcc'                              | [`MatthewsCorrelationCoefficient`](Metrics.md#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'miss_rate'                        | [`FalseNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.FalseNegativeRate)                           | False        |                         |
| 'model_bias'                       | [`ModelBias`](Metrics.md#prob_conf_mat.metrics._metrics.ModelBias)                                           | False        |                         |
| 'negative_likelihood_ratio'        | [`NegativeLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'negative_predictive_value'        | [`NegativePredictiveValue`](Metrics.md#prob_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'nlr'                              | [`NegativeLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.NegativeLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'npv'                              | [`NegativePredictiveValue`](Metrics.md#prob_conf_mat.metrics._metrics.NegativePredictiveValue)               | False        |                         |
| 'p4'                               | [`P4`](Metrics.md#prob_conf_mat.metrics._metrics.P4)                                                         | False        |                         |
| 'phi'                              | [`MatthewsCorrelationCoefficient`](Metrics.md#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'phi_coefficient'                  | [`MatthewsCorrelationCoefficient`](Metrics.md#prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient) | True         | matthews_corrcoef       |
| 'plr'                              | [`PositiveLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_likelihood_ratio'        | [`PositiveLikelihoodRatio`](Metrics.md#prob_conf_mat.metrics._metrics.PositiveLikelihoodRatio)               | False        | class_likelihood_ratios |
| 'positive_predictive_value'        | [`PositivePredictiveValue`](Metrics.md#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'ppv'                              | [`PositivePredictiveValue`](Metrics.md#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'precision'                        | [`PositivePredictiveValue`](Metrics.md#prob_conf_mat.metrics._metrics.PositivePredictiveValue)               | False        |                         |
| 'prev_thresh'                      | [`PrevalenceThreshold`](Metrics.md#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'prevalence'                       | [`Prevalence`](Metrics.md#prob_conf_mat.metrics._metrics.Prevalence)                                         | False        |                         |
| 'prevalence_threshold'             | [`PrevalenceThreshold`](Metrics.md#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'pt'                               | [`PrevalenceThreshold`](Metrics.md#prob_conf_mat.metrics._metrics.PrevalenceThreshold)                       | False        |                         |
| 'recall'                           | [`TruePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'selectivity'                      | [`TrueNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'sensitivity'                      | [`TruePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'specificity'                      | [`TrueNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'threat_score'                     | [`JaccardIndex`](Metrics.md#prob_conf_mat.metrics._metrics.JaccardIndex)                                     | False        | jaccard_score           |
| 'tnr'                              | [`TrueNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'tpr'                              | [`TruePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'true_negative_rate'               | [`TrueNegativeRate`](Metrics.md#prob_conf_mat.metrics._metrics.TrueNegativeRate)                             | False        |                         |
| 'true_positive_rate'               | [`TruePositiveRate`](Metrics.md#prob_conf_mat.metrics._metrics.TruePositiveRate)                             | False        |                         |
| 'youden_j'                         | [`Informedness`](Metrics.md#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |
| 'youdenj'                          | [`Informedness`](Metrics.md#prob_conf_mat.metrics._metrics.Informedness)                                     | False        |                         |

## Abstract Base Class

::: prob_conf_mat.metrics.abc.Metric
    options:
        heading_level: 3

::: prob_conf_mat.metrics.abc.AveragedMetric
    options:
        heading_level: 3

## Metric Instances

::: prob_conf_mat.metrics._metrics.DiagMass
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.Prevalence
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.ModelBias
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.TruePositiveRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.FalseNegativeRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.PositivePredictiveValue
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.FalseDiscoveryRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.FalsePositiveRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.TrueNegativeRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.FalseOmissionRate
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.NegativePredictiveValue
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.Accuracy
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.BalancedAccuracy
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.MatthewsCorrelationCoefficient
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.CohensKappa
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.F1
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.FBeta
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.Informedness
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.Markedness
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.P4
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.JaccardIndex
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.PositiveLikelihoodRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.LogPositiveLikelihoodRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.NegativeLikelihoodRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.LogNegativeLikelihoodRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.DiagnosticOddsRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.LogDiagnosticOddsRatio
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics._metrics.PrevalenceThreshold
    options:
        heading_level: 3
        show_root_heading: true
        show_root_toc_entry: true
        show_category_heading: false
        show_symbol_type_toc: true
        summary:
                attributes: false
                functions: false
                modules: false
        members:
                - aliases
                - bounds
                - dependencies
                - is_multiclass
                - sklearn_equivalent
        show_labels: false
        group_by_category: false
