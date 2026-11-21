These metrics are all experimental, are less well known, and have received much less testing than the other metrics in `prob_conf_mat`.

To enable experimental metrics, import the `enable_experimental` function, and run it:
```python
from prob_conf_mat.metrics.experimental import enable_experimental

enable_experimental()
```

Trying to add an experimental metric to a [Study][Study] before running `enable_experimental()` will result in an error.

## Ordinal

The following metrics were defined for [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_data) experiments, and are not suitable for nominal classification problems more often associated with confusion matrices.

These metrics assume an ordering of the classes in the experiment. An example of this is a [Likert scale](https://en.wikipedia.org/wiki/Likert_scale):

``` txt
[Strongly Agree, Agree, Neutral, Disagree, Strongly Disagree]
```

While nominal metrics can be used for ordinal regression experiments, ordinal metrics should not be used for evaluating nominal classifiers.

Read more in [How To/Ordinal Regression](../../how_to/ordinal_regression.md)

::: prob_conf_mat.metrics.experimental.ordinal_regression.MeanAbsoluteError
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

::: prob_conf_mat.metrics.experimental.ordinal_regression.MeanSquaredError
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

::: prob_conf_mat.metrics.experimental.ordinal_regression.RootMeanSquaredError
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

::: prob_conf_mat.metrics.experimental.ordinal_regression.OffByOneAccuracy
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

::: prob_conf_mat.metrics.experimental.ordinal_regression.PolynomialWeightedKappa
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

## F1 Gain

Introduced [Flach & Kull (2015). Precision-Recall-Gain Curves: PR Analysis Done Right](https://proceedings.neurips.cc/paper_files/paper/2015/hash/33e8075e9970de0cfea955afd4644bb2-Abstract.html), these metrics compare precision, recall, and F1 against an always-positive classifier, which always beats a classifier that predicts the positive class with probability $p$. Besides a lot of other properties, the metrics have the benefit that a score of 0 corresponds to baseline performance, and a score of 1 corresponds to perfect classification. As such, this metric is useful for tasks with high class imbalance, and is comparable across datasets.

It should be noted that this metric is not unbounded from below, meaning it is only useful when all compared classifiers achieve at least baseline performance.

::: prob_conf_mat.metrics.experimental.f1_gain.PrecisionGain
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

::: prob_conf_mat.metrics.experimental.f1_gain.RecallGain
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

::: prob_conf_mat.metrics.experimental.f1_gain.F1Gain
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

::: prob_conf_mat.metrics.experimental.f1_gain.FBetaGain
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
