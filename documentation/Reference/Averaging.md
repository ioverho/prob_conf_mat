## Aliases

The following lists all implemented metric averaging methods, by alias. These can be used when composing metrics using [metric syntax strings](http://ioverho.github.io/prob_conf_mat/How%20To%20Guides/metric_syntax.html).

| Alias              | Metric                                                                        | sklearn   |
|--------------------|-------------------------------------------------------------------------------|-----------|
| 'binary'           | [`SelectPositiveClass`](#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'geom'             | [`GeometricMean`](#prob_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'geometric'        | [`GeometricMean`](#prob_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'harm'             | [`HarmonicMean`](#prob_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'harmonic'         | [`HarmonicMean`](#prob_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'macro'            | [`MacroAverage`](#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'macro_average'    | [`MacroAverage`](#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'mean'             | [`MacroAverage`](#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'micro'            | [`WeightedAverage`](#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'micro_average'    | [`WeightedAverage`](#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'select'           | [`SelectPositiveClass`](#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'select_positive'  | [`SelectPositiveClass`](#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'weighted'         | [`WeightedAverage`](#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'weighted_average' | [`WeightedAverage`](#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |

## Abstract Base Class

::: prob_conf_mat.metrics.abc.Averaging
    options:
        heading_level: 3
        members:
            - "dependencies"
            - "sklearn_equivalent"
            - "aliases"
            - "compute_average"

## Metric Instances

::: prob_conf_mat.metrics.averaging.MacroAverage
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
                - dependencies
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics.averaging.WeightedAverage
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
                - dependencies
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics.averaging.SelectPositiveClass
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
                - dependencies
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics.averaging.HarmonicMean
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
                - dependencies
                - sklearn_equivalent
        show_labels: false
        group_by_category: false

::: prob_conf_mat.metrics.averaging.GeometricMean
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
                - dependencies
                - sklearn_equivalent
        show_labels: false
        group_by_category: false
