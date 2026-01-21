# Overview

The following lists all implemented metric averaging methods, by alias. These can be used when composing metrics using [metric syntax strings](../../how_to/metric_syntax/).

| Alias              | Metric                                                                        | sklearn   |
|--------------------|-------------------------------------------------------------------------------|-----------|
| 'binary'           | [`SelectPositiveClass`](./averaging_methods/#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'geom'             | [`GeometricMean`](./averaging_methods/#prob_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'geometric'        | [`GeometricMean`](./averaging_methods/#prob_conf_mat.metrics.averaging.GeometricMean)             |           |
| 'harm'             | [`HarmonicMean`](./averaging_methods/#prob_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'harmonic'         | [`HarmonicMean`](./averaging_methods/#prob_conf_mat.metrics.averaging.HarmonicMean)               |           |
| 'macro'            | [`MacroAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'macro_average'    | [`MacroAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'mean'             | [`MacroAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.MacroAverage)               | macro     |
| 'micro'            | [`WeightedAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'micro_average'    | [`WeightedAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'select'           | [`SelectPositiveClass`](./averaging_methods/#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'select_positive'  | [`SelectPositiveClass`](./averaging_methods/#prob_conf_mat.metrics.averaging.SelectPositiveClass) | binary    |
| 'weighted'         | [`WeightedAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
| 'weighted_average' | [`WeightedAverage`](./averaging_methods/#prob_conf_mat.metrics.averaging.WeightedAverage)         | weighted  |
