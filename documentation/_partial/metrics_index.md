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

@@metrics_table@@

## Averaging

The following lists all implemented metric averaging methods, by alias

@@averaging_table@@
