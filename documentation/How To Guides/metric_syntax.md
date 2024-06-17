# Metric Syntax

TODO: rewrite this to reflect the updated library

The metric syntax provides a handy interface to creating unique metric functions not hard-coded into the library. Any (binary) metric function can be combined with an aggregation function. Some metric functions and aggregation functions also require arguments. Rather than having the user search for the required metric and aggregation function, one only needs to pass a single string and have the library return the required function.

A valid metric syntax string consists of (in order):

1. The metric identifier
2. Any keyword arguments that need to be passed to the metric function
3. Optionally, an `@` symbol
4. Optionally, the aggregation function identifier
5. Optionally, any keyword arguments that need to be passed to the metric function

No spaces should be used. Instead, keywords arguments start with a `+` prepended to the key, followed by a `=` and the value.

All together:

```text
metric_identifier+arg1=2+arg2=foo@aggregation_identifier+arg1=None+arg2=2.0
```

Only the metric function identifier is necessary, all other aspects are optional.

## Properties

- The metric name should be one of the registered functions
- Only keyword arguments are allowed
- Keywords have keys specified using a `+` their value using a prepended `=`
- Only numeric (float, int) or string arguments are accepted. The string "None" maps to `None`
- Any keyword arguments before the aggregation symbol `@` are passed to the metric function, and any after to the aggregation function
- The order of the keyword arguments does not matter, as long as they appear in the correct block

## Examples

The are meant to illustrate the flexibility of the metric syntax. The defined metrics are not necessarily useful

1. The MCC score

    ```text
    mcc
    ```

2. The F3-score

    ```text
    fbeta+beta=3.0@binary+positive_class=2
    ```

3. Macro-averaged precision

    ```text
    ppv@macro
    ```

4. The geometric mean of the P4 scores

    ```text
    p4@geometric
    ```

5. The DOR for the third class only

    ```text
    dor@binary+positive_class=2
    ```

6. The F2-score for the 1st class only

    ```text
    fbeta+beta=2.0@binary+positive_class=1
    ```
