---
title: 'Compose a Metric'
---

# Metric Syntax

To add a metric to a study, it's best to use the [`Study.add_metric`](../reference/Study#prob_conf_mat.study.Study.add_metric) method:

```
study.add_metric(
    metric="acc",
    ...
)
```

The specific metric added depends entirely on the sting passed, which should be in metric syntax form. A valid metric syntax string consists of (in order):

1. The metric name. This can be any [alias assigned to the metric](../reference/metrics/overview).
2. Optionally, any keyword arguments that need to be passed to the metric function
3. Optionally, an `@` symbol
4. Optionally, the aggregation function identifier
5. Optionally, any keyword arguments that need to be passed to the averaging function. This can be any [alias assigned to the averaging method](../reference/averaging/overview.md).

No spaces should be used. Instead, keywords arguments start with a `+` prepended to the key, followed by a `=` and the value.

The benefit of this is that any metric-averaging composed function can now be defined succinctly, without the user having to create these metric instances themselves.

## Examples

Some examples might make understanding the metric syntax strings a lot easier.

1. The MCC score

    ```text
    mcc
    ```

2. The F3-score

    ```text
    fbeta+beta=3.0
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

7. The ~~macro-averaged MCC score~~ MCC score

    ```text
    mcc@macro
    ```

    Multi-class metric will just ignore any averaging parameters

## Backus-Naur Form

The following describes the metric syntax string in informal [Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form). Each `<...>` tag implies a non-terminal node that should be replaced by some other value. All values between quotations, `"..."`, are terminal, and are not replaced.

```text
<metric>        ::= <alias><metric-kwargs>*<averaging>?
<alias>         ::= "acc"|"ba"|"f1"|...
<metric-kwargs> ::= "+"<key>"="<value>
<averaging>     ::= "@"<avg-alias><avg-kwargs>*
<avg-alias>     ::= "macro"|"micro"|"geometric"|...
<avg-kwargs>    ::= "+"<key>"="<value>
```

Here `...` is meant to denote the existence of many other literal values. Quantifier `*` means the preceding value occurs 0-n times, whereas `?` means the preceding value occurs 0-1 times (i.e., it's optional).
