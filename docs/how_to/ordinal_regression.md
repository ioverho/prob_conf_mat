---
title: 'Evaluate Ordinal Regression'
---

So far, we have looked at metrics for [nominal](https://en.wikipedia.org/wiki/Level_of_measurement#Nominal_level) classifiers. These map some input $x$ into a set of disjoint, independent classes $y\in\mathcal{Y}$. In these experiments, there is no natural ordering between the classes.

In [ordinal](https://en.wikipedia.org/wiki/Level_of_measurement#Ordinal_scale) regression or classification experiments, however, the classes $\mathcal{Y}$ do have a natural ordering, and can be sorted. An example of this is a [Likert scale](https://en.wikipedia.org/wiki/Likert_scale):

```txt
[Strongly Agree, Agree, Neutral, Disagree, Strongly Disagree]
```

These types of experiments fall in between classification and regression. As such, using confusion matrices to analyze such experiments is not common, however, it is not impossible either.

In `prob_conf_mat`, we have implemented [several ordinal regression metrics](../reference/metrics/metrics/#ordinal). Since these impose an ordering on the classes, these metrics are not suitable for the evaluation nominal classifiers. However, all the nominal metrics defined so far can be sensibly used for ordinal regression experiments.

These ordinal metrics use the same syntax as the nominal ones, and can thus be combined with any metric averaging function. In ordinal regression problems, the `min` and `max` averaging operators are especially common. For example, `f1@min` gives $\min\{\mathtt{F1}(C,y) | y\in\mathcal{Y}\}$.
