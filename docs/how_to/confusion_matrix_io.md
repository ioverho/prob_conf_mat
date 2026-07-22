---
title: "Add a Confusion Matrix"
---

???+ abstract "TLDR"
    If you already have a confusion matrix (e.g. from `sklearn.metrics.confusion_matrix`), pass it directly to the `confusion_matrix` argument.

    If you have raw predictions, pass `y_true` and `y_pred` and let `prob_conf_mat` build the matrix for you.

    If your confusion matrix lives in a `.csv` file, load it first with [`load_csv`][prob_conf_mat.io.load_csv].

    In all cases, ground-truth conditions go on the **rows** and model predictions on the **columns**, exactly like scikit-learn.

Every experiment in a [`Study`][prob_conf_mat.study.Study] is defined by a single confusion matrix. Getting the confusion matrix into the study is done through the [`Study.add_experiment`][prob_conf_mat.study.Study.add_experiment] method. In practice you will rarely need to touch the [`prob_conf_mat.io`][prob_conf_mat.io] module directly.

All confusion matrices are validated before they are stored in the study (see [Validation](#validation)).

## From an array

If you already have a confusion matrix as a `list`, `numpy.ndarray`, or any other array-like, you can pass it directly through the `confusion_matrix` argument:

```python
import sklearn.metrics
from prob_conf_mat import Study

confusion_matrix = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

study = Study(seed=0)
study.add_experiment(
    experiment_name="my_model/fold_0",
    confusion_matrix=confusion_matrix,
    prevalence_prior="ones",
    confusion_prior="zeros",
)
```

Nested Python lists work just as well:

```python
study.add_experiment(
    experiment_name="my_model/fold_0",
    confusion_matrix=[[10, 2], [1, 12]],
)
```

## From raw predictions

Most of the time, though, you won't have a confusion matrix yet. Rather, you will have two aligned arrays of ground-truth labels and predicted labels. In that case, pass them as `y_true` and `y_pred`, and `prob_conf_mat` will construct the confusion matrix for you using [`compute_confusion_matrix`][prob_conf_mat.io.compute_confusion_matrix]:

```python
y_pred = classifier.predict(X_test)

study.add_experiment(
    experiment_name="my_model/fold_0",
    y_true=y_test,
    y_pred=y_pred,
)
```

The resulting matrix is identical to `sklearn.metrics.confusion_matrix(y_true, y_pred)`, so the two approaches above are interchangeable.

A few things to keep in mind:

- Labels are expected to be integers in the range $[0, \mathtt{num\_classes})$. Every class should appear at least once in `y_true`, otherwise validation will fail (a ground-truth class with no samples is not well defined).
- You must provide **both** `y_true` and `y_pred`
- If you pass a `confusion_matrix` *and* predictions, the predictions are ignored and a [`ConfigWarning`][prob_conf_mat.config.ConfigWarning] is emitted

## From a CSV file

When your confusion matrices are stored on disk, load them with [`load_csv`][prob_conf_mat.io.load_csv] first:

```python
from pathlib import Path
from prob_conf_mat.io import load_csv

study = Study(seed=0)

for file_path in sorted(Path("./confusion_matrices").glob("*.csv")):
    # e.g. "svm_0.csv" -> model="svm", fold="0"
    model, fold = file_path.stem.split("_")

    study.add_experiment(
        experiment_name=f"{model}/fold_{fold}",
        confusion_matrix=load_csv(location=file_path),
        prevalence_prior="ones",
        confusion_prior="zeros",
    )
```

Each CSV file should contain a single, comma-separated confusion matrix with one row per line, and no header. If your files use a different delimiter, encoding, or line terminator, [`load_csv`][prob_conf_mat.io.load_csv] exposes keyword arguments (`delimiter`, `encoding`, `lineterminator`, ...) to accommodate them.

For a complete, runnable example see the [Interfacing with the Filesystem](../getting_started/04_loading_and_saving_to_disk.html) How-To guide.

## Validation

A valid confusion matrix:

1. is 2-dimensional and square
2. contains at least 2 classes
3. contains only finite, non-negative numbers
4. has at least one sample for every ground-truth class (i.e., no all-zero rows)

Violating any of these raises a [`ConfMatIOError`][prob_conf_mat.io.ConfMatIOError]. Having at least one prediction per class (no all-zero columns) only emits a [`ConfMatIOWarning`][prob_conf_mat.io.ConfMatIOWarning], since a model that never predicts a particular class is unusual but not invalid.

## Converting between representations

Occasionally it is useful to move between a confusion matrix and its underlying `(prediction, condition)` pairs. Two helpers in [`prob_conf_mat.io`][prob_conf_mat.io] handle this round-trip:

- [`confusion_matrix_to_pred_cond`][prob_conf_mat.io.confusion_matrix_to_pred_cond] expands a confusion matrix back into an array of prediction/condition pairs.
- [`pred_cond_to_confusion_matrix`][prob_conf_mat.io.pred_cond_to_confusion_matrix] collapses such an array back into a confusion matrix.

## Next steps

- Once your experiments are loaded, choose appropriate [priors](./priors.md) for the sampling model
- Add the [metrics](./metric_syntax.md) you want to evaluate
