# Implementing Your Own Metrics, Metric Averaging

## Metrics

1. First import the base class, `Metric`, as:

    ```python
    from bayes_conf_mat.metrics.base import Metric
    ```

2. Define your class:

    ```python
    from bayes_conf_mat.metrics.base import Metric

    class FowlkesMallows(Metric):
    ```

3. Define the required class properties:
    1. `full_name (str)`: the full, human-readable name
    2. `is_multiclass (bool)`: whether the class outputs an array of metrics (where each column corresponds to a class), or a scalar describing all classes in one go
    3. `range (Tuple[float, float])`: the minimum and maximum value. Use `float(inf)` to specify infinite values. Used for plotting
    4. `dependencies (Tuple[str, ...])`: the name of any dependencies your metric might need. Make sure the dependencies have been implemented already (or implement them yourself). Used to build the computation graph. If there are no dependencies, leave an empty tuple
    5. `sklearn_equivalent (str | None)`: the name of the sklearn equivalent function. Used for documentation and unit testing
    6. `aliases (List[str])`: any aliases your metric might go by. Each alias must be unique. The most common or most informative alias should come first.
    For example:

    ```python
    from bayes_conf_mat.metrics.base import Metric

    class FowlkesMallows(Metric):
        full_name = "Fowlkes Mallows Index"
        is_multiclass = False
        range = (0.0, 1.0)
        dependencies = ("ppv", "tpr")
        sklearn_equivalent = "fowlkes_mallows_index"
        aliases = ["fowlkes_mallows", "fm"]
    ```

4. Finally, implement how the method should be computed under the `compute_metric` instance method:

    ```python
    from bayes_conf_mat.metrics.base import Metric

    class FowlkesMallows(Metric):
        full_name = "Fowlkes Mallows Index"
        is_multiclass = False
        range = (0.0, 1.0)
        dependencies = ("ppv", "tpr")
        sklearn_equivalent = "fowlkes_mallows_index"
        aliases = ["fowlkes_mallows", "fm"]

        def compute_metric(self, ppv, tpr):
            return np.sqrt(ppv * tpr)
    ```

5. [OPTIONAL] Add a docstring to explain your class and/or use `jaxtyping` to leave type hints

    ```python
    from bayes_conf_mat.metrics.base import Metric

    class FowlkesMallows(Metric):
        """Computes the Fowlkes-Mallows index (FMI).

        It is a metric that...
        """

        full_name = "Fowlkes Mallows Index"
        is_multiclass = False
        range = (0.0, 1.0)
        dependencies = ("ppv", "tpr")
        sklearn_equivalent = "fowlkes_mallows_index"
        aliases = ["fowlkes_mallows", "fm"]

        def compute_metric(
            self,
            ppv: jtyping.Float[np.ndarray, "num_samples num_classes"],
            tpr: jtyping.Float[np.ndarray, "num_samples num_classes"],
        ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
            return np.sqrt(ppv * tpr)
    ```

Once defined, the metric is automatically registered (through the magic of meta-classes). However, no validation is ran. You, and you alone are responsible for making sure the metric behaves as expected.

Since the metric is automatically registered, it can also be automatically found using the metric syntax interface:

```python
get_metric("fowlkes_mallows")
```

It can also be described using the `describe_metric` function.

## Aggregations

1. First import the base class, `Aggregation`, as:

    ```python
    from bayes_conf_mat.metrics.base import Aggregation
    ```

2. Define your class:

    ```python
    from bayes_conf_mat.metrics.base import Aggregation

    class Take2ndClass(Aggregation):
    ```

3. Define the required class properties:
    1. `full_name (str)`: the full, human-readable name
    2. `dependencies (Tuple[str, ...])`: the name of any (metric) dependencies your aggregation might need. Make sure the dependencies have been implemented already (or implement them yourself). Used to build the computation graph. If there are no dependencies, leave an empty tuple
    3. `sklearn_equivalent (str | None)`: the name of the sklearn equivalent averaging option. Used for documentation and unit testing
    4. `aliases (List[str])`: any aliases your aggregation might go by. Each alias must be unique, and should not conflict with a metric alias. The most common or most informative alias should come first.

    ```python
    from bayes_conf_mat.metrics.base import Aggregation

    class Take2ndClass(Aggregation):
        full_name = "Takes 2nd Class Value"
        dependencies = ()
        sklearn_equivalent = "binary, with positive_class=1"
        aliases = ["2nd_class", "two"]

    ```

4. Finally, implement how the method is computed under the `compute_aggregation` instance method. Note that this should *only* output scalar arrays, i.e. `jtyping.Float[np.ndarray, " num_samples"]`:

    ```python
    from bayes_conf_mat.metrics.base import Aggregation

    class Take2ndClass(Aggregation):
        full_name = "Takes 2nd Class Value"
        dependencies = ()
        sklearn_equivalent = "binary, with positive_class=1"
        aliases = ["2nd_class", "two"]    

        def compute_aggregation(self, metric_values):
            scalar_array = numpy_batched_arithmetic_mean(
                metric_values,
                axis=1,
                keepdims=False,
            )

            return scalar_array

    ```

5. [OPTIONAL] Add a docstring to explain your class and/or use `jaxtyping` to leave type hints
