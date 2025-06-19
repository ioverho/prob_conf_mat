# Extend the Library

While we aim to keep `prob_conf_mat` comprehensive enough to cover most use-cases, it's not possible for the library to be complete. For that reason, we have made it possible to easily extend the library with your own metrics, averaging methods and experiment aggregators.

We do this with a metaclass system that:

1. Enforces which methods and properties all subclasses should possess
2. Automatically registers subclasses when these are defined

In this guide, we outline some basic steps to help you implement these yourself.

## Metrics & Averaging

### Metric

1. First import the base class, `Metric`, as:

    ```python
    from prob_conf_mat.metrics.abc import Metric
    ```

2. Then define your class:

    ```python
    from prob_conf_mat.metrics.abc import Metric

    class FowlkesMallows(Metric):
    ```

3. Define the required class properties:
    1. `full_name (str)`: the full, human-readable name
    2. `is_multiclass (bool)`: whether the metric is defined only for binary classification, in which a result is return per class, or if it is also defined for multi-class classification, in which case only a single value is returned
    3. `bounds (tuple[float, float])`: the minimum and maximum value. Use `float(inf)` to specify infinite values. Used for cross-experiment aggregation and plotting
    4. `dependencies (tuple[str, ...])`: the name of any dependencies your metric might need. Make sure the dependencies have been implemented already (or implement them yourself). Used to build the computation graph. If there are no dependencies, leave an empty tuple
    5. `sklearn_equivalent (str | None)`: the name of the sklearn equivalent function. Used for documentation and unit testing
    6. `aliases (list[str])`: any aliases your metric might go by. Each alias must be unique, and should not be used by another metric

    For example:

    ```python
    from prob_conf_mat.metrics.abc import Metric

    class FowlkesMallows(Metric):
        full_name = "Fowlkes Mallows Index"
        is_multiclass = False
        bounds = (0.0, 1.0)
        dependencies = ("ppv", "tpr")
        sklearn_equivalent = "fowlkes_mallows_index"
        aliases = ["fowlkes_mallows", "fm"]
    ```

4. Finally, implement how the method should be computed using the `compute_metric` method. The output should *always* have a dimensionality of `Float[ndarray, " num_samples num_classes"]` if it is binary, or `Float[ndarray, " num_samples 1"]` if it is multi-class.

    The [Fowlkes-Mallows index](https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index) is defined as the square root of the product of the precision and recall, so we would define it as follows:

    ```python
    from prob_conf_mat.metrics.abc import Metric

    class FowlkesMallows(Metric):
        full_name = "Fowlkes Mallows Index"
        is_multiclass = False
        bounds = (0.0, 1.0)
        dependencies = ("ppv", "tpr")
        sklearn_equivalent = "fowlkes_mallows_index"
        aliases = ["fowlkes_mallows", "fm"]

        def compute_metric(
            self,
            ppv: jtyping.Float[np.ndarray, " num_samples num_classes"],
            tpr: jtyping.Float[np.ndarray, " num_samples num_classes"],
        ) -> jtyping.Float[np.ndarray, " num_samples num_classes"]:
            return np.sqrt(ppv * tpr)
    ```

    Make sure that the arguments in the signature of the `.compute_metric` method matches the dependencies. These are automatically fetched and assigned.

Once defined, the metric can also be automatically found using the metric syntax interface. The following is now completely valid:

```python
study.add_metric("fowlkes_mallows")
```

### Metric Averaging

1. First import the base class, `Averaging`, as:

    ```python
    from prob_conf_mat.metrics.abc import Averaging
    ```

2. Define your class:

    ```python
    from prob_conf_mat.metrics.abc import Averaging

    class Take2ndClass(Averaging):
    ```

3. Define the required class properties:
    1. `full_name (str)`: the full, human-readable name
    2. `dependencies (Tuple[str, ...])`: the name of any (metric) dependencies your averaging method might need. Make sure the dependencies have been implemented already (or implement them yourself). Used to build the computation graph. If there are no dependencies, leave an empty tuple
    3. `sklearn_equivalent (str | None)`: the name of the sklearn equivalent averaging option. Used for documentation and unit testing
    4. `aliases (list[str])`: any aliases your averaging method might go by. Each alias must be unique, and should not conflict with an alias used by another metric or averaging method

    For example:

    ```python
    from prob_conf_mat.metrics.abc import Averaging

    class Take2ndClass(Averaging):
        full_name = "Takes 2nd Class Value"
        dependencies = ()
        sklearn_equivalent = "binary, with positive_class=1"
        aliases = ["2nd_class", "two"]

    ```

4. Finally, implement the `compute_average` method. Note that the input is always an array of `Float[ndarray, " num_samples num_classes"]`, and it should output an array of dimensions `jtyping.Float[np.ndarray, " num_samples 1"]`:

    ```python
    from prob_conf_mat.metrics.abc import Averaging

    class Take2ndClass(Averaging):
        full_name = "Takes 2nd Class Value"
        dependencies = ()
        sklearn_equivalent = "binary, with positive_class=1"
        aliases = ["2nd_class", "two"]

        def compute_average(self, metric_values):
            scalar_array = metric_values[:, 1]

            return scalar_array

    ```

Just like with implementing your own metric, the averaging method can now be automatically found. You can use this averaging method with any pre-defined metric, for example:

```python
study.add_metric("acc@two")
```

Or, if you implemented the Fowlkes-Mallows index as above, the following is also completely valid:

```python
study.add_metric("fmi@two")
```

### Additional Parameters

If you want to add additional parameters, or introduce some notion of state into the metric or averaging method, you need to define an `__init__` method. For example,

```python
class FooBar(Metric):
    full_name = "FooBar Index"
    ...
    aliases = ["foobar"]

    def __init__(self, foo: bool = False, bar: int = 1) -> None:
        super().__init__()

        self.foo = foo
        self.bar = bar
```

This metric can now be called using the following metric syntax string:

```python
study.add_metric("foobar+foo=True+bar=2")
```

Make sure to call `super().__init__()` first though.

## Experiment Aggregation

A similar pattern was used in defining experiment aggregation methods.

1. First, import the base class

    ```python
    from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator
    ```

2. Define your class:

    ```python
    from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator

    class Take1stExperiment(ExperimentAggregator):
    ```

3. Define the required class properties:
    1. `full_name (str)`: the full, human-readable name
    2. `aliases (list[str])`: any aliases your averaging method might go by. Each alias must be unique, and should not conflict with an alias used by another experiment aggregator

    For example:

    ```python
    from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator

    class Take1stExperiment(ExperimentAggregator):
        full_name: str = "Always Takes 1st Experiment Result as Aggregate"
        aliases: list[str] = ("first", "1st")
    ```

4. Finally, implement the `aggregate` method. The first argument, `experiment_samples`, is always an array of `Float[ndarray, " num_samples num_experiments"]`, and it should output an array of dimensions `jtyping.Float[np.ndarray, " num_samples"]`. The signature should also take a `bounds: tuple[float, float]` argument, to allow for resampling. So, for example:

    ```python
    from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator

    class Take1stExperiment(ExperimentAggregator):
        full_name: str = "Always Takes 1st Experiment Result as Aggregate"
        aliases: list[str] = ("first", "1st")

        def aggregate(
            self,
            experiment_samples: jtyping.Float[np.ndarray, " num_samples num_experiments"],
            bounds: tuple[float, float],
        ) -> jtyping.Float[np.ndarray, " num_samples"]:
            return experiment_samples[:, 1]
    ```

Exactly like before, as soon as you define the method, it is possible to use this experiment aggregation method. For example, the following is completely valid code:

```python
study.add_metric("acc", aggregation="first")
```

### Additional Parameters

If you have additional parameters you need to define, or you want the experiment aggregation method to track some form of state, you will need to define an `__init__` method.

Unlike before, the parent class has a defined `__init__` method that you will need to adhere to. Specifically, the first argument should always be the RNG, and this should be passed to `super()`. For example,

```python
from prob_conf_mat.experiment_aggregation.abc import ExperimentAggregator
from prob_conf_mat.utils.rng import RNG

class Take1stExperiment(ExperimentAggregator):
    ...

    def __init__(self, rng: RNG, foo: bool = True, bar: int = 1)  -> None::
        super().__init__(rng=rng)

        self.foo = foo
        self.bar = bar
```

You can now pass these extra parameters as additional keyword arguments in the `study.add_metric` method. For example:

```python
study.add_metric("acc", aggregation="first", foo=False, bar=2)
```

## Notes

Once a class is registered, it cannot be unregistered. If you need to make changes to a custom class, either define it with completely new aliases, or restart your Python environment.
