# Tests

## Unit Tests

### Commands to run tests

#### Run all tests

##### Metrics

Tests the equivalence of the metric computation methods with their sklearn equivalent

```sh
uv run pytest ./tests/test_metric_equivalence.py
```

###### Metrics that haven't been tested

1. `FalseDiscoveryRate`:
2. `FalseOmissionRate`:
3. `Informedness`:
4. `Markedness`:
5. `NegativePredictiveValue`:
6. `P4`:

##### IO

Tests the various input output methods

```sh
uv run pytest ./tests/test_io.py
```

## Type Safety

### Static Type Checking

```sh
uv run --extra dev pyright > tests/pyright_report
```
