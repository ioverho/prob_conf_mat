# Tests

## Commands to run tests

### Run all tests

```bash
pixi run -e test test
```

### Run individual tests

If using `pixi`:

```bash
pixi run -e test python ${MY_FILE}.py
```

Or using `pytest`:

```bash
pytest -q python ${MY_FILE}.py
```

## Metrics that haven't been tested

1. `FalseDiscoveryRate`:
2. `FalseOmissionRate`:
3. `Informedness`:
4. `Markedness`:
5. `NegativePredictiveValue`:
6. `P4`:
