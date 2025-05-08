# Bayesian Confusion Matrices

Models a multi-class confusion matrices as a hierarchical Bayesian categorical distribution. Enables uncertainty estimates, experiment aggregation and hypothesis testing.

## Build

### Testing

#### Static Type Checking

```bash
uv run --extra dev pyright > tests/pyright_report
```

### Documentation

Run the `mkdocs.py` script to generate automated documentation.

```bash
python mkdocs.py
```

Make sure to add any new pages to the right spot in `mkdocs.yaml`.

Finally, test the documentation site by running

```bash
mkdocs serve
```
