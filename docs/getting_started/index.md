---
title: Getting Started
---

`prob_conf_mat` is a library for performing statistical inference with classification experiments. Given some confusion matrices, produced by some models on some test data, the library:

1. samples synthetic confusion matrices to produce a distribution over possible confusion matrices, allowing us to **quantify uncertainty**
2. **computes metrics** on the confusion matrix distribution samples to produce a distribution of metric values
3. combines the metric distributions from related experiments into **aggregated distributions**
4. performs comparisons to random models or other trained models to enable **statistical inference**

The goal of these 'Getting Started' tutorials is to enable a new user to apply `prob_conf_mat` succesfully to their own classification experiments, with minimal additional information. The tutorials have been formatted as `.ipynb` notebooks, and can be executed either locally[^1] or using a service like [Google Colab](https://colab.research.google.com/)[^1].

[^1]: in either case, we assume that `prob_conf_mat` has been installed: `pip install prob_conf_mat`.

The tutorials are structured as follows:

<div class="grid cards" markdown>

-   :material-numeric-1-box:{ .lg .middle } __Estimating Uncertainty__

    ---

    This notebook will you take through the steps of defining a [`Study`](../Reference/Study.html), adding a confusion matrix to an experiment, defining some evaluation metrics, and finally producing summary statistics about the experiment's performance

    [:lucide-notebook-text: Notebook](./01_estimating_uncertainty.html)

-   :material-numeric-2-box:{ .lg .middle } __Comparing Experiments__

    ---

    In this tutorial, we walk through comparing two experiments against each other, and produce some basic inferential statistics about which is better

    <br style="margin-top:1.1em;">

    [:lucide-notebook-text: Notebook](./02_comparing_experiments.html)

-   :material-numeric-3-box:{ .lg .middle } __Aggregating Experiments__

    ---

    Here we take a series of experiments produced by [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), and generate a distribution of the average performance using experiment aggregation. We produce some forest plots, and discuss how inter-experiment heterogeneity can affect our analysis

    [:lucide-notebook-text: Notebook](./03_aggregating_experiments.html)

-   :material-numeric-4-box:{ .lg .middle } __Loading and Saving__

    ---

    We go over how to load confusion matrices from your file system, and saving `Study` configurations to enable reproducibility

    <br style="margin-top:4.9em;">

    [:lucide-notebook-text: Notebook](./04_loading_and_saving_to_disk.html)

</div>

Each tutorial notebook assumes some knowledge of the preceding notebooks, so it's best to start at the beginning and work your way through it.
