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

1. [Estimating Uncertainty](./01_estimating_uncertainty): this notebook will you take through the steps of defining a [`Study`](../Reference/Study), adding a confusion matrix to an experiment, defining some evaluation metrics, and finally producing summary statistics about the experiment's performance
2. [Comparing Experiments](./02_comparing_experiments): in this tutorial, we walk through comparing two experiments against each other, and produce some basic inferential statistics about which is better
3. [Aggregating Experiments](./03_aggregating_experiments.ipynb): here we take a series of experiments produced by [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)), and generate a distribution of the average performance using experiment aggregation. We produce some forest plots, and discuss how inter-experiment heterogeneity can affect our analysis
4. [Interfacing with the Filesystem](./04_loading_and_saving_to_disk): finally, we go over how to load confusion matrices from your filesystem, and saving `Study` configurations to enable reproducibility

Each tutorial notoebook assumes some knowledge of the preceding notebooks, so it's best to start at the beginning and work your way through it.
