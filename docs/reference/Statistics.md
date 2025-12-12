# Statistics

A module dedicated to generating samples and computing summary statistics from samples.

## Batched Averages

Vectorized computation of various averages using a consistent interface.

::: prob_conf_mat.stats.batched_averaging
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

## Dirichlet Distribution

Optimized sampling of batched samples from independent Dirichlet distributions.

These functions tend to be a performance bottleneck, and should be as optimized as much as possible.

::: prob_conf_mat.stats.dirichlet_prior
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

::: prob_conf_mat.stats.dirichlet_distribution
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

## HDI Estimation

Tries to find the Highest Density Interval (HDI) of a posterior distribution from its samples.

::: prob_conf_mat.stats.hdi_estimation
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

## Mode Estimation

Tries to find the mode of a distribution from its samples.

::: prob_conf_mat.stats.mode_estimation
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

## Summary Statistics

Computes various summary statistics about a distribution from its samples.

::: prob_conf_mat.stats.summary
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80

## Truncated Sampling

Draws bounded samples from unbounded Scipy distributions.

This is necessary when making parametric assumptions about the distribution of metrics that have minimum and maximum values.

::: prob_conf_mat.stats.truncated_sampling
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        group_by_category: false
        show_signature: true
        show_signature_annotations: true
        separate_signature: true
        line_length: 80
