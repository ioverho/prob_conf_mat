# Compare Many Experiments

In [Getting Started/02 Comparing Experiments](../Getting%20Started/02_comparing_experiments), we discussed comparing two experiments against each other using a variety of methods. This provided lots of information about the probability of existence and significance, as well as visual intuition for how different the metric distributions of two different experiments are.

Repeating this process for *many* experiments is tedious, however: [the number of pairwise combinations scales factorially with the number of experiments](https://en.wikipedia.org/wiki/Combination)!

One setting where we can expect having to compare many experiments at once is in an open-source competition (e.g., [Kaggle](https://www.kaggle.com/)). This is exactly a scenario tested by [Tötsch & Hoffmann (2020)](https://peerj.com/articles/cs-398/)[^1], where they computed the expected winnings for the top-10 participants in the ['Recursion Cellular Image Classification'](https://www.kaggle.com/c/recursion-cellular-image-classification/overview) challenge, based on the probability that their classifier exceeded all other participants.

[^1]: Tötsch, N. & Hoffmann, D. (2020). 'Classifier uncertainty: evidence, potential impact, and probabilistic treatment'

## Competition Winnings

While the test set confusion matrices are hidden, from the accuracy scores and the size of the test set we can recreate a confusion matrix that would have produced those accuracy scores. The table for the top 10 participants might look like this:

| Rank | TeamId  | Score   | TP+TN | FP+FN | TP   | FN  | FP  | TN   |
| ---- | ------- | :-----: | :-----: | :-----: | ---- | --- | --- | ---- |
| 1    | 3467175 | 0.99763 | 15087 | 36    | 7544 | 18  | 18  | 7544 |
| 2    | 3394520 | 0.99672 | 15073 | 50    | 7537 | 25  | 25  | 7537 |
| 3    | 3338942 | 0.99596 | 15062 | 61    | 7531 | 31  | 31  | 7531 |
| 4    | 3339018 | 0.99512 | 15049 | 74    | 7525 | 37  | 37  | 7525 |
| 5    | 3338836 | 0.99498 | 15047 | 76    | 7524 | 38  | 38  | 7524 |
| 6    | 3429037 | 0.99380 | 15029 | 94    | 7515 | 47  | 47  | 7515 |
| 7    | 3346448 | 0.99296 | 15017 | 106   | 7509 | 53  | 53  | 7509 |
| 8    | 3338664 | 0.99296 | 15017 | 106   | 7509 | 53  | 53  | 7509 |
| 9    | 3338358 | 0.99282 | 15014 | 109   | 7507 | 55  | 55  | 7507 |
| 10   | 3339624 | 0.99240 | 15008 | 115   | 7504 | 58  | 58  | 7504 |

Using the [`Study.report_listwise_comparison`](../Reference/Study#prob_conf_mat.study.Study.plot_pairwise_comparison) we can request a table with the probability that each competitor's accuracy score achieved a certain rank:

```python
study.report_listwise_comparison(metric="acc")
```

| Group   | Experiment   |   Rank 1 |   Rank 2 |   Rank 3 |   Rank 4 |   Rank 5 |   Rank 6 |   Rank 7 |   Rank 8 |   Rank 9 |   Rank 10 |
|---------|--------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|
| 1       | 3467175      |   0.9297 |   0.0677 |   0.0025 |          |          |          |          |          |          |           |
| 2       | 3394520      |   0.0678 |   0.7916 |   0.1264 |   0.0090 |   0.0053 |          |          |          |          |           |
| 3       | 3338942      |   0.0024 |   0.1254 |   0.6522 |   0.1282 |   0.0902 |   0.0016 |          |          |          |           |
| 4       | 3339018      |          |   0.0137 |   0.1659 |   0.4415 |   0.3568 |   0.0191 |   0.0012 |   0.0013 |   0.0004 |           |
| 5       | 3338836      |          |   0.0016 |   0.0502 |   0.3609 |   0.4572 |   0.0997 |   0.0120 |   0.0124 |   0.0049 |    0.0010 |
| 6       | 3429037      |          |          |   0.0026 |   0.0518 |   0.0755 |   0.5209 |   0.1282 |   0.1281 |   0.0688 |    0.0241 |
| 8       | 3338664      |          |          |   0.0001 |   0.0070 |   0.0121 |   0.2024 |   0.2622 |   0.2572 |   0.1764 |    0.0825 |
| 7       | 3346448      |          |          |          |   0.0014 |   0.0024 |   0.0964 |   0.2524 |   0.2540 |   0.2381 |    0.1552 |
| 9       | 3338358      |          |          |          |   0.0002 |   0.0006 |   0.0445 |   0.2099 |   0.2116 |   0.2734 |    0.2598 |
| 10      | 3339624      |          |          |          |          |          |   0.0154 |   0.1341 |   0.1353 |   0.2379 |    0.4773 |

While we see no rank inversions, for many of the ranks, there is considerable ambiguity in many ranks. This indicates, that despite a large test set of $N=15.1k$ images, the margins between the top competitors are so narrow that it's difficult to say definitively which team achieved which rank (especially when $r>2$).

The competition organizers offered a $10,000 prize for 1st place, a $2,000 prize for 2nd place and a $1,000 prize for 3rd place. Using these probabilities we can compute that expected prize money each competitor should have received in a fair division of the competition winnings. Using the study, we can compute this by calling `study.report_expected_reward(metric='acc', rewards=[10000, 2000, 1000])`:

| Group | Experiment | E[Reward] |
| ----- | ---------- | --------: |
| 1     | 3467175    | 9435.19   |
| 2     | 3394520    | 2385.68   |
| 3     | 3338942    | 930.13    |
| 4     | 3339018    | 146.50    |
| 5     | 3338836    | 100.89    |
| 6     | 3429037    | 1.57      |
| 8     | 3338664    | 0.03      |
| 7     | 3346448    | 0.01      |
| 9     | 3338358    | 0.00      |
| 10    | 3339624    | 0.00      |


So while ranks 1 & 2 clearly did deserve the lion's share of the competition winnings, rank 4 & 5 comparatively deserved substantially more than the 0 they received.

The following figure (source code can be found in [Explanation/A Replication of Tötsch, N. & Hoffmann, D. (2020). 'Classifier uncertainty: evidence, potential impact, and probabilistic treatment' ](../Explanation/totsch_replication.ipynb)) summarizes the situation:

<picture>
  <img alt="Metric distribution" src="../assets/figures/replication/totsch_fig_6.svg" width="80%" style="display: block;margin-left: auto;margin-right: auto; max-width: 500;">
</picture>
