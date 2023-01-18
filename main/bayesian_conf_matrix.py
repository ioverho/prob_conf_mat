import json
import typing
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np

from io import load_file
from stats import (
    compute_confusion_matrix,
    get_batched_confusion_matrix_stats,
    summarize_posterior_samples,
)
from distributions import init_dirichlet_prior
from format import find_format_backend, flatten_summary, pandas_summary


class HierarchicalBayesConfusionMatrix(object):
    """ """

    def __init__(
        self,
        preds_fp: str,
        labels_fp: str,
        num_classes: int,
        dirichlet_prior: typing.Union[str, float],
        confidence_level: float,
    ):

        self.preds_fp = Path(preds_fp)
        self.labels_fp = Path(labels_fp)
        self.num_classes = num_classes
        if confidence_level > 0 and confidence_level < 1:
            self.alpha = 1 - confidence_level
        else:
            raise ValueError("Confidence level has to be a float in range (0, 1)")

        preds = load_file(self.preds_fp)
        labels = load_file(self.labels_fp)

        self.confusion_matrix = compute_confusion_matrix(
            preds, labels, num_classes=self.num_classes
        )
        self.N = np.sum(self.confusion_matrix)

        self.init_summary()

        self._dirichlet_prior = init_dirichlet_prior(
            dirichlet_prior, self.num_classes**2
        )

    def summarize(self, backend: typing.Optional[str] = None, verbose: bool = False):

        if backend is None:
            backend = find_format_backend(verbose)

        if backend == "pandas":
            records = flatten_summary(self.summary)
            summary_table = pandas_summary(records)

        elif backend == "json":
            summary_table = json.dumps(self.summary, indent=2)

        return summary_table

    def init_summary(self):

        conf_matrix_instance_stats = get_batched_confusion_matrix_stats(
            self.confusion_matrix[np.newaxis, :, :]
        )

        self.summary = OrderedDict()

        self.summary["Overall"] = OrderedDict()
        self.summary["Overall"]["Accuracy"] = OrderedDict(
            [("Instance", f"{conf_matrix_instance_stats['accuracy'][0]:.4f}")]
        )
        self.summary["Overall"]["F1 (Micro)"] = OrderedDict(
            [("Instance", f"{conf_matrix_instance_stats['f1_micro'][0]:.4f}")]
        )
        self.summary["Overall"]["F1 (Macro)"] = OrderedDict(
            [("Instance", f"{conf_matrix_instance_stats['f1_macro'][0]:.4f}")]
        )
        self.summary["Overall"]["MCC"] = OrderedDict(
            [("Instance", f"{conf_matrix_instance_stats['mcc'][0]:.4f}")]
        )

        for l in range(self.num_classes):
            class_str = f"Class {l}"

            self.summary[class_str] = OrderedDict()
            self.summary[class_str]["Precision"] = OrderedDict(
                [("Instance", f"{conf_matrix_instance_stats['precision'][0][l]:.4f}")]
            )
            self.summary[class_str]["Recall"] = OrderedDict(
                [("Instance", f"{conf_matrix_instance_stats['recall'][0][l]:.4f}")]
            )
            self.summary[class_str]["F1"] = OrderedDict(
                [("Instance", f"{conf_matrix_instance_stats['f1'][0][l]:.4f}")]
            )

    def sample_confusion_matrices(self, num_samples: int):

        dirichlet_samples = np.random.dirichlet(
            alpha=self.confusion_matrix.reshape(-1) + self._dirichlet_prior,
            size=num_samples,
        )

        multinomial_samples = np.apply_along_axis(
            partial(np.random.multinomial, self.N), axis=1, arr=dirichlet_samples
        )

        sampled_confusion_matrices = multinomial_samples.reshape(
            (num_samples, self.num_classes, self.num_classes)
        )

        return sampled_confusion_matrices

    def estimate_posterior(self, num_samples):

        sampled_conf_matrices = self.sample_confusion_matrices(num_samples=num_samples)

        sampled_stats = get_batched_confusion_matrix_stats(sampled_conf_matrices)

        self.summary["Overall"]["Accuracy"].update(
            summarize_posterior_samples(sampled_stats["accuracy"], alpha=self.alpha)
        )
        self.summary["Overall"]["F1 (Micro)"].update(
            summarize_posterior_samples(sampled_stats["f1_micro"], alpha=self.alpha)
        )
        self.summary["Overall"]["F1 (Macro)"].update(
            summarize_posterior_samples(sampled_stats["f1_macro"], alpha=self.alpha)
        )
        self.summary["Overall"]["MCC"].update(
            summarize_posterior_samples(sampled_stats["mcc"], alpha=self.alpha)
        )

        for l in range(self.num_classes):
            class_str = f"Class {l}"

            self.summary[class_str]["Precision"].update(
                summarize_posterior_samples(
                    sampled_stats["precision"][:, l], alpha=self.alpha
                )
            )
            self.summary[class_str]["Recall"].update(
                summarize_posterior_samples(
                    sampled_stats["recall"][:, l], alpha=self.alpha
                )
            )
            self.summary[class_str]["F1"].update(
                summarize_posterior_samples(sampled_stats["f1"][:, l], alpha=self.alpha)
            )
