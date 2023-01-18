import os
import argparse
import math
from pprint import pprint

from main.bayesian_conf_matrix import HierarchicalBayesConfusionMatrix


def estimate(args):

    print("=" * 100)
    print("    Parameters")
    print("=" * 100 + "\n")

    pprint(args)

    print("\n\n" + "=" * 100)
    print("    Estimating")
    print("=" * 100 + "\n")

    bayes_conf_matrix = HierarchicalBayesConfusionMatrix(
        preds_fp=args["preds_fp"],
        labels_fp=args["labels_fp"],
        num_classes=args["num_classes"],
        dirichlet_prior=args["dirichlet_prior"],
        confidence_level=args["confidence_level"],
        verbose=args["verbose"],
    )

    bayes_conf_matrix.estimate_posterior(num_samples=args["num_samples"])
    summary_table = bayes_conf_matrix.summarize()

    print(summary_table)

    print("\n\n" + "=" * 100)
    print("    Saving")
    print("=" * 100 + "\n")

    if args["save_dir"] is not None:
        bayes_conf_matrix.save(args["save_dir"])

    else:
        bayes_conf_matrix.save(bayes_conf_matrix.preds_fp.parent)

    return 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Estimate the metric posterior using a hierarchical Bayesian framework."
    )

    parser.add_argument("--preds_fp", type=str, help="the location of the predictions")
    parser.add_argument("--labels_fp", type=str, help="the location of the labels")
    parser.add_argument("--num_classes", type=int, help="number of classes")
    parser.add_argument(
        "--num_samples",
        default=10000,
        type=int,
        help="number of samples to draw from the posterior. Defaults to 10 000.",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="whether to print to CL what's happening",
    )
    parser.add_argument(
        "--dirichlet_prior",
        default="ones",
        help="Dirichlet prior strategy. If an, fills it for all classes. If a string, it must specify one of the implemented Dirichlet prior strategies. See `main.distributions.py` for more. Defaults to 'ones'.",
    )
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.95,
        help="Confidence level of the equal tailed credible interval. There is a `confidence_level` probability that the true metric lies within the credible interval. Defaults to 0.95.",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        type=str,
        help="the directory to save the estimated posterior to. Defaults to the parent directory of preds_fp.",
    )

    args = vars(parser.parse_args())

    estimate(args)
