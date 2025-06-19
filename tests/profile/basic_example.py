from pathlib import Path

from prob_conf_mat import Study
from prob_conf_mat.io import load_csv

if __name__ == "__main__":
    study = Study(
        seed=0,
        num_samples=10000,
        ci_probability=0.95,
    )

    study.add_metric(metric="acc", aggregation="fe_gaussian")
    study.add_metric(metric="f1", aggregation="fe_gaussian")
    study.add_metric(metric="f1@weighted", aggregation="fe_gaussian")
    study.add_metric(metric="f1@macro", aggregation="fe_gaussian")
    study.add_metric(metric="mcc", aggregation="beta")

    # Iterate over all found csv files
    conf_mat_paths = Path(
        "/home/ioverho/prob_conf_mat/documentation/Getting Started/mnist_digits",
    )
    for file_path in sorted(conf_mat_paths.glob("*.csv")):
        # Split the file name to recover the model and fold
        model, fold = file_path.stem.split("_")

        # Load in the confusion matrix using the utility function
        confusion_matrix = load_csv(location=file_path)

        # Add the experiment to the study
        study.add_experiment(
            experiment_name=f"{model}/fold_{fold}",
            confusion_matrix=confusion_matrix,
            prevalence_prior="ones",
            confusion_prior="zeros",
        )

    # Report on the aggregate F1 macro scores
    study.report_aggregated_metric_summaries(metric="f1@macro")

    # Plot the aggregated F1 macro scores for all experiment groups
    study.plot_forest_plot(metric="f1@macro")

    # Compare the aggregated F1 macro scores
    study.plot_pairwise_comparison(
        metric="f1@macro",
        experiment_a="mlp/aggregated",
        experiment_b="svm/aggregated",
        min_sig_diff=0.005,
    )

    # Plot the difference in the aggregated F1 macro scores
    study.plot_pairwise_comparison(
        metric="f1@macro",
        experiment_a="mlp/aggregated",
        experiment_b="svm/aggregated",
        min_sig_diff=0.005,
    )

    # Dump the study config
    study_config = study.to_dict()
