import math
import typing

import numpy as np

from bayes_conf_mat.experiment import Experiment
from bayes_conf_mat.report.template_handler import Template

corner = "+"
hor_pipe = "-"
vert_pipe = "|"


def simple_conf_mat(experiment):
    # Figure out how large the cells have to be to contain the class name and the largest count
    max_class_str_width = math.floor(math.log10(experiment.num_classes - 1)) + 1
    max_num_cell_width = len(str(np.max(experiment.confusion_matrix)))
    max_num_cell_width += (max_num_cell_width + 1) % 2

    # Build the header row
    pred_row = f"{'':>{max_class_str_width}}  "
    for i in range(experiment.num_classes):
        pred_row += f"{i:^{max_num_cell_width+2}d} "

    pred_row = pred_row.rstrip()

    # Build the divider string
    divider_str = f"{'':{max_class_str_width}} +"
    for _ in range(experiment.num_classes):
        divider_str += f"{hor_pipe*(max_num_cell_width+2)}{corner}"

    # Build each of the subsequent rows
    row_strs = [pred_row, divider_str]
    for i, row in enumerate(experiment.confusion_matrix):
        row_str = f"{i:>{max_class_str_width}d} {vert_pipe}"
        for val in row:
            val_str = f" {val:>{max_num_cell_width}d} "

            row_str += val_str
            row_str += vert_pipe

        row_strs.append(row_str)
        row_strs.append(divider_str)

    conf_mat_str = "\n".join(row_strs)

    return conf_mat_str


def simple_marginal(experiment, use_predictions: bool):
    # Figure out how large the cells have to be to contain the class name and the largest count
    # Must be at least 5 for the proportion
    max_class_str_width = math.floor(math.log10(experiment.num_classes - 1)) + 1
    max_num_cell_width = max(
        5, len(str(np.max(experiment.confusion_matrix))), max_class_str_width
    )
    max_num_cell_width += (max_num_cell_width + 1) % 2

    # Build the header row
    label_row = f"{'':>1}  "
    for i in range(experiment.num_classes):
        label_row += f"{i:^{max_num_cell_width+2}d} "

    label_row = label_row.rstrip()

    # Build the divider string
    divider_str = f"{'':>1} +"
    for _ in range(experiment.num_classes):
        divider_str += f"{hor_pipe*(max_num_cell_width+2)}{corner}"

    # Build each of the subsequent rows
    row_strs = [label_row, divider_str]
    if not use_predictions:
        prevalence_counts = np.sum(experiment.confusion_matrix, axis=1)
    else:
        prevalence_counts = np.sum(experiment.confusion_matrix.T, axis=1)

    count_str = f"# {vert_pipe}"
    for val in prevalence_counts:
        val_str = f" {val:>{max_num_cell_width}d} "

        count_str += val_str
        count_str += vert_pipe

    row_strs.append(count_str)
    row_strs.append(divider_str)

    prevalence_prop = prevalence_counts / prevalence_counts.sum() * 100
    count_str = f"% {vert_pipe}"
    for val in prevalence_prop:
        val_str = f" {val:>{max_num_cell_width}.2f} "

        count_str += val_str
        count_str += vert_pipe

    row_strs.append(count_str)
    row_strs.append(divider_str)

    prevalence_str = "\n".join(row_strs)

    return prevalence_str


def generate_experiment_report(
    experiment: Experiment, group_name: str, params: typing.Optional[str] = None
):
    # Fill out the experiment report template
    experiment_template = Template("experiment.txt")

    experiment_template.set(key="experiment_group_name", value=group_name)

    experiment_template.set(key="experiment_name", value=experiment.name)

    if params is not None:
        experiment_template.set(key="parameters", value=params)

    experiment_template.set(key="conf_mat", value=simple_conf_mat(experiment))

    experiment_template.set(key="total_predictions", value=experiment.num_predictions)

    experiment_template.set(
        key="cond_prevalence", value=simple_marginal(experiment, False)
    )

    experiment_template.set(
        key="pred_prevalence", value=simple_marginal(experiment, True)
    )

    return str(experiment_template)
