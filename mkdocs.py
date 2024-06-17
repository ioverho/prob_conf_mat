import logging
import sys
from pathlib import Path

from tabulate import tabulate

from bayes_conf_mat.metrics import METRIC_REGISTRY, AVERAGING_REGISTRY
from bayes_conf_mat.report.template_handler import Template

METRICS_AND_AVERAGING_CHAPTER = "References/Metrics"
METRICS_SECTION = METRICS_AND_AVERAGING_CHAPTER + "/Metrics.md"
AVERAGING_SECTION = METRICS_AND_AVERAGING_CHAPTER + "/Averaging.md"


def metrics_and_averaging_overview():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/_partial/metrics_index.md").resolve())

    # Generate a record for each metric alias
    aliases = sorted(list(METRIC_REGISTRY.items()), key=lambda x: x[0])
    aliases_index = []
    for i, (alias, metric) in enumerate(aliases):
        aliases_index += [
            [
                f"'{alias}'",
                # TODO: check that this works when hosting as well
                f"[`{metric.__name__}`](Metrics.md#{metric.__module__}.{metric.__name__})",
                metric.is_multiclass,
                # metric.bounds,
                metric.sklearn_equivalent,
            ]
        ]

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "metrics_table",
        value=tabulate(
            tabular_data=aliases_index,
            headers=[
                "Alias",
                "Metric",
                "Multiclass",
                # "Bounds",
                "sklearn",
                "Tested",
            ],
            tablefmt="github",
        ),
    )

    # Generate a record for each metric alias
    aliases = sorted(list(AVERAGING_REGISTRY.items()), key=lambda x: x[0])
    aliases_index = []
    for i, (alias, avg_method) in enumerate(aliases):
        aliases_index += [
            [
                f"'{alias}'",
                # TODO: check that this works when hosting as well
                f"[`{avg_method.__name__}`](Averaging.md#{avg_method.__module__}.{avg_method.__name__})",
                avg_method.sklearn_equivalent,
            ]
        ]

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "averaging_table",
        value=tabulate(
            tabular_data=aliases_index,
            headers=[
                "Alias",
                "Metric",
                "sklearn",
            ],
            tablefmt="github",
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{METRICS_AND_AVERAGING_CHAPTER}/index.md").write_text(
        str(template), encoding="utf-8"
    )

    logger.info(
        f"Wrote metrics & averaging index to '{METRICS_AND_AVERAGING_CHAPTER}/index.md'"
    )


def metrics():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/_partial/metrics.md").resolve())

    all_metrics = {str(metric): metric for metric in METRIC_REGISTRY.values()}

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "metrics_list",
        value="\n".join(
            f"::: {metric.__module__}.{metric.__name__}"
            for metric in all_metrics.values()
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{METRICS_SECTION}").write_text(
        str(template), encoding="utf-8"
    )

    logger.info(f"Wrote documentation for '{METRICS_SECTION}'")


def averaging():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/_partial/averaging.md").resolve())

    all_avg_methods = {
        str(avg_method): avg_method for avg_method in AVERAGING_REGISTRY.values()
    }

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "averaging_methods_list",
        value="\n".join(
            f"::: {avg_method.__module__}.{avg_method.__name__}"
            for avg_method in all_avg_methods.values()
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{AVERAGING_SECTION}").write_text(
        str(template), encoding="utf-8"
    )

    logger.info(f"Wrote averaging methods to '{AVERAGING_SECTION}'")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(funcName)s] %(message)s",
    )

    # References/Metrics/Metrics.md
    metrics()

    # References/Metrics/Averaging.md
    averaging()

    # References/Metrics/index.md
    metrics_and_averaging_overview()
