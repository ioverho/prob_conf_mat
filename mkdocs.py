import re
import logging
import sys
import typing
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path

from tabulate import tabulate

from prob_conf_mat.utils import fmt
from prob_conf_mat.metrics import METRIC_REGISTRY, AVERAGING_REGISTRY
from prob_conf_mat.experiment_aggregation import AGGREGATION_REGISTRY

REFERENCE_PART = "Reference"
METRICS_AND_AVERAGING_CHAPTER = REFERENCE_PART + "/Metrics"
METRICS_SECTION = METRICS_AND_AVERAGING_CHAPTER + "/Metrics.md"
AVERAGING_SECTION = METRICS_AND_AVERAGING_CHAPTER + "/Averaging.md"
# IO_SECTION = REFERENCE_PART + "/IO.md"
EXPERIMENT_AGGREGATION_SECTION = REFERENCE_PART + "/Experiment Aggregation/index.md"


REPL_STRING = re.compile(r"@@(.+?)@@")
TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class KwargMatch:
    key: str | None = None
    spans: list[tuple[int, int]] = field(default_factory=lambda: [])
    value: typing.Any | None = None

    def __add__(self, other):
        if self.key is None:
            self.key = other.key

        self.spans = self.spans + other.spans

        return self


class Template:
    def __init__(self, file_name: str | Path) -> None:
        self.file_path: Path = TEMPLATE_DIR / file_name
        self.name = self.file_path.stem

        self.template = self.file_path.read_text()

        self.kwargs = OrderedDict()
        for match in REPL_STRING.finditer(self.template):
            self.kwargs[match.group(1)] = self.kwargs.get(
                match.group(1),
                KwargMatch(),
            ) + KwargMatch(key=match.group(1), spans=[match.span()])

    def set(self, key: str, value):
        self.kwargs[key].value = value

    def _format_value(self, value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            return fmt(value)
        if isinstance(value, int):
            return f"{value:d}"
        try:
            return str(value)
        except Exception as e:
            raise TypeError(
                f"Unable to format {value} of type {type(value)}. Raises {e}",
            )

    def __repr__(self) -> str:
        return f"Template({self.name})"

    def __str__(self) -> str:
        filled_template = ""

        unsorted_spans = [
            (span, kwarg_match.key, kwarg_match.value)
            for kwarg_match in self.kwargs.values()
            for span in kwarg_match.spans
        ]
        sorted_spans = sorted(unsorted_spans, key=lambda x: x[0][0])

        left_pointer = 0
        for (begin, end), key, value in sorted_spans:
            filled_template += self.template[left_pointer:begin]

            if value is not None:
                filled_template += self._format_value(value)
            else:
                filled_template += f"@@{key}@@"

            left_pointer = end

        filled_template += self.template[left_pointer:]

        return filled_template


def metrics_and_averaging_overview() -> None:
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/templates/metrics_index.md").resolve())

    # Generate a record for each metric alias
    aliases = sorted(METRIC_REGISTRY.items(), key=lambda x: x[0])
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
            ],
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
            ],
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
        str(template),
        encoding="utf-8",
    )

    logger.info(
        f"Wrote metrics & averaging index to '{METRICS_AND_AVERAGING_CHAPTER}/index.md'",
    )


def metrics():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/templates/metrics.md").resolve())

    all_metrics = {str(metric): metric for metric in METRIC_REGISTRY.values()}

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "metrics_list",
        value="\n\n".join(
            (
                f"::: {metric.__module__}.{metric.__name__}"
                "\n    options:"
                "\n        heading_level: 3"
                "\n        show_root_heading: true"
                "\n        show_root_toc_entry: true"
                "\n        show_category_heading: false"
                "\n        show_symbol_type_toc: true"
                "\n        summary:"
                "\n                attributes: false"
                "\n                functions: false"
                "\n                modules: false"
                "\n        members:"
                "\n                - aliases"
                "\n                - bounds"
                "\n                - dependencies"
                "\n                - is_multiclass"
                "\n                - sklearn_equivalent"
                "\n        show_labels: false"
                "\n        group_by_category: false"
            )
            for metric in all_metrics.values()
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{METRICS_SECTION}").write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(f"Wrote documentation for '{METRICS_SECTION}'")


def averaging():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(Path("./documentation/templates/averaging.md").resolve())

    all_avg_methods = {
        str(avg_method): avg_method for avg_method in AVERAGING_REGISTRY.values()
    }

    # Complete the template
    # Creates a table with some important information as an overview
    template.set(
        "averaging_methods_list",
        value="\n\n".join(
            (
                f"::: {avg_method.__module__}.{avg_method.__name__}"
                "\n    options:"
                "\n        heading_level: 3"
                "\n        show_root_heading: true"
                "\n        show_root_toc_entry: true"
                "\n        show_category_heading: false"
                "\n        show_symbol_type_toc: true"
                "\n        summary:"
                "\n                attributes: false"
                "\n                functions: false"
                "\n                modules: false"
                "\n        members:"
                "\n                - aliases"
                "\n                - dependencies"
                "\n                - sklearn_equivalent"
                "\n        show_labels: false"
                "\n        group_by_category: false"
            )
            for avg_method in all_avg_methods.values()
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{AVERAGING_SECTION}").write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(f"Wrote averaging methods to '{AVERAGING_SECTION}'")


# def io():
#     logger = logging.getLogger(__name__)

#     # Load in the template
#     template = Template(Path("./documentation/templates/io.md").resolve())

#     # Complete the template
#     # Creates a table with some important information as an overview
#     all_io_methods = {str(io_method): io_method for io_method in IO_REGISTRY.values()}

#     template.set(
#         "io_methods_list",
#         value="\n".join(
#             f"::: {io_method.__module__}.{io_method.__name__}"
#             for io_method in all_io_methods.values()
#         ),
#     )

#     # Write the template to a md file
#     Path(f"./documentation/{IO_SECTION}").write_text(str(template), encoding="utf-8")

#     logger.info(f"Wrote IO methods to '{IO_SECTION}'")


def experiment_aggregation():
    logger = logging.getLogger(__name__)

    # Load in the template
    template = Template(
        Path("./documentation/templates/experiment_aggregation.md").resolve(),
    )

    # Complete the template
    # Creates a table with some important information as an overview
    all_agg_methods = {
        str(agg_method): agg_method for agg_method in AGGREGATION_REGISTRY.values()
    }

    template.set(
        "experiment_aggregators_list",
        value="\n".join(
            f"::: {agg_method.__module__}.{agg_method.__name__}"
            for agg_method in all_agg_methods.values()
        ),
    )

    # Write the template to a md file
    Path(f"./documentation/{EXPERIMENT_AGGREGATION_SECTION}").write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(f"Wrote IO methods to '{EXPERIMENT_AGGREGATION_SECTION}")


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

    # TODO: add documentation for IO and utils in general
    # References/IO.md
    # io()

    # References/ExperimentAggregation.md
    experiment_aggregation()
