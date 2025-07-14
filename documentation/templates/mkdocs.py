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

logger = logging.getLogger(__name__)

DOCUMENTATION_DIR = Path("documentation")

REFERENCE_PART = DOCUMENTATION_DIR / "Reference"

METRICS_AND_AVERAGING_CHAPTER = REFERENCE_PART / "Metrics"
METRICS_AND_AVERAGING_OVERVIEW = METRICS_AND_AVERAGING_CHAPTER / "index.md"
METRICS_SECTION = METRICS_AND_AVERAGING_CHAPTER / "Metrics.md"
AVERAGING_SECTION = METRICS_AND_AVERAGING_CHAPTER / "Averaging.md"

EXPERIMENT_AGGREGATION_CHAPTER = REFERENCE_PART / "Experiment Aggregation"
EXPERIMENT_AGGREGATION_SECTION = EXPERIMENT_AGGREGATION_CHAPTER / "index.md"
HETEROGENEITY_SECTION = EXPERIMENT_AGGREGATION_CHAPTER / "heterogeneity.md"

REPL_STRING = re.compile(r"@@(.+?)@@")
TEMPLATE_DIR = Path(__file__).parent


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
    # Load in the template
    template_fp = TEMPLATE_DIR / "metrics_index.md"
    if not template_fp.exists():
        raise FileNotFoundError(f"Could not find a template file at '{template_fp}'")

    logger.info(f"Metrics & Averaging - Found template at '{template_fp}'")

    template = Template(
        file_name=template_fp,
    )

    # Complete the template
    # Metrics Table ============================================================
    # Generate a record for each metric alias
    aliases = sorted(METRIC_REGISTRY.items(), key=lambda x: x[0])
    aliases_index = []
    for i, (alias, metric) in enumerate(aliases):
        aliases_index += [
            [
                f"'{alias}'",
                f"[`{metric.__name__}`](Metrics.md#{metric.__module__}.{metric.__name__})",
                metric.is_multiclass,
                # metric.bounds,
                metric.sklearn_equivalent,
            ],
        ]

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

    logger.info(
        "Metrics & Averaging - Filled parameter `metrics_table`",
    )

    # Averaging list ===========================================================
    # Generate a record for each averaging alias
    aliases = sorted(AVERAGING_REGISTRY.items(), key=lambda x: x[0])
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

    logger.info(
        "Metrics & Averaging - Filled parameter `averaging_table`",
    )

    # Write the template to a md file
    output_fp = METRICS_AND_AVERAGING_OVERVIEW.resolve()
    output_fp.write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(
        f"Metrics & Averaging - Wrote filled template to '{output_fp}'",
    )


def metrics() -> None:
    # Load in the template
    template_fp = TEMPLATE_DIR / "metrics.md"
    if not template_fp.exists():
        raise FileNotFoundError(f"Could not find a template file at '{template_fp}'")

    logger.info(f"Metrics - Found template at '{template_fp}'")

    template = Template(
        file_name=template_fp,
    )

    # Complete the template
    # Metrics list =============================================================
    all_metrics = {str(metric): metric for metric in METRIC_REGISTRY.values()}

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

    logger.info(
        "Metrics - Filled parameter `metrics_list`",
    )

    # Write the template to a md file
    output_fp = METRICS_SECTION.resolve()
    output_fp.write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(
        f"Metrics - Wrote filled template to '{output_fp}'",
    )


def averaging() -> None:
    # Load in the template
    template_fp = TEMPLATE_DIR / "averaging.md"
    if not template_fp.exists():
        raise FileNotFoundError(f"Could not find a template file at '{template_fp}'")

    logger.info(f"Averaging - Found template at '{template_fp}'")

    template = Template(
        file_name=template_fp,
    )

    # Complete the template
    # Averaging methods list ===================================================
    all_avg_methods = {
        str(avg_method): avg_method for avg_method in AVERAGING_REGISTRY.values()
    }

    template.set(
        key="averaging_methods_list",
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

    logger.info(
        "Averaging - Filled parameter `averaging_methods_list`",
    )

    # Write the template to a md file
    output_fp = AVERAGING_SECTION.resolve()
    output_fp.write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(
        f"Averaging - Wrote filled template to '{output_fp}'",
    )


def experiment_aggregation() -> None:
    # Load in the template
    template_fp = TEMPLATE_DIR / "experiment_aggregation.md"
    if not template_fp.exists():
        raise FileNotFoundError(f"Could not find a template file at '{template_fp}'")

    logger.info(f"Experiment Aggregation - Found template at '{template_fp}'")

    template = Template(
        file_name=template_fp,
    )

    # Complete the template
    # Aliases table ============================================================
    # Creates a table with some important information as an overview
    methods = []
    for alias, method in AGGREGATION_REGISTRY.items():
        methods.append(
            [
                f"'{alias}'",
                f"[{method.__name__}](#{method.__module__}.{method.__name__})",
            ],
        )

    methods = sorted(methods, key=lambda x: x[0])

    table_str = tabulate(
        tabular_data=methods,
        headers=["Alias", "Method"],
        tablefmt="github",
    )

    template.set(key="experiment_aggregators_table", value=table_str)

    logger.info("Experiment Aggregation - Filled 'experiment_aggregators_table'")

    # Write the template to an md file
    output_fp = EXPERIMENT_AGGREGATION_SECTION.resolve()
    output_fp.write_text(
        str(template),
        encoding="utf-8",
    )

    logger.info(
        f"Experiment Aggregation - Wrote filled template to '{output_fp}'",
    )


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

    # References/Experiment Aggregation/index.md
    experiment_aggregation()
