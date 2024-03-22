import typing
from collections import deque
from graphlib import TopologicalSorter

from bayes_conf_mat.metrics import get_metric
from bayes_conf_mat.metrics.base import Metric, AveragedMetric


def generate_metric_computation_schedule(
    metrics: typing.Iterable[str | typing.Type[Metric] | typing.Type[AveragedMetric]],
) -> typing.Generator[typing.Type[Metric] | typing.Type[AveragedMetric], None, None]:
    """Generates a topological ordering of the inserted metrics and their dependencies.

    Ensures no function is computed before its dependencies are available.

    Args:
        metrics (Iterable[str | Type[Metric] | Type[AggregatedMetric]]): a iterable of metrics

    Returns:
        typing.Generator[Type[Metric] | Type[AggregatedMetric]]
    """  # noqa: E501

    seen_metrics = set()
    stack = deque(metrics)
    topological_sorter = TopologicalSorter()
    while len(stack) > 0:
        metric = stack.popleft()

        if isinstance(metric, str):
            metric = get_metric(metric)

        if metric in seen_metrics:
            continue

        for dependency in metric.dependencies:
            dependency = get_metric(dependency)

            topological_sorter.add(metric, dependency)

            stack.append(dependency)

        seen_metrics.add(metric)

    computation_schedule = topological_sorter.static_order()

    return computation_schedule
