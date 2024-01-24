import typing
from collections import deque, OrderedDict
from collections.abc import Iterable
from graphlib import TopologicalSorter

from bayes_conf_mat.metrics.interface import get_metric
from bayes_conf_mat.metrics.base import RootMetric, Metric, AggregatedMetric


def generate_metric_computation_schedule(
    metrics: typing.Iterable[str | typing.Type[Metric] | typing.Type[AggregatedMetric]],
) -> typing.Generator[typing.Type[Metric] | typing.Type[AggregatedMetric], None, None]:
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


class MetricCollection:
    def __init__(
        self,
        metrics: typing.Optional[
            typing.Iterable[str | typing.Type[Metric] | typing.Type[AggregatedMetric]]
        ] = (),
    ) -> None:
        self._metrics = OrderedDict()
        self._metrics_by_alias_or_name = dict()

        if metrics is not None:
            self.add(metrics)

    def add(
        self,
        metric: str
        | typing.Type[Metric]
        | typing.Type[AggregatedMetric]
        | typing.Iterable[str | typing.Type[Metric] | typing.Type[AggregatedMetric]],
    ):
        if isinstance(metric, Iterable):
            for m in metric:
                self._add_metric(m)
        elif isinstance(metric, self.__class__):
            for m in metric.get_insert_order():
                self._add_metric(m)
        else:
            self._add_metric(metric)

    def _add_metric(self, metric: str | Metric | AggregatedMetric):
        # If the passed metric is a string, try to parse it with the `get_metric` intertface
        if isinstance(metric, str):
            metric_instance = get_metric(metric)
            self._metrics_by_alias_or_name.update({metric: metric_instance})
            # if metric_instance in self._metrics:
            #    warnings.warn(
            #        f"Metric `{metric}` already added to experiment. Skipping."
            #    )  # noqa: E501
            #    return None

        elif (
            issubclass(metric.__class__, Metric)
            or issubclass(metric.__class__, AggregatedMetric)
            or issubclass(metric.__class__, RootMetric)
        ):
            metric_instance = metric

        else:
            raise TypeError(
                f"Metric must be of type `str`, or a subclass of `Metric` or `AggregatedMetric`, not {metric}: {type(metric)}"  # noqa: E501
            )

        self._metrics.update(((metric_instance, None),))
        self._metrics_by_alias_or_name.update({metric_instance.name: metric_instance})
        self._metrics_by_alias_or_name.update(
            {alias: metric_instance for alias in metric_instance.aliases}
        )

    def __getitem__(self, key: str):
        return self._metrics_by_alias_or_name[key]

    def get_insert_order(self):
        return list(self._metrics.keys())

    def get_compute_order(self):
        topologically_sorted = generate_metric_computation_schedule(
            self.get_insert_order()
        )

        return MetricCollection(topologically_sorted)

    def __iter__(self):
        for metric in tuple(self.get_insert_order()):
            yield metric

    def __len__(self):
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"MetricCollection({list(self._metrics.keys())})"
