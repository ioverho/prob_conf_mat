from __future__ import annotations

import typing
from collections import deque, OrderedDict
from graphlib import TopologicalSorter
from functools import cache

if typing.TYPE_CHECKING:
    from prob_conf_mat.utils.typing import MetricLike

from prob_conf_mat.metrics.interface import get_metric
from prob_conf_mat.metrics.abc import RootMetric, Metric, AveragedMetric


@cache
def generate_metric_computation_schedule(
    metrics: typing.Sequence[MetricLike],
) -> typing.Sequence[MetricLike]:
    """Generates a topological ordering of the inserted metrics and their dependencies.

    Ensures no function is computed before its dependencies are available.

    Args:
        metrics (Iterable[str | Type[Metric] | Type[AggregatedMetric]]): a iterable of metrics

    Returns:
        typing.Generator[Type[Metric] | Type[AggregatedMetric]]
    """

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

    return tuple(computation_schedule)


class MetricCollection:
    """Endows a list of metrics with various needed properties, like:
        - metric syntax interfacing
        - redundancy checking
        - topological sorting

    Args:
        metrics (typing.Collection[str | MetricLike | typing.Iterable[str | MetricLike] | typing.Self], optional): the initial collection of metrics. Defaults to ().
    """

    def __init__(
        self,
        metrics: typing.Optional[
            str | MetricLike | typing.Iterable[str | MetricLike] | typing.Self
        ] = (),
    ) -> None:
        self._metrics = OrderedDict()
        self._metrics_by_alias_or_name = dict()

        if metrics is not None:
            self.add(metrics)

    def add(
        self,
        metric: str
        | MetricLike
        | typing.Iterable[str | MetricLike]
        | typing.Self,
    ) -> None:
        """Adds a metric to the metric collection. The 'metric' must be one of:
            - a valid metric syntax string
            - an instance of `Metric` or `AveragedMetric`
            - an iterable of the above two
            - a `MetricCollection`

        Args:
            metric (typing.Collection[str | MetricLike | typing.Iterable[str | MetricLike] | typing.Self], optional): the metric to be added
        """

        # If metric is a str or MetricLike
        if (
            isinstance(metric, str)
            or issubclass(metric.__class__, Metric)
            or issubclass(metric.__class__, AveragedMetric)
            or issubclass(metric.__class__, RootMetric)
        ):
            self._add_metric(metric=metric) # type: ignore

        # If metric is a list of MetricLikes
        elif (
            isinstance(metric, list)
            or isinstance(metric, set)
            or isinstance(metric, tuple)
        ):
            for m in metric:
                self.add(metric=m)

        # If metric is a MetricCollection
        elif isinstance(metric, self.__class__):
            for m in metric.get_insert_order():
                self.add(metric=m)

        # Otherwise
        else:
            raise ValueError(
                f"Cannot process input of type `{type(metric)}` into a MetricCollection."
            )

    def _add_metric(self, metric: str | MetricLike) -> None:
        # Convert str to a MetricLike
        if isinstance(metric, str):
            metric_instance = get_metric(metric)
            self._metrics_by_alias_or_name.update({metric: metric_instance})

        # Simply store a MetricLike
        elif (
            issubclass(metric.__class__, Metric)
            or issubclass(metric.__class__, AveragedMetric)
            or issubclass(metric.__class__, RootMetric)
        ):
            metric_instance = metric

        # Otherwise
        else:
            raise TypeError(
                f"Metric must be of type `str`, or a subclass of `Metric` or `AggregatedMetric`, not {metric}: {type(metric)}"  # noqa: E501
            )

        self._metrics.update(((metric_instance, None),))
        self._metrics_by_alias_or_name.update({metric_instance.name: metric_instance})
        #self._metrics_by_alias_or_name.update(
        #    {alias: metric_instance for alias in metric_instance.aliases}
        #)

    def get_insert_order(self) -> tuple[MetricLike]:
        return tuple(self._metrics.keys())

    def get_compute_order(self) -> "MetricCollection":
        topologically_sorted = generate_metric_computation_schedule(
            self.get_insert_order()
        )

        return MetricCollection(metrics=topologically_sorted)

    def __getitem__(self, key: str | MetricLike):
        return self._metrics_by_alias_or_name[key]

    def __iter__(self) -> typing.Generator[MetricLike]:
        for metric in self.get_insert_order():
            yield metric

    def __len__(self):
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"MetricCollection({list(self._metrics.keys())})"
