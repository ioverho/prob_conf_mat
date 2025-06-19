import typing

from prob_conf_mat.metrics import RootMetric, Metric, AveragedMetric

MetricType: typing.TypeVar = typing.TypeVar('MetricType', bound=Metric)
AveragedMetricType: typing.TypeVar = typing.TypeVar('AveragedMetricType', bound=AveragedMetric)
RootMetricType: typing.TypeVar = typing.TypeVar('RootMetricType', bound=RootMetric)

MetricLike: typing.TypeAlias = typing.Union[RootMetric, Metric, AveragedMetric]
