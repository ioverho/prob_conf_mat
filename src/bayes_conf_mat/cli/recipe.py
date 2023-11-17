import re
from warnings import warn
import typing

import numpy as np
import jaxtyping as jtyping

from src.metrics import IMPLEMENTED_METRICS, IMPLEMENTED_CLASS_AGGREGATIONS


class MetricRecipe:
    def __init__(
        self, metric_name: str, class_aggregation: typing.Optional[str], **kwargs
    ):
        self.metric_name = metric_name
        self.full_name = IMPLEMENTED_METRICS[self.metric_name]["name"]
        self._is_multiclass = IMPLEMENTED_METRICS[self.metric_name]["multiclass"]
        self._order = IMPLEMENTED_METRICS[self.metric_name]["order"]
        self.metric_function = IMPLEMENTED_METRICS[self.metric_name]["function"]

        self.class_aggregation = class_aggregation
        if self.class_aggregation is not None:
            self.class_aggregation_function = IMPLEMENTED_CLASS_AGGREGATIONS[
                class_aggregation
            ]["function"]

        if self._is_multiclass and self.class_aggregation is not None:
            warn(
                f"An aggregation function has been provided, but metric {self.metric_name} is already multiclass!"  # noqa: E501
            )

        if self._order == "first":
            self._access_name = "_".join(self.metric_full_name.split(" ")).lower()

        self.kwargs = kwargs

    @property
    def multiclass_output(self):
        if self._is_multiclass or self.class_aggregation is not None:
            return False
        else:
            return True

    def __call__(
        self,
        confusion_matrix_samples: jtyping.Float[
            np.ndarray, "num_samples support_size support_size"
        ],
    ):
        if self._order == "first":
            return getattr(confusion_matrix_samples, self._access_name)

        elif not self._is_multiclass:
            raw_metrics = self.metric_function(confusion_matrix_samples, **self.kwargs)

            if self.class_aggregation == "weighted":
                return self.class_aggregation_function(
                    raw_metrics, convex_weights=confusion_matrix_samples.p_condition
                )
            elif self.class_aggregation is not None:
                return self.class_aggregation_function(raw_metrics)

        else:
            raw_metrics = self.metric_function(confusion_matrix_samples, **self.kwargs)

        return raw_metrics

    def __repr__(self):
        metric_str = self.full_name

        if (not self._is_multiclass) and self.class_aggregation is not None:
            metric_str += f"@{self.class_aggregation}"

        if len(self.kwargs) != 0:
            for k, v in self.kwargs.items():
                metric_str += f"+{k}={v}"

        return metric_str


def metric_str_to_recipe(metric_str):
    metric_name = re.search(
        r"^[0-9a-z\_]+",
        metric_str,
    )
    if metric_name is not None:
        metric_name = metric_name.group(0)

        if metric_name not in IMPLEMENTED_METRICS:
            raise ValueError(
                f"Metric {metric_name} not found in {list(IMPLEMENTED_METRICS.keys())}"
            )
    else:
        raise ValueError(f"Could not parse '{metric_str}'. Please check format.")

    class_aggregation = re.findall(
        r"\@([a-z0-9]+)",
        metric_str,
    )
    if len(class_aggregation) == 0:
        class_aggregation = None

    elif len(class_aggregation) == 1:
        class_aggregation = class_aggregation[0]
        if class_aggregation not in IMPLEMENTED_CLASS_AGGREGATIONS:
            raise ValueError(
                f"Class aggregation method {class_aggregation} not found in {list(IMPLEMENTED_CLASS_AGGREGATIONS.keys())}"  # noqa: E501
            )

    else:
        raise ValueError(
            "More than one class aggregation method found. Please provide separate metric srings for each."  # noqa: E501
        )

    expected_kwargs = IMPLEMENTED_METRICS[metric_name]["kwargs"]

    kwargs = dict()
    found_kwargs = re.findall(
        r"\+([^@^+]+)",
        metric_str,
    )
    for kwarg in found_kwargs:
        k, v = kwarg.split("=")

        if k not in expected_kwargs:
            warn(f"'{metric_str}' found unexpected kwarg '{k}'. Skipping.")
        else:
            kwargs.update({k: expected_kwargs[k](v)})

    return MetricRecipe(
        metric_name=metric_name, class_aggregation=class_aggregation, **kwargs
    )
