import typing
import warnings

from prob_conf_mat.metrics.abc import Averaging, Metric


class _ExperimentalMetricsState:
    """Holds the global opt-in state for experimental metrics.

    Mutating a class attribute avoids having to rebind a module-level name.
    """

    enabled: typing.ClassVar[bool] = False


class ExperimentalWarning(UserWarning):
    """Emitted when experimental features are enabled."""


class ExperimentalMetricsError(ValueError):
    """Raised when an experimental metrics are used without opting in.

    Use the `prob_conf_mat.metrics.experimental.enable_experimental_metrics()` function
    to enable experimental metrics.
    """

    def __init__(
        self,
        msg: str = (
            "This is an experimental metric. "
            "Experimental metrics are disabled by default. "
            "Use the `prob_conf_mat.metrics.experimental.enable_experimental_metrics()` function "
            "to enable experimental metrics."
        ),
        *args,  # pyright: ignore[reportMissingParameterType]
        **kwargs,  # pyright: ignore[reportMissingParameterType]
    ):
        super().__init__(msg, *args, **kwargs)


def enable_experimental_metrics(enable: bool = True) -> None:  # noqa: FBT001, FBT002
    """Globally enable (or disable) experimental metrics and averaging methods.

    Experimental metrics have not received the same level of testing as the
    rest of the library. Use with care.
    """
    _ExperimentalMetricsState.enabled = enable

    if _ExperimentalMetricsState.enabled:
        warnings.warn(
            (
                "Experimental metrics enabled. These have not been tested to the same "
                "standard as the rest of prob_conf_mat."
            ),
            category=ExperimentalWarning,
            stacklevel=2,
        )


def is_experimental_metrics_enabled() -> bool:
    """Checks whether experimental features have been enabled."""
    return _ExperimentalMetricsState.enabled


def mark_experimental(*, since: str, reason: str = ""):
    """Mark a class as experimental."""

    def decorator(cls):  # pyright: ignore[reportMissingParameterType]
        cls._experimental = True
        cls._experimental_since = since
        cls._experimental_reason = reason
        return cls

    return decorator


def is_experimental(metric: Metric | Averaging) -> bool:
    """Checks whether Metric of Averaging Method (or composed Metric) is experimental."""
    return hasattr(metric, "_experimental")
