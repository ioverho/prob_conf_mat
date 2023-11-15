import typing

IMPLEMENTED_SIMPLE_METRICS: typing.Dict[str, typing.Callable] = dict()
IMPLEMENTED_COMPLEX_METRICS: typing.Dict[str, typing.Callable] = dict()


def register_simple_metric(
    identifier: str,
    full_name: str,
    range: typing.Tuple[float, float],
):
    def wrapper(func):
        # Add the function's attributes
        func.identifier = identifier
        func.full_name = full_name
        func.range = range

        # Register the function
        if identifier in IMPLEMENTED_SIMPLE_METRICS:
            raise ValueError(
                f"'{identifier}' already exists as {IMPLEMENTED_SIMPLE_METRICS[identifier]}. Please use a unique string for the indentifier."  # noqa: E501
            )
        else:
            IMPLEMENTED_SIMPLE_METRICS[identifier] = func

        return func

    return wrapper


def register_complex_metric(
    identifier: str,
    full_name: str,
    is_multiclass: bool,
    range: typing.Tuple[float, float],
    required_simple_metrics: typing.Tuple[str],
):
    """Function wrapper for registering a metric function and assigning some attributes to it.

    Keeps the interface functional, while providing some OOP aspects.

    Args:
        full_name (str): full name of metric, i.e. name of the Wikipedia entry
        identifier (str): short name of metric, i.e. name typed in CLI
        is_multiclass (bool): is metric per-class, or multiclass
        range (typing.Tuple[float, float]): extrema of possible values
    """  # noqa: E501

    def wrapper(func):
        # Add the function's attributes
        func.identifier = identifier
        func.full_name = full_name
        func.is_multiclass = is_multiclass
        func.range = range
        func.required_simple_metrics = required_simple_metrics

        # Check that the identifier is unique
        if identifier in IMPLEMENTED_SIMPLE_METRICS:
            raise ValueError(
                f"'{identifier}' is already a simple metric identifier for: ({IMPLEMENTED_SIMPLE_METRICS['identifier'].full_name}, {IMPLEMENTED_SIMPLE_METRICS['identifier'].__name__}). Please use a unique string for the indentifier."  # noqa: E501
            )

        if identifier in IMPLEMENTED_COMPLEX_METRICS:
            raise ValueError(
                f"'{identifier}' is already a complex metric identifier for: ({IMPLEMENTED_COMPLEX_METRICS['identifier'].full_name}, {IMPLEMENTED_COMPLEX_METRICS['identifier'].__name__}). Please use a unique string for the indentifier."  # noqa: E501
            )

        # Check that the required simple metrics are actually implemented
        for required_identifier in required_simple_metrics:
            if required_identifier not in IMPLEMENTED_SIMPLE_METRICS:
                raise ValueError(
                    f"'{required_identifier}' not recognized as a simple metric. Currently registered are: {IMPLEMENTED_SIMPLE_METRICS.keys()}"
                )

        # Register the function
        IMPLEMENTED_COMPLEX_METRICS[identifier] = func

        return func

    return wrapper
