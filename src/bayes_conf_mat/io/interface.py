import typing

from bayes_conf_mat.io.base import IO_REGISTRY


def get_io(format: str, location: str = None, **kwargs) -> typing.Callable:
    """Gets a function that matches the IO pattern described.

    Does *not* fetch the data yet. To fetch the data (and have the
    confusion matrix validated), call the load method, like:

    ```python
    io_method = get_io(...)
    confusion_matrix = io_method.load()
    ```

    Args:
        format (str): the IO format, must be registed in the `IO_REGISTRY`
        location (str): the location of the input file. Defaults to `None`, which will raise an error unless the format is `in_memory`

    Returns:
        typing.Callable: the initialized IO function
    """
    if format not in IO_REGISTRY:
        raise ValueError(
            f"Parameter `aggregation` must be a registered IO method. Currently: {format}. Must be one of {set(IO_REGISTRY.keys())}"
        )

    io_method = IO_REGISTRY[format](location=location, **kwargs)

    io_method._init_params = dict(format=format, location=location, **kwargs)

    return io_method
