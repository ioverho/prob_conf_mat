import typing

from bayes_conf_mat.io.base import IO_REGISTRY


def get_io(format: str, location: str, **kwargs) -> typing.Callable:
    """Gets a function that matches the IO pattern described.

    Does *not* fetch the data yet. To fetch the data (and have the
    confusion matrix validated), call the output, like:

    ```python
    io_method = get_io(...)
    confusion_matrix = io_method()
    ```

    Args:
        format (str): the IO format, must be registed in the `IO_REGISTRY`
        location (str): the location of the input file

    Returns:
        typing.Callable: the initialized IO function
    """
    io_method = IO_REGISTRY[format](location=location, **kwargs)
    return io_method
