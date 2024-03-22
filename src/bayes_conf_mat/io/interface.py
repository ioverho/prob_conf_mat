from bayes_conf_mat.io.base import IO_REGISTRY


def get_io(format: str, location: str, **kwargs):
    io_method = IO_REGISTRY[format](location=location, **kwargs)
    return io_method
