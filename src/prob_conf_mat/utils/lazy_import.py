# import sys
# import importlib.util

# # TODO: find an alternative for this function
# def lazy_import(name):
#     """Only imports a module when needed.

#     Avoids polluting the namespace, and should cut down startup time

#     Taken from: https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
#     """
#     spec = importlib.util.find_spec(name)

#     loader = importlib.util.LazyLoader(spec.loader)

#     spec.loader = loader

#     module = importlib.util.module_from_spec(spec)

#     sys.modules[name] = module

#     loader.exec_module(module)

#     return module
