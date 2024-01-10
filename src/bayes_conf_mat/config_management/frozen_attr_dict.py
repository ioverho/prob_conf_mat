# TODO: move this to utils folder
# Just a helper class, not that important
from types import SimpleNamespace


class FrozenAttrDict(SimpleNamespace):
    def __init__(self, d: dict):
        super().__init__()

        for k, v in d.items():
            v = self._convert_to_immutable(v)

            super().__setattr__(k, v)

    def _convert_to_immutable(self, attr):
        if isinstance(attr, list):
            return tuple(self._convert_to_immutable(a) for a in attr)
        elif isinstance(attr, dict) or hasattr(attr, "items"):
            return FrozenAttrDict(attr)
        else:
            return attr

    def __setattr__(self, name, value):
        raise AttributeError("This class is immutable.")

    def __repr__(self):
        return f"FrozenAttrDict({super().__dict__})"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()
