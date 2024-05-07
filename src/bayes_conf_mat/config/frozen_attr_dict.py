from collections import OrderedDict
from typing import Any

import strictyaml


class Config:
    """Wraps the output of strictyaml in a more usable class.

    Allows for accessing elements as attributes (i.e. the stuff we know is there)
    or by iterating over the elements (i.e. the stuff we need to discover).

    Only other implemented attribute is an `as_yaml` method for getting back the
    original YAML document.

    Should be frozen/immutable.

    Args:
        yaml_config (strictyaml.representation.YAML): _description_
    """

    def __init__(self, yaml_config: strictyaml.representation.YAML):
        self._config = FrozenAttrDict(yaml_config.data)
        self._yaml_doc = yaml_config.as_yaml()

    def __getattr__(self, name: str):
        if name == "_attrs":
            return self._config._attrs
        elif name == "_config" or name == "_yaml_doc":
            return super().__getattribute__(name)
        else:
            try:
                return self._attrs[name]
            except KeyError:
                raise AttributeError(name)

    def __setattr__(self, name, value):
        # Only allow setting the _attrs attribute once
        # Other than that, this class is immutable
        if (name == "_config" or name == "_yaml_doc") and not hasattr(self, name):
            return super().__setattr__(name, value)
        else:
            raise AttributeError("Class is frozen.")

    def as_yaml(self):
        return self._yaml_doc

    def __repr__(self):
        return f"Config({self.name})"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return len(self._attrs)

    def keys(self):
        return tuple(self._attrs.keys())

    def values(self):
        return tuple(self._attrs.values())

    def items(self):
        return tuple(self._attrs.items())

    def get(self, key, default):
        if key in self._attrs:
            return self._attrs[key]
        else:
            return default

    def __getitem__(self, key):
        return self._attrs[key]

    def __hash__(self) -> int:
        return hash(self.items())

    def __contains__(self, key):
        return key in self._attrs


class FrozenAttrDict:
    def __init__(self, d: dict):
        self._attrs = OrderedDict()
        for k, v in d.items():
            v = self._convert_to_immutable(v)

            self._attrs[k] = v

    def _convert_to_immutable(self, obj):
        if isinstance(obj, FrozenAttrDict):
            return obj
        elif isinstance(obj, list):
            return tuple(self._convert_to_immutable(a) for a in obj)
        elif isinstance(obj, set):
            return frozenset(self._convert_to_immutable(a) for a in obj)
        elif isinstance(obj, dict) or hasattr(obj, "items"):
            return FrozenAttrDict(obj)
        else:
            return obj

    def __setattr__(self, name, value):
        # Only allow setting the _attrs attribute once
        # Other than that, this class is immutable
        if name == "_attrs" and not hasattr(self, "_attrs"):
            return super().__setattr__(name, value)
        else:
            raise AttributeError("Class is frozen.")

    def __getattr__(self, name: str) -> Any:
        if name == "_attrs":
            super().__getattribute__(name)
        else:
            try:
                return self._attrs[name]
            except KeyError:
                raise AttributeError(name)

    def __repr__(self):
        return f"FrozenAttrDict(keys={list(self._attrs.keys())})"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return len(self._attrs)

    def keys(self):
        return tuple(self._attrs.keys())

    def values(self):
        return tuple(self._attrs.values())

    def items(self):
        return tuple(self._attrs.items())

    def get(self, key, default):
        if key in self._attrs:
            return self._attrs[key]
        else:
            return default

    def __getitem__(self, key):
        return self._attrs[key]

    def __hash__(self) -> int:
        return hash(self.items())

    def __contains__(self, key):
        return key in self._attrs
