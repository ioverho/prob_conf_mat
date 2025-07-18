import typing
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class NotInCache:
    """The requested element is not in the cache."""

    pass


class NestedCache:
    """Recursive defaultdict for caching things recursively.

    Examples:
        >>> cache = NestedCache()

        >>> cache.load(["foo", "bar"])
        KeyError: "Keys ['foo', 'bar'] not in cache"

        >>> cache.cache(["foo", "bar"], "baz")

        >>> cache.load(["foo", "bar"])
        'baz'

    """

    def __init__(self):
        self._cache = self.nested_dict()
        self.fingerprint = None

    def nested_dict(self):
        """Generates a recursive default dict."""
        return defaultdict(self.nested_dict)

    def __getitem__(self, key):
        return self._cache[key]

    def _get(self, dd, keys, i: int):
        # Recursively look up elements in cache
        # First check current layer
        obj = dd.get(keys[i], NotInCache)

        # If key not in cache, or we've exhausted the
        # list of keys, return what came out of the cache
        if obj is NotInCache or i == len(keys) - 1:
            return obj

        # Otherwise, keep going down the cache until we
        # exhaust the list of keys
        return self._get(obj, keys=keys, i=i + 1)

    def load(
        self,
        fingerprint: str,
        keys,
        default=NotInCache,
    ) -> type[NotInCache] | typing.Any:
        """Loads the element stored at keys in the cache."""
        if fingerprint != self.fingerprint:
            return default

        # Safe method for getting value without altering
        # the cache
        result = self._get(dd=self._cache, keys=keys, i=0)

        if result is NotInCache:
            return default
        return result

    def _set(self, dd, value, keys, i: int):
        # Recursively set elements in cache
        # If we've exhausted the list of keys,
        # set the value
        if i == len(keys) - 1:
            dd[keys[i]] = value

        # Otherwise, keep going down the cache until we
        # exhaust the list of keys
        else:
            self._set(dd[keys[i]], value=value, keys=keys, i=i + 1)

    def cache(self, fingerprint: str, keys: list[typing.Any], value) -> None:
        """Caches an element at keys."""
        if fingerprint != self.fingerprint:
            self.clean()
            self.fingerprint = fingerprint

        # Method for setting value
        self._set(dd=self._cache, keys=keys, value=value, i=0)

    def clean(self) -> None:
        """Destroys the current cache state."""
        raise NotImplementedError

    def clear(self) -> None:
        """Alias of clean."""
        self.clean()

    def empty(self) -> None:
        """Alias of clean."""
        self.clean()

    def isin(self, fingerprint: str, keys: list[typing.Any]) -> bool:
        """Checks if there is an element at keys."""
        if fingerprint != self.fingerprint:
            return False

        result = self._get(dd=self._cache, keys=keys, i=0)

        return result is not NotInCache


class InMemoryCache(NestedCache):
    """A recursive cache that is stored entirely in-memory."""
    def clean(self) -> None:
        """Destroys the current cache state."""
        self._cache = self.nested_dict()


# TODO: add support for filesystem caching
# class PickleCache(NestedCache):
#     def __init__(self):
#         raise NotImplementedError

#     def clean(self):
#         def _clean_directory(dir: Path):
#             for obj in dir.glob("*"):
#                 if obj.is_dir():
#                     _clean_directory(obj)
#                 else:
#                     obj.unlink()

#             dir.rmdir()

#         if isinstance(self.cache_location, Path) and self.cache_location.exists():
#             # warnings.warn(f"Cleaning cache dir: {self.cache_location}")

#             _clean_directory(self.cache_location)
#         else:
#             raise ValueError("Could not clean cache dir, does not exist.")
