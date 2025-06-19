def changes_state(method):
    """Whenever the state of a study changes, make sure to clean the cache"""

    def inner(self, *args, **kwargs):
        self.cache.clean()
        return method(self, *args, **kwargs)

    return inner
