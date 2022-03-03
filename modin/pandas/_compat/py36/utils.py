def create_stat_method(name):
    def stat_method(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._stat_operation(name, axis, skipna, level, numeric_only, **kwargs)

    return stat_method
