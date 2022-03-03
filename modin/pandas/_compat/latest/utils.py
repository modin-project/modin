from pandas._libs.lib import no_default


def create_stat_method(name):
    def stat_method(
        self,
        axis: "int | None | NoDefault" = no_default,
        skipna=True,
        level=None,
        numeric_only=None,
        **kwargs,
    ):
        return self._stat_operation(name, axis, skipna, level, numeric_only, **kwargs)

    return stat_method
