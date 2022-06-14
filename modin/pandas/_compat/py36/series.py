import numpy as np

from ..abc.series import BaseCompatibilitySeries


class Python36CompatibilitySeries(BaseCompatibilitySeries):
    def apply(
        self, func, convert_dtype=True, args=(), **kwds
    ):  # noqa: PR01, RT01, D200
        return self._apply(func, convert_dtype=convert_dtype, args=args, **kwds)

    def between(self, left, right, inclusive=True):  # noqa: PR01, RT01, D200
        """
        Return boolean Series equivalent to left <= series <= right.
        """
        return self._between(left, right, inclusive=inclusive)

    def kurt(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        return self._kurt(
            axis=axis, skipna=skipna, level=level, numeric_only=numeric_only, **kwargs
        )

    def mask(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):
        return self._mask(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        return self._prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def reindex(self, index=None, **kwargs):  # noqa: PR01, RT01, D200
        return self._reindex(index=index, **kwargs)

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):  # noqa: PR01, RT01, D200
        return self._replace(
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )

    def reset_index(
        self, level=None, drop=False, name=None, inplace=False
    ):  # noqa: PR01, RT01, D200
        return self._reset_index(level=level, drop=drop, name=name, inplace=inplace)

    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        return self._sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def to_frame(self, name=None) -> "DataFrame":  # noqa: PR01, RT01, D200
        return self._to_frame(name=name)

    def where(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):  # noqa: PR01, RT01, D200
        return self._where(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
        )
