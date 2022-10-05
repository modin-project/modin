# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Module for 'latest pandas' compatibility layer for Series."""

from typing import IO, Hashable, TYPE_CHECKING

import numpy as np
import pandas
from pandas.util._validators import validate_bool_kwarg
from pandas._libs.lib import no_default, NoDefault
from pandas._typing import Axis, Suffixes

from ..abc.series import BaseCompatibleSeries

if TYPE_CHECKING:
    from modin.pandas.dataframe import DataFrame


class LatestCompatibleSeries(BaseCompatibleSeries):
    """Compatibility layer for 'latest pandas' for Series."""

    def apply(self, func, convert_dtype=True, args=(), **kwargs):
        return self._apply(func, convert_dtype=convert_dtype, args=args, **kwargs)

    def between(self, left, right, inclusive="both"):  # noqa: PR01, RT01, D200
        return self._between(left, right, inclusive=inclusive)

    def compare(
        self,
        other,
        align_axis: Axis = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: Suffixes = ("self", "other"),
    ):
        return self._compare(
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )

    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        return self._idxmax(axis=axis, skipna=skipna)

    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        return self._idxmin(axis=axis, skipna=skipna)

    def info(
        self,
        verbose: "bool | None" = None,
        buf: "IO[str] | None" = None,
        max_cols: "int | None" = None,
        memory_usage: "bool | str | None" = None,
        show_counts: "bool" = True,
    ):
        return self._default_to_pandas(
            pandas.Series.info,
            verbose=verbose,
            buf=buf,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
        )

    def factorize(self, sort=False, na_sentinel=no_default, use_na_sentinel=no_default):
        return self._factorize(
            sort=sort, na_sentinel=na_sentinel, use_na_sentinel=use_na_sentinel
        )

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=no_default,
        squeeze=no_default,
        observed=False,
        dropna: bool = True,
    ):
        return self._groupby(
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            dropna=dropna,
        )

    def kurt(
        self,
        axis: "Axis | None | NoDefault" = no_default,
        skipna=True,
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
        errors=no_default,
        try_cast=no_default,
    ):
        return self._default_to_pandas(
            pandas.Series.mask,
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
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def reindex(self, *args, **kwargs):  # noqa: PR01, RT01, D200
        return self._reindex(*args, **kwargs)

    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace=False,
        limit=None,
        regex=False,
        method: "str | NoDefault" = no_default,
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
        self,
        level=None,
        drop=False,
        name=no_default,
        inplace=False,
        allow_duplicates=False,
    ):  # noqa: PR01, RT01, D200
        return self._reset_index(
            level=level,
            drop=drop,
            name=name,
            inplace=inplace,
            allow_duplicates=allow_duplicates,
        )

    def sum(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        return self._sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def to_frame(
        self, name: "Hashable" = no_default
    ) -> "DataFrame":  # noqa: PR01, RT01, D200
        return self._to_frame(name=name)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ):  # noqa: PR01, RT01, D200
        return self._value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )

    def where(
        self,
        cond,
        other=no_default,
        inplace=False,
        axis=None,
        level=None,
        errors=no_default,
        try_cast=no_default,
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
