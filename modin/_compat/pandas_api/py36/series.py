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

"""Module for 'Python 3.6 pandas' compatibility layer for Series."""

from typing import TYPE_CHECKING

import numpy as np
import pandas

from ..abc.series import BaseCompatibleSeries
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.pandas import DataFrame


@_inherit_docstrings(pandas.Series)
class Python36CompatibleSeries(BaseCompatibleSeries):
    """Compatibility layer for 'Python 3.6 pandas' for Series."""

    def apply(
        self, func, convert_dtype=True, args=(), **kwds
    ):  # noqa: PR01, RT01, D200
        return self._apply(func, convert_dtype=convert_dtype, args=args, **kwds)

    def between(self, left, right, inclusive=True):  # noqa: PR01, RT01, D200
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
        if numeric_only is True:
            raise NotImplementedError("Series.sum does not implement numeric_only")
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
