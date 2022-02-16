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

"""Implement Rolling public API."""

from typing import Optional
from pandas.core.dtypes.common import (
    is_list_like,
)
import pandas.core.window.rolling
import pandas.core.resample
import pandas.core.generic

from modin.utils import _inherit_docstrings


@_inherit_docstrings(
    pandas.core.window.rolling.Rolling,
    excluded=[pandas.core.window.rolling.Rolling.__init__],
)
class Rolling(object):
    def __init__(
        self,
        dataframe,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        method="single",
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.rolling_args = [
            window,
            min_periods,
            center,
            win_type,
            on,
            axis,
            closed,
            method,
        ]

    def count(self):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_count(self.rolling_args)
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_sum(
                self.rolling_args, *args, **kwargs
            )
        )

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_mean(
                self.rolling_args, *args, **kwargs
            )
        )

    def median(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_median(
                self.rolling_args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_var(
                self.rolling_args, ddof, *args, **kwargs
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_std(
                self.rolling_args, ddof, *args, **kwargs
            )
        )

    def min(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_min(
                self.rolling_args, *args, **kwargs
            )
        )

    def max(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_max(
                self.rolling_args, *args, **kwargs
            )
        )

    def corr(self, other=None, pairwise=None, *args, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_corr(
                self.rolling_args, other, pairwise, *args, **kwargs
            )
        )

    def cov(self, other=None, pairwise=None, ddof: Optional[int] = 1, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_cov(
                self.rolling_args, other, pairwise, ddof, **kwargs
            )
        )

    def skew(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_skew(
                self.rolling_args, **kwargs
            )
        )

    def kurt(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_kurt(
                self.rolling_args, **kwargs
            )
        )

    def apply(
        self,
        func,
        raw=False,
        engine="cython",
        engine_kwargs=None,
        args=None,
        kwargs=None,
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_apply(
                self.rolling_args,
                func,
                raw,
                engine,
                engine_kwargs,
                args,
                kwargs,
            )
        )

    def aggregate(
        self,
        func,
        *args,
        **kwargs,
    ):
        from .dataframe import DataFrame

        dataframe = DataFrame(
            query_compiler=self._query_compiler.rolling_aggregate(
                self.rolling_args,
                func,
                *args,
                **kwargs,
            )
        )
        if isinstance(self._dataframe, DataFrame):
            return dataframe
        elif is_list_like(func):
            dataframe.columns = dataframe.columns.droplevel()
            return dataframe
        else:
            return dataframe.squeeze()

    agg = aggregate

    def quantile(self, quantile, interpolation="linear", **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_quantile(
                self.rolling_args, quantile, interpolation, **kwargs
            )
        )
