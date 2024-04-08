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

"""Implement Window and Rolling public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import pandas.core.window.rolling
from pandas.core.dtypes.common import is_list_like

from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.core.storage_formats import BaseQueryCompiler
    from modin.pandas import DataFrame, Series


@_inherit_docstrings(pandas.core.window.rolling.Window)
class Window(ClassLogger):
    _dataframe: Union[DataFrame, Series]
    _query_compiler: BaseQueryCompiler

    def __init__(
        self,
        dataframe: Union[DataFrame, Series],
        window=None,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method="single",
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.window_kwargs = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "axis": axis,
            "closed": closed,
            "step": step,
            "method": method,
        }
        self.axis = axis

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_mean(
                self.axis, self.window_kwargs, *args, **kwargs
            )
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_sum(
                self.axis, self.window_kwargs, *args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_var(
                self.axis, self.window_kwargs, ddof, *args, **kwargs
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_std(
                self.axis, self.window_kwargs, ddof, *args, **kwargs
            )
        )


@_inherit_docstrings(
    pandas.core.window.rolling.Rolling,
    excluded=[pandas.core.window.rolling.Rolling.__init__],
)
class Rolling(ClassLogger):
    def __init__(
        self,
        dataframe,
        window=None,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
        step=None,
        method="single",
    ):
        if step is not None:
            raise NotImplementedError("step parameter is not implemented yet.")
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.rolling_kwargs = {
            "window": window,
            "min_periods": min_periods,
            "center": center,
            "win_type": win_type,
            "on": on,
            "axis": axis,
            "closed": closed,
            "step": step,
            "method": method,
        }
        self.axis = axis

    def _call_qc_method(self, method_name, *args, **kwargs):
        """
        Call a query compiler method for the specified rolling aggregation.

        Parameters
        ----------
        method_name : str
            Name of the aggregation.
        *args : tuple
            Positional arguments to pass to the query compiler method.
        **kwargs : dict
            Keyword arguments to pass to the query compiler method.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler holding the result of the aggregation.
        """
        qc_method = getattr(self._query_compiler, f"rolling_{method_name}")
        return qc_method(self.axis, self.rolling_kwargs, *args, **kwargs)

    def _aggregate(self, method_name, *args, **kwargs):
        """
        Run the specified rolling aggregation.

        Parameters
        ----------
        method_name : str
            Name of the aggregation.
        *args : tuple
            Positional arguments to pass to the aggregation.
        **kwargs : dict
            Keyword arguments to pass to the aggregation.

        Returns
        -------
        DataFrame or Series
            Result of the aggregation.
        """
        qc_result = self._call_qc_method(method_name, *args, **kwargs)
        return self._dataframe.__constructor__(query_compiler=qc_result)

    def count(self):
        return self._aggregate("count")

    def sem(self, *args, **kwargs):
        return self._aggregate("sem", *args, **kwargs)

    def sum(self, *args, **kwargs):
        return self._aggregate("sum", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._aggregate("mean", *args, **kwargs)

    def median(self, **kwargs):
        return self._aggregate("median", **kwargs)

    def var(self, ddof=1, *args, **kwargs):
        return self._aggregate("var", ddof, *args, **kwargs)

    def std(self, ddof=1, *args, **kwargs):
        return self._aggregate("std", ddof, *args, **kwargs)

    def min(self, *args, **kwargs):
        return self._aggregate("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self._aggregate("max", *args, **kwargs)

    def corr(self, other=None, pairwise=None, *args, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._aggregate("corr", other, pairwise, *args, **kwargs)

    def cov(self, other=None, pairwise=None, ddof: Optional[int] = 1, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._aggregate("cov", other, pairwise, ddof, **kwargs)

    def skew(self, **kwargs):
        return self._aggregate("skew", **kwargs)

    def kurt(self, **kwargs):
        return self._aggregate("kurt", **kwargs)

    def apply(
        self,
        func,
        raw=False,
        engine="cython",
        engine_kwargs=None,
        args=None,
        kwargs=None,
    ):
        func = cast_function_modin2pandas(func)
        return self._aggregate("apply", func, raw, engine, engine_kwargs, args, kwargs)

    def aggregate(
        self,
        func,
        *args,
        **kwargs,
    ):
        from .dataframe import DataFrame

        dataframe = DataFrame(
            query_compiler=self._call_qc_method(
                "aggregate",
                func,
                *args,
                **kwargs,
            )
        )
        if isinstance(self._dataframe, DataFrame):
            return dataframe
        elif is_list_like(func) and dataframe.columns.nlevels > 1:
            dataframe.columns = dataframe.columns.droplevel()
            return dataframe
        else:
            return dataframe.squeeze()

    agg = aggregate

    def quantile(self, q, interpolation="linear", **kwargs):
        return self._aggregate("quantile", q, interpolation, **kwargs)

    def rank(
        self, method="average", ascending=True, pct=False, numeric_only=False, **kwargs
    ):
        return self._aggregate("rank", method, ascending, pct, numeric_only, **kwargs)


@_inherit_docstrings(Rolling)
class RollingGroupby(Rolling):
    def __init__(self, groupby_obj, *args, **kwargs):
        self._as_index = groupby_obj._kwargs.get("as_index", True)
        self._groupby_obj = (
            groupby_obj if self._as_index else groupby_obj._override(as_index=True)
        )
        super().__init__(self._groupby_obj._df, *args, **kwargs)

    def sem(self, *args, **kwargs):
        ErrorMessage.mismatch_with_pandas(
            operation="RollingGroupby.sem() when 'as_index=False'",
            message=(
                "The group columns won't be involved in the aggregation.\n"
                + "See this gh-issue for more information: https://github.com/modin-project/modin/issues/6291"
            ),
        )
        return super().sem(*args, **kwargs)

    def corr(self, other=None, pairwise=None, *args, **kwargs):
        # pandas behavior is that it always assumes that 'as_index=True' for the '.corr()' method
        return super().corr(
            *args, as_index=True, other=other, pairwise=pairwise, **kwargs
        )

    def cov(self, other=None, pairwise=None, ddof: Optional[int] = 1, **kwargs):
        # pandas behavior is that it always assumes that 'as_index=True' for the '.cov()' method
        return super().cov(as_index=True, other=other, pairwise=pairwise, **kwargs)

    def _aggregate(self, method_name, *args, as_index=None, **kwargs):
        """
        Run the specified rolling aggregation.

        Parameters
        ----------
        method_name : str
            Name of the aggregation.
        *args : tuple
            Positional arguments to pass to the aggregation.
        as_index : bool, optional
            Whether the result should have the group labels as index levels or as columns.
            If not specified the parameter value will be taken from groupby kwargs.
        **kwargs : dict
            Keyword arguments to pass to the aggregation.

        Returns
        -------
        DataFrame or Series
            Result of the aggregation.
        """
        res = self._groupby_obj._wrap_aggregation(
            qc_method=type(self._query_compiler).groupby_rolling,
            numeric_only=False,
            agg_args=args,
            agg_kwargs=kwargs,
            agg_func=method_name,
            rolling_kwargs=self.rolling_kwargs,
        )

        if as_index is None:
            as_index = self._as_index

        if not as_index:
            res = res.reset_index(
                level=[i for i in range(len(self._groupby_obj._internal_by))],
                drop=False,
            )

        return res

    def _call_qc_method(self, method_name, *args, **kwargs):
        return self._aggregate(method_name, *args, **kwargs)._query_compiler


@_inherit_docstrings(
    pandas.core.window.expanding.Expanding,
    excluded=[pandas.core.window.expanding.Expanding.__init__],
)
class Expanding(ClassLogger):
    def __init__(self, dataframe, min_periods=1, axis=0, method="single"):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.expanding_args = [min_periods, axis, method]
        self.axis = axis

    def aggregate(self, func, *args, **kwargs):
        from .dataframe import DataFrame

        dataframe = DataFrame(
            query_compiler=self._query_compiler.expanding_aggregate(
                self.axis, self.expanding_args, func, *args, **kwargs
            )
        )
        if isinstance(self._dataframe, DataFrame):
            return dataframe
        elif is_list_like(func):
            dataframe.columns = dataframe.columns.droplevel()
            return dataframe
        else:
            return dataframe.squeeze()

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_sum(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def min(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_min(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def max(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_max(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_mean(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def median(self, numeric_only=False, engine=None, engine_kwargs=None, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_median(
                self.axis,
                self.expanding_args,
                numeric_only=numeric_only,
                engine=engine,
                engine_kwargs=engine_kwargs,
                **kwargs,
            )
        )

    def var(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_var(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def std(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_std(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def count(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_count(
                self.axis, self.expanding_args, *args, **kwargs
            )
        )

    def cov(self, other=None, pairwise=None, ddof=1, numeric_only=False, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_cov(
                self.axis,
                self.expanding_args,
                squeeze_self=isinstance(self._dataframe, Series),
                squeeze_other=isinstance(other, Series),
                other=(
                    other._query_compiler
                    if isinstance(other, (Series, DataFrame))
                    else other
                ),
                pairwise=pairwise,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def corr(self, other=None, pairwise=None, ddof=1, numeric_only=False, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_corr(
                self.axis,
                self.expanding_args,
                squeeze_self=isinstance(self._dataframe, Series),
                squeeze_other=isinstance(other, Series),
                other=(
                    other._query_compiler
                    if isinstance(other, (Series, DataFrame))
                    else other
                ),
                pairwise=pairwise,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def sem(self, ddof=1, numeric_only=False, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_sem(
                self.axis,
                self.expanding_args,
                ddof=ddof,
                numeric_only=numeric_only,
                *args,
                **kwargs,
            )
        )

    def skew(self, numeric_only=False, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_skew(
                self.axis, self.expanding_args, numeric_only=numeric_only, **kwargs
            )
        )

    def kurt(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_kurt(
                self.axis, self.expanding_args, **kwargs
            )
        )

    def quantile(self, q, interpolation="linear", **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_quantile(
                self.axis, self.expanding_args, q, interpolation, **kwargs
            )
        )

    def rank(
        self, method="average", ascending=True, pct=False, numeric_only=False, **kwargs
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.expanding_rank(
                self.axis,
                self.expanding_args,
                method,
                ascending,
                pct,
                numeric_only,
                **kwargs,
            )
        )
