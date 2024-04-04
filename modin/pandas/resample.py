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

"""Implement Resampler public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas
import pandas.core.resample
from pandas._libs import lib
from pandas.core.dtypes.common import is_list_like

from modin.logging import ClassLogger
from modin.pandas.utils import cast_function_modin2pandas
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.core.storage_formats import BaseQueryCompiler
    from modin.pandas import DataFrame, Series


@_inherit_docstrings(pandas.core.resample.Resampler)
class Resampler(ClassLogger):
    _dataframe: Union[DataFrame, Series]
    _query_compiler: BaseQueryCompiler

    def __init__(
        self,
        dataframe: Union[DataFrame, Series],
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        on=None,
        level=None,
        origin="start_day",
        offset=None,
        group_keys=lib.no_default,
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.axis = self._dataframe._get_axis_number(axis)
        self.resample_kwargs = {
            "rule": rule,
            "axis": axis,
            "closed": closed,
            "label": label,
            "convention": convention,
            "kind": kind,
            "on": on,
            "level": level,
            "origin": origin,
            "offset": offset,
            "group_keys": group_keys,
        }
        self.__groups = self._get_groups()

    def _get_groups(self):
        """
        Compute the resampled groups.

        Returns
        -------
        PandasGroupby
            Groups as specified by resampling arguments.
        """
        df = self._dataframe if self.axis == 0 else self._dataframe.T
        convention = self.resample_kwargs["convention"]
        groups = df.groupby(
            pandas.Grouper(
                key=self.resample_kwargs["on"],
                freq=self.resample_kwargs["rule"],
                closed=self.resample_kwargs["closed"],
                label=self.resample_kwargs["label"],
                convention=convention if convention is not lib.no_default else "start",
                level=self.resample_kwargs["level"],
                origin=self.resample_kwargs["origin"],
                offset=self.resample_kwargs["offset"],
            ),
            group_keys=self.resample_kwargs["group_keys"],
        )
        return groups

    def __getitem__(self, key):
        """
        Get ``Resampler`` based on `key` columns of original dataframe.

        Parameters
        ----------
        key : str or list
            String or list of selections.

        Returns
        -------
        modin.pandas.BasePandasDataset
            New ``Resampler`` based on `key` columns subset
            of the original dataframe.
        """

        def _get_new_resampler(key):
            subset = self._dataframe[key]
            resampler = type(self)(subset, **self.resample_kwargs)
            return resampler

        from .series import Series

        if isinstance(
            key, (list, tuple, Series, pandas.Series, pandas.Index, np.ndarray)
        ):
            if len(self._dataframe.columns.intersection(key)) != len(set(key)):
                missed_keys = list(set(key).difference(self._dataframe.columns))
                raise KeyError(f"Columns not found: {str(sorted(missed_keys))[1:-1]}")
            return _get_new_resampler(list(key))

        if key not in self._dataframe:
            raise KeyError(f"Column not found: {key}")

        return _get_new_resampler(key)

    @property
    def groups(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.DataFrame.resample(df, **self.resample_kwargs).groups
        )

    @property
    def indices(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.DataFrame.resample(df, **self.resample_kwargs).indices
        )

    def get_group(self, name, obj=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_get_group(
                self.resample_kwargs, name, obj
            )
        )

    def apply(self, func, *args, **kwargs):
        func = cast_function_modin2pandas(func)
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            query_comp_op = self._query_compiler.resample_app_df
        else:
            query_comp_op = self._query_compiler.resample_app_ser

        dataframe = DataFrame(
            query_compiler=query_comp_op(
                self.resample_kwargs,
                func,
                *args,
                **kwargs,
            )
        )
        if is_list_like(func) or isinstance(self._dataframe, DataFrame):
            return dataframe
        else:
            if len(dataframe.index) == 1:
                return dataframe.iloc[0]
            else:
                return dataframe.squeeze()

    def aggregate(self, func, *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            query_comp_op = self._query_compiler.resample_agg_df
        else:
            query_comp_op = self._query_compiler.resample_agg_ser

        dataframe = DataFrame(
            query_compiler=query_comp_op(
                self.resample_kwargs,
                func,
                *args,
                **kwargs,
            )
        )
        if is_list_like(func) or isinstance(self._dataframe, DataFrame):
            return dataframe
        else:
            if len(dataframe.index) == 1:
                return dataframe.iloc[0]
            else:
                return dataframe.squeeze()

    def transform(self, arg, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_transform(
                self.resample_kwargs, arg, *args, **kwargs
            )
        )

    def pipe(self, func, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_pipe(
                self.resample_kwargs, func, *args, **kwargs
            )
        )

    def ffill(self, limit=None):
        return self.fillna(method="ffill", limit=limit)

    def bfill(self, limit=None):
        return self.fillna(method="bfill", limit=limit)

    def nearest(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_nearest(
                self.resample_kwargs, limit
            )
        )

    def fillna(self, method, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_fillna(
                self.resample_kwargs, method, limit
            )
        )

    def asfreq(self, fill_value=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_asfreq(
                self.resample_kwargs, fill_value
            )
        )

    def interpolate(
        self,
        method="linear",
        *,
        axis=0,
        limit=None,
        inplace=False,
        limit_direction: Optional[str] = None,
        limit_area=None,
        downcast=lib.no_default,
        **kwargs,
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_interpolate(
                self.resample_kwargs,
                method,
                axis=axis,
                limit=limit,
                inplace=inplace,
                limit_direction=limit_direction,
                limit_area=limit_area,
                downcast=downcast,
                **kwargs,
            )
        )

    def count(self):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_count(self.resample_kwargs)
        )

    def nunique(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_nunique(
                self.resample_kwargs, *args, **kwargs
            )
        )

    def first(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_first(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def last(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_last(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def max(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_max(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_mean(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def median(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_median(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def min(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_min(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def ohlc(self, *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_df(
                    self.resample_kwargs,
                    *args,
                    **kwargs,
                )
            )
        else:
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_ser(
                    self.resample_kwargs,
                    *args,
                    **kwargs,
                )
            )

    def prod(self, min_count=0, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_prod(
                self.resample_kwargs, min_count=min_count, *args, **kwargs
            )
        )

    def size(self):
        from .series import Series

        output_series = Series(
            query_compiler=self._query_compiler.resample_size(self.resample_kwargs)
        )
        if not isinstance(self._dataframe, Series):
            # If input is a DataFrame, rename output Series to None
            return output_series.rename(None)
        return output_series

    def sem(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_sem(
                self.resample_kwargs,
                *args,
                **kwargs,
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_std(
                self.resample_kwargs, *args, ddof=ddof, **kwargs
            )
        )

    def sum(self, min_count=0, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_sum(
                self.resample_kwargs, min_count=min_count, *args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_var(
                self.resample_kwargs, *args, ddof=ddof, **kwargs
            )
        )

    def quantile(self, q=0.5, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_quantile(
                self.resample_kwargs, q, **kwargs
            )
        )
