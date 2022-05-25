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

import numpy as np
import pandas
import pandas.core.resample
from pandas._typing import (
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
)
from pandas.core.dtypes.common import is_list_like
from typing import Optional, Union
from modin.utils import _inherit_docstrings
from modin.logging import LoggerMetaClass


@_inherit_docstrings(pandas.core.resample.Resampler)
class Resampler(object, metaclass=LoggerMetaClass):
    def __init__(
        self,
        dataframe,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=0,
        on=None,
        level=None,
        origin: Union[str, TimestampConvertibleTypes] = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        axis = self._dataframe._get_axis_number(axis)
        self.resample_kwargs = {
            "rule": rule,
            "axis": axis,
            "closed": closed,
            "label": label,
            "convention": convention,
            "kind": kind,
            "loffset": loffset,
            "base": base,
            "on": on,
            "level": level,
            "origin": origin,
            "offset": offset,
        }
        self.__groups = self.__get_groups(**self.resample_kwargs)

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

    def __get_groups(
        self,
        rule,
        axis,
        closed,
        label,
        convention,
        kind,
        loffset,
        base,
        on,
        level,
        origin,
        offset,
    ):
        if axis == 0:
            df = self._dataframe
        else:
            df = self._dataframe.T
        groups = df.groupby(
            pandas.Grouper(
                key=on,
                freq=rule,
                closed=closed,
                label=label,
                convention=convention,
                loffset=loffset,
                base=base,
                level=level,
                origin=origin,
                offset=offset,
            )
        )
        return groups

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
        if self.resample_kwargs["axis"] == 0:
            result = self.__groups.get_group(name)
        else:
            result = self.__groups.get_group(name).T
        return result

    def apply(self, func, *args, **kwargs):
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
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_ffill(
                self.resample_kwargs, limit
            )
        )

    def backfill(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_backfill(
                self.resample_kwargs, limit
            )
        )

    def bfill(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_bfill(
                self.resample_kwargs, limit
            )
        )

    def pad(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_pad(
                self.resample_kwargs, limit
            )
        )

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
        axis=0,
        limit=None,
        inplace=False,
        limit_direction: Optional[str] = None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_interpolate(
                self.resample_kwargs,
                method,
                axis,
                limit,
                inplace,
                limit_direction,
                limit_area,
                downcast,
                **kwargs,
            )
        )

    def count(self):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_count(self.resample_kwargs)
        )

    def nunique(self, _method="nunique", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_nunique(
                self.resample_kwargs, _method, *args, **kwargs
            )
        )

    def first(self, _method="first", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_first(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def last(self, _method="last", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_last(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def max(self, _method="max", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_max(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def mean(self, _method="mean", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_mean(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def median(self, _method="median", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_median(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def min(self, _method="min", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_min(
                self.resample_kwargs,
                _method,
                *args,
                **kwargs,
            )
        )

    def ohlc(self, _method="ohlc", *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_df(
                    self.resample_kwargs,
                    _method,
                    *args,
                    **kwargs,
                )
            )
        else:
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_ser(
                    self.resample_kwargs,
                    _method,
                    *args,
                    **kwargs,
                )
            )

    def prod(self, _method="prod", min_count=0, *args, **kwargs):
        if self.resample_kwargs["axis"] == 0:
            result = self.__groups.prod(min_count=min_count, *args, **kwargs)
        else:
            result = self.__groups.prod(min_count=min_count, *args, **kwargs).T
        return result

    def size(self):
        from .series import Series

        return Series(
            query_compiler=self._query_compiler.resample_size(self.resample_kwargs)
        )

    def sem(self, _method="sem", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_sem(
                self.resample_kwargs,
                _method,
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

    def sum(self, _method="sum", min_count=0, *args, **kwargs):
        if self.resample_kwargs["axis"] == 0:
            result = self.__groups.sum(min_count=min_count, *args, **kwargs)
        else:
            result = self.__groups.sum(min_count=min_count, *args, **kwargs).T
        return result

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
