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

"""Implement GroupBy public API as pandas does."""

import numpy as np
import pandas
import pandas.core.groupby
from pandas.core.dtypes.common import is_list_like
from pandas.core.aggregation import reconstruct_func
from pandas._libs.lib import no_default
import pandas.core.common as com
from types import BuiltinFunctionType
from collections.abc import Iterable

from modin.error_message import ErrorMessage
from modin.utils import (
    _inherit_docstrings,
    try_cast_to_pandas,
    wrap_udf_function,
    hashable,
    wrap_into_list,
)
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.config import IsExperimental
from .series import Series
from .utils import is_label


@_inherit_docstrings(pandas.core.groupby.DataFrameGroupBy)
class DataFrameGroupBy(object):
    def __init__(
        self,
        df,
        by,
        axis,
        level,
        as_index,
        sort,
        group_keys,
        squeeze,
        idx_name,
        drop,
        **kwargs,
    ):
        self._axis = axis
        self._idx_name = idx_name
        self._df = df
        self._query_compiler = self._df._query_compiler
        self._columns = self._query_compiler.columns
        self._by = by
        self._drop = drop

        if (
            level is None
            and is_list_like(by)
            or isinstance(by, type(self._query_compiler))
        ):
            # This tells us whether or not there are multiple columns/rows in the groupby
            self._is_multi_by = (
                isinstance(by, type(self._query_compiler)) and len(by.columns) > 1
            ) or (
                not isinstance(by, type(self._query_compiler))
                and axis == 0
                and all(
                    (hashable(obj) and obj in self._query_compiler.columns)
                    or isinstance(obj, type(self._query_compiler))
                    or is_list_like(obj)
                    for obj in self._by
                )
            )
        else:
            self._is_multi_by = False
        self._level = level
        self._kwargs = {
            "level": level,
            "sort": sort,
            "as_index": as_index,
            "group_keys": group_keys,
        }
        self._squeeze = squeeze
        self._kwargs.update(kwargs)

    _index_grouped_cache = None

    def __getattr__(self, key):
        """
        Alter regular attribute access, looks up the name in the columns.

        Parameters
        ----------
        key : str
            Attribute name.

        Returns
        -------
        The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self._columns:
                return self.__getitem__(key)
            raise e

    @property
    def ngroups(self):
        return len(self)

    def skew(self, **kwargs):
        return self._apply_agg_function(lambda df: df.skew(**kwargs))

    def ffill(self, limit=None):
        return self._default_to_pandas(lambda df: df.ffill(limit=limit))

    def sem(self, ddof=1):
        return self._default_to_pandas(lambda df: df.sem(ddof=ddof))

    def mean(self, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.mean(*args, **kwargs))

    def any(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_any,
            lambda df, **kwargs: df.any(**kwargs),
            numeric_only=False,
            **kwargs,
        )

    @property
    def plot(self):  # pragma: no cover
        return self._default_to_pandas(lambda df: df.plot)

    def ohlc(self):
        return self._default_to_pandas(lambda df: df.ohlc())

    def __bytes__(self):
        """
        Convert DataFrameGroupBy object into a python2-style byte string.

        Returns
        -------
        bytearray
            Byte array representation of `self`.

        Notes
        -----
        Deprecated and removed in pandas and will be likely removed in Modin.
        """
        return self._default_to_pandas(lambda df: df.__bytes__())

    @property
    def tshift(self):
        return self._default_to_pandas(lambda df: df.tshift)

    @property
    def groups(self):
        return self._index_grouped

    def min(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_min,
            lambda df, **kwargs: df.min(**kwargs),
            numeric_only=False,
            **kwargs,
        )

    def idxmax(self):
        return self._default_to_pandas(lambda df: df.idxmax())

    @property
    def ndim(self):
        """
        Return 2.

        Returns
        -------
        int
            Returns 2.

        Notes
        -----
        Deprecated and removed in pandas and will be likely removed in Modin.
        """
        return 2  # ndim is always 2 for DataFrames

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        def _shift(data, periods, freq, axis, fill_value, is_set_nan_rows=True):
            from .dataframe import DataFrame

            result = data.shift(periods, freq, axis, fill_value)

            if (
                is_set_nan_rows
                and isinstance(self._by, BaseQueryCompiler)
                and (
                    # Check using `issubset` is effective only in case of MultiIndex
                    set(self._by.columns).issubset(list(data.columns))
                    if isinstance(self._by.columns, pandas.MultiIndex)
                    else len(
                        self._by.columns.unique()
                        .sort_values()
                        .difference(data.columns.unique().sort_values())
                    )
                    == 0
                )
                and DataFrame(query_compiler=self._by.isna()).any(axis=None)
            ):
                mask_nan_rows = data[self._by.columns].isna().any(axis=1)
                # drop NaN groups
                result = result.loc[~mask_nan_rows]
            return result

        if freq is None and axis == 1 and self._axis == 0:
            result = _shift(self._df, periods, freq, axis, fill_value)
        elif (
            freq is not None
            and axis == 0
            and self._axis == 0
            and isinstance(self._by, BaseQueryCompiler)
        ):
            result = _shift(
                self._df, periods, freq, axis, fill_value, is_set_nan_rows=False
            )
            new_idx_lvl_arrays = np.concatenate(
                [self._df[self._by.columns].values.T, [list(result.index)]]
            )
            result.index = pandas.MultiIndex.from_arrays(
                new_idx_lvl_arrays,
                names=[col_name for col_name in self._by.columns]
                + [result._query_compiler.get_index_name()],
            )
            result = result.dropna(subset=self._by.columns).sort_index()
        else:
            result = self._apply_agg_function(
                lambda df: df.shift(periods, freq, axis, fill_value)
            )
            result._query_compiler.set_index_name(None)
        return result

    def nth(self, n, dropna=None):
        return self._default_to_pandas(lambda df: df.nth(n, dropna=dropna))

    def cumsum(self, axis=0, *args, **kwargs):
        result = self._apply_agg_function(lambda df: df.cumsum(axis, *args, **kwargs))
        # pandas does not name the index on cumsum
        result._query_compiler.set_index_name(None)
        return result

    @property
    def indices(self):
        return self._index_grouped

    def pct_change(self):
        return self._default_to_pandas(lambda df: df.pct_change())

    def filter(self, func, dropna=True, *args, **kwargs):
        return self._default_to_pandas(
            lambda df: df.filter(func, dropna=dropna, *args, **kwargs)
        )

    def cummax(self, axis=0, **kwargs):
        result = self._apply_agg_function(lambda df: df.cummax(axis, **kwargs))
        # pandas does not name the index on cummax
        result._query_compiler.set_index_name(None)
        return result

    def apply(self, func, *args, **kwargs):
        if not isinstance(func, BuiltinFunctionType):
            func = wrap_udf_function(func)
        return self._apply_agg_function(lambda df: df.apply(func, *args, **kwargs))

    @property
    def dtypes(self):
        if self._axis == 1:
            raise ValueError("Cannot call dtypes on groupby with axis=1")
        if not self._as_index:
            return self.apply(lambda df: df.dtypes)
        else:
            return self._apply_agg_function(lambda df: df.dtypes)

    def first(self, **kwargs):
        return self._default_to_pandas(lambda df: df.first(**kwargs))

    def backfill(self, limit=None):
        return self.bfill(limit)

    _internal_by_cache = no_default

    # TODO: since python 3.9:
    # @cached_property
    @property
    def _internal_by(self):
        """
        Get only those components of 'by' that are column labels of the source frame.

        Returns
        -------
        tuple of labels
        """
        if self._internal_by_cache is not no_default:
            return self._internal_by_cache

        internal_by = tuple()
        if self._drop:
            if is_list_like(self._by):
                internal_by = tuple(by for by in self._by if isinstance(by, str))
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=not isinstance(self._by, BaseQueryCompiler),
                    extra_log=f"When 'drop' is True, 'by' must be either list-like or a QueryCompiler, met: {type(self._by)}.",
                )
                internal_by = tuple(self._by.columns)

        self._internal_by_cache = internal_by
        return internal_by

    def __getitem__(self, key):
        """
        Implement indexing operation on a DataFrameGroupBy object.

        Parameters
        ----------
        key : list or str
            Names of columns to use as subset of original object.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            Result of indexing operation.

        Raises
        ------
        NotImplementedError
            Column lookups on GroupBy with arbitrary Series in by is not yet supported.
        """
        # These parameters are common for building the resulted Series or DataFrame groupby object
        kwargs = {
            **self._kwargs.copy(),
            "by": self._by,
            "axis": self._axis,
            "idx_name": self._idx_name,
            "squeeze": self._squeeze,
        }
        # The rules of type deduction for the resulted object is the following:
        #   1. If `key` is a list-like or `as_index is False`, then the resulted object is a DataFrameGroupBy
        #   2. Otherwise, the resulted object is SeriesGroupBy
        #   3. Result type does not depend on the `by` origin
        # Examples:
        #   - drop: any, as_index: any, __getitem__(key: list_like) -> DataFrameGroupBy
        #   - drop: any, as_index: False, __getitem__(key: any) -> DataFrameGroupBy
        #   - drop: any, as_index: True, __getitem__(key: label) -> SeriesGroupBy
        if is_list_like(key):
            make_dataframe = True
        else:
            if self._as_index:
                make_dataframe = False
            else:
                make_dataframe = True
                key = [key]
        if make_dataframe:
            internal_by = frozenset(self._internal_by)
            if len(internal_by.intersection(key)) != 0:
                ErrorMessage.missmatch_with_pandas(
                    operation="GroupBy.__getitem__",
                    message=(
                        "intersection of the selection and 'by' columns is not yet supported, "
                        "to achieve the desired result rewrite the original code from:\n"
                        "df.groupby('by_column')['by_column']\n"
                        "to the:\n"
                        "df.groupby(df['by_column'].copy())['by_column']"
                    ),
                )
            cols_to_grab = internal_by.union(key)
            key = [col for col in self._df.columns if col in cols_to_grab]
            return DataFrameGroupBy(
                self._df[key],
                drop=self._drop,
                **kwargs,
            )
        if (
            self._is_multi_by
            and isinstance(self._by, list)
            and not all(hashable(o) and o in self._df for o in self._by)
        ):
            raise NotImplementedError(
                "Column lookups on GroupBy with arbitrary Series in by"
                " is not yet supported."
            )
        return SeriesGroupBy(
            self._df[key],
            drop=False,
            **kwargs,
        )

    def cummin(self, axis=0, **kwargs):
        result = self._apply_agg_function(lambda df: df.cummin(axis=axis, **kwargs))
        # pandas does not name the index on cummin
        result._query_compiler.set_index_name(None)
        return result

    def bfill(self, limit=None):
        return self._default_to_pandas(lambda df: df.bfill(limit=limit))

    def idxmin(self):
        return self._default_to_pandas(lambda df: df.idxmin())

    def prod(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_prod,
            lambda df, **kwargs: df.prod(**kwargs),
            **kwargs,
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.std(ddof, *args, **kwargs))

    def aggregate(self, func=None, *args, **kwargs):
        if self._axis != 0:
            # This is not implemented in pandas,
            # so we throw a different message
            raise NotImplementedError("axis other than 0 is not supported")

        if (
            callable(func)
            and isinstance(func, BuiltinFunctionType)
            and func.__name__ in dir(self)
        ):
            func = func.__name__

        relabeling_required = False
        if isinstance(func, dict) or func is None:

            def try_get_str_func(fn):
                if not isinstance(fn, str) and isinstance(fn, Iterable):
                    return [try_get_str_func(f) for f in fn]
                return fn.__name__ if callable(fn) and fn.__name__ in dir(self) else fn

            relabeling_required, func_dict, new_columns, order = reconstruct_func(
                func, **kwargs
            )
            func_dict = {col: try_get_str_func(fn) for col, fn in func_dict.items()}

            if any(i not in self._df.columns for i in func_dict.keys()):
                from pandas.core.base import SpecificationError

                raise SpecificationError("nested renamer is not supported")
            if func is None:
                kwargs = {}
            func = func_dict
        elif is_list_like(func):
            return self._default_to_pandas(
                lambda df, *args, **kwargs: df.aggregate(func, *args, **kwargs),
                *args,
                **kwargs,
            )
        elif callable(func):
            return self._apply_agg_function(
                lambda grp, *args, **kwargs: grp.aggregate(func, *args, **kwargs),
                *args,
                **kwargs,
            )
        elif isinstance(func, str):
            # Using "getattr" here masks possible AttributeError which we throw
            # in __getattr__, so we should call __getattr__ directly instead.
            agg_func = self.__getattr__(func)
            if callable(agg_func):
                return agg_func(*args, **kwargs)

        result = self._apply_agg_function(
            func,
            *args,
            **kwargs,
        )

        if relabeling_required:
            if not self._as_index:
                nby_cols = len(result.columns) - len(new_columns)
                order = np.concatenate([np.arange(nby_cols), order + nby_cols])
                by_cols = result.columns[:nby_cols]
                new_columns = pandas.Index(new_columns)
                if by_cols.nlevels != new_columns.nlevels:
                    by_cols = by_cols.remove_unused_levels()
                    empty_levels = [
                        i
                        for i, level in enumerate(by_cols.levels)
                        if len(level) == 1 and level[0] == ""
                    ]
                    by_cols = by_cols.droplevel(empty_levels)
                new_columns = by_cols.append(new_columns)
            result = result.iloc[:, order]
            result.columns = new_columns
        return result

    agg = aggregate

    def last(self, **kwargs):
        return self._default_to_pandas(lambda df: df.last(**kwargs))

    def mad(self, **kwargs):
        return self._default_to_pandas(lambda df: df.mad(**kwargs))

    def rank(self, **kwargs):
        result = self._apply_agg_function(lambda df: df.rank(**kwargs))
        # pandas does not name the index on rank
        result._query_compiler.set_index_name(None)
        return result

    @property
    def corrwith(self):
        return self._default_to_pandas(lambda df: df.corrwith)

    def pad(self, limit=None):
        return self._default_to_pandas(lambda df: df.pad(limit=limit))

    def max(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_max,
            lambda df, **kwargs: df.max(**kwargs),
            numeric_only=False,
            **kwargs,
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.var(ddof, *args, **kwargs))

    def get_group(self, name, obj=None):
        return self._default_to_pandas(lambda df: df.get_group(name, obj=obj))

    def __len__(self):
        return len(self._index_grouped)

    def all(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_all,
            lambda df, **kwargs: df.all(**kwargs),
            numeric_only=False,
            **kwargs,
        )

    def size(self):
        if self._axis == 1:
            return DataFrameGroupBy(
                self._df.T.iloc[:, [0]],
                self._by,
                0,
                drop=self._drop,
                idx_name=self._idx_name,
                squeeze=self._squeeze,
                **self._kwargs,
            ).size()
        work_object = type(self)(
            self._df,
            self._by,
            self._axis,
            drop=False,
            idx_name=None,
            squeeze=self._squeeze,
            **self._kwargs,
        )
        result = work_object._wrap_aggregation(
            type(work_object._query_compiler).groupby_size,
            lambda df: df.size(),
            numeric_only=False,
        )
        if not isinstance(result, Series):
            result = result.squeeze(axis=1)
        if not self._kwargs.get("as_index") and not isinstance(result, Series):
            result = result.rename(columns={0: "size"})
            result = (
                result.rename(columns={"__reduced__": "index"})
                if "__reduced__" in result.columns
                else result
            )
        elif isinstance(self._df, Series):
            result.name = self._df.name
        else:
            result.name = None
        return result.fillna(0)

    def sum(self, **kwargs):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_sum,
            lambda df, **kwargs: df.sum(**kwargs),
            **kwargs,
        )

    def describe(self, **kwargs):
        return self._default_to_pandas(lambda df: df.describe(**kwargs))

    def boxplot(
        self,
        grouped,
        subplots=True,
        column=None,
        fontsize=None,
        rot=0,
        grid=True,
        ax=None,
        figsize=None,
        layout=None,
        **kwargs,
    ):
        return self._default_to_pandas(
            lambda df: df.boxplot(
                grouped,
                subplots=subplots,
                column=column,
                fontsize=fontsize,
                rot=rot,
                grid=grid,
                ax=ax,
                figsize=figsize,
                layout=layout,
                **kwargs,
            )
        )

    def ngroup(self, ascending=True):
        return self._default_to_pandas(lambda df: df.ngroup(ascending))

    def nunique(self, dropna=True):
        return self._apply_agg_function(lambda df: df.nunique(dropna))

    def resample(self, rule, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.resample(rule, *args, **kwargs))

    def median(self, **kwargs):
        return self._apply_agg_function(lambda df: df.median(**kwargs))

    def head(self, n=5):
        return self._default_to_pandas(lambda df: df.head(n))

    def cumprod(self, axis=0, *args, **kwargs):
        result = self._apply_agg_function(lambda df: df.cumprod(axis, *args, **kwargs))
        # pandas does not name the index on cumprod
        result._query_compiler.set_index_name(None)
        return result

    def __iter__(self):
        return self._iter.__iter__()

    def cov(self):
        return self._default_to_pandas(lambda df: df.cov())

    def transform(self, func, *args, **kwargs):
        result = self._apply_agg_function(
            lambda df: df.transform(func, *args, **kwargs)
        )
        # pandas does not name the index on transform
        result._query_compiler.set_index_name(None)
        return result

    def corr(self, **kwargs):
        return self._default_to_pandas(lambda df: df.corr(**kwargs))

    def fillna(self, **kwargs):
        new_groupby_kwargs = self._kwargs.copy()
        new_groupby_kwargs["as_index"] = True
        work_object = type(self)(
            df=self._df,
            by=self._by,
            axis=self._axis,
            idx_name=self._idx_name,
            drop=self._drop,
            squeeze=self._squeeze,
            **new_groupby_kwargs,
        )
        result = work_object._apply_agg_function(lambda df: df.fillna(**kwargs))
        # pandas does not name the index on fillna
        result._query_compiler.set_index_name(None)
        return result

    def count(self, **kwargs):
        result = self._wrap_aggregation(
            type(self._query_compiler).groupby_count,
            lambda df, **kwargs: df.count(**kwargs),
            numeric_only=False,
            **kwargs,
        )
        # pandas do it in case of Series
        if isinstance(result, Series):
            result = result.fillna(0)
        return result

    def pipe(self, func, *args, **kwargs):
        return com.pipe(self, func, *args, **kwargs)

    def cumcount(self, ascending=True):
        result = self._default_to_pandas(lambda df: df.cumcount(ascending=ascending))
        # pandas does not name the index on cumcount
        result._query_compiler.set_index_name(None)
        return result

    def tail(self, n=5):
        return self._default_to_pandas(lambda df: df.tail(n))

    # expanding and rolling are unique cases and need to likely be handled
    # separately. They do not appear to be commonly used.
    def expanding(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.expanding(*args, **kwargs))

    def rolling(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.rolling(*args, **kwargs))

    def hist(self):
        return self._default_to_pandas(lambda df: df.hist())

    def quantile(self, q=0.5, **kwargs):
        if is_list_like(q):
            return self._default_to_pandas(lambda df: df.quantile(q=q, **kwargs))

        return self._apply_agg_function(lambda df: df.quantile(q, **kwargs))

    def diff(self):
        return self._default_to_pandas(lambda df: df.diff())

    def take(self, **kwargs):
        return self._default_to_pandas(lambda df: df.take(**kwargs))

    @property
    def _index(self):
        """
        Get index value.

        Returns
        -------
        pandas.Index
            Index value.
        """
        return self._query_compiler.index

    @property
    def _sort(self):
        """
        Get sort parameter value.

        Returns
        -------
        bool
            Value of sort parameter used to create DataFrameGroupBy object.
        """
        return self._kwargs.get("sort")

    @property
    def _as_index(self):
        """
        Get as_index parameter value.

        Returns
        -------
        bool
            Value of as_index parameter used to create DataFrameGroupBy object.
        """
        return self._kwargs.get("as_index")

    @property
    def _iter(self):
        """
        Construct a tuple of (group_id, DataFrame) tuples to allow iteration over groups.

        Returns
        -------
        generator
            Generator expression of GroupBy object broken down into tuples for iteration.
        """
        from .dataframe import DataFrame

        group_ids = self._index_grouped.keys()
        if self._axis == 0:
            return (
                (
                    k,
                    DataFrame(
                        query_compiler=self._query_compiler.getitem_row_array(
                            self._index.get_indexer_for(self._index_grouped[k].unique())
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )
        else:
            return (
                (
                    k,
                    DataFrame(
                        query_compiler=self._query_compiler.getitem_column_array(
                            self._index_grouped[k].unique()
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )

    @property
    def _index_grouped(self):
        """
        Construct an index of group IDs.

        Returns
        -------
        dict
            A dict of {group name -> group labels} values.

        See Also
        --------
        pandas.core.groupby.GroupBy.groups
        """
        if self._index_grouped_cache is None:
            # Splitting level-by and column-by since we serialize them in a different ways
            by = None
            level = []
            if self._level is not None:
                level = self._level
                if not isinstance(level, list):
                    level = [level]
            elif isinstance(self._by, list):
                by = []
                for o in self._by:
                    if hashable(o) and o in self._query_compiler.get_index_names(
                        self._axis
                    ):
                        level.append(o)
                    else:
                        by.append(o)
            else:
                by = self._by

            is_multi_by = self._is_multi_by or (by is not None and len(level) > 0)

            if hasattr(self._by, "columns") and is_multi_by:
                by = list(self._by.columns)

            if is_multi_by:
                # Because we are doing a collect (to_pandas) here and then groupby, we
                # end up using pandas implementation. Add the warning so the user is
                # aware.
                ErrorMessage.catch_bugs_and_request_email(self._axis == 1)
                ErrorMessage.default_to_pandas("Groupby with multiple columns")
                if isinstance(by, list) and all(
                    is_label(self._df, o, self._axis) for o in by
                ):
                    pandas_df = self._df._query_compiler.getitem_column_array(
                        by
                    ).to_pandas()
                else:
                    by = try_cast_to_pandas(by, squeeze=True)
                    pandas_df = self._df._to_pandas()
                by = wrap_into_list(by, level)
                self._index_grouped_cache = pandas_df.groupby(by=by).groups
            else:
                if isinstance(self._by, type(self._query_compiler)):
                    by = self._by.to_pandas().squeeze().values
                elif self._by is None:
                    index = self._query_compiler.get_axis(self._axis)
                    levels_to_drop = [
                        i
                        for i, name in enumerate(index.names)
                        if name not in level and i not in level
                    ]
                    by = index.droplevel(levels_to_drop)
                    if isinstance(by, pandas.MultiIndex):
                        by = by.reorder_levels(level)
                else:
                    by = self._by
                if self._axis == 0:
                    self._index_grouped_cache = self._index.groupby(by)
                else:
                    self._index_grouped_cache = self._columns.groupby(by)
        return self._index_grouped_cache

    def _wrap_aggregation(
        self, qc_method, default_func, drop=True, numeric_only=True, **kwargs
    ):
        """
        Perform common metadata transformations and apply groupby functions.

        Parameters
        ----------
        qc_method : callable
            The query compiler method to call.
        default_func : callable
            The function to call if we need to default to pandas.
        drop : bool, default: True
            Whether or not to the grouping columns should be dropped on this operation.
        numeric_only : bool, default: True
            True for numeric only computations, False otherwise.
        **kwargs : dict
            The keyword arguments to be passed to the calling function.

        Returns
        -------
        DataFrame or Series
            Returns the same type as `self._df`.
        """
        if self._axis != 0:
            return self._default_to_pandas(default_func, **kwargs)
        # For aggregations, pandas behavior does this for the result.
        # For other operations it does not, so we wait until there is an aggregation to
        # actually perform this operation.
        if not self._is_multi_by and drop and self._drop and self._as_index:
            groupby_qc = self._query_compiler.drop(columns=self._by.columns)
        else:
            groupby_qc = self._query_compiler

        result = type(self._df)(
            query_compiler=qc_method(
                groupby_qc,
                by=self._by,
                axis=self._axis,
                groupby_args=self._kwargs,
                map_args=kwargs,
                reduce_args=kwargs,
                numeric_only=numeric_only,
                drop=self._drop,
            )
        )
        if self._squeeze:
            return result.squeeze()
        return result

    def _apply_agg_function(self, f, *args, **kwargs):
        """
        Perform aggregation and combine stages based on a given function.

        Parameters
        ----------
        f : callable
            The function to apply to each group.
        *args : list
            Extra positional arguments to pass to `f`.
        **kwargs : dict
            Extra keyword arguments to pass to `f`.

        Returns
        -------
        DataFrame
            A new combined DataFrame with the result of all groups.
        """
        assert callable(f) or isinstance(
            f, dict
        ), "'{0}' object is not callable and not a dict".format(type(f))

        new_manager = self._query_compiler.groupby_agg(
            by=self._by,
            is_multi_by=self._is_multi_by,
            axis=self._axis,
            agg_func=f,
            agg_args=args,
            agg_kwargs=kwargs,
            groupby_kwargs=self._kwargs,
            drop=self._drop,
        )
        if self._idx_name is not None and self._as_index:
            new_manager.set_index_name(self._idx_name)
        result = type(self._df)(query_compiler=new_manager)
        if result._query_compiler.get_index_name() == "__reduced__":
            result._query_compiler.set_index_name(None)
        if self._squeeze:
            return result.squeeze()
        return result

    def _default_to_pandas(self, f, *args, **kwargs):
        """
        Execute function `f` in default-to-pandas way.

        Parameters
        ----------
        f : callable
            The function to apply to each group.
        *args : list
            Extra positional arguments to pass to `f`.
        **kwargs : dict
            Extra keyword arguments to pass to `f`.

        Returns
        -------
        modin.pandas.DataFrame
            A new Modin DataFrame with the result of the pandas function.
        """
        if (
            isinstance(self._by, type(self._query_compiler))
            and len(self._by.columns) == 1
        ):
            by = self._by.columns[0] if self._drop else self._by.to_pandas().squeeze()
        # converting QC 'by' to a list of column labels only if this 'by' comes from the self (if drop is True)
        elif self._drop and isinstance(self._by, type(self._query_compiler)):
            by = list(self._by.columns)
        else:
            by = self._by

        by = try_cast_to_pandas(by, squeeze=True)
        # Since 'by' may be a 2D query compiler holding columns to group by,
        # to_pandas will also produce a pandas DataFrame containing them.
        # So splitting 2D 'by' into a list of 1D Series using 'GroupBy.validate_by':
        by = GroupBy.validate_by(by)

        def groupby_on_multiple_columns(df, *args, **kwargs):
            return f(
                df.groupby(
                    by=by, axis=self._axis, squeeze=self._squeeze, **self._kwargs
                ),
                *args,
                **kwargs,
            )

        return self._df._default_to_pandas(groupby_on_multiple_columns, *args, **kwargs)


@_inherit_docstrings(pandas.core.groupby.SeriesGroupBy)
class SeriesGroupBy(DataFrameGroupBy):
    @property
    def ndim(self):
        """
        Return 1.

        Returns
        -------
        int
            Returns 1.

        Notes
        -----
        Deprecated and removed in pandas and will be likely removed in Modin.
        """
        return 1  # ndim is always 1 for Series

    @property
    def _iter(self):
        """
        Construct a tuple of (group_id, Series) tuples to allow iteration over groups.

        Returns
        -------
        generator
            Generator expression of GroupBy object broken down into tuples for iteration.
        """
        group_ids = self._index_grouped.keys()
        if self._axis == 0:
            return (
                (
                    k,
                    Series(
                        query_compiler=self._query_compiler.getitem_row_array(
                            self._index.get_indexer_for(self._index_grouped[k].unique())
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )
        else:
            return (
                (
                    k,
                    Series(
                        query_compiler=self._query_compiler.getitem_column_array(
                            self._index_grouped[k].unique()
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )


if IsExperimental.get():
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(DataFrameGroupBy, "make_dataframe_groupby_wrapper")
