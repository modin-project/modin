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

import warnings

import numpy as np
import pandas
from pandas.core.apply import reconstruct_func
from pandas.errors import SpecificationError
import pandas.core.groupby
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype, is_integer
from pandas._libs.lib import no_default
import pandas.core.common as com
from types import BuiltinFunctionType
from collections.abc import Iterable

from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import (
    _inherit_docstrings,
    try_cast_to_pandas,
    wrap_udf_function,
    hashable,
    wrap_into_list,
    MODIN_UNNAMED_SERIES_LABEL,
)
from modin.pandas.utils import cast_function_modin2pandas
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.config import IsExperimental
from .series import Series
from .utils import is_label


_DEFAULT_BEHAVIOUR = {
    "__class__",
    "__getitem__",
    "__init__",
    "__iter__",
    "_as_index",
    "_axis",
    "_by",
    "_check_index",
    "_check_index_name",
    "_columns",
    "_compute_index_grouped",
    "_default_to_pandas",
    "_df",
    "_drop",
    "_groups_cache",
    "_idx_name",
    "_index",
    "_indices_cache",
    "_internal_by",
    "_internal_by_cache",
    "_is_multi_by",
    "_iter",
    "_kwargs",
    "_level",
    "_pandas_class",
    "_query_compiler",
    "_sort",
    "_squeeze",
    "_wrap_aggregation",
}


@_inherit_docstrings(pandas.core.groupby.DataFrameGroupBy)
class DataFrameGroupBy(ClassLogger):
    _pandas_class = pandas.core.groupby.DataFrameGroupBy

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

    def __override(self, **kwargs):
        new_kw = dict(
            df=self._df,
            by=self._by,
            axis=self._axis,
            squeeze=self._squeeze,
            idx_name=self._idx_name,
            drop=self._drop,
            **self._kwargs,
        )
        new_kw.update(kwargs)
        return type(self)(**new_kw)

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
        except AttributeError as err:
            if key in self._columns:
                return self.__getitem__(key)
            raise err

    # TODO: `.__getattribute__` overriding is broken in experimental mode. We should
    # remove this branching one it's fixed:
    # https://github.com/modin-project/modin/issues/5536
    if not IsExperimental.get():

        def __getattribute__(self, item):
            attr = super().__getattribute__(item)
            if (
                item not in _DEFAULT_BEHAVIOUR
                and not self._query_compiler.lazy_execution
            ):
                # We default to pandas on empty DataFrames. This avoids a large amount of
                # pain in underlying implementation and returns a result immediately rather
                # than dealing with the edge cases that empty DataFrames have.
                if (
                    callable(attr)
                    and self._df.empty
                    and hasattr(self._pandas_class, item)
                ):

                    def default_handler(*args, **kwargs):
                        return self._default_to_pandas(item, *args, **kwargs)

                    return default_handler
            return attr

    @property
    def ngroups(self):
        return len(self)

    def skew(self, *args, **kwargs):
        # The 'skew' aggregation is less tolerant to non-numeric columns than others
        # (i.e. it doesn't allow numeric categoricals), thus dropping non-numeric
        # columns here since `._wrap_aggregation(numeric_only=True, ...)` is not enough
        if self.ndim == 2:
            by_cols = self._internal_by
            mask_cols = [
                col
                for col, dtype in self._df.dtypes.items()
                if is_numeric_dtype(dtype) or col in by_cols
            ]
            if not self._df.columns.equals(mask_cols):
                masked_df = self._df[mask_cols]
                masked_obj = type(self)(
                    df=masked_df,
                    by=self._by,
                    axis=self._axis,
                    idx_name=self._idx_name,
                    drop=self._drop,
                    squeeze=self._squeeze,
                    **self._kwargs,
                )
            else:
                masked_obj = self
        else:
            masked_obj = self

        return masked_obj._wrap_aggregation(
            type(masked_obj._query_compiler).groupby_skew,
            agg_args=args,
            agg_kwargs=kwargs,
            # Don't want to try to drop non-numeric columns for the second time
            numeric_only=False,
        )

    def ffill(self, limit=None):
        ErrorMessage.single_warning(
            ".ffill() is implemented using .fillna() in Modin, "
            + "which can be impacted by pandas bug https://github.com/pandas-dev/pandas/issues/43412 "
            + "on dataframes with duplicated indices"
        )
        return self.fillna(limit=limit, method="ffill")

    def sem(self, ddof=1, numeric_only=no_default):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_sem,
            agg_kwargs=dict(ddof=ddof),
            numeric_only=numeric_only,
        )

    def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
        return self._default_to_pandas(
            lambda df: df.sample(
                n=n,
                frac=frac,
                replace=replace,
                weights=weights,
                random_state=random_state,
            )
        )

    def ewm(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.ewm(*args, **kwargs))

    def value_counts(
        self,
        subset=None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ):
        return self._default_to_pandas(
            lambda df: df.value_counts(
                subset=subset,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                dropna=dropna,
            )
        )

    def mean(self, numeric_only=no_default, engine="cython", engine_kwargs=None):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.mean(
                    numeric_only=numeric_only,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            )
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_mean,
                agg_kwargs=dict(
                    numeric_only=None if numeric_only is no_default else numeric_only,
                ),
                numeric_only=numeric_only,
            )
        )

    def any(self, skipna=True):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_any,
            numeric_only=False,
            agg_kwargs=dict(skipna=skipna),
        )

    @property
    def plot(self):  # pragma: no cover
        return self._default_to_pandas(lambda df: df.plot)

    def ohlc(self):
        from .dataframe import DataFrame

        return DataFrame(
            query_compiler=self._query_compiler.groupby_ohlc(
                by=self._by,
                axis=self._axis,
                groupby_kwargs=self._kwargs,
                agg_args=[],
                agg_kwargs={},
                is_df=isinstance(self._df, DataFrame),
            ),
        )

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

    _groups_cache = no_default

    # TODO: since python 3.9:
    # @cached_property
    @property
    def groups(self):
        if self._groups_cache is not no_default:
            return self._groups_cache

        self._groups_cache = self._compute_index_grouped(numerical=False)
        return self._groups_cache

    def min(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.min(
                    numeric_only=numeric_only,
                    min_count=min_count,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            )
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_min,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
        )

    def max(self, numeric_only=False, min_count=-1, engine=None, engine_kwargs=None):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.max(
                    numeric_only=numeric_only,
                    min_count=min_count,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            )
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_max,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
        )

    def idxmax(self, axis=0, skipna=True, numeric_only=no_default):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_idxmax,
            agg_kwargs=dict(axis=axis, skipna=skipna),
            numeric_only=numeric_only,
        )

    def idxmin(self, axis=0, skipna=True, numeric_only=no_default):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_idxmin,
            agg_kwargs=dict(axis=axis, skipna=skipna),
            numeric_only=numeric_only,
        )

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
            result = result.dropna(subset=self._by.columns)
            if self._sort:
                result = result.sort_values(list(self._by.columns), axis=axis)
            else:
                result = result.sort_index()
        else:
            result = self._check_index_name(
                self._wrap_aggregation(
                    type(self._query_compiler).groupby_shift,
                    numeric_only=False,
                    agg_kwargs=dict(
                        periods=periods, freq=freq, axis=axis, fill_value=fill_value
                    ),
                )
            )
        return result

    def nth(self, n, dropna=None):
        # TODO: what we really should do is create a GroupByNthSelector to mimic
        # pandas behavior and then implement some of these methods there.
        # Adapted error checking from pandas
        if dropna:
            if not is_integer(n):
                raise ValueError("dropna option only supported for an integer argument")

            if dropna not in ("any", "all"):
                # Note: when agg-ing picker doesn't raise this, just returns NaN
                raise ValueError(
                    "For a DataFrame or Series groupby.nth, dropna must be "
                    + "either None, 'any' or 'all', "
                    + f"(was passed {dropna})."
                )

        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_nth,
                numeric_only=False,
                agg_kwargs=dict(n=n, dropna=dropna),
            )
        )

    def cumsum(self, axis=0, *args, **kwargs):
        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_cumsum,
                agg_args=args,
                agg_kwargs=dict(axis=axis, **kwargs),
                numeric_only=True,
            )
        )

    _indices_cache = no_default

    # TODO: since python 3.9:
    # @cached_property
    @property
    def indices(self):
        if self._indices_cache is not no_default:
            return self._indices_cache

        self._indices_cache = self._compute_index_grouped(numerical=True)
        return self._indices_cache

    @_inherit_docstrings(pandas.core.groupby.DataFrameGroupBy.pct_change)
    def pct_change(self, periods=1, fill_method="ffill", limit=None, freq=None, axis=0):
        from .dataframe import DataFrame

        # Should check for API level errors
        # Attempting to match pandas error behavior here
        if not isinstance(periods, int):
            raise TypeError(f"periods must be an int. got {type(periods)} instead")

        if isinstance(self._df, Series):
            if not is_numeric_dtype(self._df.dtypes):
                raise TypeError(
                    f"unsupported operand type for -: got {self._df.dtypes}"
                )
        elif isinstance(self._df, DataFrame) and axis == 0:
            for col, dtype in self._df.dtypes.items():
                # can't calculate change on non-numeric columns, so check for
                # non-numeric columns that are not included in the `by`
                if not is_numeric_dtype(dtype) and not (
                    isinstance(self._by, BaseQueryCompiler) and col in self._by.columns
                ):
                    raise TypeError(f"unsupported operand type for -: got {dtype}")

        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_pct_change,
                agg_kwargs=dict(
                    periods=periods,
                    fill_method=fill_method,
                    limit=limit,
                    freq=freq,
                    axis=axis,
                ),
            )
        )

    def filter(self, func, dropna=True, *args, **kwargs):
        return self._default_to_pandas(
            lambda df: df.filter(func, dropna=dropna, *args, **kwargs)
        )

    def cummax(self, axis=0, numeric_only=False, **kwargs):
        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_cummax,
                agg_kwargs=dict(axis=axis, **kwargs),
                numeric_only=numeric_only,
            )
        )

    def apply(self, func, *args, **kwargs):
        func = cast_function_modin2pandas(func)
        if not isinstance(func, BuiltinFunctionType):
            func = wrap_udf_function(func)

        return self._check_index(
            self._wrap_aggregation(
                qc_method=type(self._query_compiler).groupby_agg,
                numeric_only=False,
                agg_func=func,
                agg_args=args,
                agg_kwargs=kwargs,
                how="group_wise",
            )
        )

    @property
    def dtypes(self):
        if self._axis == 1:
            raise ValueError("Cannot call dtypes on groupby with axis=1")
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_dtypes,
                numeric_only=False,
            )
        )

    def first(self, numeric_only=False, min_count=-1):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_first,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
        )

    def last(self, numeric_only=False, min_count=-1):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_last,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
        )

    def backfill(self, limit=None):
        warnings.warn(
            (
                "backfill is deprecated and will be removed in a future version. "
                + "Use bfill instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
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
                internal_by_list = []
                for by in self._by:
                    if isinstance(by, str):
                        internal_by_list.append(by)
                    elif isinstance(by, pandas.Grouper):
                        internal_by_list.append(by.key)
                internal_by = tuple(internal_by_list)
            elif isinstance(self._by, pandas.Grouper):
                internal_by = tuple([self._by.key])
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=not isinstance(self._by, BaseQueryCompiler),
                    extra_log=f"When 'drop' is True, 'by' must be either list-like, Grouper, or a QueryCompiler, met: {type(self._by)}.",
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
                        + "to achieve the desired result rewrite the original code from:\n"
                        + "df.groupby('by_column')['by_column']\n"
                        + "to the:\n"
                        + "df.groupby(df['by_column'].copy())['by_column']"
                    ),
                )
            # We need to maintain order of the columns in key, using a set doesn't
            # maintain order.
            # We use dictionaries since they maintain insertion order as of 3.7,
            # and its faster to call dict.update than it is to loop through `key`
            # and select only the elements which aren't in `cols_to_grab`.
            cols_to_grab = dict.fromkeys(self._internal_by)
            cols_to_grab.update(dict.fromkeys(key))
            key = [col for col in cols_to_grab.keys() if col in self._df.columns]
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
                + " is not yet supported."
            )
        return SeriesGroupBy(
            self._df[key],
            drop=False,
            **kwargs,
        )

    def cummin(self, axis=0, numeric_only=False, **kwargs):
        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_cummin,
                agg_kwargs=dict(axis=axis, **kwargs),
                numeric_only=numeric_only,
            )
        )

    def bfill(self, limit=None):
        ErrorMessage.single_warning(
            ".bfill() is implemented using .fillna() in Modin, "
            + "which can be impacted by pandas bug https://github.com/pandas-dev/pandas/issues/43412 "
            + "on dataframes with duplicated indices"
        )
        return self.fillna(limit=limit, method="bfill")

    def prod(self, numeric_only=no_default, min_count=0):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_prod,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
        )

    def std(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=no_default):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.std(
                    ddof=ddof,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                    numeric_only=numeric_only,
                )
            )
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_std,
            agg_kwargs=dict(ddof=ddof),
            numeric_only=numeric_only,
        )

    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.aggregate(
                    func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
                )
            )
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

        do_relabel = None
        if isinstance(func, dict) or func is None:
            relabeling_required, func_dict, new_columns, order = reconstruct_func(
                func, **kwargs
            )

            if relabeling_required:

                def do_relabel(obj_to_relabel):
                    new_order, new_columns_idx = order, pandas.Index(new_columns)
                    if not self._as_index:
                        nby_cols = len(obj_to_relabel.columns) - len(new_columns_idx)
                        new_order = np.concatenate(
                            [np.arange(nby_cols), new_order + nby_cols]
                        )
                        by_cols = obj_to_relabel.columns[:nby_cols]
                        if by_cols.nlevels != new_columns_idx.nlevels:
                            by_cols = by_cols.remove_unused_levels()
                            empty_levels = [
                                i
                                for i, level in enumerate(by_cols.levels)
                                if len(level) == 1 and level[0] == ""
                            ]
                            by_cols = by_cols.droplevel(empty_levels)
                        new_columns_idx = by_cols.append(new_columns_idx)
                    result = obj_to_relabel.iloc[:, new_order]
                    result.columns = new_columns_idx
                    return result

            if any(isinstance(fn, list) for fn in func_dict.values()):
                # multicolumn case
                # putting functions in a `list` allows to achieve multicolumn in each partition
                func_dict = {
                    col: fn if isinstance(fn, list) else [fn]
                    for col, fn in func_dict.items()
                }
            if (
                relabeling_required
                and not self._as_index
                and any(col in func_dict for col in self._internal_by)
            ):
                ErrorMessage.missmatch_with_pandas(
                    operation="GroupBy.aggregate(**dictionary_renaming_aggregation)",
                    message=(
                        "intersection of the columns to aggregate and 'by' is not yet supported when 'as_index=False', "
                        + "columns with group names of the intersection will not be presented in the result. "
                        + "To achieve the desired result rewrite the original code from:\n"
                        + "df.groupby('by_column', as_index=False).agg(agg_func=('by_column', agg_func))\n"
                        + "to the:\n"
                        + "df.groupby('by_column').agg(agg_func=('by_column', agg_func)).reset_index()"
                    ),
                )

            if any(i not in self._df.columns for i in func_dict.keys()):
                raise SpecificationError("nested renamer is not supported")
            if func is None:
                kwargs = {}
            func = func_dict
        elif is_list_like(func):
            # for list-list aggregation pandas always puts
            # groups as index in the result, ignoring as_index,
            # so we have to reset it to default value
            return self.__override(as_index=True)._wrap_aggregation(
                qc_method=type(self._query_compiler).groupby_agg,
                numeric_only=False,
                agg_func=func,
                agg_args=args,
                agg_kwargs=kwargs,
                how="axis_wise",
            )
        elif callable(func):
            return self._check_index(
                self._wrap_aggregation(
                    qc_method=type(self._query_compiler).groupby_agg,
                    numeric_only=False,
                    agg_func=func,
                    agg_args=args,
                    agg_kwargs=kwargs,
                    how="axis_wise",
                )
            )
        elif isinstance(func, str):
            # Using "getattr" here masks possible AttributeError which we throw
            # in __getattr__, so we should call __getattr__ directly instead.
            agg_func = self.__getattr__(func)
            if callable(agg_func):
                return agg_func(*args, **kwargs)

        result = self._wrap_aggregation(
            qc_method=type(self._query_compiler).groupby_agg,
            numeric_only=False,
            agg_func=func,
            agg_args=args,
            agg_kwargs=kwargs,
            how="axis_wise",
        )
        return do_relabel(result) if do_relabel else result

    agg = aggregate

    def mad(self, **kwargs):
        warnings.warn(
            (
                "The 'mad' method is deprecated and will be removed in a future version. "
                + "To compute the same result, you may do `(df - df.mean()).abs().mean()`."
            ),
            FutureWarning,
            stacklevel=2,
        )
        return self._default_to_pandas(lambda df: df.mad(**kwargs))

    def rank(
        self, method="average", ascending=True, na_option="keep", pct=False, axis=0
    ):
        result = self._wrap_aggregation(
            type(self._query_compiler).groupby_rank,
            agg_kwargs=dict(
                method=method,
                ascending=ascending,
                na_option=na_option,
                pct=pct,
                axis=axis,
            ),
            numeric_only=False,
        )
        # pandas does not name the index on rank
        result._query_compiler.set_index_name(None)
        return result

    @property
    def corrwith(self):
        return self._default_to_pandas(lambda df: df.corrwith)

    def pad(self, limit=None):
        ErrorMessage.single_warning(
            ".pad() is implemented using .fillna() in Modin, "
            + "which can be impacted by pandas bug https://github.com/pandas-dev/pandas/issues/43412 "
            + "on dataframes with duplicated indices"
        )
        return self.fillna(limit=limit, method="pad")

    def var(self, ddof=1, engine=None, engine_kwargs=None, numeric_only=no_default):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.var(
                    ddof=ddof,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                    numeric_only=numeric_only,
                )
            )
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_var,
            agg_kwargs=dict(ddof=ddof),
            numeric_only=numeric_only,
        )

    def get_group(self, name, obj=None):
        work_object = self.__override(
            df=obj if obj is not None else self._df, as_index=True
        )

        return work_object._check_index(
            work_object._wrap_aggregation(
                qc_method=type(work_object._query_compiler).groupby_get_group,
                numeric_only=False,
                agg_kwargs=dict(name=name),
            )
        )

    def __len__(self):
        return len(self.indices)

    def all(self, skipna=True):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_all,
            numeric_only=False,
            agg_kwargs=dict(skipna=skipna),
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
            numeric_only=False,
        )
        if not isinstance(result, Series):
            result = result.squeeze(axis=1)
        if not self._kwargs.get("as_index") and not isinstance(result, Series):
            result = (
                result.rename(columns={MODIN_UNNAMED_SERIES_LABEL: "index"})
                if MODIN_UNNAMED_SERIES_LABEL in result.columns
                else result
            )
        elif isinstance(self._df, Series):
            result.name = self._df.name
        else:
            result.name = None
        return result

    def sum(
        self, numeric_only=no_default, min_count=0, engine=None, engine_kwargs=None
    ):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.sum(
                    numeric_only=numeric_only,
                    min_count=min_count,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            )

        return self._wrap_aggregation(
            type(self._query_compiler).groupby_sum,
            agg_kwargs=dict(min_count=min_count),
            numeric_only=numeric_only,
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
        sharex=False,
        sharey=True,
        backend=None,
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
                sharex=sharex,
                sharey=sharey,
                backend=backend,
                **kwargs,
            )
        )

    def ngroup(self, ascending=True):
        result = self._wrap_aggregation(
            type(self._query_compiler).groupby_ngroup,
            numeric_only=False,
            agg_kwargs=dict(ascending=ascending),
        )
        if not isinstance(result, Series):
            # The result should always be a Series with name None and type int64
            result = result.squeeze(axis=1)
        # TODO: this might not hold in the future
        result.name = None
        return result

    def nunique(self, dropna=True):
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_nunique,
                numeric_only=False,
                agg_kwargs=dict(dropna=dropna),
            )
        )

    def resample(self, rule, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.resample(rule, *args, **kwargs))

    def median(self, numeric_only=no_default):
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_median,
                numeric_only=numeric_only,
            )
        )

    def head(self, n=5):
        # groupby().head()/.tail() ignore as_index, so override it to True
        work_object = self.__override(as_index=True)

        return work_object._check_index(
            work_object._wrap_aggregation(
                type(work_object._query_compiler).groupby_head,
                agg_kwargs=dict(n=n),
                numeric_only=False,
            )
        )

    def cumprod(self, axis=0, *args, **kwargs):
        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_cumprod,
                agg_args=args,
                agg_kwargs=dict(axis=axis, **kwargs),
                numeric_only=True,
            )
        )

    def __iter__(self):
        return self._iter.__iter__()

    def cov(self, min_periods=None, ddof=1, numeric_only=True):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_cov,
            agg_kwargs=dict(min_periods=min_periods, ddof=ddof),
            numeric_only=numeric_only,
        )

    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        if engine not in ("cython", None) and engine_kwargs is not None:
            return self._default_to_pandas(
                lambda df: df.transform(
                    func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
                )
            )

        return self._check_index_name(
            self._wrap_aggregation(
                qc_method=type(self._query_compiler).groupby_agg,
                numeric_only=False,
                agg_func=func,
                agg_args=args,
                agg_kwargs=kwargs,
                how="transform",
            )
        )

    def corr(self, method="pearson", min_periods=None, numeric_only=True):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_corr,
            agg_kwargs=dict(method=method, min_periods=min_periods),
            numeric_only=numeric_only,
        )

    def fillna(self, *args, **kwargs):
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
        return work_object._check_index_name(
            work_object._wrap_aggregation(
                type(self._query_compiler).groupby_fillna,
                numeric_only=False,
                agg_args=args,
                agg_kwargs=kwargs,
            )
        )

    def count(self):
        return self._wrap_aggregation(
            type(self._query_compiler).groupby_count,
            numeric_only=False,
        )

    def pipe(self, func, *args, **kwargs):
        return com.pipe(self, func, *args, **kwargs)

    def cumcount(self, ascending=True):
        result = self._wrap_aggregation(
            type(self._query_compiler).groupby_cumcount,
            numeric_only=False,
            agg_kwargs=dict(ascending=ascending),
        )
        if not isinstance(result, Series):
            # The result should always be a Series with name None and type int64
            result = result.squeeze(axis=1)
            result.name = None
        return result

    def tail(self, n=5):
        # groupby().head()/.tail() ignore as_index, so override it to True
        work_object = self.__override(as_index=True)
        return work_object._check_index(
            work_object._wrap_aggregation(
                type(work_object._query_compiler).groupby_tail,
                agg_kwargs=dict(n=n),
                numeric_only=False,
            )
        )

    # expanding and rolling are unique cases and need to likely be handled
    # separately. They do not appear to be commonly used.
    def expanding(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.expanding(*args, **kwargs))

    def rolling(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.rolling(*args, **kwargs))

    def hist(self):
        return self._default_to_pandas(lambda df: df.hist())

    def quantile(self, q=0.5, interpolation="linear", numeric_only=no_default):
        # TODO: handle list-like cases properly
        if numeric_only is no_default:
            numeric_only = True
        # We normally handle `numeric_only` by masking non-numeric columns; however
        # pandas errors out if there are only non-numeric columns and `numeric_only=True`
        # for groupby.quantile.
        if numeric_only:
            if all(
                [not is_numeric_dtype(dtype) for dtype in self._query_compiler.dtypes]
            ):
                raise TypeError(
                    f"'quantile' cannot be performed against '{self._query_compiler.dtypes[0]}' dtypes!"
                )
        if is_list_like(q):
            return self._default_to_pandas(
                lambda df: df.quantile(q=q, interpolation=interpolation)
            )

        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_quantile,
                numeric_only=numeric_only,
                agg_kwargs=dict(q=q, interpolation=interpolation),
            )
        )

    def diff(self, periods=1, axis=0):
        from .dataframe import DataFrame

        # Should check for API level errors
        # Attempting to match pandas error behavior here
        if not isinstance(periods, int):
            raise TypeError(f"periods must be an int. got {type(periods)} instead")

        if isinstance(self._df, Series):
            if not is_numeric_dtype(self._df.dtypes):
                raise TypeError(
                    f"unsupported operand type for -: got {self._df.dtypes}"
                )
        elif isinstance(self._df, DataFrame) and axis == 0:
            for col, dtype in self._df.dtypes.items():
                # can't calculate diff on non-numeric columns, so check for non-numeric
                # columns that are not included in the `by`
                if not is_numeric_dtype(dtype) and not (
                    isinstance(self._by, BaseQueryCompiler) and col in self._by.columns
                ):
                    raise TypeError(f"unsupported operand type for -: got {dtype}")

        return self._check_index_name(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_diff,
                agg_kwargs=dict(
                    periods=periods,
                    axis=axis,
                ),
            )
        )

    def take(self, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.take(*args, **kwargs))

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

        indices = self.indices
        group_ids = indices.keys()
        if self._axis == 0:
            return (
                (
                    k,
                    DataFrame(
                        query_compiler=self._query_compiler.getitem_row_array(
                            indices[k]
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
                            indices[k], numeric=True
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )

    def _compute_index_grouped(self, numerical=False):
        """
        Construct an index of group IDs.

        Parameters
        ----------
        numerical : bool, default: False
            Whether a group indices should be positional (True) or label-based (False).

        Returns
        -------
        dict
            A dict of {group name -> group indices} values.

        See Also
        --------
        pandas.core.groupby.GroupBy.groups
        """
        # We end up using pure pandas to compute group indices, so raising a warning
        ErrorMessage.default_to_pandas("Group indices computation")

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
        # `dropna` param is the only one that matters for the group indices result
        dropna = self._kwargs.get("dropna", True)

        if isinstance(self._by, BaseQueryCompiler) and is_multi_by:
            by = list(self._by.columns)

        if is_multi_by:
            # Because we are doing a collect (to_pandas) here and then groupby, we
            # end up using pandas implementation. Add the warning so the user is
            # aware.
            ErrorMessage.catch_bugs_and_request_email(self._axis == 1)
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
            groupby_obj = pandas_df.groupby(by=by, dropna=dropna)
            return groupby_obj.indices if numerical else groupby_obj.groups
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
            axis_labels = self._query_compiler.get_axis(self._axis)
            if numerical:
                # Since we want positional indices of the groups, we want to group
                # on a `RangeIndex`, not on the actual index labels
                axis_labels = pandas.RangeIndex(len(axis_labels))
            # `pandas.Index.groupby` doesn't take any parameters except `by`.
            # Have to convert an Index to a Series to be able to process `dropna=False`:
            if dropna:
                return axis_labels.groupby(by)
            else:
                groupby_obj = axis_labels.to_series().groupby(by, dropna=dropna)
                return groupby_obj.indices if numerical else groupby_obj.groups

    def _wrap_aggregation(
        self,
        qc_method,
        numeric_only=None,
        agg_args=None,
        agg_kwargs=None,
        **kwargs,
    ):
        """
        Perform common metadata transformations and apply groupby functions.

        Parameters
        ----------
        qc_method : callable
            The query compiler method to call.
        numeric_only : {None, True, False}, default: None
            Specifies whether to aggregate non numeric columns:
                - True: include only numeric columns (including categories that holds a numeric dtype)
                - False: include all columns
                - None: infer the parameter, ``False`` if there are no numeric types in the frame,
                  ``True`` otherwise.
        agg_args : list-like, optional
            Positional arguments to pass to the aggregation function.
        agg_kwargs : dict-like, optional
            Keyword arguments to pass to the aggregation function.
        **kwargs : dict
            Keyword arguments to pass to the specified query compiler's method.

        Returns
        -------
        DataFrame or Series
            Returns the same type as `self._df`.
        """
        agg_args = tuple() if agg_args is None else agg_args
        agg_kwargs = dict() if agg_kwargs is None else agg_kwargs

        if numeric_only is None or numeric_only is no_default:
            # pandas behavior: if `numeric_only` wasn't explicitly specified then
            # the parameter is considered to be `False` if there are no numeric types
            # in the frame and `True` otherwise.
            numeric_only = any(
                is_numeric_dtype(dtype) for dtype in self._query_compiler.dtypes
            )

        if numeric_only and self.ndim == 2:
            by_cols = self._internal_by
            mask_cols = [
                col
                for col, dtype in self._query_compiler.dtypes.items()
                if (
                    is_numeric_dtype(dtype)
                    or (
                        isinstance(dtype, pandas.CategoricalDtype)
                        and is_numeric_dtype(dtype.categories.dtype)
                    )
                    or col in by_cols
                )
            ]
            groupby_qc = self._query_compiler.getitem_column_array(mask_cols)
        else:
            groupby_qc = self._query_compiler

        result = type(self._df)(
            query_compiler=qc_method(
                groupby_qc,
                by=self._by,
                axis=self._axis,
                groupby_kwargs=self._kwargs,
                agg_args=agg_args,
                agg_kwargs=agg_kwargs,
                drop=self._drop,
                **kwargs,
            )
        )
        if self._squeeze:
            return result.squeeze()
        return result

    def _check_index(self, result):
        """
        Check the result of groupby aggregation on the need of resetting index.

        Parameters
        ----------
        result : DataFrame
            Group by aggregation result.

        Returns
        -------
        DataFrame
        """
        if self._by is None and not self._as_index:
            # This is a workaround to align behavior with pandas. In this case pandas
            # resets index, but Modin doesn't do that. More details are in https://github.com/modin-project/modin/issues/3716.
            result.reset_index(drop=True, inplace=True)

        return result

    def _check_index_name(self, result):
        """
        Check the result of groupby aggregation on the need of resetting index name.

        Parameters
        ----------
        result : DataFrame
            Group by aggregation result.

        Returns
        -------
        DataFrame
        """
        if self._by is not None:
            # pandas does not name the index for this case
            result._query_compiler.set_index_name(None)
        return result

    def _default_to_pandas(self, f, *args, **kwargs):
        """
        Execute function `f` in default-to-pandas way.

        Parameters
        ----------
        f : callable or str
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
            groupby_obj = df.groupby(
                by=by, axis=self._axis, squeeze=self._squeeze, **self._kwargs
            )

            if callable(f):
                return f(groupby_obj, *args, **kwargs)
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=not isinstance(f, str)
                )
                attribute = getattr(groupby_obj, f)
                if callable(attribute):
                    return attribute(*args, **kwargs)
                return attribute

        return self._df._default_to_pandas(groupby_on_multiple_columns, *args, **kwargs)


@_inherit_docstrings(pandas.core.groupby.SeriesGroupBy)
class SeriesGroupBy(DataFrameGroupBy):
    _pandas_class = pandas.core.groupby.SeriesGroupBy

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
        indices = self.indices
        group_ids = indices.keys()
        if self._axis == 0:
            return (
                (
                    k,
                    Series(
                        query_compiler=self._query_compiler.getitem_row_array(
                            indices[k]
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
                            indices[k], numeric=True
                        )
                    ),
                )
                for k in (sorted(group_ids) if self._sort else group_ids)
            )

    def _try_get_str_func(self, fn):
        """
        Try to convert a groupby aggregation function to a string or list of such.

        Parameters
        ----------
        fn : callable, str, or Iterable

        Returns
        -------
        str, list
            If `fn` is a callable, return its name if it's a method of the groupby
            object, otherwise return `fn` itself. If `fn` is a string, return it.
            If `fn` is an Iterable, return a list of _try_get_str_func applied to
            each element of `fn`.
        """
        if not isinstance(fn, str) and isinstance(fn, Iterable):
            return [self._try_get_str_func(f) for f in fn]
        if fn is np.max:
            # np.max is called "amax", so it's not a method of the groupby object.
            return "amax"
        elif fn is np.min:
            # np.min is called "amin", so it's not a method of the groupby object.
            return "amin"
        return fn.__name__ if callable(fn) and fn.__name__ in dir(self) else fn

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ):
        return self._default_to_pandas(
            lambda ser: ser.value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
        )

    @property
    def is_monotonic_decreasing(self):
        return self._default_to_pandas(lambda ser: ser.is_monotonic_decreasing)

    @property
    def is_monotonic_increasing(self):
        return self._default_to_pandas(lambda ser: ser.is_monotonic_increasing)

    @property
    def dtype(self):
        return self._default_to_pandas(lambda ser: ser.dtype)

    def unique(self):
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_unique,
                numeric_only=False,
            )
        )

    def nlargest(self, n=5, keep="first"):
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_nlargest,
                agg_kwargs=dict(n=n, keep=keep),
                numeric_only=True,
            )
        )

    def nsmallest(self, n=5, keep="first"):
        return self._check_index(
            self._wrap_aggregation(
                type(self._query_compiler).groupby_nsmallest,
                agg_kwargs=dict(n=n, keep=keep),
                numeric_only=True,
            )
        )

    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        engine_default = engine is None and engine_kwargs is None
        if isinstance(func, dict) and engine_default:
            raise SpecificationError("nested renamer is not supported")
        elif is_list_like(func) and engine_default:
            from .dataframe import DataFrame

            result = DataFrame(
                query_compiler=self._query_compiler.groupby_agg(
                    by=self._by,
                    agg_func=func,
                    axis=self._axis,
                    groupby_kwargs=self._kwargs,
                    agg_args=args,
                    agg_kwargs=kwargs,
                )
            )
            # query compiler always gives result a multiindex on the axis with the
            # function names, but series always gets a regular index on the columns
            # because there is no need to identify which original column's aggregation
            # the new column represents. alternatively we could give the query compiler
            # a hint that it's for a series, not a dataframe.
            maybe_squeezed = result.squeeze() if self._squeeze else result
            return maybe_squeezed.set_axis(labels=self._try_get_str_func(func), axis=1)
        else:
            return super().aggregate(
                func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
            )

    agg = aggregate


if IsExperimental.get():
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(DataFrameGroupBy, "make_dataframe_groupby_wrapper")
