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

import pandas
import pandas.core.groupby
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com

from modin.error_message import ErrorMessage
from .utils import _inherit_docstrings


@_inherit_docstrings(
    pandas.core.groupby.DataFrameGroupBy,
    excluded=[
        pandas.core.groupby.DataFrameGroupBy,
        pandas.core.groupby.DataFrameGroupBy.__init__,
    ],
)
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
        by_type=None,
        **kwargs
    ):
        self._axis = axis
        self._idx_name = idx_name
        self._df = df
        self._query_compiler = self._df._query_compiler
        self._index = self._query_compiler.index
        self._columns = self._query_compiler.columns
        self._by = by
        self._drop = drop
        self._by_type = by_type if by_type else type(self._by)

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
                and all(obj in self._query_compiler.columns for obj in self._by)
            )
        else:
            self._is_multi_by = False
        self._level = level
        self._kwargs = {
            "level": level,
            "sort": sort,
            "as_index": as_index,
            "group_keys": group_keys,
            "squeeze": squeeze,
        }
        self._kwargs.update(kwargs)

    @property
    def _sort(self):
        return self._kwargs.get("sort")

    @property
    def _as_index(self):
        return self._kwargs.get("as_index")

    def __getattr__(self, key):
        """Afer regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self._columns:
                return self._default_to_pandas(lambda df: df.__getitem__(key))
            raise e

    _index_grouped_cache = None

    @property
    def _index_grouped(self):
        if self._index_grouped_cache is None:
            if hasattr(self._by, "columns") and len(self._by.columns) > 1:
                by = list(self._by.columns)
                is_multi_by = True
            else:
                by = self._by
                is_multi_by = self._is_multi_by
            if is_multi_by:
                # Because we are doing a collect (to_pandas) here and then groupby, we
                # end up using pandas implementation. Add the warning so the user is
                # aware.
                ErrorMessage.catch_bugs_and_request_email(self._axis == 1)
                ErrorMessage.default_to_pandas("Groupby with multiple columns")
                self._index_grouped_cache = {
                    k: v.index
                    for k, v in self._df._query_compiler.getitem_column_array(by)
                    .to_pandas()
                    .groupby(by=by)
                }
            else:
                if isinstance(self._by, type(self._query_compiler)):
                    by = self._by.to_pandas().squeeze()
                else:
                    by = self._by
                if self._axis == 0:
                    self._index_grouped_cache = self._index.groupby(by)
                else:
                    self._index_grouped_cache = self._columns.groupby(by)
        return self._index_grouped_cache

    @property
    def _iter(self):
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
        return self._groupby_reduce(
            lambda df: df.any(**kwargs), None, numeric_only=False
        )

    @property
    def plot(self):  # pragma: no cover
        return self._default_to_pandas(lambda df: df.plot)

    def ohlc(self):
        return self._default_to_pandas(lambda df: df.ohlc())

    def __bytes__(self):
        return self._default_to_pandas(lambda df: df.__bytes__())

    @property
    def tshift(self):
        return self._default_to_pandas(lambda df: df.tshift)

    @property
    def groups(self):
        return self._index_grouped

    def min(self, **kwargs):
        return self._groupby_reduce(
            lambda df: df.min(**kwargs), None, numeric_only=False
        )

    def idxmax(self):
        return self._default_to_pandas(lambda df: df.idxmax())

    @property
    def ndim(self):
        return 2  # ndim is always 2 for DataFrames

    def shift(self, periods=1, freq=None, axis=0):
        return self._default_to_pandas(
            lambda df: df.shift(periods=periods, freq=freq, axis=axis)
        )

    def nth(self, n, dropna=None):
        return self._default_to_pandas(lambda df: df.nth(n, dropna=dropna))

    def cumsum(self, axis=0, *args, **kwargs):
        result = self._apply_agg_function(lambda df: df.cumsum(axis, *args, **kwargs))
        # pandas does not name the index on cumsum
        result.index.name = None
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
        result.index.name = None
        return result

    def apply(self, func, *args, **kwargs):
        return self._apply_agg_function(
            lambda df: df.apply(func, *args, **kwargs), drop=self._as_index
        )

    @property
    def dtypes(self):
        if self._axis == 1:
            raise ValueError("Cannot call dtypes on groupby with axis=1")
        return self._apply_agg_function(lambda df: df.dtypes, drop=self._as_index)

    def first(self, **kwargs):
        return self._default_to_pandas(lambda df: df.first(**kwargs))

    def backfill(self, limit=None):
        return self.bfill(limit)

    def __getitem__(self, key):
        kwargs = self._kwargs.copy()
        if self._drop:
            if not isinstance(key, list):
                key = [key]
                kwargs["squeeze"] = True
        # When `as_index` is False, pandas will always convert to a `DataFrame`, we
        # convert to a list here so that the result will be a `DataFrame`.
        elif not self._as_index and not isinstance(key, list):
            key = [key]
        if isinstance(key, list):
            return DataFrameGroupBy(
                self._df[key],
                self._by,
                self._axis,
                idx_name=self._idx_name,
                drop=self._drop,
                by_type=self._by_type,
                **kwargs
            )
        return SeriesGroupBy(
            self._df[key],
            self._by,
            self._axis,
            idx_name=self._idx_name,
            drop=False,
            by_type=self._by_type,
            **kwargs
        )

    def cummin(self, axis=0, **kwargs):
        result = self._apply_agg_function(lambda df: df.cummin(axis=axis, **kwargs))
        # pandas does not name the index on cummin
        result.index.name = None
        return result

    def bfill(self, limit=None):
        return self._default_to_pandas(lambda df: df.bfill(limit=limit))

    def idxmin(self):
        return self._default_to_pandas(lambda df: df.idxmin())

    def prod(self, **kwargs):
        return self._groupby_reduce(lambda df: df.prod(**kwargs), None)

    def std(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.std(ddof, *args, **kwargs))

    def aggregate(self, arg, *args, **kwargs):
        if self._axis != 0:
            # This is not implemented in pandas,
            # so we throw a different message
            raise NotImplementedError("axis other than 0 is not supported")

        if is_list_like(arg):
            return self._default_to_pandas(
                lambda df: df.aggregate(arg, *args, **kwargs)
            )
        return self._apply_agg_function(
            lambda df: df.aggregate(arg, *args, **kwargs), drop=self._as_index
        )

    def last(self, **kwargs):
        return self._default_to_pandas(lambda df: df.last(**kwargs))

    def mad(self, **kwargs):
        return self._default_to_pandas(lambda df: df.mad(**kwargs))

    def rank(self, **kwargs):
        result = self._apply_agg_function(lambda df: df.rank(**kwargs))
        # pandas does not name the index on rank
        result.index.name = None
        return result

    @property
    def corrwith(self):
        return self._default_to_pandas(lambda df: df.corrwith)

    def pad(self, limit=None):
        return self._default_to_pandas(lambda df: df.pad(limit=limit))

    def max(self, **kwargs):
        return self._groupby_reduce(
            lambda df: df.max(**kwargs), None, numeric_only=False
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.var(ddof, *args, **kwargs))

    def get_group(self, name, obj=None):
        return self._default_to_pandas(lambda df: df.get_group(name, obj=obj))

    def __len__(self):
        return len(self._index_grouped)

    def all(self, **kwargs):
        return self._groupby_reduce(
            lambda df: df.all(**kwargs), None, numeric_only=False
        )

    def size(self):
        return pandas.Series({k: len(v) for k, v in self._index_grouped.items()})

    def sum(self, **kwargs):
        return self._groupby_reduce(lambda df: df.sum(**kwargs), None)

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
        **kwargs
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
                **kwargs
            )
        )

    def ngroup(self, ascending=True):
        return self._default_to_pandas(lambda df: df.ngroup(ascending))

    def nunique(self, dropna=True):
        return self._apply_agg_function(lambda df: df.nunique(dropna), drop=False)

    def resample(self, rule, *args, **kwargs):
        return self._default_to_pandas(lambda df: df.resample(rule, *args, **kwargs))

    def median(self, **kwargs):
        return self._apply_agg_function(lambda df: df.median(**kwargs))

    def head(self, n=5):
        return self._default_to_pandas(lambda df: df.head(n))

    def cumprod(self, axis=0, *args, **kwargs):
        result = self._apply_agg_function(lambda df: df.cumprod(axis, *args, **kwargs))
        # pandas does not name the index on cumprod
        result.index.name = None
        return result

    def __iter__(self):
        return self._iter.__iter__()

    def agg(self, arg, *args, **kwargs):
        return self.aggregate(arg, *args, **kwargs)

    def cov(self):
        return self._default_to_pandas(lambda df: df.cov())

    def transform(self, func, *args, **kwargs):
        result = self._apply_agg_function(
            lambda df: df.transform(func, *args, **kwargs)
        )
        # pandas does not name the index on transform
        result.index.name = None
        return result

    def corr(self, **kwargs):
        return self._default_to_pandas(lambda df: df.corr(**kwargs))

    def fillna(self, **kwargs):
        result = self._apply_agg_function(
            lambda df: df.fillna(**kwargs), drop=self._as_index
        )
        # pandas does not name the index on fillna
        result.index.name = None
        return result

    def count(self, **kwargs):
        return self._groupby_reduce(
            lambda df: df.count(**kwargs),
            lambda df: df.sum(**kwargs),
            numeric_only=False,
        )

    def pipe(self, func, *args, **kwargs):
        return com.pipe(self, func, *args, **kwargs)

    def cumcount(self, ascending=True):
        result = self._default_to_pandas(lambda df: df.cumcount(ascending=ascending))
        # pandas does not name the index on cumcount
        result.index.name = None
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
        import numpy as np

        if self.ndim > 1:
            if self._df.dtypes.map(lambda x: x == np.dtype("O")).any():
                raise TypeError(
                    "'quantile' cannot be performed against 'object' dtypes!"
                )
        elif self._df.dtypes == np.dtype("O"):
            raise TypeError("'quantile' cannot be performed against 'object' dtypes!")
        if is_list_like(q):
            return self._default_to_pandas(lambda df: df.quantile(q=q, **kwargs))

        return self._apply_agg_function(lambda df: df.quantile(q, **kwargs))

    def diff(self):
        return self._default_to_pandas(lambda df: df.diff())

    def take(self, **kwargs):
        return self._default_to_pandas(lambda df: df.take(**kwargs))

    def _groupby_reduce(
        self, map_func, reduce_func, drop=True, numeric_only=True, **kwargs
    ):
        if self._is_multi_by and not isinstance(self._by, type(self._query_compiler)):
            return self._default_to_pandas(map_func, **kwargs)
        if not isinstance(self._by, type(self._query_compiler)):
            return self._apply_agg_function(map_func, drop=drop, **kwargs)

        # For aggregations, pandas behavior does this for the result.
        # For other operations it does not, so we wait until there is an aggregation to
        # actually perform this operation.
        if self._idx_name is not None and drop and self._drop:
            groupby_qc = self._query_compiler.drop(columns=[self._idx_name])
        else:
            groupby_qc = self._query_compiler

        result = type(self._df)(
            query_compiler=groupby_qc.groupby_reduce(
                self._by,
                self._axis,
                self._kwargs,
                map_func,
                kwargs,
                reduce_func=reduce_func,
                reduce_args=kwargs,
                numeric_only=numeric_only,
                drop=self._drop,
            )
        )
        if self._kwargs.get("squeeze", False):
            return result.squeeze()
        return result

    def _apply_agg_function(self, f, drop=True, **kwargs):
        """Perform aggregation and combine stages based on a given function.

        Args:
            f: The function to apply to each group.

        Returns:
             A new combined DataFrame with the result of all groups.
        """
        assert callable(f), "'{0}' object is not callable".format(type(f))

        if self._is_multi_by:
            return self._default_to_pandas(f, **kwargs)

        if isinstance(self._by, type(self._query_compiler)):
            by = self._by.to_pandas().squeeze()
        else:
            by = self._by

        # For aggregations, pandas behavior does this for the result.
        # For other operations it does not, so we wait until there is an aggregation to
        # actually perform this operation.
        if self._idx_name is not None and drop and self._drop:
            groupby_qc = self._query_compiler.drop(columns=[self._idx_name])
        else:
            groupby_qc = self._query_compiler
        new_manager = groupby_qc.groupby_agg(
            by, self._axis, f, self._kwargs, kwargs, drop=self._drop
        )
        if self._idx_name is not None and self._as_index:
            new_manager.index.name = self._idx_name
        result = type(self._df)(query_compiler=new_manager)
        if self._kwargs.get("squeeze", False):
            return result.squeeze()
        return result

    def _default_to_pandas(self, f, **kwargs):
        """Defailts the execution of this function to pandas.

        Args:
            f: The function to apply to each group.

        Returns:
             A new Modin DataFrame with the result of the pandas function.
        """
        if (
            isinstance(self._by, type(self._query_compiler))
            and len(self._by.columns) == 1
        ):
            by = (
                self._by.columns[0]
                if self._by_type is str
                else self._by.to_pandas().squeeze()
            )
        elif isinstance(self._by, type(self._query_compiler)):
            by = list(self._by.columns)
        else:
            by = self._by

        def groupby_on_multiple_columns(df):
            return f(df.groupby(by=by, axis=self._axis, **self._kwargs), **kwargs)

        return self._df._default_to_pandas(groupby_on_multiple_columns)


class SeriesGroupBy(DataFrameGroupBy):
    @property
    def ndim(self):
        return 1  # ndim is always 1 for Series

    @property
    def _iter(self):
        from .dataframe import Series

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
