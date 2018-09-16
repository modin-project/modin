from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import pandas.core.groupby
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com

import numpy as np
import ray

from .concat import concat
from .index_metadata import _IndexMetadata
from .utils import _inherit_docstrings, _reindex_helper, post_task_gc


@_inherit_docstrings(
    pandas.core.groupby.DataFrameGroupBy,
    excluded=[
        pandas.core.groupby.DataFrameGroupBy,
        pandas.core.groupby.DataFrameGroupBy.__init__
    ])
class DataFrameGroupBy(object):
    def __init__(self, df, by, axis, level, as_index, sort, group_keys,
                 squeeze, **kwargs):

        self._columns = df.columns
        self._index = df.index
        self._axis = axis

        self._df = df
        self._by = by
        self._level = level
        self._as_index = as_index
        self._sort = sort
        self._group_keys = group_keys
        self._squeeze = squeeze

        self._row_metadata = df._row_metadata
        self._col_metadata = df._col_metadata

        if axis == 0:
            self._partitions = df._block_partitions.T
        else:
            self._partitions = df._block_partitions

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
                raise NotImplementedError(
                    "SeriesGroupBy is not implemented."
                    "To contribute to Pandas on Ray, please visit "
                    "github.com/modin-project/modin.")
            raise e

    _index_grouped_cache = None

    @property
    def _index_grouped(self):
        if self._index_grouped_cache is None:
            if self._axis == 0:
                self._index_grouped_cache = pandas.Series(
                    np.zeros(len(self._index), dtype=np.uint8),
                    index=self._index).groupby(
                        by=self._by, sort=self._sort)
            else:
                self._index_grouped_cache = pandas.Series(
                    np.zeros(len(self._columns), dtype=np.uint8),
                    index=self._columns).groupby(
                        by=self._by, sort=self._sort)

        return self._index_grouped_cache

    _keys_and_values_cache = None

    @property
    def _keys_and_values(self):
        if self._keys_and_values_cache is None:
            self._keys_and_values_cache = \
                [(k, v) for k, v in self._index_grouped]
        return self._keys_and_values_cache

    @property
    def _grouped_partitions(self):

        # It is expensive to put this multiple times, so let's just put it once
        remote_by = ray.put(self._by)
        remote_index = \
            [ray.put(v.index) for _, v in
             self._df._col_metadata._coord_df.copy().groupby(by='partition')] \
            if self._axis == 0 \
            else [ray.put(v.index) for _, v in
                  self._df._row_metadata._coord_df.copy()
                      .groupby(by='partition')]

        if len(self._index_grouped) > 1:
            return zip(*(groupby._submit(
                args=(remote_index[i], remote_by, self._axis, self._level,
                      self._as_index, self._sort, self._group_keys,
                      self._squeeze) + tuple(part.tolist()),
                num_return_vals=len(self._index_grouped))
                         for i, part in enumerate(self._partitions)))
        elif self._axis == 0:
            return [self._df._col_partitions]
        else:
            return [self._df._row_partitions]

    @property
    def _iter(self):
        from .dataframe import DataFrame

        if self._axis == 0:
            return ((self._keys_and_values[i][0],
                     DataFrame(
                         col_partitions=part,
                         columns=self._columns,
                         index=self._keys_and_values[i][1].index,
                         col_metadata=self._col_metadata))
                    for i, part in enumerate(self._grouped_partitions))
        else:
            return ((self._keys_and_values[i][0],
                     DataFrame(
                         row_partitions=part,
                         columns=self._keys_and_values[i][1].index,
                         index=self._index,
                         row_metadata=self._row_metadata))
                    for i, part in enumerate(self._grouped_partitions))

    @property
    def ngroups(self):
        return len(self)

    def skew(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _skew_remote.remote(df, self._axis, kwargs))

    def ffill(self, limit=None):
        return self._apply_df_function(
            lambda df: df.ffill(axis=self._axis, limit=limit))

    def sem(self, ddof=1):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def mean(self, *args, **kwargs):
        return self._apply_agg_function(
            lambda df: _mean_remote.remote(df, self._axis, kwargs, *args))

    def any(self):
        return self._apply_agg_function(
            lambda df: _any_remote.remote(df, self._axis))

    @property
    def plot(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def ohlc(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __bytes__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def tshift(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def groups(self):
        return {k: pandas.Index(v) for k, v in self._keys_and_values}

    def min(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _min_remote.remote(df, self._axis, kwargs))

    def idxmax(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    @property
    def ndim(self):
        return 2  # ndim is always 2 for DataFrames

    def shift(self, periods=1, freq=None, axis=0):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def nth(self, n, dropna=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def cumsum(self, axis=0, *args, **kwargs):
        return self._apply_df_function(
            lambda df: df.cumsum(axis, *args, **kwargs))

    @property
    def indices(self):
        return dict(self._keys_and_values)

    def pct_change(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def filter(self, func, dropna=True, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def cummax(self, axis=0, **kwargs):
        return self._apply_df_function(lambda df: df.cummax(axis, **kwargs))

    def apply(self, func, *args, **kwargs):
        def apply_helper(df):
            return df.apply(func, axis=self._axis, *args, **kwargs)

        result = [func(v) for k, v in self._iter]
        if self._axis == 0:
            if isinstance(result[0], pandas.Series):
                # Applied an aggregation function
                new_df = concat(result, axis=1).T
                new_df.columns = self._columns
                new_df.index = [k for k, v in self._iter]
            else:
                new_df = concat(result, axis=self._axis)
                new_df._block_partitions = np.array([
                    _reindex_helper._submit(
                        args=tuple([
                            new_df.index, self._index, self._axis ^ 1,
                            len(new_df._block_partitions)
                        ] + block.tolist()),
                        num_return_vals=len(new_df._block_partitions))
                    for block in new_df._block_partitions.T
                ]).T
                new_df.index = self._index
                new_df._row_metadata = \
                    _IndexMetadata(new_df._block_partitions[:, 0],
                                   index=new_df.index, axis=0)
        else:
            if isinstance(result[0], pandas.Series):
                # Applied an aggregation function
                new_df = concat(result, axis=1)
                new_df.columns = [k for k, v in self._iter]
                new_df.index = self._index
            else:
                new_df = concat(result, axis=self._axis)
                new_df._block_partitions = np.array([
                    _reindex_helper._submit(
                        args=tuple([
                            new_df.columns, self._columns, self._axis ^ 1,
                            new_df._block_partitions.shape[1]
                        ] + block.tolist()),
                        num_return_vals=new_df._block_partitions.shape[1])
                    for block in new_df._block_partitions
                ])
                new_df.columns = self._columns
                new_df._col_metadata = \
                    _IndexMetadata(new_df._block_partitions[0, :],
                                   index=new_df.columns, axis=1)
        return new_df

    @property
    def dtypes(self):
        if self._axis == 1:
            raise ValueError("Cannot call dtypes on groupby with axis=1")
        return self._apply_agg_function(lambda df: _dtypes_remote.remote(df))

    def first(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def backfill(self, limit=None):
        return self.bfill(limit)

    def __getitem__(self, key):
        # This operation requires a SeriesGroupBy Object
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def cummin(self, axis=0, **kwargs):
        return self._apply_df_function(
            lambda df: df.cummin(axis=axis, **kwargs))

    def bfill(self, limit=None):
        return self._apply_df_function(
            lambda df: df.bfill(axis=self._axis, limit=limit))

    def idxmin(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def prod(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _prod_remote.remote(df, self._axis, kwargs))

    def std(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(
            lambda df: _std_remote.remote(df, self._axis, ddof, kwargs, *args))

    def aggregate(self, arg, *args, **kwargs):
        if self._axis != 0:
            # This is not implemented in pandas,
            # so we throw a different message
            raise NotImplementedError("axis other than 0 is not supported")

        if is_list_like(arg):
            raise NotImplementedError(
                "This requires Multi-level index to be implemented. "
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")
        return self._apply_agg_function(
            lambda df: _agg_remote.remote(df, self._axis, arg, kwargs, *args))

    def last(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def mad(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def rank(self):
        return self._apply_df_function(lambda df: df.rank(axis=self._axis))

    @property
    def corrwith(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def pad(self, limit=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def max(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _max_remote.remote(df, self._axis, kwargs))

    def var(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(
            lambda df: _var_remote.remote(df, self._axis, ddof, kwargs, *args))

    def get_group(self, name, obj=None):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def __len__(self):
        return len(self._index_grouped)

    def all(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _all_remote.remote(df, kwargs))

    def size(self):
        return self._apply_agg_function(lambda df: _size_remote.remote(df))

    def sum(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _sum_remote.remote(df, self._axis, kwargs))

    def __unicode__(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def describe(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def boxplot(self,
                grouped,
                subplots=True,
                column=None,
                fontsize=None,
                rot=0,
                grid=True,
                ax=None,
                figsize=None,
                layout=None,
                **kwds):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def ngroup(self, ascending=True):
        return self._index_grouped.ngroup(ascending)

    def nunique(self, dropna=True):
        return self._apply_agg_function(
            lambda df: _nunique_remote.remote(df, self._axis, dropna))

    def resample(self, rule, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def median(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _median_remote.remote(df, self._axis, kwargs))

    def head(self, n=5):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def cumprod(self, axis=0, *args, **kwargs):
        return self._apply_df_function(
            lambda df: df.cumprod(axis, *args, **kwargs))

    def __iter__(self):
        return self._iter.__iter__()

    def agg(self, arg, *args, **kwargs):
        return self.aggregate(arg, *args, **kwargs)

    def cov(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def transform(self, func, *args, **kwargs):
        return self._apply_df_function(
            lambda df: df.transform(func, *args, **kwargs))

    def corr(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def fillna(self, **kwargs):
        return self._apply_df_function(
            lambda df: df.fillna(axis=self._axis, **kwargs))

    def count(self, **kwargs):
        return self._apply_agg_function(
            lambda df: _count_remote.remote(df, self._axis, kwargs))

    def pipe(self, func, *args, **kwargs):
        return com._pipe(self, func, *args, **kwargs)

    def cumcount(self, ascending=True):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def tail(self, n=5):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    # expanding and rolling are unique cases and need to likely be handled
    # separately. They do not appear to be commonly used.
    def expanding(self, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def rolling(self, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def hist(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def quantile(self, q=0.5, **kwargs):
        if is_list_like(q):
            raise NotImplementedError(
                "This requires Multi-level index to be implemented. "
                "To contribute to Pandas on Ray, please visit "
                "github.com/modin-project/modin.")

        return self._apply_agg_function(
            lambda df: _quantile_remote.remote(df, self._axis, q, kwargs))

    def diff(self):
        raise NotImplementedError(
            "To contribute to Pandas on Ray, please visit "
            "github.com/modin-project/modin.")

    def take(self, **kwargs):
        return self._apply_df_function(lambda df: df.take(**kwargs))

    def _apply_agg_function(self, f, index=None):
        """Perform aggregation and combine stages based on a given function.

        Args:
            f: The function to apply to each group. f must be a remote
                function.

        Returns:
             A new combined DataFrame with the result of all groups.
        """
        assert callable(f), "\'{0}\' object is not callable".format(type(f))

        blocks = np.array([[f(part) for part in group_of_parts]
                           for group_of_parts in self._grouped_partitions])

        from .dataframe import DataFrame
        if self._axis == 0:
            return DataFrame(
                block_partitions=blocks,
                columns=self._columns,
                index=index
                if index is not None else [k for k, _ in self._index_grouped])
        else:
            return DataFrame(
                block_partitions=blocks.T,
                index=self._index,
                columns=index
                if index is not None else [k for k, _ in self._index_grouped])

    def _apply_df_function(self, f, concat_axis=None):
        assert callable(f), "\'{0}\' object is not callable".format(type(f))

        result = [f(v) for k, v in self._iter]
        concat_axis = self._axis if concat_axis is None else concat_axis

        new_df = concat(result, axis=concat_axis)

        if self._axis == 0:
            new_df._block_partitions = np.array([
                _reindex_helper._submit(
                    args=tuple([
                        new_df.index, self._index, 1,
                        len(new_df._block_partitions)
                    ] + block.tolist()),
                    num_return_vals=len(new_df._block_partitions))
                for block in new_df._block_partitions.T
            ]).T
            new_df.index = self._index
            new_df._row_metadata = \
                _IndexMetadata(new_df._block_partitions[:, 0],
                               index=new_df.index, axis=0)
        else:
            new_df._block_partitions = np.array([
                _reindex_helper._submit(
                    args=tuple([
                        new_df.columns, self._columns, 0, new_df.
                        _block_partitions.shape[1]
                    ] + block.tolist()),
                    num_return_vals=new_df._block_partitions.shape[1])
                for block in new_df._block_partitions
            ])
            new_df.columns = self._columns
            new_df._col_metadata = \
                _IndexMetadata(new_df._block_partitions[0, :],
                               index=new_df.columns, axis=1)

        return new_df


@ray.remote
@post_task_gc
def groupby(index, by, axis, level, as_index, sort, group_keys, squeeze, *df):

    df = pandas.concat(df, axis=axis)

    if axis == 0:
        df.columns = index
    else:
        df.index = index
    return [
        v for k, v in df.groupby(
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze)
    ]


@ray.remote
def _sum_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.sum(axis=axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _skew_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.skew(axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _mean_remote(df, axis, kwargs, *args):
    result = pandas.DataFrame(df.mean(axis, *args, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _any_remote(df, axis):
    result = pandas.DataFrame(df.any(axis))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _min_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.min(axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _dtypes_remote(df):
    return pandas.DataFrame(df.dtypes).T


@ray.remote
def _prod_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.prod(axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _std_remote(df, axis, ddof, kwargs, *args):
    result = pandas.DataFrame(df.std(axis=axis, ddof=ddof, *args, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _max_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.max(axis=axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _var_remote(df, axis, ddof, kwargs, *args):
    result = pandas.DataFrame(df.var(axis=axis, ddof=ddof, *args, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _all_remote(df, kwargs):
    return pandas.DataFrame(df.all(**kwargs)).T


@ray.remote
def _size_remote(df):
    return pandas.DataFrame(df.size).T


@ray.remote
def _nunique_remote(df, axis, dropna):
    result = pandas.DataFrame(df.nunique(axis=axis, dropna=dropna))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _median_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.median(axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _count_remote(df, axis, kwargs):
    result = pandas.DataFrame(df.count(axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _quantile_remote(df, axis, q, kwargs):
    result = pandas.DataFrame(df.quantile(q=q, axis=axis, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result


@ray.remote
def _agg_remote(df, axis, arg, kwargs, *args):
    result = pandas.DataFrame(df.agg(arg, axis=axis, *args, **kwargs))
    if axis == 0:
        return result.T
    else:
        return result
