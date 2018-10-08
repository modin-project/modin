from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import pandas.core.groupby
from pandas.core.dtypes.common import is_list_like
import pandas.core.common as com

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
        self, df, by, axis, level, as_index, sort, group_keys, squeeze, **kwargs
    ):

        self._axis = axis
        self._data_manager = df._data_manager
        self._index = self._data_manager.index
        self._columns = self._data_manager.columns
        self._by = by
        self._level = level
        self._kwargs = {
            "sort": sort,
            "as_index": as_index,
            "group_keys": group_keys,
            "squeeze": squeeze,
        }

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
                    "To contribute to Modin, please visit "
                    "github.com/modin-project/modin."
                )
            raise e

    _index_grouped_cache = None

    @property
    def _index_grouped(self):
        if self._index_grouped_cache is None:
            if self._axis == 0:
                self._index_grouped_cache = self._index.groupby(self._by)
            else:
                self._index_grouped_cache = self._columns.groupby(self._by)
        return self._index_grouped_cache

    _keys_and_values_cache = None

    @property
    def _keys_and_values(self):
        if self._keys_and_values_cache is None:
            self._keys_and_values_cache = list(self._index_grouped.items())
            if self._sort:
                self._keys_and_values_cache.sort()
        return self._keys_and_values_cache

    @property
    def _iter(self):
        from .dataframe import DataFrame

        if self._axis == 0:
            return (
                (
                    k,
                    DataFrame(
                        data_manager=self._data_manager.getitem_row_array(
                            self._index_grouped[k]
                        )
                    ),
                )
                for k, _ in self._keys_and_values
            )
        else:
            return (
                (
                    k,
                    DataFrame(
                        data_manager=self._data_manager.getitem_column_array(
                            self._index_grouped[k]
                        )
                    ),
                )
                for k, _ in self._keys_and_values
            )

    @property
    def ngroups(self):
        return len(self)

    def skew(self, **kwargs):
        return self._apply_agg_function(lambda df: df.skew(**kwargs))

    def ffill(self, limit=None):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def sem(self, ddof=1):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def mean(self, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.mean(*args, **kwargs))

    def any(self):
        return self._apply_agg_function(lambda df: df.any())

    @property
    def plot(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def ohlc(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def __bytes__(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    @property
    def tshift(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    @property
    def groups(self):
        return self._index_grouped

    def min(self, **kwargs):
        return self._apply_agg_function(lambda df: df.min(**kwargs))

    def idxmax(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    @property
    def ndim(self):
        return 2  # ndim is always 2 for DataFrames

    def shift(self, periods=1, freq=None, axis=0):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def nth(self, n, dropna=None):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def cumsum(self, axis=0, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.cumsum(axis, *args, **kwargs))

    @property
    def indices(self):
        return dict(self._keys_and_values)

    def pct_change(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def filter(self, func, dropna=True, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def cummax(self, axis=0, **kwargs):
        return self._apply_agg_function(lambda df: df.cummax(axis, **kwargs))

    def apply(self, func, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.apply(func, *args, **kwargs))

    @property
    def dtypes(self):
        if self._axis == 1:
            raise ValueError("Cannot call dtypes on groupby with axis=1")
        return self._apply_agg_function(lambda df: df.dtypes)

    def first(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def backfill(self, limit=None):
        return self.bfill(limit)

    def __getitem__(self, key):
        # This operation requires a SeriesGroupBy Object
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def cummin(self, axis=0, **kwargs):
        return self._apply_agg_function(lambda df: df.cummin(axis=axis, **kwargs))

    def bfill(self, limit=None):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def idxmin(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def prod(self, **kwargs):
        return self._apply_agg_function(lambda df: df.prod(**kwargs))

    def std(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.std(ddof, *args, **kwargs))

    def aggregate(self, arg, *args, **kwargs):
        if self._axis != 0:
            # This is not implemented in pandas,
            # so we throw a different message
            raise NotImplementedError("axis other than 0 is not supported")

        if is_list_like(arg):
            raise NotImplementedError(
                "This requires Multi-level index to be implemented. "
                "To contribute to Modin, please visit "
                "github.com/modin-project/modin."
            )
        return self._apply_agg_function(lambda df: df.aggregate(arg, *args, **kwargs))

    def last(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def mad(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def rank(self):
        return self._apply_agg_function(lambda df: df.rank())

    @property
    def corrwith(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def pad(self, limit=None):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def max(self, **kwargs):
        return self._apply_agg_function(lambda df: df.max(**kwargs))

    def var(self, ddof=1, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.var(ddof, *args, **kwargs))

    def get_group(self, name, obj=None):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def __len__(self):
        return len(self._index_grouped)

    def all(self, **kwargs):
        return self._apply_agg_function(lambda df: df.all(**kwargs))

    def size(self):
        return self._apply_agg_function(lambda df: df.size())

    def sum(self, **kwargs):
        return self._apply_agg_function(lambda df: df.sum(**kwargs))

    def __unicode__(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def describe(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

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
        **kwds
    ):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def ngroup(self, ascending=True):
        index = self._index if not self._axis else self._columns
        return (
            pandas.Series(index=index)
            .groupby(by=self._by, **self._kwargs)
            .ngroup(ascending)
        )

    def nunique(self, dropna=True):
        return self._apply_agg_function(lambda df: df.nunique(dropna))

    def resample(self, rule, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def median(self, **kwargs):
        return self._apply_agg_function(lambda df: df.median(**kwargs))

    def head(self, n=5):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def cumprod(self, axis=0, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.cumprod(axis, *args, **kwargs))

    def __iter__(self):
        return self._iter.__iter__()

    def agg(self, arg, *args, **kwargs):
        return self.aggregate(arg, *args, **kwargs)

    def cov(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def transform(self, func, *args, **kwargs):
        return self._apply_agg_function(lambda df: df.transform(func, *args, **kwargs))

    def corr(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def fillna(self, **kwargs):
        return self._apply_agg_function(lambda df: df.fillna(**kwargs))

    def count(self, **kwargs):
        return self._apply_agg_function(lambda df: df.count(**kwargs))

    def pipe(self, func, *args, **kwargs):
        return com._pipe(self, func, *args, **kwargs)

    def cumcount(self, ascending=True):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def tail(self, n=5):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    # expanding and rolling are unique cases and need to likely be handled
    # separately. They do not appear to be commonly used.
    def expanding(self, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def rolling(self, *args, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def hist(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def quantile(self, q=0.5, **kwargs):
        if is_list_like(q):
            raise NotImplementedError(
                "This requires Multi-level index to be implemented. "
                "To contribute to Modin, please visit "
                "github.com/modin-project/modin."
            )

        return self._apply_agg_function(lambda df: df.quantile(q, **kwargs))

    def diff(self):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def take(self, **kwargs):
        raise NotImplementedError(
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )

    def _apply_agg_function(self, f, **kwargs):
        """Perform aggregation and combine stages based on a given function.

        Args:
            f: The function to apply to each group.

        Returns:
             A new combined DataFrame with the result of all groups.
        """
        assert callable(f), "'{0}' object is not callable".format(type(f))
        from .dataframe import DataFrame

        new_manager = self._data_manager.groupby_agg(
            self._by, self._axis, f, self._kwargs, kwargs
        )
        return DataFrame(data_manager=new_manager)
