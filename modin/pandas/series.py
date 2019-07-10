from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from pandas.compat import string_types
from pandas.core.common import is_bool_indexer
from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_scalar,
    is_string_like,
)
import sys
import warnings

from .base import BasePandasDataset
from .iterator import PartitionIterator
from .utils import _inherit_docstrings
from .utils import from_pandas, to_pandas


@_inherit_docstrings(pandas.Series, excluded=[pandas.Series, pandas.Series.__init__])
class Series(BasePandasDataset):
    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=False,
        fastpath=False,
        query_compiler=None,
    ):
        """Constructor for a Series object.

        Args:
            series_oids ([ObjectID]): The list of remote Series objects.
        """
        if query_compiler is None:
            warnings.warn(
                "Distributing {} object. This may take some time.".format(type(data))
            )
            if name is None:
                name = "__reduced__"
            query_compiler = from_pandas(
                pandas.DataFrame(
                    pandas.Series(
                        data=data,
                        index=index,
                        dtype=dtype,
                        name=name,
                        copy=copy,
                        fastpath=fastpath,
                    )
                )
            )._query_compiler
        if len(query_compiler.columns) != 1 or (
            len(query_compiler.index) == 1 and query_compiler.index[0] == "__reduced__"
        ):
            query_compiler = query_compiler.transpose()
        self._query_compiler = query_compiler

    def _get_name(self):
        name = self._query_compiler.columns[0]
        if name == "__reduced__":
            return None
        return name

    def _set_name(self, name):
        if name is None:
            name = "__reduced__"
        self._query_compiler.columns = [name]

    name = property(_get_name, _set_name)
    _parent = None

    def _reduce_dimension(self, query_compiler):
        return query_compiler.to_pandas().squeeze()

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        return self

    def _validate_dtypes_min_max(self, axis, numeric_only):
        return self

    def _validate_dtypes(self, numeric_only=False):
        pass

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace and (
            len(new_query_compiler.columns) == 1 or len(new_query_compiler.index) == 1
        ):
            return Series(query_compiler=new_query_compiler)
        elif not inplace:
            # This can happen with things like `reset_index` where we can add columns.
            from .dataframe import DataFrame

            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _prepare_inter_op(self, other):
        if isinstance(other, Series):
            new_self = self.copy()
            new_self.name = "__reduced__"
            new_other = other.copy()
            new_other.name = "__reduced__"
        else:
            new_self = self
            new_other = other
        return new_self, new_other

    def __add__(self, right):
        return self.add(right)

    def __and__(self, other):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__and__(new_other)

    def __array__(self, dtype=None):
        return super(Series, self).__array__(dtype).flatten()

    def __array_prepare__(self, result, context=None):  # pragma: no cover
        return self._default_to_pandas(
            pandas.Series.__array_prepare__, result, context=context
        )

    @property
    def __array_priority__(self):  # pragma: no cover
        return self._to_pandas().__array_priority__

    def __bytes__(self):
        return self._default_to_pandas(pandas.Series.__bytes__)

    def __contains__(self, key):
        return key in self.index

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        return self.copy(deep=True)

    def __delitem__(self, key):
        if key not in self.keys():
            raise KeyError(key)
        self.drop(labels=key, inplace=True)

    def __div__(self, right):
        return self.div(right)

    def __divmod__(self, right):
        return self.divmod(right)

    def __float__(self):
        return float(self.squeeze())

    def __floordiv__(self, right):
        return self.floordiv(right)

    def _getitem(self, key):
        if isinstance(key, Series) and key.dtype == np.bool:
            # This ends up being significantly faster than looping through and getting
            # each item individually.
            key = key._to_pandas()
        if is_bool_indexer(key):
            return self.__constructor__(
                query_compiler=self._query_compiler.getitem_row_array(
                    pandas.RangeIndex(len(self.index))[key]
                )
            )
        # TODO: More efficiently handle `tuple` case for `Series.__getitem__`
        if isinstance(key, tuple):
            return self._default_to_pandas(pandas.Series.__getitem__, key)
        else:
            if not is_list_like(key):
                reduce_dimension = True
                key = [key]
            else:
                reduce_dimension = False
            # The check for whether or not `key` is in `keys()` will throw a TypeError
            # if the object is not hashable. When that happens, we just use the `iloc`.
            try:
                if all(k in self.keys() for k in key):
                    result = self._query_compiler.getitem_row_array(
                        self.index.get_indexer_for(key)
                    )
                else:
                    result = self._query_compiler.getitem_row_array(key)
            except TypeError:
                result = self._query_compiler.getitem_row_array(key)
        if reduce_dimension:
            return self._reduce_dimension(result)
        return self.__constructor__(query_compiler=result)

    def __int__(self):
        return int(self.squeeze())

    def __iter__(self):
        return self._to_pandas().__iter__()

    def __mod__(self, right):
        return self.mod(right)

    def __mul__(self, right):
        return self.mul(right)

    def __or__(self, other):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__or__(new_other)

    def __pow__(self, right):
        return self.pow(right)

    def __repr__(self):
        num_rows = pandas.get_option("max_rows") or 60
        num_cols = pandas.get_option("max_columns") or 20
        temp_df = self._build_repr_df(num_rows, num_cols)
        if isinstance(temp_df, pandas.DataFrame):
            temp_df = temp_df.iloc[:, 0]
        temp_str = repr(temp_df)
        if self.name is not None:
            name_str = "Name: {}, ".format(str(self.name))
        else:
            name_str = ""
        if len(self.index) > num_rows:
            len_str = "Length: {}, ".format(len(self.index))
        else:
            len_str = ""
        dtype_str = "dtype: {}".format(temp_str.rsplit("dtype: ", 1)[-1])
        if len(self) == 0:
            return "Series([], {}{}".format(name_str, dtype_str)
        return temp_str.rsplit("\nName:", 1)[0] + "\n{}{}{}".format(
            name_str, len_str, dtype_str
        )

    def __round__(self, decimals=0):
        return self._create_or_update_from_compiler(
            self._query_compiler.round(decimals=decimals)
        )

    def __setitem__(self, key, value):
        if key not in self.keys():
            raise KeyError(key)
        self._create_or_update_from_compiler(
            self._query_compiler.setitem(1, key, value), inplace=True
        )

    def __sub__(self, right):
        return self.sub(right)

    def __truediv__(self, right):
        return self.truediv(right)

    __iadd__ = __add__
    __imul__ = __add__
    __ipow__ = __pow__
    __isub__ = __sub__
    __itruediv__ = __truediv__

    @property
    def values(self):
        """Create a numpy array with the values from this Series.

        Returns:
            The numpy representation of this object.
        """
        return super(Series, self).to_numpy().flatten()

    def __xor__(self, other):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__xor__(new_other)

    def add(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).add(
            new_other, level=level, fill_value=fill_value, axis=axis
        )

    def add_prefix(self, prefix):
        """Add a prefix to each of the column names.

        Returns:
            A new Series containing the new column names.
        """
        return Series(query_compiler=self._query_compiler.add_prefix(prefix, axis=0))

    def add_suffix(self, suffix):
        """Add a suffix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return Series(query_compiler=self._query_compiler.add_suffix(suffix, axis=0))

    def append(self, to_append, ignore_index=False, verify_integrity=False):
        """Append another DataFrame/list/Series to this one.

        Args:
            to_append: The object to append to this.
            ignore_index: Ignore the index on appending.
            verify_integrity: Verify the integrity of the index on completion.

        Returns:
            A new DataFrame containing the concatenated values.
        """
        from .dataframe import DataFrame

        bad_type_msg = (
            'cannot concatenate object of type "{}"; only pd.Series, '
            "pd.DataFrame, and pd.Panel (deprecated) objs are valid"
        )
        if isinstance(to_append, list):
            if not all(isinstance(o, BasePandasDataset) for o in to_append):
                raise TypeError(
                    bad_type_msg.format(
                        type(
                            next(
                                o
                                for o in to_append
                                if not isinstance(o, BasePandasDataset)
                            )
                        )
                    )
                )
            elif all(isinstance(o, Series) for o in to_append):
                self.name = None
                for i in range(len(to_append)):
                    to_append[i].name = None
                    to_append[i] = to_append[i]._query_compiler
            else:
                # Matching pandas behavior of naming the Series columns 0
                self.name = 0
                for i in range(len(to_append)):
                    if isinstance(to_append[i], Series):
                        to_append[i].name = 0
                        to_append[i] = DataFrame(to_append[i])
                return DataFrame(self.copy()).append(
                    to_append,
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                )
        elif isinstance(to_append, Series):
            self.name = None
            to_append.name = None
            to_append = [to_append._query_compiler]
        elif isinstance(to_append, DataFrame):
            self.name = 0
            return DataFrame(self.copy()).append(
                to_append, ignore_index=ignore_index, verify_integrity=verify_integrity
            )
        else:
            raise TypeError(bad_type_msg.format(type(to_append)))
        # If ignore_index is False, by definition the Index will be correct.
        # We also do this first to ensure that we don't waste compute/memory.
        if verify_integrity and not ignore_index:
            appended_index = (
                self.index.append(to_append.index)
                if not isinstance(to_append, list)
                else self.index.append([o.index for o in to_append])
            )
            is_valid = next((False for idx in appended_index.duplicated() if idx), True)
            if not is_valid:
                raise ValueError(
                    "Indexes have overlapping values: {}".format(
                        appended_index[appended_index.duplicated()]
                    )
                )
        query_compiler = self._query_compiler.concat(
            0, to_append, ignore_index=ignore_index, sort=None
        )
        if len(query_compiler.columns) > 1:
            return DataFrame(query_compiler=query_compiler)
        else:
            return Series(query_compiler=query_compiler)

    def apply(self, func, convert_dtype=True, args=(), **kwds):
        # apply and aggregate have slightly different behaviors, so we have to use
        # each one separately to determine the correct return type. In the case of
        # `agg`, the axis is set, but it is not required for the computation, so we use
        # it to determine which function to run.
        if kwds.pop("axis", None) is not None:
            apply_func = "agg"
        else:
            apply_func = "apply"

        # This is the simplest way to determine the return type, but there are checks
        # in pandas that verify that some results are created. This is a challenge for
        # empty DataFrames, but fortunately they only happen when the `func` type is
        # a list or a dictionary, which means that the return type won't change from
        # type(self), so we catch that error and use `self.__name__` for the return
        # type.
        # Because a `Series` cannot be empty in pandas, we create a "dummy" `Series` to
        # do the error checking and determining the return type.
        try:
            return_type = type(
                getattr(pandas.Series([""], index=self.index[:1]), apply_func)(
                    func, *args, **kwds
                )
            ).__name__
        except Exception:
            return_type = self.__name__
        if (
            isinstance(func, string_types)
            or is_list_like(func)
            or return_type not in ["DataFrame", "Series"]
        ):
            query_compiler = super(Series, self).apply(func, *args, **kwds)
            # Sometimes we can return a scalar here
            if not isinstance(query_compiler, type(self._query_compiler)):
                return query_compiler
        else:
            # handle ufuncs and lambdas
            if kwds or args and not isinstance(func, np.ufunc):

                def f(x):
                    return func(x, *args, **kwds)

            else:
                f = func
            with np.errstate(all="ignore"):
                if isinstance(f, np.ufunc):
                    return f(self)
                query_compiler = self.map(f)._query_compiler
        if return_type not in ["DataFrame", "Series"]:
            return query_compiler.to_pandas().squeeze()
        else:
            result = getattr(sys.modules[self.__module__], return_type)(
                query_compiler=query_compiler
            )
            if result.name == self.index[0]:
                result.name = None
            return result

    def argmax(self, axis=0, skipna=True, *args, **kwargs):
        # Series and DataFrame have a different behavior for `skipna`
        if skipna is None:
            skipna = True
        return self.idxmax(axis=axis, skipna=skipna, *args, **kwargs)

    def argmin(self, axis=0, skipna=True, *args, **kwargs):
        # Series and DataFrame have a different behavior for `skipna`
        if skipna is None:
            skipna = True
        return self.idxmin(axis=axis, skipna=skipna, *args, **kwargs)

    def argsort(self, axis=0, kind="quicksort", order=None):
        return self._default_to_pandas(
            pandas.Series.argsort, axis=axis, kind=kind, order=order
        )

    def array(self):
        return self._default_to_pandas(pandas.Series.array)

    def autocorr(self, lag=1):
        return self._default_to_pandas(pandas.Series.autocorr, lag=lag)

    def between(self, left, right, inclusive=True):
        return self._default_to_pandas(
            pandas.Series.between, left, right, inclusive=inclusive
        )

    def combine(self, other, func, fill_value=None):
        return super(Series, self).combine(other, func, fill_value=fill_value)

    def compound(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas(
            pandas.Series.compound, axis=axis, skipna=skipna, level=level
        )

    def compress(self, condition, *args, **kwargs):
        return self._default_to_pandas(
            pandas.Series.compress, condition, *args, **kwargs
        )

    def convert_objects(
        self,
        convert_dates=True,
        convert_numeric=False,
        convert_timedeltas=True,
        copy=True,
    ):
        return self._default_to_pandas(
            pandas.Series.convert_objects,
            convert_dates=convert_dates,
            convert_numeric=convert_numeric,
            convert_timedeltas=convert_timedeltas,
            copy=copy,
        )

    def corr(self, other, method="pearson", min_periods=None):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas(
            pandas.Series.corr, other, method=method, min_periods=min_periods
        )

    def count(self, level=None):
        return super(Series, self).count(level=level)

    def cov(self, other, min_periods=None):
        if isinstance(other, BasePandasDataset):
            other = other._to_pandas()
        return self._default_to_pandas(
            pandas.Series.cov, other, min_periods=min_periods
        )

    def describe(self, percentiles=None, include=None, exclude=None):
        # Pandas ignores the `include` and `exclude` for Series for some reason.
        return super(Series, self).describe(percentiles=percentiles)

    def diff(self, periods=1):
        return super(Series, self).diff(periods=periods, axis=0)

    def divmod(self, other, level=None, fill_value=None, axis=0):
        return self._default_to_pandas(
            pandas.Series.divmod, other, level=level, fill_value=fill_value, axis=axis
        )

    def drop_duplicates(self, keep="first", inplace=False):
        return super(Series, self).drop_duplicates(keep=keep, inplace=inplace)

    def dropna(self, axis=0, inplace=False, **kwargs):
        kwargs.pop("how", None)
        if kwargs:
            raise TypeError(
                "dropna() got an unexpected keyword "
                'argument "{0}"'.format(list(kwargs.keys())[0])
            )
        return super(Series, self).dropna(axis=axis, inplace=inplace)

    def duplicated(self, keep="first"):
        return super(Series, self).duplicated(keep=keep)

    def eq(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).eq(new_other, level=level, axis=axis)

    def equals(self, other):
        return (
            self.name == other.name
            and self.index.equals(other.index)
            and self.eq(other).all()
        )

    def factorize(self, sort=False, na_sentinel=-1):
        return self._default_to_pandas(
            pandas.Series.factorize, sort=sort, na_sentinel=na_sentinel
        )

    def floordiv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).floordiv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def from_array(
        self, arr, index=None, name=None, dtype=None, copy=False, fastpath=False
    ):
        raise NotImplementedError("Not Yet implemented.")

    def from_csv(
        self,
        path,
        sep=",",
        parse_dates=True,
        header=None,
        index_col=0,
        encoding=None,
        infer_datetime_format=False,
    ):
        return super(Series, self).from_csv(
            path,
            sep=sep,
            parse_dates=parse_dates,
            header=header,
            index_col=index_col,
            encoding=encoding,
            infer_datetime_format=infer_datetime_format,
        )

    def ge(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).ge(new_other, level=level, axis=axis)

    def get_value(self, label, takeable=False):
        return self._default_to_pandas(
            pandas.Series.get_value, label, takeable=takeable
        )

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        **kwargs
    ):
        return self._default_to_pandas(
            pandas.Series.groupby,
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            **kwargs
        )

    def gt(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).gt(new_other, level=level, axis=axis)

    def head(self, n=5):
        if n == 0:
            return Series(dtype=self.dtype)
        return super(Series, self).head(n)

    def hist(
        self,
        by=None,
        ax=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        yrot=None,
        figsize=None,
        bins=10,
        **kwds
    ):
        return self._default_to_pandas(
            pandas.Series.hist,
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            **kwds
        )

    def idxmax(self, axis=0, skipna=True, *args, **kwargs):
        if skipna is None:
            skipna = True
        return super(Series, self).idxmax(axis=axis, skipna=skipna, *args, **kwargs)

    def idxmin(self, axis=0, skipna=True, *args, **kwargs):
        if skipna is None:
            skipna = True
        return super(Series, self).idxmin(axis=axis, skipna=skipna, *args, **kwargs)

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction="forward",
        limit_area=None,
        downcast=None,
        **kwargs
    ):
        return self._default_to_pandas(
            pandas.Series.interpolate,
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs
        )

    def item(self):
        return self[0]

    def items(self):
        index_iter = iter(self.index)

        def item_builder(df):
            s = df.iloc[:, 0]
            s.index = [next(index_iter)]
            s.name = self.name
            return s.items()

        partition_iterator = PartitionIterator(self._query_compiler, 0, item_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):
        return self.items()

    def keys(self):
        return self.index

    def le(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).le(new_other, level=level, axis=axis)

    def lt(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).lt(new_other, level=level, axis=axis)

    def map(self, arg, na_action=None):
        return self.__constructor__(
            query_compiler=self._query_compiler._map_partitions(
                lambda df: pandas.DataFrame(df.iloc[:, 0].map(arg, na_action=na_action))
            )
        )

    def memory_usage(self, index=True, deep=False):
        if index:
            result = self._reduce_dimension(
                self._query_compiler.memory_usage(index=False, deep=deep)
            )
            index_value = self.index.memory_usage(deep=deep)
            return result + index_value
        return super(Series, self).memory_usage(index=index, deep=deep)

    def mod(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).mod(
            new_other, level=level, fill_value=None, axis=axis
        )

    def mul(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).mul(
            new_other, level=level, fill_value=None, axis=axis
        )

    multiply = rmul = mul

    def ne(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).ne(new_other, level=level, axis=axis)

    def nlargest(self, n=5, keep="first"):
        return self._default_to_pandas(pandas.Series.nlargest, n=n, keep=keep)

    def nonzero(self):
        return self.to_numpy().nonzero()

    def nsmallest(self, n=5, keep="first"):
        return self._default_to_pandas(pandas.Series.nsmallest, n=n, keep=keep)

    @property
    def plot(
        self,
        kind="line",
        ax=None,
        figsize=None,
        use_index=True,
        title=None,
        grid=None,
        legend=False,
        style=None,
        logx=False,
        logy=False,
        loglog=False,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        rot=None,
        fontsize=None,
        colormap=None,
        table=False,
        yerr=None,
        xerr=None,
        label=None,
        secondary_y=False,
        **kwds
    ):
        return self._to_pandas().plot

    def pow(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).pow(
            new_other, level=level, fill_value=None, axis=axis
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        axis = self._get_axis_number(axis)
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan
        return super(Series, self).prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    product = prod

    def ptp(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._default_to_pandas(
            pandas.Series.ptp,
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs
        )

    def put(self, *args, **kwargs):
        return self._default_to_pandas(pandas.Series.put, *args, **kwargs)

    radd = add

    def ravel(self, order="C"):
        return self._default_to_pandas(pandas.Series.ravel, order=order)

    def reindex(self, index=None, **kwargs):
        method = kwargs.pop("method", None)
        level = kwargs.pop("level", None)
        copy = kwargs.pop("copy", True)
        limit = kwargs.pop("limit", None)
        tolerance = kwargs.pop("tolerance", None)
        fill_value = kwargs.pop("fill_value", None)
        if kwargs:
            raise TypeError(
                "reindex() got an unexpected keyword "
                'argument "{0}"'.format(list(kwargs.keys())[0])
            )
        return super(Series, self).reindex(
            index=index,
            method=method,
            level=level,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
            fill_value=fill_value,
        )

    def reindex_axis(self, labels, axis=0, **kwargs):
        if axis != 0:
            raise ValueError("cannot reindex series on non-zero axis!")
        return self.reindex(index=labels, **kwargs)

    def rename(self, index=None, **kwargs):
        non_mapping = is_scalar(index) or (
            is_list_like(index) and not is_dict_like(index)
        )
        if non_mapping:
            if kwargs.get("inplace", False):
                self.name = index
            else:
                self_cp = self.copy()
                self_cp.name = index
                return self_cp
        else:
            from .dataframe import DataFrame

            result = DataFrame(self.copy()).rename(index=index, **kwargs).squeeze()
            result.name = self.name
            return result

    def reorder_levels(self, order):
        return self._default_to_pandas(pandas.Series.reorder_levels, order)

    def repeat(self, repeats, axis=None):
        return self._default_to_pandas(pandas.Series.repeat, repeats, axis=axis)

    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        if drop and level is None:
            new_idx = pandas.RangeIndex(len(self.index))
            if inplace:
                self.index = new_idx
                self.name = name or self.name
            else:
                result = self.copy()
                result.index = new_idx
                result.name = name or self.name
                return result
        elif not drop and inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series to create a DataFrame"
            )
        else:
            obj = self.copy()
            if name is not None:
                obj.name = name
            from .dataframe import DataFrame

            return DataFrame(self.copy()).reset_index(
                level=level, drop=drop, inplace=inplace
            )

    def rdivmod(self, other, level=None, fill_value=None, axis=0):
        return self._default_to_pandas(
            pandas.Series.rdivmod, other, level=level, fill_value=fill_value, axis=axis
        )

    def rfloordiv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rfloordiv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rmod(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rmod(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rpow(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rpow(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rsub(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rsub(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rtruediv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rtruediv(
            new_other, level=level, fill_value=None, axis=axis
        )

    rdiv = rtruediv

    def searchsorted(self, value, side="left", sorter=None):
        return self._default_to_pandas(
            pandas.Series.searchsorted, value, side=side, sorter=sorter
        )

    def set_value(self, label, value, takeable=False):
        return self._default_to_pandas("set_value", label, value, takeable=takeable)

    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
    ):
        from .dataframe import DataFrame

        # When we convert to a DataFrame, the name is automatically converted to 0 if it
        # is None, so we do this to avoid a KeyError.
        by = self.name if self.name is not None else 0
        result = (
            DataFrame(self.copy())
            .sort_values(
                by=by,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
            )
            .squeeze(axis=1)
        )
        result.name = self.name
        return self._create_or_update_from_compiler(
            result._query_compiler, inplace=inplace
        )

    def sparse(self, data=None):
        return self._default_to_pandas(pandas.Series.sparse, data=data)

    def squeeze(self, axis=None):
        if axis is not None:
            # Validate `axis`
            pandas.Series._get_axis_number(axis)
        if len(self.index) == 1:
            return self._reduce_dimension(self._query_compiler)
        else:
            return self.copy()

    def sub(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).sub(
            new_other, level=level, fill_value=None, axis=axis
        )

    subtract = sub

    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        axis = self._get_axis_number(axis)
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan
        return super(Series, self).sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    def swaplevel(self, i=-2, j=-1, copy=True):
        return self._default_to_pandas("swaplevel", i=i, j=j, copy=copy)

    def tail(self, n=5):
        if n == 0:
            return Series(dtype=self.dtype)
        return super(Series, self).tail(n)

    def to_frame(self, name=None):
        from .dataframe import DataFrame

        self_cp = self.copy()
        if name is not None:
            self_cp.name = name
        return DataFrame(self.copy())

    def to_list(self):
        return self._default_to_pandas(pandas.Series.to_list)

    def to_numpy(self, dtype=None, copy=False):
        """Convert the Series to a NumPy array.

        Args:
            dtype: The dtype to pass to numpy.asarray()
            copy: Whether to ensure that the returned value is a not a view on another
                array.

        Returns:
            A numpy array.
        """
        return super(Series, self).to_numpy(dtype, copy).flatten()

    tolist = to_list

    # TODO(williamma12): When we implement to_timestamp, have this call the version
    # in base.py
    def to_period(self, freq=None, copy=True):
        return self._default_to_pandas("to_period", freq=freq, copy=copy)

    def to_string(
        self,
        buf=None,
        na_rep="NaN",
        float_format=None,
        header=True,
        index=True,
        length=False,
        dtype=False,
        name=False,
        max_rows=None,
    ):
        return self._default_to_pandas(
            pandas.Series.to_string,
            buf=buf,
            na_rep=na_rep,
            float_format=float_format,
            header=header,
            index=index,
            length=length,
            dtype=dtype,
            name=name,
            max_rows=max_rows,
        )

    # TODO(williamma12): When we implement to_timestamp, have this call the version
    # in base.py
    def to_timestamp(self, freq=None, how="start", copy=True):
        return self._default_to_pandas("to_timestamp", freq=freq, how=how, copy=copy)

    def transpose(self, *args, **kwargs):
        return self

    T = property(transpose)

    def truediv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).truediv(
            new_other, level=level, fill_value=None, axis=axis
        )

    div = divide = truediv

    def truncate(self, before=None, after=None, axis=None, copy=True):
        return self._default_to_pandas(
            pandas.Series.truncate, before=before, after=after, axis=axis, copy=copy
        )

    def unique(self):
        return self._default_to_pandas(pandas.Series.unique)

    def update(self, other):
        return self._default_to_pandas(pandas.Series.update, other)

    def valid(self, inplace=False, **kwargs):
        return self._default_to_pandas(pandas.Series.valid, inplace=inplace, **kwargs)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ):
        return self._default_to_pandas(
            pandas.Series.value_counts,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            bins=bins,
            dropna=dropna,
        )

    def view(self, dtype=None):
        return self._default_to_pandas(pandas.Series.view, dtype=dtype)

    def where(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        raise_on_error=None,
    ):
        if isinstance(other, Series):
            other = to_pandas(other)
        return self._default_to_pandas(
            pandas.Series.where,
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
            raise_on_error=raise_on_error,
        )

    def xs(self, key, axis=0, level=None, drop_level=True):  # pragma: no cover
        raise NotImplementedError("Not Yet implemented.")

    @property
    def asobject(self):
        # We cannot default to pandas without a named function to call.
        def asobject(df):
            return df.asobject

        return self._default_to_pandas(asobject)

    @property
    def axes(self):
        return [self.index]

    @property
    def base(self):
        # We cannot default to pandas without a named function to call.
        def base(df):
            return df.base

        return self._default_to_pandas(base)

    @property
    def cat(self):
        return self._default_to_pandas(pandas.Series.cat)

    @property
    def data(self):
        # We cannot default to pandas without a named function to call.
        def data(df):
            return df.data

        return self._default_to_pandas(data)

    @property
    def dt(self):
        return self._default_to_pandas(pandas.Series.dt)

    @property
    def dtype(self):
        return self._query_compiler.dtypes.squeeze()

    dtypes = dtype

    @property
    def empty(self):
        return len(self.index) == 0

    @property
    def flags(self):
        # We cannot default to pandas without a named function to call.
        def flags(df):
            return df.flags

        return self._default_to_pandas(flags)

    @property
    def ftype(self):
        return "{}:dense".format(self.dtype)

    ftypes = ftype

    @property
    def hasnans(self):
        return self.isna().sum() > 0

    @property
    def imag(self):
        # We cannot default to pandas without a named function to call.
        def imag(df):
            return df.imag

        return self._default_to_pandas(imag)

    @property
    def is_monotonic(self):
        # We cannot default to pandas without a named function to call.
        def is_monotonic(df):
            return df.is_monotonic

        return self._default_to_pandas(is_monotonic)

    @property
    def is_monotonic_decreasing(self):
        # We cannot default to pandas without a named function to call.
        def is_monotonic_decreasing(df):
            return df.is_monotonic

        return self._default_to_pandas(is_monotonic_decreasing)

    @property
    def is_monotonic_increasing(self):
        # We cannot default to pandas without a named function to call.
        def is_monotonic_increasing(df):
            return df.is_monotonic

        return self._default_to_pandas(is_monotonic_increasing)

    @property
    def is_unique(self):
        # We cannot default to pandas without a named function to call.
        def is_unique(df):
            return df.is_unique

        return self._default_to_pandas(is_unique)

    @property
    def itemsize(self):
        # We cannot default to pandas without a named function to call.
        def itemsize(df):
            return df.itemsize

        return self._default_to_pandas(itemsize)

    @property
    def nbytes(self):
        # We cannot default to pandas without a named function to call.
        def nbytes(df):
            return df.nbytes

        return self._default_to_pandas(nbytes)

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this Series.
        """
        # Series have an invariant that requires they be 1 dimension.
        return 1

    @property
    def real(self):
        # We cannot default to pandas without a named function to call.
        def real(df):
            return df.real

        return self._default_to_pandas(real)

    @property
    def shape(self):
        return (len(self),)

    @property
    def strides(self):
        # We cannot default to pandas without a named function to call.
        def strides(df):
            return df.strides

        return self._default_to_pandas(strides)

    @property
    def str(self):
        return StringMethods(self)

    def _to_pandas(self):
        df = self._query_compiler.to_pandas()
        series = df[df.columns[0]]
        if series.name == "__reduced__":
            series.name = None
        return series


class StringMethods(object):
    def __init__(self, series):
        # Check if dtypes is objects

        self._series = series
        self._query_compiler = series._query_compiler

    def cat(self, others=None, sep=None, na_rep=None, join=None):
        if isinstance(others, Series):
            others = others._to_pandas()
        return self._default_to_pandas(
            pandas.Series.str.cat, others=others, sep=sep, na_rep=na_rep, join=join
        )

    def split(self, pat=None, n=-1, expand=False):
        if not pat and pat is not None and pandas.compat.PY3:
            raise ValueError("split() requires a non-empty pattern match.")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.split, pat=pat, n=n, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_split(
                    pat=pat, n=n, expand=expand
                )
            )

    def rsplit(self, pat=None, n=-1, expand=False):
        if not pat and pat is not None:
            raise ValueError("rsplit() requires a non-empty pattern match.")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.rsplit, pat=pat, n=n, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_rsplit(
                    pat=pat, n=n, expand=expand
                )
            )

    def get(self, i):
        return Series(query_compiler=self._query_compiler.str_get(i))

    def join(self, sep):
        if sep is None:
            raise AttributeError("'NoneType' object has no attribute 'join'")
        return Series(query_compiler=self._query_compiler.str_join(sep))

    def get_dummies(self, sep="|"):
        return self._default_to_pandas(pandas.Series.str.get_dummies, sep=sep)

    def contains(self, pat, case=True, flags=0, na=np.NaN, regex=True):
        if pat is None and not case:
            raise AttributeError("'NoneType' object has no attribute 'upper'")
        return Series(
            query_compiler=self._query_compiler.str_contains(
                pat, case=case, flags=flags, na=na, regex=regex
            )
        )

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True):
        if not (is_string_like(repl) or callable(repl)):
            raise TypeError("repl must be a string or callable")
        return Series(
            query_compiler=self._query_compiler.str_replace(
                pat, repl, n=n, case=case, flags=flags, regex=regex
            )
        )

    def repeats(self, repeats):
        return Series(query_compiler=self._query_compiler.str_repeats(repeats))

    def pad(self, width, side="left", fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_pad(
                width, side=side, fillchar=fillchar
            )
        )

    def center(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_center(width, fillchar=fillchar)
        )

    def ljust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_ljust(width, fillchar=fillchar)
        )

    def rjust(self, width, fillchar=" "):
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")
        return Series(
            query_compiler=self._query_compiler.str_rjust(width, fillchar=fillchar)
        )

    def zfill(self, width):
        return Series(query_compiler=self._query_compiler.str_zfill(width))

    def wrap(self, width, **kwargs):
        if width <= 0:
            raise ValueError("invalid width {} (must be > 0)".format(width))
        return Series(query_compiler=self._query_compiler.str_wrap(width, **kwargs))

    def slice(self, start=None, stop=None, step=None):
        if step == 0:
            raise ValueError("slice step cannot be zero")
        return Series(
            query_compiler=self._query_compiler.str_slice(
                start=start, stop=stop, step=step
            )
        )

    def slice_replace(self, start=None, stop=None, repl=None):
        return Series(
            query_compiler=self._query_compiler.str_slice_replace(
                start=start, stop=stop, repl=repl
            )
        )

    def count(self, pat, flags=0, **kwargs):
        import re

        if not isinstance(pat, (str, re._pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_count(pat, flags=flags, **kwargs)
        )

    def startswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_startswith(pat, na=na))

    def endswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_endswith(pat, na=na))

    def findall(self, pat, flags=0, **kwargs):
        import re

        if not isinstance(pat, (str, re._pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_findall(pat, flags=flags, **kwargs)
        )

    def match(self, pat, case=True, flags=0, na=np.NaN):
        import re

        if not isinstance(pat, (str, re._pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_match(pat, flags=flags, na=na)
        )

    def extract(self, pat, flags=0, expand=True):
        return self._default_to_pandas(
            pandas.Series.str.extract, pat, flags=flags, expand=expand
        )

    def extractall(self, pat, flags=0):
        return self._default_to_pandas(pandas.Series.str.extractall, pat, flags=flags)

    def len(self):
        return Series(query_compiler=self._query_compiler.str_len())

    def strip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_strip(to_strip=to_strip))

    def rstrip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_rstrip(to_strip=to_strip))

    def lstrip(self, to_strip=None):
        return Series(query_compiler=self._query_compiler.str_lstrip(to_strip=to_strip))

    def partition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.partition, sep=sep, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_partition(
                    sep=sep, expand=expand
                )
            )

    def rpartition(self, sep=" ", expand=True):
        if sep is not None and len(sep) == 0:
            raise ValueError("empty separator")

        if expand:
            return self._default_to_pandas(
                pandas.Series.str.rpartition, sep=sep, expand=expand
            )
        else:
            return Series(
                query_compiler=self._query_compiler.str_rpartition(
                    sep=sep, expand=expand
                )
            )

    def lower(self):
        return Series(query_compiler=self._query_compiler.str_lower())

    def upper(self):
        return Series(query_compiler=self._query_compiler.str_upper())

    def find(self, sub, start=0, end=None):
        if not isinstance(sub, pandas.compat.string_types):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_find(sub, start=start, end=end)
        )

    def rfind(self, sub, start=0, end=None):
        if not isinstance(sub, pandas.compat.string_types):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end)
        )

    def index(self, sub, start=0, end=None):
        if not isinstance(sub, pandas.compat.string_types):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_index(sub, start=start, end=end)
        )

    def rindex(self, sub, start=0, end=None):
        if not isinstance(sub, pandas.compat.string_types):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_rindex(sub, start=start, end=end)
        )

    def capitalize(self):
        return Series(query_compiler=self._query_compiler.str_capitalize())

    def swapcase(self):
        return Series(query_compiler=self._query_compiler.str_swapcase())

    def normalize(self, form):
        return Series(query_compiler=self._query_compiler.str_normalize(form))

    def translate(self, table, deletechars=None):
        if pandas.compat.PY3:
            if deletechars is not None:
                raise ValueError(
                    "deletechars is not a valid argument for "
                    "str.translate in python 3. You should simply "
                    "specify character deletions in the table "
                    "argument"
                )
        return Series(
            query_compiler=self._query_compiler.str_translate(
                table, deletechars=deletechars
            )
        )

    def isalnum(self):
        return Series(query_compiler=self._query_compiler.str_isalnum())

    def isalpha(self):
        return Series(query_compiler=self._query_compiler.str_isalpha())

    def isdigit(self):
        return Series(query_compiler=self._query_compiler.str_isdigit())

    def isspace(self):
        return Series(query_compiler=self._query_compiler.str_isspace())

    def islower(self):
        return Series(query_compiler=self._query_compiler.str_islower())

    def isupper(self):
        return Series(query_compiler=self._query_compiler.str_isupper())

    def istitle(self):
        return Series(query_compiler=self._query_compiler.str_istitle())

    def isnumeric(self):
        return Series(query_compiler=self._query_compiler.str_isnumeric())

    def isdecimal(self):
        return Series(query_compiler=self._query_compiler.str_isdecimal())

    def _default_to_pandas(self, op, *args, **kwargs):
        return self._series._default_to_pandas(
            lambda series: op(series.str, *args, **kwargs)
        )
