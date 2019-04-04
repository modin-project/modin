from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from pandas.core.dtypes.common import is_dict_like, is_list_like, is_scalar
from .base import BasePandasDataset
from .iterator import PartitionIterator
from .utils import _inherit_docstrings
from .utils import from_pandas


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
        if len(query_compiler.columns) != 1:
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
        pass

    def _validate_dtypes_min_max(self, axis, numeric_only):
        pass

    def _validate_dtypes(self, numeric_only=False):
        pass

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace and len(new_query_compiler.columns) == 1:
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

    def __array_prepare__(self, result, context=None):
        return self._to_pandas().__array_prepare__(result, context=context)

    @property
    def __array_priority__(self):
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
        raise NotImplementedError("Not Yet implemented.")

    def __div__(self, right):
        return self.div(right)

    def __divmod__(self, right):
        return self // right, self % right

    def __float__(self):
        return float(self.squeeze())

    def __floordiv__(self, right):
        return self.floordiv(right)

    def __getitem__(self, key):
        if (
            key in self.keys()
            or is_list_like(key)
            and all(k in self.keys() for k in key)
        ):
            return self.loc[key]
        else:
            return self.iloc[key]

    def __iadd__(self, other):
        return self.add(other)

    def __imul__(self, other):
        return self.mul(other)

    def __int__(self):
        return int(self.squeeze())

    def __ipow__(self, other):
        return self.pow(other)

    def __isub__(self, other):
        return self.sub(other)

    def __iter__(self):
        return self._to_pandas().__iter__()

    def __itruediv__(self, other):
        return self.truediv(other)

    def __mod__(self, right):
        return self.mod(right)

    def __mul__(self, right):
        return self.mul(right)

    def __pow__(self, right):
        return self.pow(right)

    def __repr__(self):
        # In the future, we can have this be configurable, just like Pandas.
        num_rows = 60
        num_cols = 30
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
        raise NotImplementedError("Not Yet implemented.")

    def __sub__(self, right):
        return self.sub(right)

    def __truediv__(self, right):
        return self.truediv(right)

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

    def align(
        self,
        other,
        join="outer",
        axis=None,
        level=None,
        copy=True,
        fill_value=None,
        method=None,
        limit=None,
        fill_axis=0,
        broadcast_axis=None,
    ):
        raise NotImplementedError("Not Yet implemented.")

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
            else:
                # Matching pandas behavior of naming the Series columns 0
                self.name = 0
                for i in range(len(to_append)):
                    if isinstance(to_append[i], Series):
                        to_append[i].name = 0
                        to_append[i] = DataFrame(to_append[i])
                return DataFrame(self).append(
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
            return DataFrame(self).append(
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

    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        return self.idxmax(axis=axis, skipna=skipna)

    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        return self.idxmin(axis=axis, skipna=skipna)

    def argsort(self, axis=0, kind="quicksort", order=None):
        return self._default_to_pandas(
            pandas.Series.argsort, axis=axis, kind=kind, order=order
        )

    def autocorr(self, lag=1):
        return self._default_to_pandas(pandas.Series.autocorr, lag=lag)

    def between(self, left, right, inclusive=True):
        return self._default_to_pandas(
            pandas.Series.between, left, right, inclusive=inclusive
        )

    def compound(self, axis=None, skipna=None, level=None):
        return self._default_to_pandas(
            pandas.Series.compound, axis=axis, skipna=skipna, level=level
        )

    def compress(self, condition, *args, **kwargs):
        return self._default_to_pandas(
            pandas.Series.compress, condition, *args, **kwargs
        )

    def consolidate(self, inplace=False):
        self._create_or_update_from_compiler(
            self._default_to_pandas(
                pandas.Series.consolidate, inplace=inplace
            )._query_compiler,
            inplace=inplace,
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
        return self._default_to_pandas(
            pandas.Series.corr, other, method=method, min_periods=min_periods
        )

    def cov(self, other, min_periods=None):
        return self._default_to_pandas(
            pandas.Series.cov, other, min_periods=min_periods
        )

    def describe(self, percentiles=None, include=None, exclude=None):
        # Pandas ignores the `include` and `exclude` for Series for some reason.
        return super(Series, self).describe(percentiles=percentiles)

    def diff(self, periods=1):
        return super(Series, self).diff(periods=periods, axis=0)

    def drop(self, labels, axis=0, level=None, inplace=False, errors="raise"):
        return super(Series, self).drop(
            labels, axis=axis, level=level, inplace=inplace, errors=errors
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
        raise NotImplementedError("Not Yet implemented.")

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
        **kwargs
    ):
        raise NotImplementedError("Not Yet implemented.")

    def gt(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).gt(new_other, level=level, axis=axis)

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

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction="forward",
        downcast=None,
        **kwargs
    ):
        raise NotImplementedError("Not Yet implemented.")

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

            return DataFrame(self).rename(index=index, **kwargs).squeeze()

    def reorder_levels(self, order):
        return self._default_to_pandas(pandas.Series.reorder_levels, order)

    def repeat(self, repeats, *args, **kwargs):
        return self._default_to_pandas(pandas.Series.repeat, repeats, *args, **kwargs)

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
        elif drop and inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series " "to create a DataFrame"
            )
        else:
            obj = self.copy()
            if name is not None:
                obj.name = name
            return super(Series, obj).reset_index(
                level=level, drop=drop, inplace=inplace
            )

    def reshape(self, *args, **kwargs):
        return self._default_to_pandas(*args, **kwargs)

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

    def searchsorted(self, value, side="left", sorter=None):
        return self._default_to_pandas(
            pandas.Series.searchsorted, value, side=side, sorter=sorter
        )

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

    def to_frame(self, name=None):
        from .dataframe import DataFrame

        self_cp = self.copy()
        if name is not None:
            self_cp.name = name
        return DataFrame(self)

    def tolist(self):
        return self._default_to_pandas(pandas.Series.to_list())

    def transpose(self, *args, **kwargs):
        return self

    T = property(transpose)

    def truediv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).truediv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def truncate(self, before=None, after=None, axis=None, copy=True):
        return self._default_to_pandas(
            pandas.Series.truncate, before=before, after=after, axis=axis, copy=copy
        )

    def unique(self):
        return self._default_to_pandas(pandas.Series.unique)

    def upandasate(self, other):
        return self._default_to_pandas(pandas.Series.upandasate, other)

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
        try_cast=False,
        raise_on_error=True,
    ):
        return self._default_to_pandas(
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            try_cast=try_cast,
            raise_on_error=raise_on_error,
        )

    def xs(key, axis=0, level=None, drop_level=True):
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
    def data(self):
        # We cannot default to pandas without a named function to call.
        def data(df):
            return df.data

        return self._default_to_pandas(data)

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
    def is_copy(self):
        raise NotImplementedError("Not Yet implemented.")

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

    def _to_pandas(self):
        df = self._query_compiler.to_pandas()
        series = df[df.columns[0]]
        if series.name == "__reduced__":
            series.name = None
        return series
