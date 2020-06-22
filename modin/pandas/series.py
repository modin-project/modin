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

import numpy as np
import pandas
from pandas.core.common import apply_if_callable, is_bool_indexer
import pandas._libs.lib as lib
from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
    is_scalar,
)
import sys
import warnings

from .base import BasePandasDataset
from .iterator import PartitionIterator
from .utils import _inherit_docstrings
from .utils import from_pandas, to_pandas

if sys.version_info[0] == 3 and sys.version_info[1] >= 7:
    # Python >= 3.7
    from re import Pattern as _pattern_type
else:
    # Python <= 3.6
    from re import _pattern_type


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
        """
    One-dimensional ndarray with axis labels (including time series).

    Args:
        data: Contains data stored in Series.
        index: Values must be hashable and have the same length as `data`.
        dtype: Data type for the output Series. If not specified, this will be
            inferred from `data`.
        name: The name to give to the Series.
        copy: Copy input data.
        query_compiler: A query compiler object to create the Series from.
    """
        if isinstance(data, type(self)):
            query_compiler = data._query_compiler.copy()
            if index is not None:
                if any(i not in data.index for i in index):
                    raise NotImplementedError(
                        "Passing non-existent columns or index values to constructor "
                        "not yet implemented."
                    )
                query_compiler = data.loc[index]._query_compiler
        if query_compiler is None:
            # Defaulting to pandas
            warnings.warn(
                "Distributing {} object. This may take some time.".format(type(data))
            )
            if name is None:
                name = "__reduced__"
                if isinstance(data, pandas.Series) and data.name is not None:
                    name = data.name

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
        if name is not None:
            self._query_compiler = self._query_compiler
            self.name = name

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

    def __radd__(self, left):
        return self.add(left)

    def __and__(self, other):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__and__(new_other)

    def __array__(self, dtype=None):
        return super(Series, self).__array__(dtype).flatten()

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

    def __rdiv__(self, left):
        return self.rdiv(left)

    def __divmod__(self, right):
        return self.divmod(right)

    def __rdivmod__(self, left):
        return self.rdivmod(left)

    def __float__(self):
        return float(self.squeeze())

    def __floordiv__(self, right):
        return self.floordiv(right)

    def __rfloordiv__(self, right):
        return self.rfloordiv(right)

    def _getitem(self, key):
        key = apply_if_callable(key, self)
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

    def __getattr__(self, key):
        """After regular attribute access, looks up the name in the index

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.index:
                return self[key]
            raise e

    def __int__(self):
        return int(self.squeeze())

    def __iter__(self):
        return self._to_pandas().__iter__()

    def __mod__(self, right):
        return self.mod(right)

    def __rmod__(self, left):
        return self.rmod(left)

    def __mul__(self, right):
        return self.mul(right)

    def __rmul__(self, left):
        return self.rmul(left)

    def __or__(self, other):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__or__(new_other)

    def __pow__(self, right):
        return self.pow(right)

    def __rpow__(self, left):
        return self.rpow(left)

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
        # Propagate changes back to parent so that column in dataframe had the same contents
        if self._parent is not None:
            self._parent[self.name] = self

    def __sub__(self, right):
        return self.sub(right)

    def __rsub__(self, left):
        return self.rsub(left)

    def __truediv__(self, right):
        return self.truediv(right)

    def __rtruediv__(self, left):
        return self.rtruediv(left)

    __iadd__ = __add__
    __imul__ = __add__
    __ipow__ = __pow__
    __isub__ = __sub__
    __itruediv__ = __truediv__

    @property
    def values(self):
        """Create a NumPy array with the values from this Series.

        Returns:
            The NumPy representation of this object.
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
            isinstance(func, str)
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

    def argmax(self, axis=None, skipna=True, *args, **kwargs):
        result = self.idxmax(axis=axis, skipna=skipna, *args, **kwargs)
        if np.isnan(result) or result is pandas.NA:
            result = -1
        return result

    def argmin(self, axis=None, skipna=True, *args, **kwargs):
        result = self.idxmin(axis=axis, skipna=skipna, *args, **kwargs)
        if np.isnan(result) or result is pandas.NA:
            result = -1
        return result

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
        return super(Series, self).combine(
            other, lambda s1, s2: s1.combine(s2, func, fill_value=fill_value)
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
        """
        Compute covariance with Series, excluding missing values.

        Parameters
        ----------
        other : Series
            Series with which to compute the covariance.
        min_periods : int, optional
            Minimum number of observations needed to have a valid result.

        Returns
        -------
        float
            Covariance between Series and other normalized by N-1
            (unbiased estimator).

        Notes
        -----
        Covariance floating point precision may slightly differ from pandas.
        """
        this, other = self.align(other, join="inner", copy=False)
        this = self.__constructor__(this)
        other = self.__constructor__(other)
        if len(this) == 0:
            return np.nan

        if len(this) != len(other):
            raise ValueError("Operands must have same size")

        if min_periods is None:
            min_periods = 1

        valid = this.notna() & other.notna()
        if not valid.all():
            this = this[valid]
            other = other[valid]

        if len(this) < min_periods:
            return np.nan

        this = this.astype(dtype="float64")
        other = other.astype(dtype="float64")

        this -= this.mean()
        other -= other.mean()

        other = other.__constructor__(query_compiler=other._query_compiler.conj())
        result = this * other / (len(this) - 1)
        result = result.sum()
        return result

    def describe(self, percentiles=None, include=None, exclude=None):
        # Pandas ignores the `include` and `exclude` for Series for some reason.
        return super(Series, self).describe(percentiles=percentiles)

    def diff(self, periods=1):
        return super(Series, self).diff(periods=periods, axis=0)

    def divmod(self, other, level=None, fill_value=None, axis=0):
        return self._default_to_pandas(
            pandas.Series.divmod, other, level=level, fill_value=fill_value, axis=axis
        )

    def dot(self, other):
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame, or the Series and
        each columns of an array.

        It can also be called using `self @ other` in Python >= 3.5.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series or numpy.ndarray
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame or a numpy.ndarray between the Series
            and each columns of the numpy array.

        See Also
        --------
        DataFrame.dot: Compute the matrix product with the DataFrame.
        Series.mul: Multiplication of series and other, element-wise.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.
        """
        if isinstance(other, BasePandasDataset):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("Matrices are not aligned")

            qc = other.reindex(index=common)._query_compiler
            if isinstance(other, Series):
                return self._reduce_dimension(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=True, squeeze_other=True
                    )
                )
            else:
                return self.__constructor__(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=True, squeeze_other=False
                    )
                )

        other = np.asarray(other)
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                "Dot product shape mismatch, {} vs {}".format(self.shape, other.shape)
            )

        if len(other.shape) > 1:
            return (
                self._query_compiler.dot(other, squeeze_self=True).to_numpy().squeeze()
            )

        return self._reduce_dimension(
            query_compiler=self._query_compiler.dot(other, squeeze_self=True)
        )

    def drop_duplicates(self, keep="first", inplace=False):
        return super(Series, self).drop_duplicates(keep=keep, inplace=inplace)

    def dropna(self, axis=0, inplace=False, how=None):
        return super(Series, self).dropna(axis=axis, inplace=inplace)

    def duplicated(self, keep="first"):
        return self.to_frame().duplicated(keep=keep)

    def eq(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).eq(new_other, level=level, axis=axis)

    def equals(self, other):
        return (
            self.name == other.name
            and self.index.equals(other.index)
            and self.eq(other).all()
        )

    def explode(self):
        return self._default_to_pandas(pandas.Series.explode)

    def factorize(self, sort=False, na_sentinel=-1):
        return self._default_to_pandas(
            pandas.Series.factorize, sort=sort, na_sentinel=na_sentinel
        )

    def floordiv(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).floordiv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def ge(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).ge(new_other, level=level, axis=axis)

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
    ):
        from .groupby import SeriesGroupBy

        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        # SeriesGroupBy expects a query compiler object if it is available
        if isinstance(by, Series):
            by = by._query_compiler
        elif callable(by):
            by = by(self.index)
        elif by is None and level is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        return SeriesGroupBy(
            self,
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            squeeze,
            idx_name=None,
            observed=observed,
            drop=False,
        )

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
        **kwds,
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
            **kwds,
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
        **kwargs,
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
            **kwargs,
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
        if not callable(arg) and hasattr(arg, "get"):
            mapper = arg

            def arg(s):
                return mapper.get(s, np.nan)

        return self.__constructor__(
            query_compiler=self._query_compiler.applymap(
                lambda s: arg(s)
                if pandas.isnull(s) is not True or na_action is None
                else s
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

    def mode(self, dropna=True):
        return super(Series, self).mode(numeric_only=False, dropna=dropna)

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

    def nsmallest(self, n=5, keep="first"):
        """
        Return the smallest `n` elements.
        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.
        keep : {'first', 'last', 'all'}, default 'first'
            When there are duplicate values that cannot all fit in a
            Series of `n` elements:
            - ``first`` : return the first `n` occurrences in order
                of appearance.
            - ``last`` : return the last `n` occurrences in reverse
                order of appearance.
            - ``all`` : keep all occurrences. This can result in a Series of
                size larger than `n`.
        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.
        """
        return Series(query_compiler=self._query_compiler.nsmallest(n=n, keep=keep))

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
        **kwds,
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
        **kwargs,
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
            **kwargs,
        )

    product = prod
    radd = add

    def ravel(self, order="C"):
        """
        Returns the flattened containing data as ndarray.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional

        Returns
        ----------
        numpy.ndarray or ndarray-like
            Flattened data of the Series.

        """
        data = self._query_compiler.to_numpy().flatten(order=order)
        if isinstance(self.dtype, pandas.CategoricalDtype):
            data = pandas.Categorical(data, dtype=self.dtype)

        return data

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

    def rename(
        self,
        index=None,
        *,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):
        non_mapping = is_scalar(index) or (
            is_list_like(index) and not is_dict_like(index)
        )
        if non_mapping:
            if inplace:
                self.name = index
            else:
                self_cp = self.copy()
                self_cp.name = index
                return self_cp
        else:
            from .dataframe import DataFrame

            result = DataFrame(self.copy()).rename(index=index).squeeze()
            result.name = self.name
            return result

    def repeat(self, repeats, axis=None):
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        Series
            Newly created Series with repeated elements.
        """
        if (isinstance(repeats, int) and repeats == 0) or (
            is_list_like(repeats) and len(repeats) == 1 and repeats[0] == 0
        ):
            return self.__constructor__()

        return self.__constructor__(query_compiler=self._query_compiler.repeat(repeats))

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

    def quantile(self, q=0.5, interpolation="linear"):
        return super(Series, self).quantile(
            q=q, numeric_only=False, interpolation=interpolation
        )

    def reorder_levels(self, order):
        return super(Series, self).reorder_levels(order)

    def searchsorted(self, value, side="left", sorter=None):
        return self._default_to_pandas(
            pandas.Series.searchsorted, value, side=side, sorter=sorter
        )

    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
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

    @property
    def sparse(self):
        return self._default_to_pandas(pandas.Series.sparse)

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
        **kwargs,
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
            **kwargs,
        )

    def swaplevel(self, i=-2, j=-1, copy=True):
        return self._default_to_pandas("swaplevel", i=i, j=j, copy=copy)

    def take(self, indices, axis=0, is_copy=None, **kwargs):
        return super(Series, self).take(indices, axis=axis, is_copy=is_copy, **kwargs)

    def _to_datetime(self, **kwargs):
        """
        Convert `self` to datetime.

        Returns
        -------
        datetime
            Series: Series of datetime64 dtype
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.to_datetime(**kwargs)
        )

    def _to_numeric(self, **kwargs):
        """
        Convert `self` to numeric.

        Returns
        -------
        numeric
            Series: Series of numeric dtype
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.to_numeric(**kwargs)
        )

    def to_dict(self, into=dict):  # pragma: no cover
        return self._default_to_pandas("to_dict", into=into)

    def to_frame(self, name=None):
        from .dataframe import DataFrame

        self_cp = self.copy()
        if name is not None:
            self_cp.name = name
        return DataFrame(self_cp)

    def to_list(self):
        return self._default_to_pandas(pandas.Series.to_list)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default, **kwargs):
        """Convert the Series to a NumPy array.

        Args:
            dtype: The dtype to pass to numpy.asarray()
            copy: Whether to ensure that the returned value is a not a view on another
                array.

        Returns:
            A NumPy array.
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
        min_rows=None,
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
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        Returns
        -------
        ndarray
            The unique values returned as a NumPy array.
        """
        return self._query_compiler.unique().to_numpy().squeeze()

    def update(self, other):
        """
        Modify Series in place using non-NA values from passed
        Series. Aligns on index.

        Parameters
        ----------
        other : Series, or object coercible into Series
        """
        if not isinstance(other, Series):
            other = Series(other)
        query_compiler = self._query_compiler.series_update(other._query_compiler)
        self._update_inplace(new_query_compiler=query_compiler)

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
        return self.__constructor__(
            query_compiler=self._query_compiler.series_view(dtype=dtype)
        )

    def where(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
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
        )

    def xs(self, key, axis=0, level=None, drop_level=True):  # pragma: no cover
        raise NotImplementedError("Not Yet implemented.")

    @property
    def attrs(self):
        def attrs(df):
            return df.attrs

        self._default_to_pandas(attrs)

    @property
    def axes(self):
        return [self.index]

    @property
    def cat(self):
        return self._default_to_pandas(pandas.Series.cat)

    @property
    def dt(self):
        return DatetimeProperties(self)

    @property
    def dtype(self):
        return self._query_compiler.dtypes.squeeze()

    dtypes = dtype

    @property
    def empty(self):
        return len(self.index) == 0

    @property
    def hasnans(self):
        return self.isna().sum() > 0

    @property
    def is_monotonic(self):
        """Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
            bool
        """
        return self._reduce_dimension(self._query_compiler.is_monotonic())

    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic_decreasing.

        Returns
        -------
            bool
        """
        return self._reduce_dimension(self._query_compiler.is_monotonic_decreasing())

    @property
    def is_unique(self):
        """Check if Series has no duplicate values.

        Returns
        -------
        bool
            True if there is no duplicates in Series, False otherwise.

        """
        return self.nunique(dropna=False) == len(self)

    @property
    def nbytes(self):
        """
        Returns the number of bytes in the containing data.
        """
        return self.memory_usage(index=False)

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this Series.
        """
        # Series have an invariant that requires they be 1 dimension.
        return 1

    def nunique(self, dropna=True):
        return super(Series, self).nunique(dropna=dropna)

    @property
    def shape(self):
        return (len(self),)

    @property
    def str(self):
        return StringMethods(self)

    def _to_pandas(self):
        df = self._query_compiler.to_pandas()
        series = df[df.columns[0]]
        if self._query_compiler.columns[0] == "__reduced__":
            series.name = None
        return series


class DatetimeProperties(object):
    def __init__(self, series):
        self._series = series
        self._query_compiler = series._query_compiler

    @property
    def date(self):
        return Series(query_compiler=self._query_compiler.dt_date())

    @property
    def time(self):
        return Series(query_compiler=self._query_compiler.dt_time())

    @property
    def timetz(self):
        return Series(query_compiler=self._query_compiler.dt_timetz())

    @property
    def year(self):
        return Series(query_compiler=self._query_compiler.dt_year())

    @property
    def month(self):
        return Series(query_compiler=self._query_compiler.dt_month())

    @property
    def day(self):
        return Series(query_compiler=self._query_compiler.dt_day())

    @property
    def hour(self):
        return Series(query_compiler=self._query_compiler.dt_hour())

    @property
    def minute(self):
        return Series(query_compiler=self._query_compiler.dt_minute())

    @property
    def second(self):
        return Series(query_compiler=self._query_compiler.dt_second())

    @property
    def microsecond(self):
        return Series(query_compiler=self._query_compiler.dt_microsecond())

    @property
    def nanosecond(self):
        return Series(query_compiler=self._query_compiler.dt_nanosecond())

    @property
    def week(self):
        return Series(query_compiler=self._query_compiler.dt_week())

    @property
    def weekofyear(self):
        return Series(query_compiler=self._query_compiler.dt_weekofyear())

    @property
    def dayofweek(self):
        return Series(query_compiler=self._query_compiler.dt_dayofweek())

    @property
    def weekday(self):
        return Series(query_compiler=self._query_compiler.dt_weekday())

    @property
    def dayofyear(self):
        return Series(query_compiler=self._query_compiler.dt_dayofyear())

    @property
    def quarter(self):
        return Series(query_compiler=self._query_compiler.dt_quarter())

    @property
    def is_month_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_month_start())

    @property
    def is_month_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_month_end())

    @property
    def is_quarter_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_quarter_start())

    @property
    def is_quarter_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_quarter_end())

    @property
    def is_year_start(self):
        return Series(query_compiler=self._query_compiler.dt_is_year_start())

    @property
    def is_year_end(self):
        return Series(query_compiler=self._query_compiler.dt_is_year_end())

    @property
    def is_leap_year(self):
        return Series(query_compiler=self._query_compiler.dt_is_leap_year())

    @property
    def daysinmonth(self):
        return Series(query_compiler=self._query_compiler.dt_daysinmonth())

    @property
    def days_in_month(self):
        return Series(query_compiler=self._query_compiler.dt_days_in_month())

    @property
    def tz(self):
        return self._query_compiler.dt_tz().to_pandas().squeeze()

    @property
    def freq(self):
        return self._query_compiler.dt_freq().to_pandas().squeeze()

    def to_period(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_to_period(*args, **kwargs))

    def to_pydatetime(self):
        return Series(query_compiler=self._query_compiler.dt_to_pydatetime()).to_numpy()

    def tz_localize(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_tz_localize(*args, **kwargs)
        )

    def tz_convert(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_tz_convert(*args, **kwargs)
        )

    def normalize(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_normalize(*args, **kwargs))

    def strftime(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_strftime(*args, **kwargs))

    def round(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_round(*args, **kwargs))

    def floor(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_floor(*args, **kwargs))

    def ceil(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_ceil(*args, **kwargs))

    def month_name(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_month_name(*args, **kwargs)
        )

    def day_name(self, *args, **kwargs):
        return Series(query_compiler=self._query_compiler.dt_day_name(*args, **kwargs))

    def total_seconds(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_total_seconds(*args, **kwargs)
        )

    def to_pytimedelta(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.Series.dt.to_pytimedelta(df.squeeze().dt)
        )

    @property
    def seconds(self):
        return Series(query_compiler=self._query_compiler.dt_seconds())

    @property
    def days(self):
        return Series(query_compiler=self._query_compiler.dt_days())

    @property
    def microseconds(self):
        return Series(query_compiler=self._query_compiler.dt_microseconds())

    @property
    def nanoseconds(self):
        return Series(query_compiler=self._query_compiler.dt_nanoseconds())

    @property
    def components(self):
        from .dataframe import DataFrame

        return DataFrame(query_compiler=self._query_compiler.dt_components())

    @property
    def qyear(self):
        return Series(query_compiler=self._query_compiler.dt_qyear())

    @property
    def start_time(self):
        return Series(query_compiler=self._query_compiler.dt_start_time())

    @property
    def end_time(self):
        return Series(query_compiler=self._query_compiler.dt_end_time())

    def to_timestamp(self, *args, **kwargs):
        return Series(
            query_compiler=self._query_compiler.dt_to_timestamp(*args, **kwargs)
        )


class StringMethods(object):
    def __init__(self, series):
        # Check if dtypes is objects

        self._series = series
        self._query_compiler = series._query_compiler

    def casefold(self):
        return self._default_to_pandas(pandas.Series.str.casefold)

    def cat(self, others=None, sep=None, na_rep=None, join=None):
        if isinstance(others, Series):
            others = others._to_pandas()
        return self._default_to_pandas(
            pandas.Series.str.cat, others=others, sep=sep, na_rep=na_rep, join=join
        )

    def decode(self, encoding, errors="strict"):
        return self._default_to_pandas(
            pandas.Series.str.decode, encoding, errors=errors
        )

    def split(self, pat=None, n=-1, expand=False):
        if not pat and pat is not None:
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
        if not (isinstance(repl, str) or callable(repl)):
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
        if not isinstance(pat, (str, _pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_count(pat, flags=flags, **kwargs)
        )

    def startswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_startswith(pat, na=na))

    def encode(self, encoding, errors="strict"):
        return self._default_to_pandas(
            pandas.Series.str.encode, encoding, errors=errors
        )

    def endswith(self, pat, na=np.NaN):
        return Series(query_compiler=self._query_compiler.str_endswith(pat, na=na))

    def findall(self, pat, flags=0, **kwargs):
        if not isinstance(pat, (str, _pattern_type)):
            raise TypeError("first argument must be string or compiled pattern")
        return Series(
            query_compiler=self._query_compiler.str_findall(pat, flags=flags, **kwargs)
        )

    def match(self, pat, case=True, flags=0, na=np.NaN):
        if not isinstance(pat, (str, _pattern_type)):
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

    def repeat(self, repeats):
        return self._default_to_pandas(pandas.Series.str.repeat, repeats)

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

    def title(self):
        return Series(query_compiler=self._query_compiler.str_title())

    def find(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_find(sub, start=start, end=end)
        )

    def rfind(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_rfind(sub, start=start, end=end)
        )

    def index(self, sub, start=0, end=None):
        if not isinstance(sub, str):
            raise TypeError(
                "expected a string object, not {0}".format(type(sub).__name__)
            )
        return Series(
            query_compiler=self._query_compiler.str_index(sub, start=start, end=end)
        )

    def rindex(self, sub, start=0, end=None):
        if not isinstance(sub, str):
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

    def translate(self, table):
        return Series(query_compiler=self._query_compiler.str_translate(table))

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
