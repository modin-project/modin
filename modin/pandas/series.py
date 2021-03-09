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

"""
Implement Series public API as Pandas does.

Almost all docstrings for public and magic methods should be inherited from Pandas
for better maintability. So some codes are ignored in pydocstyle check:
    - D101: missing docstring in class
    - D102: missing docstring in public method
    - D105: missing docstring in magic method
Manually add documentation for methods which are not presented in pandas.
"""

import numpy as np
import pandas
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.common import (
    is_dict_like,
    is_list_like,
)
from pandas._libs.lib import no_default
from pandas._typing import IndexKeyFunc
import sys
from typing import Union, Optional
import warnings

from modin.utils import _inherit_docstrings, to_pandas, Engine
from modin.config import IsExperimental, PersistentPickle
from .base import BasePandasDataset, _ATTRS_NO_LOOKUP
from .iterator import PartitionIterator
from .utils import from_pandas, is_scalar
from .accessor import CachedAccessor, SparseAccessor
from . import _update_engine


@_inherit_docstrings(pandas.Series, excluded=[pandas.Series.__init__])
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

        TODO: add types.

        Parameters
        ----------
        data:
            Contains data stored in Series.
        index:
            Values must be hashable and have the same length as `data`.
        dtype:
            Data type for the output Series. If not specified, this will be
            inferred from `data`.
        name:
            The name to give to the Series.
        copy:
            Copy input data.
        query_compiler: query_compiler
            A query compiler object to create the Series from.
        """
        Engine.subscribe(_update_engine)
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
        self._query_compiler = query_compiler.columnarize()
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
    # Parent axis denotes axis that was used to select series in a parent dataframe.
    # If _parent_axis == 0, then it means that index axis was used via df.loc[row]
    # indexing operations and assignments should be done to rows of parent.
    # If _parent_axis == 1 it means that column axis was used via df[column] and assignments
    # should be done to columns of parent.
    _parent_axis = 0

    def __add__(self, right):
        return self.add(right)

    def __radd__(self, left):
        return self.add(left)

    def __and__(self, other):
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__and__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__and__(new_other)

    def __rand__(self, other):
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__rand__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__rand__(new_other)

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

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key not in _ATTRS_NO_LOOKUP and key in self.index:
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
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__or__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__or__(new_other)

    def __ror__(self, other):
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__ror__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__ror__(new_other)

    def __xor__(self, other):
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__xor__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__xor__(new_other)

    def __rxor__(self, other):
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__rxor__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__rxor__(new_other)

    def __pow__(self, right):
        return self.pow(right)

    def __rpow__(self, left):
        return self.rpow(left)

    def __repr__(self):
        num_rows = pandas.get_option("max_rows") or 60
        num_cols = pandas.get_option("max_columns") or 20
        temp_df = self._build_repr_df(num_rows, num_cols)
        if isinstance(temp_df, pandas.DataFrame) and not temp_df.empty:
            temp_df = temp_df.iloc[:, 0]
        temp_str = repr(temp_df)
        freq_str = (
            "Freq: {}, ".format(self.index.freqstr)
            if isinstance(self.index, pandas.DatetimeIndex)
            else ""
        )
        if self.name is not None:
            name_str = "Name: {}, ".format(str(self.name))
        else:
            name_str = ""
        if len(self.index) > num_rows:
            len_str = "Length: {}, ".format(len(self.index))
        else:
            len_str = ""
        dtype_str = "dtype: {}".format(
            str(self.dtype) + ")"
            if temp_df.empty
            else temp_str.rsplit("dtype: ", 1)[-1]
        )
        if len(self) == 0:
            return "Series([], {}{}{}".format(freq_str, name_str, dtype_str)
        return temp_str.rsplit("\n", 1)[0] + "\n{}{}{}{}".format(
            freq_str, name_str, len_str, dtype_str
        )

    def __round__(self, decimals=0):
        return self._create_or_update_from_compiler(
            self._query_compiler.round(decimals=decimals)
        )

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            self._setitem_slice(key, value)
        else:
            self.loc[key] = value

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
        return super(Series, self).to_numpy().flatten()

    def add(self, other, level=None, fill_value=None, axis=0):
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).add(
            new_other, level=level, fill_value=fill_value, axis=axis
        )

    def add_prefix(self, prefix):
        return Series(query_compiler=self._query_compiler.add_prefix(prefix, axis=0))

    def add_suffix(self, suffix):
        return Series(query_compiler=self._query_compiler.add_suffix(suffix, axis=0))

    def append(self, to_append, ignore_index=False, verify_integrity=False):
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
        # type(self), so we catch that error and use `type(self).__name__` for the return
        # type.
        # We create a "dummy" `Series` to do the error checking and determining
        # the return type.
        try:
            return_type = type(
                getattr(pandas.Series("", index=self.index[:1]), apply_func)(
                    func, *args, **kwds
                )
            ).__name__
        except Exception:
            try:
                return_type = type(
                    getattr(pandas.Series(0, index=self.index[:1]), apply_func)(
                        func, *args, **kwds
                    )
                ).__name__
            except Exception:
                return_type = type(self).__name__
        if (
            isinstance(func, str)
            or is_list_like(func)
            or return_type not in ["DataFrame", "Series"]
        ):
            result = super(Series, self).apply(func, *args, **kwds)
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
                result = self.map(f)._query_compiler
        if return_type not in ["DataFrame", "Series"]:
            # sometimes result can be not a query_compiler, but scalar (for example
            # for sum or count functions)
            if isinstance(result, type(self._query_compiler)):
                return result.to_pandas().squeeze()
            else:
                return result
        else:
            result = getattr(sys.modules[self.__module__], return_type)(
                query_compiler=result
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

    def autocorr(self, lag=1):
        return self.corr(self.shift(lag))

    def between(self, left, right, inclusive=True):
        return self._default_to_pandas(
            pandas.Series.between, left, right, inclusive=inclusive
        )

    def combine(self, other, func, fill_value=None):
        return super(Series, self).combine(
            other, lambda s1, s2: s1.combine(s2, func, fill_value=fill_value)
        )

    def compare(
        self,
        other: "Series",
        align_axis: Union[str, int] = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
    ):
        if not isinstance(other, Series):
            raise TypeError(f"Cannot compare Series to {type(other)}")
        result = self.to_frame().compare(
            other.to_frame(),
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
        )
        if align_axis == "columns" or align_axis == 1:
            # Pandas.DataFrame.Compare returns a dataframe with a multidimensional index object as the
            # columns so we have to change column object back.
            result.columns = pandas.Index(["self", "other"])
        else:
            result = result.squeeze().rename(None)
        return result

    def corr(self, other, method="pearson", min_periods=None):
        if method == "pearson":
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
            result = np.array([result.sum()])

            stddev_this = ((this * this) / (len(this) - 1)).sum()
            stddev_other = ((other * other) / (len(other) - 1)).sum()

            stddev_this = np.array([np.sqrt(stddev_this)])
            stddev_other = np.array([np.sqrt(stddev_other)])

            result /= stddev_this * stddev_other

            np.clip(result.real, -1, 1, out=result.real)
            if np.iscomplexobj(result):
                np.clip(result.imag, -1, 1, out=result.imag)
            return result[0]

        return self.__constructor__(
            query_compiler=self._query_compiler.default_to_pandas(
                pandas.Series.corr,
                other._query_compiler,
                method=method,
                min_periods=min_periods,
            )
        )

    def count(self, level=None):
        return super(Series, self).count(level=level)

    def cov(self, other, min_periods=None, ddof: Optional[int] = 1):
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
        result = this * other / (len(this) - ddof)
        result = result.sum()
        return result

    def describe(
        self, percentiles=None, include=None, exclude=None, datetime_is_numeric=False
    ):
        # Pandas ignores the `include` and `exclude` for Series for some reason.
        return super(Series, self).describe(
            percentiles=percentiles, datetime_is_numeric=datetime_is_numeric
        )

    def diff(self, periods=1):
        return super(Series, self).diff(periods=periods, axis=0)

    def divmod(self, other, level=None, fill_value=None, axis=0):
        return self._default_to_pandas(
            pandas.Series.divmod, other, level=level, fill_value=fill_value, axis=axis
        )

    def dot(self, other):
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

    def explode(self, ignore_index: bool = False):
        return self._default_to_pandas(pandas.Series.explode, ignore_index=ignore_index)

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
        squeeze: bool = no_default,
        observed=False,
        dropna: bool = True,
    ):
        if squeeze is not no_default:
            warnings.warn(
                (
                    "The `squeeze` parameter is deprecated and "
                    "will be removed in a future version."
                ),
                FutureWarning,
                stacklevel=2,
            )
        else:
            squeeze = False

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
            dropna=dropna,
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
        limit_direction: Optional[str] = None,
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
        def item_builder(s):
            return s.name, s.squeeze()

        partition_iterator = PartitionIterator(self.to_frame(), 0, item_builder)
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
        return Series(query_compiler=self._query_compiler.nsmallest(n=n, keep=keep))

    def slice_shift(self, periods=1, axis=0):
        if periods == 0:
            return self.copy()

        if axis == "index" or axis == 0:
            if abs(periods) >= len(self.index):
                return Series(dtype=self.dtype)
            else:
                if periods > 0:
                    new_index = self.index.drop(labels=self.index[:periods])
                    new_df = self.drop(self.index[-periods:])
                else:
                    new_index = self.index.drop(labels=self.index[periods:])
                    new_df = self.drop(self.index[:-periods])

                new_df.index = new_index
                return new_df
        else:
            raise ValueError(
                "No axis named {axis} for object type {type}".format(
                    axis=axis, type=type(self)
                )
            )

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        return super(type(self), self).shift(
            periods=periods, freq=freq, axis=axis, fill_value=fill_value
        )

    def unstack(self, level=-1, fill_value=None):
        from .dataframe import DataFrame

        result = DataFrame(
            query_compiler=self._query_compiler.unstack(level, fill_value)
        )

        return result.droplevel(0, axis=1) if result.columns.nlevels > 1 else result

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
        if level is not None:
            if (
                not self._query_compiler.has_multiindex(axis=axis)
                and level > 0
                or level < -1
                and level != self.index.name
            ):
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
            return self.groupby(level=level, axis=axis, sort=False).prod(
                numeric_only=numeric_only, min_count=min_count, **kwargs
            )
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan

        data = self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=True)
        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.prod_min_count(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.prod(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    product = prod
    radd = add

    def ravel(self, order="C"):
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

            result = DataFrame(self.copy()).rename(index=index).squeeze(axis=1)
            result.name = self.name
            return result

    def repeat(self, repeats, axis=None):
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

            return DataFrame(obj).reset_index(level=level, drop=drop, inplace=inplace)

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

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.replace(
            to_replace=to_replace,
            value=value,
            inplace=False,
            limit=limit,
            regex=regex,
            method=method,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def searchsorted(self, value, side="left", sorter=None):
        searchsorted_qc = self._query_compiler
        if sorter is not None:
            # `iloc` method works slowly (https://github.com/modin-project/modin/issues/1903),
            # so _default_to_pandas is used for now
            # searchsorted_qc = self.iloc[sorter].reset_index(drop=True)._query_compiler
            # sorter = None
            return self._default_to_pandas(
                pandas.Series.searchsorted, value, side=side, sorter=sorter
            )
        # searchsorted should return item number irrespective of Series index, so
        # Series.index is always set to pandas.RangeIndex, which can be easily processed
        # on the query_compiler level
        if not isinstance(searchsorted_qc.index, pandas.RangeIndex):
            searchsorted_qc = searchsorted_qc.reset_index(drop=True)

        result = self.__constructor__(
            query_compiler=searchsorted_qc.searchsorted(
                value=value, side=side, sorter=sorter
            )
        ).squeeze()

        # matching Pandas output
        if not is_scalar(value) and not is_list_like(result):
            result = np.array([result])
        elif isinstance(result, type(self)):
            result = result.to_numpy()

        return result

    def sort_values(
        self,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
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
                ignore_index=ignore_index,
                key=key,
            )
            .squeeze(axis=1)
        )
        result.name = self.name
        return self._create_or_update_from_compiler(
            result._query_compiler, inplace=inplace
        )

    sparse = CachedAccessor("sparse", SparseAccessor)

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
        if level is not None:
            if (
                not self._query_compiler.has_multiindex(axis=axis)
                and level > 0
                or level < -1
                and level != self.index.name
            ):
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
            return self.groupby(level=level, axis=axis, sort=False).sum(
                numeric_only=numeric_only, min_count=min_count, **kwargs
            )

        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan

        data = self._validate_dtypes_sum_prod_mean(
            axis, numeric_only, ignore_axis=False
        )
        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.sum_min_count(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.sum(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    def swaplevel(self, i=-2, j=-1, copy=True):
        return self._default_to_pandas("swaplevel", i=i, j=j, copy=copy)

    def take(self, indices, axis=0, is_copy=None, **kwargs):
        return super(Series, self).take(indices, axis=axis, is_copy=is_copy, **kwargs)

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

    def to_numpy(self, dtype=None, copy=False, na_value=no_default, **kwargs):
        return (
            super(Series, self)
            .to_numpy(
                dtype=dtype,
                copy=copy,
                na_value=na_value,
            )
            .flatten()
        )

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
        return self.__constructor__(
            query_compiler=self._query_compiler.unique()
        ).to_numpy()

    def update(self, other):
        if not isinstance(other, Series):
            other = Series(other)
        query_compiler = self._query_compiler.series_update(other._query_compiler)
        self._update_inplace(new_query_compiler=query_compiler)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ):
        return self.__constructor__(
            query_compiler=self._query_compiler.value_counts(
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
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

        return self._default_to_pandas(attrs)

    @property
    def array(self):
        def array(df):
            return df.array

        return self._default_to_pandas(array)

    @property
    def axes(self):
        return [self.index]

    @property
    def cat(self):
        from .series_utils import CategoryMethods

        return CategoryMethods(self)

    @property
    def dt(self):
        from .series_utils import DatetimeProperties

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
        return self._reduce_dimension(self._query_compiler.is_monotonic_increasing())

    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self):
        return self._reduce_dimension(self._query_compiler.is_monotonic_decreasing())

    @property
    def is_unique(self):
        return self.nunique(dropna=False) == len(self)

    @property
    def nbytes(self):
        return self.memory_usage(index=False)

    @property
    def ndim(self):
        # Series have an invariant that requires they be 1 dimension.
        return 1

    def nunique(self, dropna=True):
        return super(Series, self).nunique(dropna=dropna)

    @property
    def shape(self):
        return (len(self),)

    @property
    def str(self):
        from .series_utils import StringMethods

        return StringMethods(self)

    def _to_pandas(self):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        df = self._query_compiler.to_pandas()
        series = df[df.columns[0]]
        if self._query_compiler.columns[0] == "__reduced__":
            series.name = None
        return series

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

    def _reduce_dimension(self, query_compiler):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        return query_compiler.to_pandas().squeeze()

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        return self

    def _validate_dtypes_min_max(self, axis, numeric_only):
        return self

    def _validate_dtypes(self, numeric_only=False):
        pass

    def _get_numeric_data(self, axis: int):
        # `numeric_only` parameter does not supported by Series, so this method
        # doesn't do anything
        return self

    def _update_inplace(self, new_query_compiler):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        super(Series, self)._update_inplace(new_query_compiler=new_query_compiler)
        # Propagate changes back to parent so that column in dataframe had the same contents
        if self._parent is not None:
            if self._parent_axis == 0:
                self._parent.loc[self.name] = self
            else:
                self._parent[self.name] = self

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """
        Return or update a DataFrame given new query_compiler.

        TODO: add description for parameters.

        Parameters
        ----------
        new_query_compiler: query_compiler
        inplace: bool

        Returns
        -------
        Dataframe
        """
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace and new_query_compiler.is_series_like():
            return Series(query_compiler=new_query_compiler)
        elif not inplace:
            # This can happen with things like `reset_index` where we can add columns.
            from .dataframe import DataFrame

            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _prepare_inter_op(self, other):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        if isinstance(other, Series):
            new_self = self.copy()
            new_other = other.copy()
            if self.name == other.name:
                new_self.name = new_other.name = self.name
            else:
                new_self.name = new_other.name = "__reduced__"
        else:
            new_self = self
            new_other = other
        return new_self, new_other

    def _getitem(self, key):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
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

    # Persistance support methods - BEGIN
    @classmethod
    def _inflate_light(cls, query_compiler, name):
        return cls(query_compiler=query_compiler, name=name)

    @classmethod
    def _inflate_full(cls, pandas_series):
        return cls(data=pandas_series)

    def __reduce__(self):
        self._query_compiler.finalize()
        if PersistentPickle.get():
            return self._inflate_full, (self._to_pandas(),)
        return self._inflate_light, (self._query_compiler, self.name)

    # Persistance support methods - END


if IsExperimental.get():
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(Series, "make_series_wrapper")
