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

"""Module houses ``DataFrame`` class, that is distributed version of ``pandas.DataFrame``."""

import pandas
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.common import (
    infer_dtype_from_object,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
)
from pandas.util._validators import validate_bool_kwarg
from pandas.io.formats.printing import pprint_thing
from pandas._libs.lib import no_default
from pandas._typing import StorageOptions

import re
import itertools
import functools
import numpy as np
import sys
from typing import IO, Optional, Union, Iterator
import warnings

from modin.pandas import Categorical
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings, to_pandas, hashable
from modin.config import Engine, IsExperimental, PersistentPickle
from .utils import (
    from_pandas,
    from_non_pandas,
)
from . import _update_engine
from .iterator import PartitionIterator
from .series import Series
from .base import BasePandasDataset, _ATTRS_NO_LOOKUP
from .groupby import DataFrameGroupBy
from .accessor import CachedAccessor, SparseFrameAccessor


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame.__init__], apilink="pandas.DataFrame"
)
class EmptyDataFrame(DataFrame):
    """
    Modin ``EmptyDataFrame`` class to handle empty dataframe.

    Inherit common for ``DataFrame``-s functionality from the
    ``DataFrame`` class.

    Parameters
    ----------
    data : DataFrame, Series, pandas.DataFrame, ndarray, Iterable or dict, optional
        Dict can contain ``Series``, arrays, constants, dataclass or list-like objects.
        If data is a dict, column order follows insertion-order.
    index : Index or array-like, optional
        Index to use for resulting frame. Will default to ``RangeIndex`` if no
        indexing information part of input data and no index provided.
    columns : Index or array-like, optional
        Column labels to use for resulting frame. Will default to
        ``RangeIndex`` if no column labels are provided.
    dtype : str, np.dtype, or pandas.ExtensionDtype, optional
        Data type to force. Only a single dtype is allowed. If None, infer.
    copy : bool, default: False
        Copy data from inputs. Only affects ``pandas.DataFrame`` / 2d ndarray input.
    query_compiler : BaseQueryCompiler, optional
        A query compiler object to create the ``DataFrame`` from.

    Notes
    -----
    ``DataFrame`` can be created either from passed `data` or `query_compiler`. If both
    parameters are provided, data source will be prioritized in the next order:

    1) Modin ``DataFrame`` or ``Series`` passed with `data` parameter.
    2) Query compiler from the `query_compiler` parameter.
    3) Various pandas/NumPy/Python data structures passed with `data` parameter.

    The last option is less desirable since import of such data structures is very
    inefficient, please use previously created Modin structures from the fist two
    options or import data using highly efficient Modin IO tools (for example
    ``pd.read_csv``).
    """

    _pandas_class = pandas.DataFrame

    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=None,
        query_compiler=None,
    ):
        super().__init__()

    columns = property(super()._get_columns, super()._set_columns)

    def to_dataframe(self, df):
        """
        Convert pandas output to ``DataFrame`` or ``EmptyDataFrame`` as relevant.
        """
        if not self.is_empty():
            return DataFrame(data, index, columns, dtype, copy, query_compiler)
        return EmptyDataFrame(df)

    def _default_to_pandas(self, op, *args, **kwargs):
        # TODO: Clean this up by calling the super()._default_to_pandas(),
        # checking the return type, and calling to_empty_type().
        """
        Convert dataset to pandas type and call a pandas function on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed to `op`.
        **kwargs : dict
            Additional keywords arguments to be passed to `op`.

        Returns
        -------
        object
            Result of operation.
        """
        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)
        pandas_obj = self._to_pandas()
        if callable(op):
            result = op(pandas_obj, *args, **kwargs)
        elif isinstance(op, str):
            # The inner `getattr` is ensuring that we are treating this object (whether
            # it is a DataFrame, Series, etc.) as a pandas object. The outer `getattr`
            # will get the operation (`op`) from the pandas version of the class and run
            # it on the object after we have converted it to pandas.
            result = getattr(self._pandas_class, op)(pandas_obj, *args, **kwargs)
        else:
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=True,
                extra_log="{} is an unsupported operation".format(op),
            )
        # SparseDataFrames cannot be serialized by arrow and cause problems for Modin.
        # For now we will use pandas.
        if isinstance(result, type(self)) and not isinstance(
            result, (pandas.SparseDataFrame, pandas.SparseSeries)
        ):
            return self._create_or_update_from_compiler(
                result, inplace=kwargs.get("inplace", False)
            )
        elif isinstance(result, pandas.DataFrame):
            return to_dataframe(result)
        elif isinstance(result, pandas.Series):
            from .series import Series

            return Series(result)
        # inplace
        elif result is None:
            return self._create_or_update_from_compiler(
                getattr(pd, type(pandas_obj).__name__)(pandas_obj)._query_compiler,
                inplace=True,
            )
        else:
            try:
                if (
                    isinstance(result, (list, tuple))
                    and len(result) == 2
                    and isinstance(result[0], pandas.DataFrame)
                ):
                    # Some operations split the DataFrame into two (e.g. align). We need to wrap
                    # both of the returned results
                    if isinstance(result[1], pandas.DataFrame):
                        second = self.__constructor__(result[1])
                    else:
                        second = result[1]
                    return self.__constructor__(result[0]), second
                else:
                    return to_dataframe(result)
            except TypeError:
                return to_dataframe(result)

    def drop_duplicates(
        self, subset=None, keep="first", inplace=False, ignore_index=False
    ):  # noqa: PR01, RT01, D200
        """
        Return ``DataFrame`` with duplicate rows removed.
        """
        return self._default_to_pandas(
            pandas.DataFrame.drop_duplicates,
            subset=subset,
            keep=keep,
            inplace=inplace,
            ignore_index=ignore_index,
        )

    def duplicated(self, subset=None, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return boolean ``Series`` denoting duplicate rows.
        """
        return self._default_to_pandas(
            pandas.DataFrame.duplicated,
            subset=subset,
            keep=keep,
        )

    def add_prefix(self, prefix):  # noqa: PR01, RT01, D200
        """
        Prefix labels with string `prefix`.
        """
        return EmptyDataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):  # noqa: PR01, RT01, D200
        """
        Suffix labels with string `suffix`.
        """
        return EmptyDataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(
        self, func, na_action: Optional[str] = None, **kwargs
    ):  # noqa: PR01, RT01, D200
        """
        Apply a function to a ``DataFrame`` elementwise.
        """
        return self._default_to_pandas(
            pandas.DataFrame.applymap,
            func=func,
            na_action=na_action,
            **kwargs,
        )

    def apply(
        self, func, axis=0, raw=False, result_type=None, args=(), **kwargs
    ):  # noqa: PR01, RT01, D200
        """
        Apply a function along an axis of the ``DataFrame``.
        """
        return self._default_to_pandas(
            pandas.DataFrame.apply,
            func=func,
            axis=axis,
            raw=raw,
            result_type=result_type,
            args=args,
            **kwargs,
        )

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
    ):  # noqa: PR01, RT01, D200
        """
        Group ``DataFrame`` using a mapper or by a ``Series`` of columns.
        """
        if squeeze is not no_default:
            warnings.warn(
                (
                    "The `squeeze` parameter is deprecated and "
                    + "will be removed in a future version."
                ),
                FutureWarning,
                stacklevel=2,
            )
        else:
            squeeze = False

        return self._default_to_pandas(
            pandas.DataFrame.groupby,
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            squeeze=squeeze,
            observed=observed,
            dropna=dropna
        )

    def transpose(self, copy=False, *args):  # noqa: PR01, RT01, D200
        """
        Transpose index and columns.
        """
        return self._default_to_pandas(
            pandas.DataFrame.transpose,
            copy=copy,
            *args
        )

    T = property(transpose)

    def add(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get addition of ``DataFrame`` and `other`, element-wise (binary operator `add`).
        """
        return self._default_to_pandas(
            "add",
            other=other,
            axis=axis,
            level=level,
            fill_value=fill_value
        )

    def append(
        self, other, ignore_index=False, verify_integrity=False, sort=False
    ):  # noqa: PR01, RT01, D200
        """
        Append rows of `other` to the end of caller, returning a new object.
        """
        if isinstance(other, Series):
            other = pandas.Series(other)
        elif isinstance(other, DataFrame):
            other = pandas.DataFrame(other)
        return self._default_to_pandas(
            pandas.DataFrame.append,
            other=other,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort
        )

    def assign(self, **kwargs):  # noqa: PR01, RT01, D200
        """
        Assign new columns to a ``DataFrame``.
        """
        return self._default_to_pandas(pandas.DataFrame.assign, **kwargs)

    def boxplot(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize=None,
        rot=0,
        grid=True,
        figsize=None,
        layout=None,
        return_type=None,
        backend=None,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Make a box plot from ``DataFrame`` columns.
        """
        return self._default_to_pandas(
            pandas.DataFrame.boxplot,
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            backend=backend,
            **kwargs,
        )

    def combine(
        self, other, func, fill_value=None, overwrite=True
    ):  # noqa: PR01, RT01, D200
        """
        Perform column-wise combine with another ``DataFrame``.
        """
        return self._default_to_pandas(
            pandas.DataFrame.combine,
            other=pandas.DataFrame(other),
            func=func,
            fill_value=fill_value,
            overwrite=overwrite,
        )

    def compare(
        self,
        other: "DataFrame",
        align_axis: Union[str, int] = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
    ) -> "DataFrame":  # noqa: PR01, RT01, D200
        """
        Compare to another ``DataFrame`` and show the differences.
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"Cannot compare DataFrame to {type(other)}")
        return self._default_to_pandas(
            pandas.DataFrame.compare,
            other=pandas.DataFrame(other),
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
        )

    def corr(self, method="pearson", min_periods=1):  # noqa: PR01, RT01, D200
        """
        Compute pairwise correlation of columns, excluding NA/null values.
        """
        return self._default_to_pandas(
            pandas.DataFrame.corr,
            method=method,
            min_periods=min_periods,
        )

    def corrwith(
        self, other, axis=0, drop=False, method="pearson"
    ):  # noqa: PR01, RT01, D200
        """
        Compute pairwise correlation.
        """
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.corrwith,
            other=other,
            axis=axis,
            drop=drop,
            method=method,
        )

    def cov(self, min_periods=None, ddof: Optional[int] = 1):  # noqa: PR01, RT01, D200
        """
        Compute pairwise covariance of columns, excluding NA/null values.
        """
        return self._default_to_pandas(
            pandas.DataFrame.cov,
            min_periods=min_periods,
            ddof=ddof,
        )

    def dot(self, other):  # noqa: PR01, RT01, D200
        """
        Compute the matrix multiplication between the ``DataFrame`` and `other`.
        """
        if isinstance(other, DataFrame):
            other = pandas.DataFrame(other)
        elif isinstance(other, Series):
            other = pandas.Series(other)
        return self._default_to_pandas("dot", other=other)

    def eq(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Perform equality comparison of ``DataFrame`` and `other` (binary operator `eq`).
        """
        if isinstance(other, DataFrame):
            other = pandas.DataFrame(other)
        return self._default_to_pandas(
            "eq",
            other=other,
            axis=axis,
            level=level,
        )

    def equals(self, other):  # noqa: PR01, RT01, D200
        """
        Test whether two objects contain the same elements.
        """
        if isinstance(other, DataFrame):
            other = pandas.DataFrame(other)
        return self._default_to_pandas(pandas.DataFrame.equals, other=other)

    def eval(self, expr, inplace=False, **kwargs):  # noqa: PR01, RT01, D200
        """
        Evaluate a string describing operations on ``DataFrame`` columns.
        """
        return self._default_to_pandas(
            "eval",
            expr=expr,
            inplace=inplace,
            **kwargs,
        )

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):  # noqa: PR01, RT01, D200
        """
        Fill NA/NaN values using the specified method.
        """
        return self._default_to_pandas(
            pandas.DataFrame.fillna,
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )

    def floordiv(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get integer division of ``DataFrame`` and `other`, element-wise (binary operator `floordiv`).
        """
        if isinstance(other, DataFrame):
            other = pandas.DataFrame(other)
        elif isinstance(other, Series):
            other = pandas.Series(other)
        return self._default_to_pandas(
            "floordiv",
            other=other,
            axis=axis,
            level=level,
            fill_value=fill_value,
        )

    # Test Above - narenk

    @classmethod
    def from_dict(
        cls, data, orient="columns", dtype=None, columns=None
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Construct ``DataFrame`` from dict of array-like or dicts.
        """
        return from_pandas(
            pandas.DataFrame.from_dict(
                data, orient=orient, dtype=dtype, columns=columns
            )
        )

    @classmethod
    def from_records(
        cls,
        data,
        index=None,
        exclude=None,
        columns=None,
        coerce_float=False,
        nrows=None,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Convert structured or record ndarray to ``DataFrame``.
        """
        return from_pandas(
            pandas.DataFrame.from_records(
                data,
                index=index,
                exclude=exclude,
                columns=columns,
                coerce_float=coerce_float,
                nrows=nrows,
            )
        )

    def ge(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get greater than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._default_to_pandas(pandas.DataFrame.ge, other, axis, level)

    def gt(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get greater than comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._default_to_pandas(pandas.DataFrame.gt, other, axis, level)

    def hist(
        self,
        column=None,
        by=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        yrot=None,
        ax=None,
        sharex=False,
        sharey=False,
        figsize=None,
        layout=None,
        bins=10,
        **kwds,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Make a histogram of the ``DataFrame``.
        """
        return self._default_to_pandas(
            pandas.DataFrame.hist,
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            **kwds,
        )

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[IO[str]] = None,
        max_cols: Optional[int] = None,
        memory_usage: Optional[Union[bool, str]] = None,
        show_counts: Optional[bool] = None,
        null_counts: Optional[bool] = None,
    ):  # noqa: PR01, D200
        """
        Print a concise summary of the ``DataFrame``.
        """
        return self._default_to_pandas(pandas.DataFrame.info, verbose, buf, max_cols, memory_usage, show_counts, null_counts)

    def insert(self, loc, column, value, allow_duplicates=False):  # noqa: PR01, D200
        """
        Insert column into ``DataFrame`` at specified location.
        """
        return self._default_to_pandas(pandas.DataFrame.insert, loc, column, value, allow_duplicates)

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
    ):  # noqa: PR01, RT01, D200
        """
        Fill NaN values using an interpolation method.
        """
        return self._default_to_pandas(
            pandas.DataFrame.interpolate,
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    def iterrows(self):  # noqa: D200
        """
        Iterate over ``DataFrame`` rows as (index, ``Series``) pairs.
        """
        return self._default_to_pandas(pandas.DataFrame.iterrows)

    def items(self):  # noqa: D200
        """
        Iterate over (column name, ``Series``) pairs.
        """
        return self._default_to_pandas(pandas.DataFrame.items)

    def iteritems(self):  # noqa: RT01, D200
        """
        Iterate over (column name, ``Series``) pairs.
        """
        return self._default_to_pandas(pandas.DataFrame.iteritems)

    def itertuples(self, index=True, name="Pandas"):  # noqa: PR01, D200
        """
        Iterate over ``DataFrame`` rows as ``namedtuple``-s.
        """
        return self._default_to_pandas(pandas.DataFrame.itertuples)

    def le(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get less than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `le`).
        """
        return self._default_to_pandas(pandas.DataFrame.le, other, axis, level)

    def lookup(self, row_labels, col_labels):  # noqa: PR01, RT01, D200
        """
        Label-based "fancy indexing" function for ``DataFrame``.
        """
        return self._default_to_pandas(pandas.DataFrame.lookup, row_labels, col_labels)

    def lt(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get less than comparison of ``DataFrame`` and `other`, element-wise (binary operator `le`).
        """
        return self._default_to_pandas(pandas.DataFrame.lt, other, axis, level)

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index=True,
    ):  # noqa: PR01, RT01, D200
        """
        Unpivot a ``DataFrame`` from wide to long format, optionally leaving identifiers set.
        """
        return self._default_to_pandas(pandas.DataFrame.melt, id_vars, value_vars, var_name, value_name, col_level, ignore_index)

    def memory_usage(self, index=True, deep=False):  # noqa: PR01, RT01, D200
        """
        Return the memory usage of each column in bytes.
        """
        return self._default_to_pandas(pandas.DataFrame.memory_usage, index, deep)

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    ):  # noqa: PR01, RT01, D200
        """
        Merge ``DataFrame`` or named ``Series`` objects with a database-style join.
        """
        if isinstance(right, Series):
            if right.name is None:
                raise ValueError("Cannot merge a Series without a name")
            right = pandas.Series(right)
        elif isinstance(right, DataFrame):
            right = pandas.DataFrame(right)
        else:
            raise TypeError(
                f"Can only merge Series or DataFrame objects, a {type(right)} was passed"
            )
        return self._default_to_pandas(
            pandas.DataFrame.merge,
            right,
            how,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            sort,
            suffixes,
            copy,
            indicator,
            validate,
        )

    def mod(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get modulo of ``DataFrame`` and `other`, element-wise (binary operator `mod`).
        """
        return self._default_to_pandas(pandas.DataFrame.mod, other, axis, level, fill_value)

    def mul(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get multiplication of ``DataFrame`` and `other`, element-wise (binary operator `mul`).
        """
        return self._default_to_pandas(pandas.DataFrame.mul, other, axis, level, fill_value)

    rmul = multiply = mul

    def ne(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get not equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ne`).
        """
        return self._default_to_pandas(pandas.DataFrame.ne, other, axis, level)

    def nlargest(self, n, columns, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in descending order.
        """
        return self._default_to_pandas(pandas.DataFrame.nlargest, n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in ascending order.
        """
        return self._default_to_pandas(pandas.DataFrame.nsmallest, n, columns, keep)

    def slice_shift(self, periods=1, axis=0):  # noqa: PR01, RT01, D200
        """
        Equivalent to `shift` without copying data.
        """
        return self._default_to_pandas(pandas.DataFrame.slice_shift, periods, axis)

    def unstack(self, level=-1, fill_value=None):  # noqa: PR01, RT01, D200
        """
        Pivot a level of the (necessarily hierarchical) index labels.
        """
        return self._default_to_pandas(pandas.DataFrame.unstack, level, fill_value)

    def pivot(self, index=None, columns=None, values=None):  # noqa: PR01, RT01, D200
        """
        Return reshaped ``DataFrame`` organized by given index / column values.
        """
        return self._default_to_pandas(pandas.DataFrame.pivot, index, columns, values)

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
        observed=False,
        sort=True,
    ):  # noqa: PR01, RT01, D200
        """
        Create a spreadsheet-style pivot table as a ``DataFrame``.
        """
        return self._default_to_pandas(
            pandas.DataFrame.pivot_table,
            index=index,
            values=values,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort,
        )

    @property
    def plot(
        self,
        x=None,
        y=None,
        kind="line",
        ax=None,
        subplots=False,
        sharex=None,
        sharey=False,
        layout=None,
        figsize=None,
        use_index=True,
        title=None,
        grid=None,
        legend=True,
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
        secondary_y=False,
        sort_columns=False,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Make plots of ``DataFrame``.
        """
        return self._to_pandas().plot

    def pow(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get exponential power of ``DataFrame`` and `other`, element-wise (binary operator `pow`).
        """
        return self._default_to_pandas(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    def prod(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Return the product of the values over the requested axis.
        """
        return self._default_to_pandas(pandas.DataFrame.prod, axis, skipna, level, numeric_only, min_count, **kwargs)

    product = prod
    radd = add

    def query(self, expr, inplace=False, **kwargs):  # noqa: PR01, RT01, D200
        """
        Query the columns of a ``DataFrame`` with a boolean expression.
        """
        return self._default_to_pandas(pandas.DataFrame.query, expr, inplace, **kwargs)

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=np.nan,
        limit=None,
        tolerance=None,
    ):  # noqa: PR01, RT01, D200
        """
        Conform ``DataFrame`` to new index with optional filling logic.
        """
        return self._default_to_pandas(
            pandas.DataFrame.reindex,
            labels,
            index,
            columns,
            axis,
            method,
            copy,
            level,
            fill_value,
            limit,
            tolerance,
        )

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):  # noqa: PR01, RT01, D200
        """
        Alter axes labels.
        """
        return self._default_to_pandas(
            pandas.DataFrame.rename,
            mapper=mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )

    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace: "bool" = False,
        limit=None,
        regex: "bool" = False,
        method: "str | NoDefault" = no_default,
    ):  # noqa: PR01, RT01, D200
        """
        Replace values given in `to_replace` with `value`.
        """
        return self._default_to_pandas(
            pandas.DataFrame.replace,
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )

    def rfloordiv(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get integer division of ``DataFrame`` and `other`, element-wise (binary operator `rfloordiv`).
        """
        return self._default_to_pandas(
            "rfloordiv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
        )

    def rmod(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get modulo of ``DataFrame`` and `other`, element-wise (binary operator `rmod`).
        """
        return self._binary_op(
            "rmod",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rpow(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get exponential power of ``DataFrame`` and `other`, element-wise (binary operator `rpow`).
        """
        if isinstance(other, Series):
            return self._default_to_pandas(
                "rpow", other, axis=axis, level=level, fill_value=fill_value
            )
        return self._binary_op(
            "rpow",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rsub(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get subtraction of ``DataFrame`` and `other`, element-wise (binary operator `rsub`).
        """
        return self._binary_op(
            "rsub",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rtruediv(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get floating division of ``DataFrame`` and `other`, element-wise (binary operator `rtruediv`).
        """
        return self._binary_op(
            "rtruediv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    rdiv = rtruediv

    def select_dtypes(self, include=None, exclude=None):  # noqa: PR01, RT01, D200
        """
        Return a subset of the ``DataFrame``'s columns based on the column dtypes.
        """
        # Validates arguments for whether both include and exclude are None or
        # if they are disjoint. Also invalidates string dtypes.
        pandas.DataFrame().select_dtypes(include, exclude)

        if include and not is_list_like(include):
            include = [include]
        elif include is None:
            include = []
        if exclude and not is_list_like(exclude):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        sel = tuple(map(set, (include, exclude)))
        include, exclude = map(lambda x: set(map(infer_dtype_from_object, x)), sel)
        include_these = pandas.Series(not bool(include), index=self.columns)
        exclude_these = pandas.Series(not bool(exclude), index=self.columns)

        def is_dtype_instance_mapper(column, dtype):
            return column, functools.partial(issubclass, dtype.type)

        for column, f in itertools.starmap(
            is_dtype_instance_mapper, self.dtypes.iteritems()
        ):
            if include:  # checks for the case of empty include or exclude
                include_these[column] = any(map(f, include))
            if exclude:
                exclude_these[column] = not any(map(f, exclude))

        dtype_indexer = include_these & exclude_these
        indicate = [
            i for i in range(len(dtype_indexer.values)) if not dtype_indexer.values[i]
        ]
        return self.drop(columns=self.columns[indicate], inplace=False)

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ):  # noqa: PR01, RT01, D200
        """
        Set the ``DataFrame`` index using existing columns.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]

        if any(
            isinstance(col, (pandas.Index, Series, np.ndarray, list, Iterator))
            for col in keys
        ):
            if inplace:
                frame = self
            else:
                frame = self.copy()
            if not all(
                isinstance(col, (pandas.Index, Series, np.ndarray, list, Iterator))
                for col in keys
            ):
                if drop:
                    keys = [frame.pop(k) if not is_list_like(k) else k for k in keys]
                keys = [k._to_pandas() if isinstance(k, Series) else k for k in keys]
            # These are single-threaded objects, so we might as well let pandas do the
            # calculation so that it matches.
            frame.index = (
                pandas.DataFrame(index=self.index)
                .set_index(keys, append=append, verify_integrity=verify_integrity)
                .index
            )
            if not inplace:
                return frame
            else:
                return

        missing = []
        for col in keys:
            # everything else gets tried as a key;
            # see https://github.com/pandas-dev/pandas/issues/24969
            try:
                found = col in self.columns
            except TypeError as err:
                raise TypeError(
                    'The parameter "keys" may be a column key, one-dimensional '
                    + "array, or a list containing only valid column keys and "
                    + f"one-dimensional arrays. Received column of type {type(col)}"
                ) from err
            else:
                if not found:
                    missing.append(col)
        if missing:
            raise KeyError(f"None of {missing} are in the columns")

        new_query_compiler = self._query_compiler.set_index_from_columns(
            keys, drop=drop, append=append
        )

        if verify_integrity and not new_query_compiler.index.is_unique:
            duplicates = new_query_compiler.index[
                new_query_compiler.index.duplicated()
            ].unique()
            raise ValueError(f"Index has duplicate keys: {duplicates}")

        return self._create_or_update_from_compiler(new_query_compiler, inplace=inplace)

    sparse = CachedAccessor("sparse", SparseFrameAccessor)

    def squeeze(self, axis=None):  # noqa: PR01, RT01, D200
        """
        Squeeze 1 dimensional axis objects into scalars.
        """
        axis = self._get_axis_number(axis) if axis is not None else None
        if axis is None and (len(self.columns) == 1 or len(self.index) == 1):
            return Series(query_compiler=self._query_compiler).squeeze()
        if axis == 1 and len(self.columns) == 1:
            return Series(query_compiler=self._query_compiler)
        if axis == 0 and len(self.index) == 1:
            return Series(query_compiler=self.T._query_compiler)
        else:
            return self.copy()

    def stack(self, level=-1, dropna=True):  # noqa: PR01, RT01, D200
        """
        Stack the prescribed level(s) from columns to index.
        """
        if not isinstance(self.columns, pandas.MultiIndex) or (
            isinstance(self.columns, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.columns.nlevels
        ):
            return self._reduce_dimension(
                query_compiler=self._query_compiler.stack(level, dropna)
            )
        else:
            return DataFrame(query_compiler=self._query_compiler.stack(level, dropna))

    def sub(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get subtraction of ``DataFrame`` and `other`, element-wise (binary operator `sub`).
        """
        return self._binary_op(
            "sub",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    subtract = sub

    def sum(
        self,
        axis=None,
        skipna=True,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Return the sum of the values over the requested axis.
        """
        axis = self._get_axis_number(axis)
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        axis_to_apply = self.columns if axis else self.index
        if (
            skipna is not False
            and numeric_only is None
            and min_count > len(axis_to_apply)
        ):
            new_index = self.columns if not axis else self.index
            return Series(
                [np.nan] * len(new_index), index=new_index, dtype=np.dtype("object")
            )

        data = self._validate_dtypes_sum_prod_mean(
            axis, numeric_only, ignore_axis=False
        )
        if level is not None:
            if (
                not self._query_compiler.has_multiindex(axis=axis)
                and level > 0
                or level < -1
                and level != self.index.name
            ):
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
            return self.groupby(level=level, axis=axis, sort=False).sum(
                numeric_only=numeric_only, min_count=min_count
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

    def to_feather(self, path, **kwargs):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Write a ``DataFrame`` to the binary Feather format.
        """
        return self._default_to_pandas(pandas.DataFrame.to_feather, path, **kwargs)

    def to_gbq(
        self,
        destination_table,
        project_id=None,
        chunksize=None,
        reauth=False,
        if_exists="fail",
        auth_local_webserver=False,
        table_schema=None,
        location=None,
        progress_bar=True,
        credentials=None,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Write a ``DataFrame`` to a Google BigQuery table.
        """
        return self._default_to_pandas(
            pandas.DataFrame.to_gbq,
            destination_table,
            project_id=project_id,
            chunksize=chunksize,
            reauth=reauth,
            if_exists=if_exists,
            auth_local_webserver=auth_local_webserver,
            table_schema=table_schema,
            location=location,
            progress_bar=progress_bar,
            credentials=credentials,
        )

    def to_html(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        bold_rows=True,
        classes=None,
        escape=True,
        notebook=False,
        border=None,
        table_id=None,
        render_links=False,
        encoding=None,
    ):  # noqa: PR01, RT01, D200
        """
        Render a ``DataFrame`` as an HTML table.
        """
        return self._default_to_pandas(
            pandas.DataFrame.to_html,
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            bold_rows=bold_rows,
            classes=classes,
            escape=escape,
            notebook=notebook,
            border=border,
            table_id=table_id,
            render_links=render_links,
            encoding=None,
        )

    def to_parquet(
        self,
        path=None,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions = None,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Write a DataFrame to the binary parquet format.
        """
        config = {
            "path": path,
            "engine": engine,
            "compression": compression,
            "index": index,
            "partition_cols": partition_cols,
            "storage_options": storage_options,
        }
        new_query_compiler = self._query_compiler

        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_parquet(new_query_compiler, **config, **kwargs)

    def to_period(
        self, freq=None, axis=0, copy=True
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Convert ``DataFrame`` from ``DatetimeIndex`` to ``PeriodIndex``.
        """
        return super(DataFrame, self).to_period(freq=freq, axis=axis, copy=copy)

    def to_records(
        self, index=True, column_dtypes=None, index_dtypes=None
    ):  # noqa: PR01, RT01, D200
        """
        Convert ``DataFrame`` to a NumPy record array.
        """
        return self._default_to_pandas(
            pandas.DataFrame.to_records,
            index=index,
            column_dtypes=column_dtypes,
            index_dtypes=index_dtypes,
        )

    def to_stata(
        self,
        path: "FilePath | WriteBuffer[bytes]",
        convert_dates: "dict[Hashable, str] | None" = None,
        write_index: "bool" = True,
        byteorder: "str | None" = None,
        time_stamp: "datetime.datetime | None" = None,
        data_label: "str | None" = None,
        variable_labels: "dict[Hashable, str] | None" = None,
        version: "int | None" = 114,
        convert_strl: "Sequence[Hashable] | None" = None,
        compression: "CompressionOptions" = "infer",
        storage_options: "StorageOptions" = None,
        *,
        value_labels: "dict[Hashable, dict[float | int, str]] | None" = None,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Export ``DataFrame`` object to Stata data format.
        """
        return self._default_to_pandas(
            pandas.DataFrame.to_stata,
            path,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            version=version,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
            value_labels=value_labels,
        )

    def to_timestamp(
        self, freq=None, how="start", axis=0, copy=True
    ):  # noqa: PR01, RT01, D200
        """
        Cast to DatetimeIndex of timestamps, at *beginning* of period.
        """
        return super(DataFrame, self).to_timestamp(
            freq=freq, how=how, axis=axis, copy=copy
        )

    def to_xml(
        self,
        path_or_buffer=None,
        index=True,
        root_name="data",
        row_name="row",
        na_rep=None,
        attr_cols=None,
        elem_cols=None,
        namespaces=None,
        prefix=None,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=True,
        parser="lxml",
        stylesheet=None,
        compression="infer",
        storage_options=None,
    ):  # noqa: PR01, RT01, D200
        """
        Render a DataFrame to an XML document.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.default_to_pandas(
                pandas.DataFrame.to_xml,
                path_or_buffer=path_or_buffer,
                index=index,
                root_name=root_name,
                row_name=row_name,
                na_rep=na_rep,
                attr_cols=attr_cols,
                elem_cols=elem_cols,
                namespaces=namespaces,
                prefix=prefix,
                encoding=encoding,
                xml_declaration=xml_declaration,
                pretty_print=pretty_print,
                parser=parser,
                stylesheet=stylesheet,
                compression=compression,
                storage_options=storage_options,
            )
        )

    def truediv(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
        """
        Get floating division of ``DataFrame`` and `other`, element-wise (binary operator `truediv`).
        """
        return self._binary_op(
            "truediv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    div = divide = truediv

    def update(
        self, other, join="left", overwrite=True, filter_func=None, errors="ignore"
    ):  # noqa: PR01, RT01, D200
        """
        Modify in place using non-NA values from another ``DataFrame``.
        """
        if not isinstance(other, DataFrame):
            other = DataFrame(other)
        query_compiler = self._query_compiler.df_update(
            other._query_compiler,
            join=join,
            overwrite=overwrite,
            filter_func=filter_func,
            errors=errors,
        )
        self._update_inplace(new_query_compiler=query_compiler)

    def where(
        self,
        cond,
        other=no_default,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=no_default,
    ):  # noqa: PR01, RT01, D200
        """
        Replace values where the condition is False.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(other, pandas.Series) and axis is None:
            raise ValueError("Must specify axis=0 or 1")
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            if isinstance(cond, DataFrame):
                cond = cond._query_compiler.to_pandas()
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.where,
                cond,
                other=other,
                inplace=False,
                axis=axis,
                level=level,
                errors=errors,
                try_cast=try_cast,
            )
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        axis = self._get_axis_number(axis)
        cond = cond(self) if callable(cond) else cond

        if not isinstance(cond, DataFrame):
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            cond = DataFrame(cond, index=self.index, columns=self.columns)
        if isinstance(other, DataFrame):
            other = other._query_compiler
        elif isinstance(other, pandas.Series):
            other = other.reindex(self.index if not axis else self.columns)
        else:
            index = self.index if not axis else self.columns
            other = pandas.Series(other, index=index)
        query_compiler = self._query_compiler.where(
            cond._query_compiler, other, axis=axis, level=level
        )
        return self._create_or_update_from_compiler(query_compiler, inplace)

    def xs(self, key, axis=0, level=None, drop_level=True):  # noqa: PR01, RT01, D200
        """
        Return cross-section from the ``DataFrame``.
        """
        return self._default_to_pandas(
            pandas.DataFrame.xs, key, axis=axis, level=level, drop_level=drop_level
        )

    def _getitem_column(self, key):
        """
        Get column specified by `key`.

        Parameters
        ----------
        key : hashable
            Key that points to column to retrieve.

        Returns
        -------
        Series
            Selected column.
        """
        if key not in self.keys():
            raise KeyError("{}".format(key))
        s = DataFrame(
            query_compiler=self._query_compiler.getitem_column_array([key])
        ).squeeze(axis=1)
        if isinstance(s, Series):
            s._parent = self
            s._parent_axis = 1
        return s

    def __getattr__(self, key):
        """
        Return item identified by `key`.

        Parameters
        ----------
        key : hashable
            Key to get.

        Returns
        -------
        Any

        Notes
        -----
        First try to use `__getattribute__` method. If it fails
        try to get `key` from ``DataFrame`` fields.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key not in _ATTRS_NO_LOOKUP and key in self.columns:
                return self[key]
            raise e

    def __setattr__(self, key, value):
        """
        Set attribute `value` identified by `key`.

        Parameters
        ----------
        key : hashable
            Key to set.
        value : Any
            Value to set.
        """
        # While we let users assign to a column labeled "x" with "df.x" , there
        # are some attributes that we should assume are NOT column names and
        # therefore should follow the default Python object assignment
        # behavior. These are:
        # - anything in self.__dict__. This includes any attributes that the
        #   user has added to the dataframe with,  e.g., `df.c = 3`, and
        #   any attribute that Modin has added to the frame, e.g.
        #   `_query_compiler` and `_siblings`
        # - `_query_compiler`, which Modin initializes before it appears in
        #   __dict__
        # - `_siblings`, which Modin initializes before it appears in __dict__
        if key in ["_query_compiler", "_siblings"] or key in self.__dict__:
            pass
        elif key in self and key not in dir(self):
            self.__setitem__(key, value)
            # Note: return immediately so we don't keep this `key` as dataframe state.
            # `__getattr__` will return the columns not present in `dir(self)`, so we do not need
            # to manually track this state in the `dir`.
            return
        elif isinstance(value, pandas.Series):
            warnings.warn(
                "Modin doesn't allow columns to be created via a new attribute name - see "
                + "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access",
                UserWarning,
            )
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        """
        Set attribute `value` identified by `key`.

        Parameters
        ----------
        key : Any
            Key to set.
        value : Any
            Value to set.

        Returns
        -------
        None
        """
        if isinstance(key, slice):
            return self._setitem_slice(key, value)

        if hashable(key) and key not in self.columns:
            if isinstance(value, Series) and len(self.columns) == 0:
                # Note: column information is lost when assigning a query compiler
                prev_index = self.columns
                self._query_compiler = value._query_compiler.copy()
                # Now that the data is appended, we need to update the column name for
                # that column to `key`, otherwise the name could be incorrect.
                self.columns = prev_index.insert(0, key)
                return
            # Do new column assignment after error checks and possible value modifications
            self.insert(loc=len(self.columns), column=key, value=value)
            return

        if not hashable(key):
            if isinstance(key, DataFrame) or isinstance(key, np.ndarray):
                if isinstance(key, np.ndarray):
                    if key.shape != self.shape:
                        raise ValueError("Array must be same shape as DataFrame")
                    key = DataFrame(key, columns=self.columns)
                return self.mask(key, value, inplace=True)

            def setitem_unhashable_key(df, value):
                df[key] = value
                return df

            return self._update_inplace(
                self._default_to_pandas(setitem_unhashable_key, value)._query_compiler
            )
        if is_list_like(value):
            if isinstance(value, (pandas.DataFrame, DataFrame)):
                value = value[value.columns[0]].values
            elif isinstance(value, np.ndarray):
                assert (
                    len(value.shape) < 3
                ), "Shape of new values must be compatible with manager shape"
                value = value.T.reshape(-1)
                if len(self) > 0:
                    value = value[: len(self)]
            if not isinstance(value, (Series, Categorical)):
                value = list(value)

        if not self._query_compiler.lazy_execution and len(self.index) == 0:
            new_self = DataFrame({key: value}, columns=self.columns)
            self._update_inplace(new_self._query_compiler)
        else:
            if isinstance(value, Series):
                value = value._query_compiler
            self._update_inplace(self._query_compiler.setitem(0, key, value))

    def __iter__(self):
        """
        Iterate over info axis.

        Returns
        -------
        iterable
            Iterator of the columns names.
        """
        return iter(self.columns)

    def __contains__(self, key):
        """
        Check if `key` in the ``DataFrame.columns``.

        Parameters
        ----------
        key : hashable
            Key to check the presence in the columns.

        Returns
        -------
        bool
        """
        return self.columns.__contains__(key)

    def __round__(self, decimals=0):
        """
        Round each value in a ``DataFrame`` to the given number of decimals.

        Parameters
        ----------
        decimals : int, default: 0
            Number of decimal places to round to.

        Returns
        -------
        DataFrame
        """
        return self._default_to_pandas(pandas.DataFrame.__round__, decimals=decimals)

    def __delitem__(self, key):
        """
        Delete item identified by `key` label.

        Parameters
        ----------
        key : hashable
            Key to delete.
        """
        if key not in self:
            raise KeyError(key)
        self._update_inplace(new_query_compiler=self._query_compiler.delitem(key))

    __add__ = add
    __iadd__ = add  # pragma: no cover
    __radd__ = radd
    __mul__ = mul
    __imul__ = mul  # pragma: no cover
    __rmul__ = rmul
    __pow__ = pow
    __ipow__ = pow  # pragma: no cover
    __rpow__ = rpow
    __sub__ = sub
    __isub__ = sub  # pragma: no cover
    __rsub__ = rsub
    __floordiv__ = floordiv
    __ifloordiv__ = floordiv  # pragma: no cover
    __rfloordiv__ = rfloordiv
    __truediv__ = truediv
    __itruediv__ = truediv  # pragma: no cover
    __rtruediv__ = rtruediv
    __mod__ = mod
    __imod__ = mod  # pragma: no cover
    __rmod__ = rmod
    __rdiv__ = rdiv

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """
        Get a Modin DataFrame that implements the dataframe exchange protocol.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        nan_as_null : bool, default: False
            A keyword intended for the consumer to tell the producer
            to overwrite null values in the data with ``NaN`` (or ``NaT``).
            This currently has no effect; once support for nullable extension
            dtypes is added, this value should be propagated to columns.
        allow_copy : bool, default: True
            A keyword that defines whether or not the library is allowed
            to make a copy of the data. For example, copying data would be necessary
            if a library supports strided buffers, given that this protocol
            specifies contiguous buffers. Currently, if the flag is set to ``False``
            and a copy is needed, a ``RuntimeError`` will be raised.

        Returns
        -------
        ProtocolDataframe
            A dataframe object following the dataframe protocol specification.
        """
        return self._query_compiler.to_dataframe(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @property
    def attrs(self):  # noqa: D200
        """
        Return dictionary of global attributes of this dataset.
        """

        def attrs(df):
            return df.attrs

        self._default_to_pandas(attrs)

    @property
    def style(self):  # noqa: RT01, D200
        """
        Return a Styler object.
        """

        def style(df):
            """Define __name__ attr because properties do not have it."""
            return df.style

        return self._default_to_pandas(style)

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """
        Return or update a ``DataFrame`` with given `new_query_compiler`.

        Parameters
        ----------
        new_query_compiler : PandasQueryCompiler
            QueryCompiler to use to manage the data.
        inplace : bool, default: False
            Whether or not to perform update or creation inplace.

        Returns
        -------
        DataFrame or None
            None if update was done, ``DataFrame`` otherwise.
        """
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _get_numeric_data(self, axis: int):
        """
        Grab only numeric data from ``DataFrame``.

        Parameters
        ----------
        axis : {0, 1}
            Axis to inspect on having numeric types only.

        Returns
        -------
        DataFrame
            ``DataFrame`` with numeric data.
        """
        # Pandas ignores `numeric_only` if `axis` is 1, but we do have to drop
        # non-numeric columns if `axis` is 0.
        if axis != 0:
            return self
        return self.drop(
            columns=[
                i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
            ]
        )

    def _validate_dtypes(self, numeric_only=False):
        """
        Check that all the dtypes are the same.

        Parameters
        ----------
        numeric_only : bool, default: False
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.
        """
        dtype = self.dtypes[0]
        for t in self.dtypes:
            if numeric_only and not is_numeric_dtype(t):
                raise TypeError("{0} is not a numeric data type".format(t))
            elif not numeric_only and t != dtype:
                raise TypeError(
                    "Cannot compare type '{0}' with type '{1}'".format(t, dtype)
                )

    def _validate_dtypes_min_max(self, axis, numeric_only):
        """
        Validate data dtype for `min` and `max` methods.

        Parameters
        ----------
        axis : {0, 1}
            Axis to validate over.
        numeric_only : bool
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception.

        Returns
        -------
        DataFrame
        """
        # If our DataFrame has both numeric and non-numeric dtypes then
        # comparisons between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            axis
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot compare Numeric and Non-Numeric Types")

        return (
            self._get_numeric_data(axis)
            if numeric_only is None or numeric_only
            else self
        )

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        """
        Validate data dtype for `sum`, `prod` and `mean` methods.

        Parameters
        ----------
        axis : {0, 1}
            Axis to validate over.
        numeric_only : bool
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.
        ignore_axis : bool, default: False
            Whether or not to ignore `axis` parameter.

        Returns
        -------
        DataFrame
        """
        # We cannot add datetime types, so if we are summing a column with
        # dtype datetime64 and cannot ignore non-numeric types, we must throw a
        # TypeError.
        if (
            not axis
            and numeric_only is False
            and any(dtype == np.dtype("datetime64[ns]") for dtype in self.dtypes)
        ):
            raise TypeError("Cannot add Timestamp Types")

        # If our DataFrame has both numeric and non-numeric dtypes then
        # operations between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            (axis or ignore_axis)
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot operate on Numeric and Non-Numeric Types")

        return (
            self._get_numeric_data(axis)
            if numeric_only is None or numeric_only
            else self
        )

    def _to_pandas(self):
        """
        Convert Modin ``DataFrame`` to pandas ``DataFrame``.

        Returns
        -------
        pandas.DataFrame
        """
        return self._query_compiler.to_pandas()

    def _validate_eval_query(self, expr, **kwargs):
        """
        Validate the arguments of ``eval`` and ``query`` functions.

        Parameters
        ----------
        expr : str
            The expression to evaluate. This string cannot contain any
            Python statements, only Python expressions.
        **kwargs : dict
            Optional arguments of ``eval`` and ``query`` functions.
        """
        if isinstance(expr, str) and expr == "":
            raise ValueError("expr cannot be an empty string")

        if isinstance(expr, str) and "not" in expr:
            if "parser" in kwargs and kwargs["parser"] == "python":
                ErrorMessage.not_implemented(
                    "'Not' nodes are not implemented."
                )  # pragma: no cover

    def _reduce_dimension(self, query_compiler):
        """
        Reduce the dimension of data from the `query_compiler`.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to retrieve the data.

        Returns
        -------
        Series
        """
        return Series(query_compiler=query_compiler)

    def _set_axis_name(self, name, axis=0, inplace=False):
        """
        Alter the name or names of the axis.

        Parameters
        ----------
        name : str or list of str
            Name for the Index, or list of names for the MultiIndex.
        axis : str or int, default: 0
            The axis to set the label.
            0 or 'index' for the index, 1 or 'columns' for the columns.
        inplace : bool, default: False
            Whether to modify `self` directly or return a copy.

        Returns
        -------
        DataFrame or None
        """
        axis = self._get_axis_number(axis)
        renamed = self if inplace else self.copy()
        if axis == 0:
            renamed.index = renamed.index.set_names(name)
        else:
            renamed.columns = renamed.columns.set_names(name)
        if not inplace:
            return renamed

    def _to_datetime(self, **kwargs):
        """
        Convert `self` to datetime.

        Parameters
        ----------
        **kwargs : dict
            Optional arguments to use during query compiler's
            `to_datetime` invocation.

        Returns
        -------
        Series of datetime64 dtype
        """
        return self._reduce_dimension(
            query_compiler=self._query_compiler.to_datetime(**kwargs)
        )

    def _getitem(self, key):
        """
        Get the data specified by `key` for this ``DataFrame``.

        Parameters
        ----------
        key : callable, Series, DataFrame, np.ndarray, pandas.Index or list
            Data identifiers to retrieve.

        Returns
        -------
        Series or DataFrame
            Retrieved data.
        """
        key = apply_if_callable(key, self)
        # Shortcut if key is an actual column
        is_mi_columns = self._query_compiler.has_multiindex(axis=1)
        try:
            if key in self.columns and not is_mi_columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass
        if isinstance(key, Series):
            return DataFrame(
                query_compiler=self._query_compiler.getitem_array(key._query_compiler)
            )
        elif isinstance(key, (np.ndarray, pandas.Index, list)):
            return DataFrame(query_compiler=self._query_compiler.getitem_array(key))
        elif isinstance(key, DataFrame):
            return self.where(key)
        elif is_mi_columns:
            return self._default_to_pandas(pandas.DataFrame.__getitem__, key)
            # return self._getitem_multilevel(key)
        else:
            return self._getitem_column(key)

    # Persistance support methods - BEGIN
    @classmethod
    def _inflate_light(cls, query_compiler):
        """
        Re-creates the object from previously-serialized lightweight representation.

        The method is used for faster but not disk-storable persistence.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to use for object re-creation.

        Returns
        -------
        DataFrame
            New ``DataFrame`` based on the `query_compiler`.
        """
        return cls(query_compiler=query_compiler)

    @classmethod
    def _inflate_full(cls, pandas_df):
        """
        Re-creates the object from previously-serialized disk-storable representation.

        Parameters
        ----------
        pandas_df : pandas.DataFrame
            Data to use for object re-creation.

        Returns
        -------
        DataFrame
            New ``DataFrame`` based on the `pandas_df`.
        """
        return cls(data=from_pandas(pandas_df))

    def __reduce__(self):
        self._query_compiler.finalize()
        if PersistentPickle.get():
            return self._inflate_full, (self._to_pandas(),)
        return self._inflate_light, (self._query_compiler,)

    # Persistance support methods - END


if IsExperimental.get():
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(DataFrame, "make_dataframe_wrapper")
