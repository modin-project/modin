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

from __future__ import annotations

import datetime
import functools
import itertools
import os
import re
import sys
import warnings
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Hashable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import (
    CompressionOptions,
    FilePath,
    IndexLabel,
    Scalar,
    StorageOptions,
    WriteBuffer,
)
from pandas.core.common import apply_if_callable, get_cython_func
from pandas.core.computation.eval import _check_engine
from pandas.core.dtypes.common import (
    infer_dtype_from_object,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.io.formats.info import DataFrameInfo
from pandas.util._validators import validate_bool_kwarg

from modin.config import PersistentPickle
from modin.error_message import ErrorMessage
from modin.logging import disable_logging
from modin.pandas import Categorical
from modin.pandas.io import from_non_pandas, from_pandas, to_pandas
from modin.utils import (
    MODIN_UNNAMED_SERIES_LABEL,
    _inherit_docstrings,
    expanduser_path_arg,
    hashable,
    import_optional_dependency,
    try_cast_to_pandas,
)

from .accessor import CachedAccessor, SparseFrameAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .groupby import DataFrameGroupBy
from .iterator import PartitionIterator
from .series import Series
from .utils import (
    SET_DATAFRAME_ATTRIBUTE_WARNING,
    _doc_binary_op,
    cast_function_modin2pandas,
)

if TYPE_CHECKING:
    from modin.core.storage_formats import BaseQueryCompiler

# Dictionary of extensions assigned to this class
_DATAFRAME_EXTENSIONS_ = {}


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame.__init__], apilink="pandas.DataFrame"
)
class DataFrame(BasePandasDataset):
    """
    Modin distributed representation of ``pandas.DataFrame``.

    Internally, the data can be divided into partitions along both columns and rows
    in order to parallelize computations and utilize the user's hardware as much as possible.

    Inherit common for ``DataFrame``-s and ``Series`` functionality from the
    `BasePandasDataset` class.

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
        query_compiler: BaseQueryCompiler = None,
    ) -> None:
        from modin.numpy import array

        # Siblings are other dataframes that share the same query compiler. We
        # use this list to update inplace when there is a shallow copy.
        self._siblings = []
        if isinstance(data, (DataFrame, Series)):
            self._query_compiler = data._query_compiler.copy()
            if index is not None and any(i not in data.index for i in index):
                raise NotImplementedError(
                    "Passing non-existant columns or index values to constructor not"
                    + " yet implemented."
                )
            if isinstance(data, Series):
                # We set the column name if it is not in the provided Series
                if data.name is None:
                    self.columns = [0] if columns is None else columns
                # If the columns provided are not in the named Series, pandas clears
                # the DataFrame and sets columns to the columns provided.
                elif columns is not None and data.name not in columns:
                    self._query_compiler = from_pandas(
                        pandas.DataFrame(columns=columns)
                    )._query_compiler
                if index is not None:
                    self._query_compiler = data.loc[index]._query_compiler
            elif columns is None and index is None:
                data._add_sibling(self)
            else:
                if columns is not None and any(i not in data.columns for i in columns):
                    raise NotImplementedError(
                        "Passing non-existant columns or index values to constructor not"
                        + " yet implemented."
                    )
                if index is None:
                    index = slice(None)
                if columns is None:
                    columns = slice(None)
                self._query_compiler = data.loc[index, columns]._query_compiler
        elif isinstance(data, array):
            self._query_compiler = data._query_compiler.copy()
            if copy is not None and not copy:
                data._add_sibling(self)
            if columns is not None and not isinstance(columns, pandas.Index):
                columns = pandas.Index(columns)
            if columns is not None:
                obj_with_new_columns = self.set_axis(columns, axis=1, copy=False)
                self._update_inplace(obj_with_new_columns._query_compiler)
            if index is not None:
                obj_with_new_index = self.set_axis(index, axis=0, copy=False)
                self._update_inplace(obj_with_new_index._query_compiler)
            if dtype is not None:
                casted_obj = self.astype(dtype, copy=False)
                self._query_compiler = casted_obj._query_compiler
        # Check type of data and use appropriate constructor
        elif query_compiler is None:
            distributed_frame = from_non_pandas(data, index, columns, dtype)
            if distributed_frame is not None:
                self._query_compiler = distributed_frame._query_compiler
                return

            if isinstance(data, pandas.Index):
                pass
            elif (
                is_list_like(data)
                and not is_dict_like(data)
                and not isinstance(data, np.ndarray)
            ):
                old_dtype = getattr(data, "dtype", None)
                values = [
                    obj._to_pandas() if isinstance(obj, Series) else obj for obj in data
                ]
                try:
                    data = type(data)(values, dtype=old_dtype)
                except TypeError:
                    data = values
            elif is_dict_like(data) and not isinstance(
                data, (pandas.Series, Series, pandas.DataFrame, DataFrame)
            ):
                if columns is not None:
                    data = {key: value for key, value in data.items() if key in columns}

                if len(data) and all(isinstance(v, Series) for v in data.values()):
                    from .general import concat

                    new_qc = concat(
                        data.values(), axis=1, keys=data.keys()
                    )._query_compiler

                    if dtype is not None:
                        new_qc = new_qc.astype({col: dtype for col in new_qc.columns})
                    if index is not None:
                        new_qc = new_qc.reindex(axis=0, labels=index)
                    if columns is not None:
                        new_qc = new_qc.reindex(axis=1, labels=columns)

                    self._query_compiler = new_qc
                    return

                data = {
                    k: v._to_pandas() if isinstance(v, Series) else v
                    for k, v in data.items()
                }
            pandas_df = pandas.DataFrame(
                data=data, index=index, columns=columns, dtype=dtype, copy=copy
            )
            if pandas_df.size >= 1_000_000:
                warnings.warn(
                    "Distributing {} object. This may take some time.".format(
                        type(data)
                    )
                )
            self._query_compiler = from_pandas(pandas_df)._query_compiler
        else:
            self._query_compiler = query_compiler

    def __repr__(self) -> str:
        """
        Return a string representation for a particular ``DataFrame``.

        Returns
        -------
        str
        """
        num_rows = pandas.get_option("display.max_rows") or len(self)
        num_cols = pandas.get_option(
            "display.max_columns"
        ) or self._query_compiler.get_axis_len(1)
        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self) > num_rows or self._query_compiler.get_axis_len(1) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(
                *self.shape
            )
        else:
            return result

    def _repr_html_(self) -> str:  # pragma: no cover
        """
        Return a html representation for a particular ``DataFrame``.

        Returns
        -------
        str
        """
        num_rows = pandas.get_option("display.max_rows") or 60
        num_cols = pandas.get_option("display.max_columns") or 20

        # We use pandas _repr_html_ to get a string of the HTML representation
        # of the dataframe.
        result = self._build_repr_df(num_rows, num_cols)._repr_html_()
        if len(self) > num_rows or self._query_compiler.get_axis_len(1) > num_cols:
            # We split so that we insert our correct dataframe dimensions.
            return result.split("<p>")[
                0
            ] + "<p>{0} rows x {1} columns</p>\n</div>".format(*self.shape)
        else:
            return result

    def _get_columns(self) -> pandas.Index:
        """
        Get the columns for this ``DataFrame``.

        Returns
        -------
        pandas.Index
            The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_columns(self, new_columns) -> None:
        """
        Set the columns for this ``DataFrame``.

        Parameters
        ----------
        new_columns : list-like, Index
            The new index to set.
        """
        self._query_compiler.columns = new_columns

    columns: pandas.Index = property(_get_columns, _set_columns)

    @property
    def ndim(self) -> int:  # noqa: RT01, D200
        """
        Return the number of dimensions of the underlying data, by definition 2.
        """
        return 2

    def drop_duplicates(
        self, subset=None, *, keep="first", inplace=False, ignore_index=False
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Return ``DataFrame`` with duplicate rows removed.
        """
        return super(DataFrame, self).drop_duplicates(
            subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index
        )

    @property
    def dtypes(self) -> pandas.Series:  # noqa: RT01, D200
        """
        Return the dtypes in the ``DataFrame``.
        """
        return self._query_compiler.dtypes

    def duplicated(self, subset=None, keep="first") -> Series:  # noqa: PR01, RT01, D200
        """
        Return boolean ``Series`` denoting duplicate rows.
        """
        df = self[subset] if subset is not None else self
        new_qc = df._query_compiler.duplicated(keep=keep)
        duplicates = self._reduce_dimension(new_qc)
        return duplicates

    @property
    def empty(self) -> bool:  # noqa: RT01, D200
        """
        Indicate whether ``DataFrame`` is empty.
        """
        return self._query_compiler.get_axis_len(1) == 0 or len(self) == 0

    @property
    def axes(self) -> list[pandas.Index]:  # noqa: RT01, D200
        """
        Return a list representing the axes of the ``DataFrame``.
        """
        return [self.index, self.columns]

    @property
    def shape(self) -> tuple[int, int]:  # noqa: RT01, D200
        """
        Return a tuple representing the dimensionality of the ``DataFrame``.
        """
        return len(self), self._query_compiler.get_axis_len(1)

    def add_prefix(self, prefix, axis=None) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Prefix labels with string `prefix`.
        """
        axis = 1 if axis is None else self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.add_prefix(prefix, axis)
        )

    def add_suffix(self, suffix, axis=None) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Suffix labels with string `suffix`.
        """
        axis = 1 if axis is None else self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.add_suffix(suffix, axis)
        )

    def map(self, func, na_action: Optional[str] = None, **kwargs) -> DataFrame:
        if not callable(func):
            raise ValueError("'{0}' object is not callable".format(type(func)))
        return self.__constructor__(
            query_compiler=self._query_compiler.map(func, na_action=na_action, **kwargs)
        )

    def applymap(self, func, na_action: Optional[str] = None, **kwargs) -> DataFrame:
        warnings.warn(
            "DataFrame.applymap has been deprecated. Use DataFrame.map instead.",
            FutureWarning,
        )
        return self.map(func, na_action=na_action, **kwargs)

    def apply(
        self,
        func,
        axis=0,
        raw=False,
        result_type=None,
        args=(),
        by_row="compat",
        engine="python",
        engine_kwargs=None,
        **kwargs,
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Apply a function along an axis of the ``DataFrame``.
        """
        if by_row != "compat" or engine != "python" or engine_kwargs:
            # TODO: add test
            return self._default_to_pandas(
                pandas.DataFrame.apply,
                func=func,
                axis=axis,
                raw=raw,
                result_type=result_type,
                args=args,
                by_row=by_row,
                engine=engine,
                engine_kwargs=engine_kwargs,
                **kwargs,
            )

        func = cast_function_modin2pandas(func)
        axis = self._get_axis_number(axis)
        query_compiler = super(DataFrame, self).apply(
            func,
            axis=axis,
            raw=raw,
            result_type=result_type,
            args=args,
            **kwargs,
        )
        if not isinstance(query_compiler, type(self._query_compiler)):
            # A scalar was returned
            return query_compiler

        if result_type == "reduce":
            output_type = Series
        elif result_type == "broadcast":
            output_type = DataFrame
        # the 'else' branch also handles 'result_type == "expand"' since it makes the output type
        # depend on the `func` result (Series for a scalar, DataFrame for list-like)
        else:
            reduced_index = pandas.Index([MODIN_UNNAMED_SERIES_LABEL])
            if query_compiler.get_axis(axis).equals(
                reduced_index
            ) or query_compiler.get_axis(axis ^ 1).equals(reduced_index):
                output_type = Series
            else:
                output_type = DataFrame

        return output_type(query_compiler=query_compiler)

    def groupby(
        self,
        by=None,
        axis=lib.no_default,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=lib.no_default,
        dropna: bool = True,
    ):  # noqa: PR01, RT01, D200
        """
        Group ``DataFrame`` using a mapper or by a ``Series`` of columns.
        """
        if axis is not lib.no_default:
            axis = self._get_axis_number(axis)
            if axis == 1:
                warnings.warn(
                    "DataFrame.groupby with axis=1 is deprecated. Do "
                    + "`frame.T.groupby(...)` without axis instead.",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    "The 'axis' keyword in DataFrame.groupby is deprecated and "
                    + "will be removed in a future version.",
                    FutureWarning,
                )
        else:
            axis = 0

        axis = self._get_axis_number(axis)
        idx_name = None
        # Drop here indicates whether or not to drop the data column before doing the
        # groupby. The typical pandas behavior is to drop when the data came from this
        # dataframe. When a string, Series directly from this dataframe, or list of
        # strings is passed in, the data used for the groupby is dropped before the
        # groupby takes place.
        drop = False

        return_tuple_when_iterating = False
        if (
            not isinstance(by, (pandas.Series, Series))
            and is_list_like(by)
            and len(by) == 1
        ):
            by = by[0]
            return_tuple_when_iterating = True

        if callable(by):
            by = self.index.map(by)
        elif hashable(by) and not isinstance(by, (pandas.Grouper, FrozenList)):
            drop = by in self.columns
            idx_name = by
            if by is not None and by in self._query_compiler.get_index_names(axis):
                # In this case we pass the string value of the name through to the
                # partitions. This is more efficient than broadcasting the values.
                level, by = by, None
            elif level is None:
                by = self.__getitem__(by)._query_compiler
        elif isinstance(by, Series):
            drop = by._parent is self
            idx_name = by.name
            by = by._query_compiler
        elif isinstance(by, pandas.Grouper):
            drop = by.key in self
        elif is_list_like(by):
            # fastpath for multi column groupby
            if axis == 0 and all(
                (
                    (hashable(o) and (o in self))
                    or isinstance(o, Series)
                    or (isinstance(o, pandas.Grouper) and o.key in self)
                    or (is_list_like(o) and len(o) == len(self._get_axis(axis)))
                )
                for o in by
            ):
                has_external = False
                processed_by = []

                for current_by in by:
                    if isinstance(current_by, pandas.Grouper):
                        processed_by.append(current_by)
                        has_external = True
                    elif hashable(current_by):
                        processed_by.append(current_by)
                    elif isinstance(current_by, Series):
                        if current_by._parent is self:
                            processed_by.append(current_by.name)
                        else:
                            processed_by.append(current_by._query_compiler)
                            has_external = True
                    else:
                        has_external = True
                        processed_by.append(current_by)

                by = processed_by

                if not has_external:
                    by = self[processed_by]._query_compiler

                drop = True
            else:
                mismatch = len(by) != len(self._get_axis(axis))
                if mismatch and all(
                    hashable(obj)
                    and (
                        obj in self or obj in self._query_compiler.get_index_names(axis)
                    )
                    for obj in by
                ):
                    # In the future, we will need to add logic to handle this, but for now
                    # we default to pandas in this case.
                    pass
                elif mismatch and any(
                    hashable(obj) and obj not in self.columns for obj in by
                ):
                    names = [o.name if isinstance(o, Series) else o for o in by]
                    raise KeyError(next(x for x in names if x not in self))
        return DataFrameGroupBy(
            self,
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            idx_name,
            observed=observed,
            drop=drop,
            dropna=dropna,
            return_tuple_when_iterating=return_tuple_when_iterating,
        )

    def keys(self) -> pandas.Index:  # noqa: RT01, D200
        """
        Get columns of the ``DataFrame``.
        """
        return self.columns

    def transpose(self, copy=False, *args) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Transpose index and columns.
        """
        # FIXME: Judging by pandas docs `*args` serves only compatibility purpose
        # and does not affect the result, we shouldn't pass it to the query compiler.
        return self.__constructor__(
            query_compiler=self._query_compiler.transpose(*args)
        )

    T: DataFrame = property(transpose)

    def add(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get addition of ``DataFrame`` and `other`, element-wise (binary operator `add`).
        """
        return self._binary_op(
            "add",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def assign(self, **kwargs) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Assign new columns to a ``DataFrame``.
        """
        df = self.copy()
        for k, v in kwargs.items():
            if callable(v):
                df[k] = v(df)
            else:
                df[k] = v
        return df

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
        return to_pandas(self).boxplot(
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
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Perform column-wise combine with another ``DataFrame``.
        """
        return super(DataFrame, self).combine(
            other, func, fill_value=fill_value, overwrite=overwrite
        )

    def compare(
        self,
        other,
        align_axis=1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names=("self", "other"),
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Compare to another ``DataFrame`` and show the differences.
        """
        if not isinstance(other, DataFrame):
            raise TypeError(f"Cannot compare DataFrame to {type(other)}")
        other = self._validate_other(other, 0, compare_index=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.compare(
                other,
                align_axis=align_axis,
                keep_shape=keep_shape,
                keep_equal=keep_equal,
                result_names=result_names,
            )
        )

    def corr(
        self, method="pearson", min_periods=1, numeric_only=False
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Compute pairwise correlation of columns, excluding NA/null values.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.corr(
                method=method,
                min_periods=min_periods,
                numeric_only=numeric_only,
            )
        )

    def corrwith(
        self, other, axis=0, drop=False, method="pearson", numeric_only=False
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Compute pairwise correlation.
        """
        if not isinstance(other, (Series, DataFrame)):
            raise TypeError(f"unsupported type: {type(other)}")
        return self.__constructor__(
            query_compiler=self._query_compiler.corrwith(
                other=other._query_compiler,
                axis=axis,
                drop=drop,
                method=method,
                numeric_only=numeric_only,
            )
        )

    def cov(
        self, min_periods=None, ddof: Optional[int] = 1, numeric_only=False
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Compute pairwise covariance of columns, excluding NA/null values.
        """
        cov_df = self
        if numeric_only:
            cov_df = self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )

        if min_periods is not None and min_periods > len(cov_df):
            result = np.empty((cov_df.shape[1], cov_df.shape[1]))
            result.fill(np.nan)
            return cov_df.__constructor__(result)

        return cov_df.__constructor__(
            query_compiler=cov_df._query_compiler.cov(
                min_periods=min_periods, ddof=ddof
            )
        )

    def dot(self, other) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Compute the matrix multiplication between the ``DataFrame`` and `other`.
        """
        if isinstance(other, BasePandasDataset):
            common = self.columns.union(other.index)
            if len(common) > self._query_compiler.get_axis_len(1) or len(common) > len(
                other
            ):
                raise ValueError("Matrices are not aligned")

            qc = other.reindex(index=common)._query_compiler
            if isinstance(other, DataFrame):
                return self.__constructor__(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=False, squeeze_other=False
                    )
                )
            else:
                return self._reduce_dimension(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=False, squeeze_other=True
                    )
                )

        other = np.asarray(other)
        if self.shape[1] != other.shape[0]:
            raise ValueError(
                "Dot product shape mismatch, {} vs {}".format(self.shape, other.shape)
            )

        if len(other.shape) > 1:
            return self.__constructor__(
                query_compiler=self._query_compiler.dot(other, squeeze_self=False)
            )

        return self._reduce_dimension(
            query_compiler=self._query_compiler.dot(other, squeeze_self=False)
        )

    def eq(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Perform equality comparison of ``DataFrame`` and `other` (binary operator `eq`).
        """
        return self._binary_op(
            "eq", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def equals(self, other) -> bool:  # noqa: PR01, RT01, D200
        """
        Test whether two objects contain the same elements.
        """
        if isinstance(other, pandas.DataFrame):
            # Copy into a Modin DataFrame to simplify logic below
            other = self.__constructor__(other)

        if (
            type(self) is not type(other)
            or not self.index.equals(other.index)
            or not self.columns.equals(other.columns)
        ):
            return False

        result = self.__constructor__(
            query_compiler=self._query_compiler.equals(other._query_compiler)
        )
        return result.all(axis=None)

    def _update_var_dicts_in_kwargs(self, expr, kwargs) -> None:
        """
        Copy variables with "@" prefix in `local_dict` and `global_dict` keys of kwargs.

        Parameters
        ----------
        expr : str
            The expression string to search variables with "@" prefix.
        kwargs : dict
            See the documentation for eval() for complete details on the keyword arguments accepted by query().
        """
        if "@" not in expr:
            return
        frame = sys._getframe()
        try:
            f_locals = frame.f_back.f_back.f_back.f_back.f_locals
            f_globals = frame.f_back.f_back.f_back.f_back.f_globals
        finally:
            del frame
        local_names = set(re.findall(r"@([\w]+)", expr))
        local_dict = {}
        global_dict = {}

        for name in local_names:
            for dct_out, dct_in in ((local_dict, f_locals), (global_dict, f_globals)):
                try:
                    dct_out[name] = dct_in[name]
                except KeyError:
                    pass

        if local_dict:
            local_dict.update(kwargs.get("local_dict") or {})
            kwargs["local_dict"] = local_dict
        if global_dict:
            global_dict.update(kwargs.get("global_dict") or {})
            kwargs["global_dict"] = global_dict

    def eval(self, expr, inplace=False, **kwargs):  # noqa: PR01, RT01, D200
        """
        Evaluate a string describing operations on ``DataFrame`` columns.
        """
        self._update_var_dicts_in_kwargs(expr, kwargs)
        if _check_engine(kwargs.get("engine", None)) == "numexpr":
            # on numexpr engine, pandas.eval returns np.array if input is not of pandas
            # type, so we can't use pandas eval [1]. Even if we could, pandas eval seems
            # to convert all the data to numpy and then do the numexpr add, which is
            # slow for modin. The user would not really be getting the benefit of
            # numexpr.
            # [1] https://github.com/pandas-dev/pandas/blob/934eebb532cf50e872f40638a788000be6e4dda4/pandas/core/computation/align.py#L78
            return self._default_to_pandas(
                pandas.DataFrame.eval, expr, inplace=inplace, **kwargs
            )
        return pandas.DataFrame.eval(self, expr, inplace=inplace, **kwargs)

    def fillna(
        self,
        value=None,
        *,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=lib.no_default,
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Fill NA/NaN values using the specified method.
        """
        return super(DataFrame, self).fillna(
            squeeze_self=False,
            squeeze_value=isinstance(value, Series),
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )

    def floordiv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get integer division of ``DataFrame`` and `other`, element-wise (binary operator `floordiv`).
        """
        return self._binary_op(
            "floordiv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    @classmethod
    def from_dict(
        cls, data, orient="columns", dtype=None, columns=None
    ) -> DataFrame:  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Construct ``DataFrame`` from dict of array-like or dicts.
        """
        ErrorMessage.default_to_pandas("`from_dict`")
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
    ) -> DataFrame:  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Convert structured or record ndarray to ``DataFrame``.
        """
        ErrorMessage.default_to_pandas("`from_records`")
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

    def ge(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get greater than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._binary_op(
            "ge", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def gt(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get greater than comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._binary_op(
            "gt", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def hist(
        data,
        column: IndexLabel | None = None,
        by=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        ax=None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: tuple[int, int] | None = None,
        layout: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Make a histogram of the ``DataFrame``.
        """
        return data._default_to_pandas(
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
            backend=backend,
            legend=legend,
            **kwargs,
        )

    def info(
        self,
        verbose: Optional[bool] = None,
        buf: Optional[IO[str]] = None,
        max_cols: Optional[int] = None,
        memory_usage: Optional[Union[bool, str]] = None,
        show_counts: Optional[bool] = None,
    ) -> None:  # noqa: PR01, D200
        """
        Print a concise summary of the ``DataFrame``.
        """
        info = DataFrameInfo(
            data=self,
            memory_usage=memory_usage,
        )
        info.render(
            buf=buf,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )

    def insert(
        self, loc, column, value, allow_duplicates=lib.no_default
    ) -> None:  # noqa: PR01, D200
        """
        Insert column into ``DataFrame`` at specified location.
        """
        from modin.numpy import array

        if (
            isinstance(value, (DataFrame, pandas.DataFrame))
            or isinstance(value, (array, np.ndarray))
            and len(value.shape) > 1
        ):
            if isinstance(value, (array, np.ndarray)) and value.shape[1] != 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.shape}"
                )
            elif (
                isinstance(value, (DataFrame, pandas.DataFrame)) and value.shape[1] != 1
            ):
                raise ValueError(
                    "Expected a one-dimensional object, got a DataFrame with "
                    + f"{len(value.columns)} columns instead."
                )
            value = value.squeeze(axis=1)
        if not self._query_compiler.lazy_row_count and len(self) == 0:
            if not hasattr(value, "index"):
                try:
                    value = pandas.Series(value)
                except (TypeError, ValueError, IndexError):
                    raise ValueError(
                        "Cannot insert into a DataFrame with no defined index "
                        + "and a value that cannot be converted to a "
                        + "Series"
                    )
            new_index = value.index.copy()
            new_columns = self.columns.insert(loc, column)
            new_query_compiler = self.__constructor__(
                value, index=new_index, columns=new_columns
            )._query_compiler
        elif self._query_compiler.get_axis_len(1) == 0 and loc == 0:
            new_index = self.index
            new_query_compiler = self.__constructor__(
                data=value,
                columns=[column],
                index=None if len(new_index) == 0 else new_index,
            )._query_compiler
        else:
            if (
                is_list_like(value)
                and not isinstance(value, (pandas.Series, Series))
                and len(value) != len(self)
            ):
                raise ValueError(
                    "Length of values ({}) does not match length of index ({})".format(
                        len(value), len(self)
                    )
                )
            if allow_duplicates is not True and column in self.columns:
                raise ValueError(f"cannot insert {column}, already exists")
            columns_len = self._query_compiler.get_axis_len(1)
            if not -columns_len <= loc <= columns_len:
                raise IndexError(
                    f"index {loc} is out of bounds for axis 0 with size {columns_len}"
                )
            elif loc < 0:
                raise ValueError("unbounded slice")
            if isinstance(value, (Series, array)):
                value = value._query_compiler
            new_query_compiler = self._query_compiler.insert(loc, column, value)

        self._update_inplace(new_query_compiler=new_query_compiler)

    def isna(self) -> DataFrame:
        """
        Detect missing values.

        Returns
        -------
        DataFrame
            The result of detecting missing values.
        """
        return super(DataFrame, self).isna()

    def isnull(self) -> DataFrame:
        """
        Detect missing values.

        Returns
        -------
        DataFrame
            The result of detecting missing values.
        """
        return super(DataFrame, self).isnull()

    def iterrows(self) -> Iterable[tuple[Hashable, Series]]:  # noqa: D200
        """
        Iterate over ``DataFrame`` rows as (index, ``Series``) pairs.
        """

        def iterrow_builder(s):
            """Return tuple of the given `s` parameter name and the parameter themself."""
            return s.name, s

        partition_iterator = PartitionIterator(self, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self) -> Iterable[tuple[Hashable, Series]]:  # noqa: D200
        """
        Iterate over (column name, ``Series``) pairs.
        """

        def items_builder(s):
            """Return tuple of the given `s` parameter name and the parameter themself."""
            return s.name, s

        partition_iterator = PartitionIterator(self, 1, items_builder)
        for v in partition_iterator:
            yield v

    def itertuples(
        self, index=True, name="Pandas"
    ) -> Iterable[tuple[Any, ...]]:  # noqa: PR01, D200
        """
        Iterate over ``DataFrame`` rows as ``namedtuple``-s.
        """

        def itertuples_builder(s):
            """Return the next ``namedtuple``."""
            return next(s._to_pandas().to_frame().T.itertuples(index=index, name=name))

        partition_iterator = PartitionIterator(self, 0, itertuples_builder)
        for v in partition_iterator:
            yield v

    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        validate=None,
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Join columns of another ``DataFrame``.
        """
        if on is not None and not isinstance(other, (Series, DataFrame)):
            raise ValueError(
                "Joining multiple DataFrames only supported for joining on index"
            )
        if validate is not None:
            return self._default_to_pandas(
                pandas.DataFrame.join,
                other,
                on=on,
                how=how,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                sort=sort,
                validate=validate,
            )

        if isinstance(other, Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = self.__constructor__(other)
        if on is not None or how == "cross":
            return self.__constructor__(
                query_compiler=self._query_compiler.join(
                    other._query_compiler,
                    on=on,
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
                    validate=validate,
                )
            )
        if isinstance(other, DataFrame):
            # Joining the empty DataFrames with either index or columns is
            # fast. It gives us proper error checking for the edge cases that
            # would otherwise require a lot more logic.
            new_columns = (
                pandas.DataFrame(columns=self.columns)
                .join(
                    pandas.DataFrame(columns=other.columns),
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                )
                .columns
            )
            other = [other]
        else:
            new_columns = (
                pandas.DataFrame(columns=self.columns)
                .join(
                    [pandas.DataFrame(columns=obj.columns) for obj in other],
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                )
                .columns
            )
        new_frame = self.__constructor__(
            query_compiler=self._query_compiler.concat(
                1, [obj._query_compiler for obj in other], join=how, sort=sort
            )
        )
        new_frame.columns = new_columns
        return new_frame

    def isetitem(self, loc, value) -> None:
        return self._default_to_pandas(
            pandas.DataFrame.isetitem,
            loc=loc,
            value=value,
        )

    def le(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get less than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `le`).
        """
        return self._binary_op(
            "le", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def lt(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get less than comparison of ``DataFrame`` and `other`, element-wise (binary operator `le`).
        """
        return self._binary_op(
            "lt", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index=True,
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Unpivot a ``DataFrame`` from wide to long format, optionally leaving identifiers set.
        """
        if id_vars is None:
            id_vars = []
        if not is_list_like(id_vars):
            id_vars = [id_vars]
        if value_vars is None:
            value_vars = self.columns.drop(id_vars)
        if var_name is None:
            columns_name = self._query_compiler.get_index_name(axis=1)
            var_name = columns_name if columns_name is not None else "variable"
        return self.__constructor__(
            query_compiler=self._query_compiler.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=var_name,
                value_name=value_name,
                col_level=col_level,
                ignore_index=ignore_index,
            )
        )

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
        copy=None,
        indicator=False,
        validate=None,
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Merge ``DataFrame`` or named ``Series`` objects with a database-style join.
        """
        if copy is None:
            copy = True
        if isinstance(right, Series):
            if right.name is None:
                raise ValueError("Cannot merge a Series without a name")
            else:
                right = right.to_frame()
        if not isinstance(right, DataFrame):
            raise TypeError(
                f"Can only merge Series or DataFrame objects, a {type(right)} was passed"
            )

        # If we are joining on the index and we are using
        # default parameters we can map this to a join
        if left_index and right_index and not indicator:
            return self.join(
                right, how=how, lsuffix=suffixes[0], rsuffix=suffixes[1], sort=sort
            )

        return self.__constructor__(
            query_compiler=self._query_compiler.merge(
                right._query_compiler,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                sort=sort,
                suffixes=suffixes,
                copy=copy,
                indicator=indicator,
                validate=validate,
            )
        )

    def mod(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get modulo of ``DataFrame`` and `other`, element-wise (binary operator `mod`).
        """
        return self._binary_op(
            "mod",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def mul(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get multiplication of ``DataFrame`` and `other`, element-wise (binary operator `mul`).
        """
        return self._binary_op(
            "mul",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    multiply = mul

    def rmul(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get multiplication of ``DataFrame`` and `other`, element-wise (binary operator `mul`).
        """
        return self._binary_op(
            "rmul",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def ne(
        self, other, axis="columns", level=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get not equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ne`).
        """
        return self._binary_op(
            "ne", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def nlargest(self, n, columns, keep="first") -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in descending order.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.nlargest(n, columns, keep)
        )

    def nsmallest(
        self, n, columns, keep="first"
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in ascending order.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.nsmallest(
                n=n, columns=columns, keep=keep
            )
        )

    def unstack(
        self, level=-1, fill_value=None, sort=True
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Pivot a level of the (necessarily hierarchical) index labels.
        """
        if not sort:
            # TODO: it should be easy to add support for sort == False
            return self._default_to_pandas(
                pandas.DataFrame.unstack, level=level, fill_value=fill_value, sort=sort
            )

        # This ensures that non-pandas MultiIndex objects are caught.
        is_multiindex = len(self.index.names) > 1
        if not is_multiindex or (
            is_multiindex and is_list_like(level) and len(level) == self.index.nlevels
        ):
            return self._reduce_dimension(
                query_compiler=self._query_compiler.unstack(level, fill_value)
            )
        else:
            return self.__constructor__(
                query_compiler=self._query_compiler.unstack(level, fill_value)
            )

    def pivot(
        self, *, columns, index=lib.no_default, values=lib.no_default
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Return reshaped ``DataFrame`` organized by given index / column values.
        """
        if index is lib.no_default:
            index = None
        if values is lib.no_default:
            values = None

        # if values is not specified, it should be the remaining columns not in
        # index or columns
        if values is None:
            values = list(self.columns)
            if index is not None:
                values = [v for v in values if v not in index]
            if columns is not None:
                values = [v for v in values if v not in columns]

        return self.__constructor__(
            query_compiler=self._query_compiler.pivot(
                index=index, columns=columns, values=values
            )
        )

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
        observed=lib.no_default,
        sort=True,
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Create a spreadsheet-style pivot table as a ``DataFrame``.
        """
        # Convert callable to a string aggregation name if possible
        if hashable(aggfunc):
            aggfunc = get_cython_func(aggfunc) or aggfunc

        result = self.__constructor__(
            query_compiler=self._query_compiler.pivot_table(
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
        )
        return result

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
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get exponential power of ``DataFrame`` and `other`, element-wise (binary operator `pow`).
        """
        if isinstance(other, Series):
            return self._default_to_pandas(
                "pow", other, axis=axis, level=level, fill_value=fill_value
            )
        return self._binary_op(
            "pow",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def prod(
        self,
        axis=0,
        skipna=True,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Return the product of the values over the requested axis.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        axis = self._get_axis_number(axis)

        axis_to_apply = self.columns if axis else self.index
        if (
            skipna is not False
            and numeric_only is False
            and min_count > len(axis_to_apply)
            # This fast path is only suitable for the default backend
            and self._query_compiler.get_pandas_backend() is None
        ):
            new_index = self.columns if not axis else self.index
            # >>> pd.DataFrame([1,2,3,4], dtype="int64[pyarrow]").prod(min_count=10)
            # 0    <NA>
            # dtype: int64[pyarrow]
            return Series(
                [np.nan] * len(new_index),
                index=new_index,
                dtype=pandas.api.types.pandas_dtype("float64"),
            )

        data = self._validate_dtypes_prod_mean(axis, numeric_only, ignore_axis=True)
        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.prod_min_count(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.prod(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    product = prod

    def quantile(
        self,
        q=0.5,
        axis=0,
        numeric_only=False,
        interpolation="linear",
        method="single",
    ) -> Union[DataFrame, Series]:
        return super(DataFrame, self).quantile(
            q=q,
            axis=axis,
            numeric_only=numeric_only,
            interpolation=interpolation,
            method=method,
        )

    # methods and fields we need to use pandas.DataFrame.query
    _AXIS_ORDERS = ["index", "columns"]
    _get_index_resolvers = pandas.DataFrame._get_index_resolvers

    def _get_axis_resolvers(self, axis: str) -> dict:  # noqa: GL08
        # forked from pandas because we only want to update the index if there's more
        # than one level of the index.
        # index or columns
        axis_index = getattr(self, axis)
        d = {}
        prefix = axis[0]

        for i, name in enumerate(axis_index.names):
            if name is not None:
                key = level = name
            else:
                # prefix with 'i' or 'c' depending on the input axis
                # e.g., you must do ilevel_0 for the 0th level of an unnamed
                # multiiindex
                key = f"{prefix}level_{i}"
                level = i

            level_values = axis_index.get_level_values(level)
            s = level_values.to_series()
            if axis_index.nlevels > 1:
                s.index = axis_index
            d[key] = s

        # put the index/columns itself in the dict
        if axis_index.nlevels > 2:
            dindex = axis_index
        else:
            dindex = axis_index.to_series()

        d[axis] = dindex
        return d

    def _get_cleaned_column_resolvers(self) -> dict[Hashable, Series]:  # noqa: RT01
        """
        Return the special character free column resolvers of a dataframe.

        Column names with special characters are 'cleaned up' so that they can
        be referred to by backtick quoting.
        Used in `DataFrame.eval`.

        Notes
        -----
        Copied from pandas.
        """
        from pandas.core.computation.parsing import clean_column_name

        return {
            clean_column_name(k): v for k, v in self.items() if not isinstance(k, int)
        }

    def query(
        self, expr, inplace=False, **kwargs
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Query the columns of a ``DataFrame`` with a boolean expression.
        """
        self._update_var_dicts_in_kwargs(expr, kwargs)
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        # HACK: this condition kind of breaks the idea of backend agnostic API as all queries
        # _should_ work fine for all of the engines using `pandas.DataFrame.query(...)` approach.
        # However, at this point we know that we can execute simple queries way more efficiently
        # using the QC's API directly in case of pandas backend. Ideally, we have to make it work
        # with the 'pandas.query' approach the same as good the direct QC call is. But investigating
        # and fixing the root cause of the perf difference appears to be much more complicated
        # than putting this hack here. Hopefully, we'll get rid of it soon:
        # https://github.com/modin-project/modin/issues/6499
        try:
            new_query_compiler = self._query_compiler.rowwise_query(expr, **kwargs)
        except NotImplementedError:
            # a non row-wise query was passed, falling back to pandas implementation
            new_query_compiler = pandas.DataFrame.query(
                self, expr, inplace=False, **kwargs
            )._query_compiler
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=None,
        inplace=False,
        level=None,
        errors="ignore",
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Alter axes labels.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if mapper is None and index is None and columns is None:
            raise TypeError("must pass an index to rename")
        # We have to do this with the args because of how rename handles kwargs. It
        # doesn't ignore None values passed in, so we have to filter them ourselves.
        args = locals()
        kwargs = {k: v for k, v in args.items() if v is not None and k != "self"}
        # inplace should always be true because this is just a copy, and we will use the
        # results after.
        kwargs["inplace"] = False
        if axis is not None:
            axis = self._get_axis_number(axis)
        if index is not None or (mapper is not None and axis == 0):
            new_index = pandas.DataFrame(index=self.index).rename(**kwargs).index
        else:
            new_index = None
        if columns is not None or (mapper is not None and axis == 1):
            new_columns = (
                pandas.DataFrame(columns=self.columns).rename(**kwargs).columns
            )
        else:
            new_columns = None

        if inplace:
            obj = self
        else:
            obj = self.copy()
        if new_index is not None:
            obj.index = new_index
        if new_columns is not None:
            obj.columns = new_columns

        if not inplace:
            return obj

    def reindex(
        self,
        labels=None,
        *,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=None,
        level=None,
        fill_value=np.nan,
        limit=None,
        tolerance=None,
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        axis = self._get_axis_number(axis)
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels
        return super(DataFrame, self).reindex(
            index=index,
            columns=columns,
            method=method,
            copy=copy,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
        )

    def replace(
        self,
        to_replace=None,
        value=lib.no_default,
        *,
        inplace: bool = False,
        limit=None,
        regex: bool = False,
        method: str | lib.NoDefault = lib.no_default,
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Replace values given in `to_replace` with `value`.
        """
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

    def rfloordiv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get integer division of ``DataFrame`` and `other`, element-wise (binary operator `rfloordiv`).
        """
        return self._binary_op(
            "rfloordiv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def radd(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Get addition of ``DataFrame`` and `other`, element-wise (binary operator `radd`).
        """
        return self._binary_op(
            "radd",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rmod(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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

    def select_dtypes(
        self, include=None, exclude=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
            is_dtype_instance_mapper, self.dtypes.items()
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
        self, keys, *, drop=True, append=False, inplace=False, verify_integrity=False
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
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
            if drop:
                keys = [k if is_list_like(k) else frame.pop(k) for k in keys]
            keys = try_cast_to_pandas(keys)
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
        # If the missing column is a "primitive", return the errors.
        # Otherwise we let the query compiler figure out what to do with
        # the keys
        if missing and not hasattr(missing[0], "__dict__"):
            # The keys are a primitive type
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

    def squeeze(
        self, axis=None
    ) -> Union[DataFrame, Series, Scalar]:  # noqa: PR01, RT01, D200
        """
        Squeeze 1 dimensional axis objects into scalars.
        """
        axis = self._get_axis_number(axis) if axis is not None else None
        if axis is None and (
            self._query_compiler.get_axis_len(1) == 1 or len(self) == 1
        ):
            return Series(query_compiler=self._query_compiler).squeeze()
        if axis == 1 and self._query_compiler.get_axis_len(1) == 1:
            self._query_compiler._shape_hint = "column"
            return Series(query_compiler=self._query_compiler)
        if axis == 0 and len(self) == 1:
            qc = self.T._query_compiler
            qc._shape_hint = "column"
            return Series(query_compiler=qc)
        else:
            return self.copy()

    def stack(
        self, level=-1, dropna=lib.no_default, sort=lib.no_default, future_stack=False
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Stack the prescribed level(s) from columns to index.
        """
        if future_stack:
            return self._default_to_pandas(
                pandas.DataFrame.stack,
                level=level,
                dropna=dropna,
                sort=sort,
                future_stack=future_stack,
            )

        # FutureWarnings only needed if future_stack == True
        if dropna is lib.no_default:
            dropna = True
        if sort is lib.no_default:
            sort = True

        # This ensures that non-pandas MultiIndex objects are caught.
        is_multiindex = len(self.columns.names) > 1
        if not is_multiindex or (
            is_multiindex and is_list_like(level) and len(level) == self.columns.nlevels
        ):
            return self._reduce_dimension(
                query_compiler=self._query_compiler.stack(level, dropna, sort)
            )
        else:
            return self.__constructor__(
                query_compiler=self._query_compiler.stack(level, dropna, sort)
            )

    def sub(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
        axis=0,
        skipna=True,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ) -> Series:  # noqa: PR01, RT01, D200
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        """
        Return the sum of the values over the requested axis.
        """
        axis = self._get_axis_number(axis)
        axis_to_apply = self.columns if axis else self.index
        if (
            skipna is not False
            and numeric_only is False
            and min_count > len(axis_to_apply)
            # This fast path is only suitable for the default backend
            and self._query_compiler.get_pandas_backend() is None
        ):
            new_index = self.columns if not axis else self.index
            return Series(
                [np.nan] * len(new_index),
                index=new_index,
                dtype=pandas.api.types.pandas_dtype("float64"),
            )

        # We cannot add datetime types, so if we are summing a column with
        # dtype datetime64 and cannot ignore non-numeric types, we must throw a
        # TypeError.
        if numeric_only is False and any(
            dtype == pandas.api.types.pandas_dtype("datetime64[ns]")
            for dtype in self.dtypes
        ):
            raise TypeError(
                "'DatetimeArray' with dtype datetime64[ns] does not support reduction 'sum'"
            )

        data = self._get_numeric_data(axis) if numeric_only else self

        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.sum_min_count(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.sum(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    @expanduser_path_arg("path")
    def to_feather(
        self, path, **kwargs
    ) -> None:  # pragma: no cover # noqa: PR01, RT01, D200
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
        auth_local_webserver=True,
        table_schema=None,
        location=None,
        progress_bar=True,
        credentials=None,
    ) -> None:  # pragma: no cover # noqa: PR01, RT01, D200
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

    @expanduser_path_arg("path")
    def to_orc(
        self, path=None, *, engine="pyarrow", index=None, engine_kwargs=None
    ) -> Union[bytes, None]:
        return self._default_to_pandas(
            pandas.DataFrame.to_orc,
            path=path,
            engine=engine,
            index=index,
            engine_kwargs=engine_kwargs,
        )

    @expanduser_path_arg("buf")
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
    ) -> Union[str, None]:  # noqa: PR01, RT01, D200
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

    @expanduser_path_arg("path")
    def to_parquet(
        self,
        path=None,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options: StorageOptions = None,
        **kwargs,
    ) -> Union[bytes, None]:
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_parquet(
            self._query_compiler,
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            storage_options=storage_options,
            **kwargs,
        )

    def to_period(
        self, freq=None, axis=0, copy=None
    ) -> DataFrame:  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Convert ``DataFrame`` from ``DatetimeIndex`` to ``PeriodIndex``.
        """
        return super(DataFrame, self).to_period(freq=freq, axis=axis, copy=copy)

    def to_records(
        self, index=True, column_dtypes=None, index_dtypes=None
    ) -> np.rec.recarray:  # noqa: PR01, RT01, D200
        """
        Convert ``DataFrame`` to a NumPy record array.
        """
        return self._default_to_pandas(
            pandas.DataFrame.to_records,
            index=index,
            column_dtypes=column_dtypes,
            index_dtypes=index_dtypes,
        )

    @expanduser_path_arg("path")
    def to_stata(
        self,
        path: FilePath | WriteBuffer[bytes],
        *,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime.datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        version: int | None = 114,
        convert_strl: Sequence[Hashable] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions = None,
        value_labels: dict[Hashable, dict[float | int, str]] | None = None,
    ) -> None:
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

    @expanduser_path_arg("path_or_buffer")
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
    ) -> Union[str, None]:
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_xml(
            self._query_compiler,
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

    def to_timestamp(
        self, freq=None, how="start", axis=0, copy=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Cast to DatetimeIndex of timestamps, at *beginning* of period.
        """
        return super(DataFrame, self).to_timestamp(
            freq=freq, how=how, axis=axis, copy=copy
        )

    def truediv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> DataFrame:  # noqa: PR01, RT01, D200
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
    ) -> None:  # noqa: PR01, RT01, D200
        """
        Modify in place using non-NA values from another ``DataFrame``.
        """
        if not isinstance(other, DataFrame):
            other = self.__constructor__(other)
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
        other=np.nan,
        *,
        inplace=False,
        axis=None,
        level=None,
    ) -> Union[DataFrame, None]:  # noqa: PR01, RT01, D200
        """
        Replace values where the condition is False.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(other, Series) and axis is None:
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
            )
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        cond = cond(self) if callable(cond) else cond

        if not isinstance(cond, DataFrame):
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            cond = self.__constructor__(cond, index=self.index, columns=self.columns)
        if isinstance(other, DataFrame):
            other = other._query_compiler
        else:
            """
            Only infer the axis number when ``other`` will be made into a
            series. When ``other`` is a dataframe, axis=None has a meaning
            distinct from 0 and 1, e.g. at pandas 1.4.3:

            import pandas as pd
            df = pd.DataFrame([[1,2], [3, 4]], index=[1, 0])
            cond = pd.DataFrame([[True,False], [False, True]], columns=[1, 0])
            other = pd.DataFrame([[5,6], [7,8]], columns=[1, 0])

            print(df.where(cond, other, axis=None))
            0  1
            1  1  7
            0  6  4

            print(df.where(cond, other, axis=0))

            0  1
            1  1  8
            0  5  4

            print(df.where(cond, other, axis=1))

            0  1
            1  1  5
            0  8  4
            """
            # _get_axis_number interprets lib.no_default as None, but where doesn't
            # accept lib.no_default.
            if axis == lib.no_default:
                raise ValueError(
                    "No axis named NoDefault.no_default for object type DataFrame"
                )
            axis = self._get_axis_number(axis)
            if isinstance(other, Series):
                other = other.reindex(
                    self.index if axis == 0 else self.columns
                )._query_compiler
                if other._shape_hint is None:
                    # To make the query compiler recognizable as a Series at lower levels
                    other._shape_hint = "column"
            elif is_list_like(other):
                index = self.index if axis == 0 else self.columns
                other = pandas.Series(other, index=index)
        query_compiler = self._query_compiler.where(
            cond._query_compiler, other, axis=axis, level=level
        )
        return self._create_or_update_from_compiler(query_compiler, inplace)

    def _getitem_column(self, key) -> Series:
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
        s = self.__constructor__(
            query_compiler=self._query_compiler.getitem_column_array([key])
        ).squeeze(axis=1)
        if isinstance(s, Series):
            s._parent = self
            s._parent_axis = 1
        return s

    @disable_logging
    def __getattr__(self, key) -> Any:
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
            return _DATAFRAME_EXTENSIONS_.get(key, object.__getattribute__(self, key))
        except AttributeError as err:
            if key not in _ATTRS_NO_LOOKUP and key in self.columns:
                return self[key]
            raise err

    def __setattr__(self, key, value) -> None:
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
        #   before it appears in __dict__.
        if key in ("_query_compiler", "_siblings") or key in self.__dict__:
            pass
        # we have to check for the key in `dir(self)` first in order not to trigger columns computation
        elif key not in dir(self) and key in self:
            self.__setitem__(key, value)
            # Note: return immediately so we don't keep this `key` as dataframe state.
            # `__getattr__` will return the columns not present in `dir(self)`, so we do not need
            # to manually track this state in the `dir`.
            return
        elif is_list_like(value) and key not in ["index", "columns"]:
            warnings.warn(
                SET_DATAFRAME_ATTRIBUTE_WARNING,
                UserWarning,
            )
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value) -> None:
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
            if isinstance(value, Series) and self._query_compiler.get_axis_len(1) == 0:
                # Note: column information is lost when assigning a query compiler
                prev_index = self.columns
                self._query_compiler = value._query_compiler.copy()
                # Now that the data is appended, we need to update the column name for
                # that column to `key`, otherwise the name could be incorrect.
                self.columns = prev_index.insert(0, key)
                return
            # Do new column assignment after error checks and possible value modifications
            self.insert(
                loc=self._query_compiler.get_axis_len(1), column=key, value=value
            )
            return

        if not hashable(key):
            if isinstance(key, DataFrame) or isinstance(key, np.ndarray):
                if isinstance(key, np.ndarray):
                    if key.shape != self.shape:
                        raise ValueError("Array must be same shape as DataFrame")
                    key = self.__constructor__(key, columns=self.columns)
                return self.mask(key, value, inplace=True)

            if isinstance(key, (list, pandas.Index)) and all(
                (x in self.columns for x in key)
            ):
                if is_list_like(value):
                    if not (hasattr(value, "shape") and hasattr(value, "ndim")):
                        value = np.array(value)
                    if len(key) != value.shape[-1]:
                        raise ValueError("Columns must be same length as key")
                if isinstance(value, type(self)):
                    # importing here to avoid circular import
                    from .general import concat

                    if not value.columns.equals(pandas.Index(key)):
                        # we only need to change the labels, so shallow copy here
                        value = value.copy(deep=False)
                        value.columns = key

                    # here we iterate over every column in the 'self' frame, then check if it's in the 'key'
                    # and so has to be taken from either from the 'value' or from the 'self'. After that,
                    # we concatenate those mixed column chunks and get a dataframe with updated columns
                    to_concat = []
                    # columns to take for this chunk
                    to_take = []
                    # whether columns in this chunk are in the 'key' and has to be taken from the 'value'
                    get_cols_from_value = False
                    # an object to take columns from for this chunk
                    src_obj = self
                    for col in self.columns:
                        if (col in key) != get_cols_from_value:
                            if len(to_take):
                                to_concat.append(src_obj[to_take])
                            to_take = [col]
                            get_cols_from_value = not get_cols_from_value
                            src_obj = value if get_cols_from_value else self
                        else:
                            to_take.append(col)
                    if len(to_take):
                        to_concat.append(src_obj[to_take])

                    new_qc = concat(to_concat, axis=1)._query_compiler
                else:
                    new_qc = self._query_compiler.write_items(
                        slice(None),
                        self.columns.get_indexer_for(key),
                        value,
                        need_columns_reindex=False,
                    )
                self._update_inplace(new_qc)
                # self.loc[:, key] = value
                return
            elif (
                isinstance(key, list)
                and isinstance(value, type(self))
                # Mixed case is more complicated, it's defaulting to pandas for now
                and all((x not in self.columns for x in key))
            ):
                if len(key) != len(value.columns):
                    raise ValueError("Columns must be same length as key")

                # Aligning the value's columns with the key
                if not np.array_equal(value.columns, key):
                    value = value.set_axis(key, axis=1)

                new_qc = self._query_compiler.insert_item(
                    axis=1,
                    loc=self._query_compiler.get_axis_len(1),
                    value=value._query_compiler,
                    how="left",
                )
                self._update_inplace(new_qc)
                return

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
            if not isinstance(value, (Series, Categorical, np.ndarray, list, range)):
                value = list(value)

        if not self._query_compiler.lazy_row_count and len(self) == 0:
            new_self = self.__constructor__({key: value}, columns=self.columns)
            self._update_inplace(new_self._query_compiler)
        else:
            if isinstance(value, Series):
                value = value._query_compiler
            self._update_inplace(self._query_compiler.setitem(0, key, value))

    def __iter__(self) -> Iterable[Hashable]:
        """
        Iterate over info axis.

        Returns
        -------
        iterable
            Iterator of the columns names.
        """
        return iter(self.columns)

    def __contains__(self, key) -> bool:
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

    def __round__(self, decimals=0) -> DataFrame:
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
        return self.round(decimals)

    def __delitem__(self, key) -> None:
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

    @_doc_binary_op(
        operation="integer division and modulo",
        bin_op="divmod",
        returns="tuple of two DataFrames",
    )
    def __divmod__(self, right) -> tuple[DataFrame, DataFrame]:
        return self._default_to_pandas(pandas.DataFrame.__divmod__, right)

    @_doc_binary_op(
        operation="integer division and modulo",
        bin_op="divmod",
        right="left",
        returns="tuple of two DataFrames",
    )
    def __rdivmod__(self, left) -> tuple[DataFrame, DataFrame]:
        return self._default_to_pandas(pandas.DataFrame.__rdivmod__, left)

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
        return self._query_compiler.to_interchange_dataframe(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    def __dataframe_consortium_standard__(
        self, *, api_version: str | None = None
    ):  # noqa: PR01, RT01
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of Modin.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        dataframe_api_compat = import_optional_dependency(
            "dataframe_api_compat", "implementation"
        )
        convert_to_standard_compliant_dataframe = (
            dataframe_api_compat.modin_standard.convert_to_standard_compliant_dataframe
        )
        return convert_to_standard_compliant_dataframe(self, api_version=api_version)

    @property
    def attrs(self) -> dict:  # noqa: RT01, D200
        """
        Return dictionary of global attributes of this dataset.
        """

        def attrs(df):
            return df.attrs

        return self._default_to_pandas(attrs)

    @property
    def style(self):  # noqa: RT01, D200
        """
        Return a Styler object.
        """

        def style(df):
            """Define __name__ attr because properties do not have it."""
            return df.style

        return self._default_to_pandas(style)

    def reindex_like(
        self: DataFrame,
        other,
        method=None,
        copy: Optional[bool] = None,
        limit=None,
        tolerance=None,
    ) -> DataFrame:
        if copy is None:
            copy = True
        # docs say "Same as calling .reindex(index=other.index, columns=other.columns,...).":
        # https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.DataFrame.reindex_like.html
        return self.reindex(
            index=other.index,
            columns=other.columns,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )

    def _create_or_update_from_compiler(
        self, new_query_compiler, inplace=False
    ) -> Union[DataFrame, None]:
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
        assert isinstance(
            new_query_compiler, self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return self.__constructor__(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _get_numeric_data(self, axis: int) -> DataFrame:
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

    def _validate_dtypes(self, numeric_only=False) -> None:
        """
        Check that all the dtypes are the same.

        Parameters
        ----------
        numeric_only : bool, default: False
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.
        """
        # Series.__getitem__ treating keys as positions is deprecated. In a future version,
        # integer keys will always be treated as labels (consistent with DataFrame behavior).
        # To access a value by position, use `ser.iloc[pos]`
        dtypes = self._query_compiler.get_dtypes_set()
        dtype = next(iter(dtypes))
        for t in dtypes:
            if numeric_only and not is_numeric_dtype(t):
                raise TypeError("{0} is not a numeric data type".format(t))
            elif not numeric_only and t != dtype:
                raise TypeError(
                    "Cannot compare type '{0}' with type '{1}'".format(t, dtype)
                )

    def _validate_dtypes_min_max(self, axis, numeric_only) -> DataFrame:
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
        # TypeError. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            axis
            and numeric_only is False
            and not all([is_numeric_dtype(dtype) for dtype in self.dtypes])
        ):
            raise TypeError("Cannot compare Numeric and Non-Numeric Types")

        return self._get_numeric_data(axis) if numeric_only else self

    def _validate_dtypes_prod_mean(
        self, axis, numeric_only, ignore_axis=False
    ) -> DataFrame:
        """
        Validate data dtype for `prod` and `mean` methods.

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
        # If our DataFrame has both numeric and non-numeric dtypes then
        # operations between these types do not make sense and we must raise a
        # TypeError. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            (axis or ignore_axis)
            and numeric_only is False
            and not all([is_numeric_dtype(dtype) for dtype in self.dtypes])
        ):
            raise TypeError("Cannot operate on Numeric and Non-Numeric Types")

        return self._get_numeric_data(axis) if numeric_only else self

    def _to_pandas(self) -> pandas.DataFrame:
        """
        Convert Modin ``DataFrame`` to pandas ``DataFrame``.

        Recommended conversion method: `dataframe.modin.to_pandas()`.

        Returns
        -------
        pandas.DataFrame
        """
        return self._query_compiler.to_pandas()

    def _validate_eval_query(self, expr, **kwargs) -> None:
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

    def _reduce_dimension(self, query_compiler: BaseQueryCompiler) -> Series:
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

    def _set_axis_name(self, name, axis=0, inplace=False) -> Union[DataFrame, None]:
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

    def _to_datetime(self, **kwargs) -> Series:
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

    def _getitem(self, key) -> Union[DataFrame, Series]:
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
            return self.__constructor__(
                query_compiler=self._query_compiler.getitem_array(key._query_compiler)
            )
        elif isinstance(key, (np.ndarray, pandas.Index, list)):
            return self.__constructor__(
                query_compiler=self._query_compiler.getitem_array(key)
            )
        elif isinstance(key, DataFrame):
            return self.where(key)
        elif is_mi_columns:
            return self._default_to_pandas(pandas.DataFrame.__getitem__, key)
            # return self._getitem_multilevel(key)
        else:
            return self._getitem_column(key)

    # Persistance support methods - BEGIN
    @classmethod
    def _inflate_light(cls, query_compiler, source_pid) -> DataFrame:
        """
        Re-creates the object from previously-serialized lightweight representation.

        The method is used for faster but not disk-storable persistence.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to use for object re-creation.
        source_pid : int
            Determines whether a Modin or pandas object needs to be created.
            Modin objects are created only on the main process.

        Returns
        -------
        DataFrame
            New ``DataFrame`` based on the `query_compiler`.
        """
        if os.getpid() != source_pid:
            return query_compiler.to_pandas()
        # The current logic does not involve creating Modin objects
        # and manipulation with them in worker processes
        return cls(query_compiler=query_compiler)

    @classmethod
    def _inflate_full(cls, pandas_df, source_pid) -> DataFrame:
        """
        Re-creates the object from previously-serialized disk-storable representation.

        Parameters
        ----------
        pandas_df : pandas.DataFrame
            Data to use for object re-creation.
        source_pid : int
            Determines whether a Modin or pandas object needs to be created.
            Modin objects are created only on the main process.

        Returns
        -------
        DataFrame
            New ``DataFrame`` based on the `pandas_df`.
        """
        if os.getpid() != source_pid:
            return pandas_df
        # The current logic does not involve creating Modin objects
        # and manipulation with them in worker processes
        return cls(data=from_pandas(pandas_df))

    def __reduce__(self):
        self._query_compiler.finalize()
        pid = os.getpid()
        if (
            PersistentPickle.get()
            or not self._query_compiler.support_materialization_in_worker_process()
        ):
            return self._inflate_full, (self._to_pandas(), pid)
        return self._inflate_light, (self._query_compiler, pid)

    # Persistance support methods - END
