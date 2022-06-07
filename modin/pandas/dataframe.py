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

from modin.logging import metaclass_resolver
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
class DataFrame(metaclass_resolver(BasePandasDataset)):
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
        query_compiler=None,
    ):
        # Siblings are other dataframes that share the same query compiler. We
        # use this list to update inplace when there is a shallow copy.
        self._siblings = []
        Engine.subscribe(_update_engine)
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
                        DataFrame(columns=columns)
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

        # Check type of data and use appropriate constructor
        elif query_compiler is None:
            distributed_frame = from_non_pandas(data, index, columns, dtype)
            if distributed_frame is not None:
                self._query_compiler = distributed_frame._query_compiler
                return

            warnings.warn(
                "Distributing {} object. This may take some time.".format(type(data))
            )
            if is_list_like(data) and not is_dict_like(data):
                old_dtype = getattr(data, "dtype", None)
                values = [
                    obj._to_pandas() if isinstance(obj, Series) else obj for obj in data
                ]
                if isinstance(data, np.ndarray):
                    data = np.array(values, dtype=old_dtype)
                else:
                    try:
                        data = type(data)(values, dtype=old_dtype)
                    except TypeError:
                        data = values
            elif is_dict_like(data) and not isinstance(
                data, (pandas.Series, Series, pandas.DataFrame, DataFrame)
            ):
                data = {
                    k: v._to_pandas() if isinstance(v, Series) else v
                    for k, v in data.items()
                }
            pandas_df = pandas.DataFrame(
                data=data, index=index, columns=columns, dtype=dtype, copy=copy
            )
            self._query_compiler = from_pandas(pandas_df)._query_compiler
        else:
            self._query_compiler = query_compiler

    def __repr__(self):
        """
        Return a string representation for a particular ``DataFrame``.

        Returns
        -------
        str
        """
        from pandas.io.formats import console

        num_rows = pandas.get_option("display.max_rows") or 10
        num_cols = pandas.get_option("display.max_columns") or 20
        if pandas.get_option("display.max_columns") is None and pandas.get_option(
            "display.expand_frame_repr"
        ):
            width, _ = console.get_console_size()
            width = min(width, len(self.columns))
            col_counter = 0
            i = 0
            while col_counter < width:
                col_counter += len(str(self.columns[i])) + 1
                i += 1

            num_cols = i
            i = len(self.columns) - 1
            col_counter = 0
            while col_counter < width:
                col_counter += len(str(self.columns[i])) + 1
                i -= 1

            num_cols += len(self.columns) - i
        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _repr_html_(self):  # pragma: no cover
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
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # We split so that we insert our correct dataframe dimensions.
            return result.split("<p>")[
                0
            ] + "<p>{0} rows x {1} columns</p>\n</div>".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _get_columns(self):
        """
        Get the columns for this ``DataFrame``.

        Returns
        -------
        pandas.Index
            The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_columns(self, new_columns):
        """
        Set the columns for this ``DataFrame``.

        Parameters
        ----------
        new_columns : list-like, Index
            The new index to set.
        """
        self._query_compiler.columns = new_columns

    columns = property(_get_columns, _set_columns)

    @property
    def ndim(self):  # noqa: RT01, D200
        """
        Return the number of dimensions of the underlying data, by definition 2.
        """
        return 2

    def drop_duplicates(
        self, subset=None, keep="first", inplace=False, ignore_index=False
    ):  # noqa: PR01, RT01, D200
        """
        Return ``DataFrame`` with duplicate rows removed.
        """
        return super(DataFrame, self).drop_duplicates(
            subset=subset, keep=keep, inplace=inplace, ignore_index=ignore_index
        )

    @property
    def dtypes(self):  # noqa: RT01, D200
        """
        Return the dtypes in the ``DataFrame``.
        """
        return self._query_compiler.dtypes

    def duplicated(self, subset=None, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return boolean ``Series`` denoting duplicate rows.
        """
        import hashlib

        df = self[subset] if subset is not None else self
        # if the number of columns we are checking for duplicates is larger than 1, we must
        # hash them to generate a single value that can be compared across rows.
        if len(df.columns) > 1:
            hashed = df.apply(
                lambda s: hashlib.new("md5", str(tuple(s)).encode()).hexdigest(), axis=1
            ).to_frame()
        else:
            hashed = df
        duplicates = hashed.apply(lambda s: s.duplicated(keep=keep)).squeeze(axis=1)
        # remove Series name which was assigned automatically by .apply
        duplicates.name = None
        return duplicates

    @property
    def empty(self):  # noqa: RT01, D200
        """
        Indicate whether ``DataFrame`` is empty.
        """
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def axes(self):  # noqa: RT01, D200
        """
        Return a list representing the axes of the ``DataFrame``.
        """
        return [self.index, self.columns]

    @property
    def shape(self):  # noqa: RT01, D200
        """
        Return a tuple representing the dimensionality of the ``DataFrame``.
        """
        return len(self.index), len(self.columns)

    def add_prefix(self, prefix):  # noqa: PR01, RT01, D200
        """
        Prefix labels with string `prefix`.
        """
        return DataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):  # noqa: PR01, RT01, D200
        """
        Suffix labels with string `suffix`.
        """
        return DataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(
        self, func, na_action: Optional[str] = None, **kwargs
    ):  # noqa: PR01, RT01, D200
        """
        Apply a function to a ``DataFrame`` elementwise.
        """
        if not callable(func):
            raise ValueError("'{0}' object is not callable".format(type(func)))
        return DataFrame(
            query_compiler=self._query_compiler.applymap(func, na_action, **kwargs)
        )

    def apply(
        self, func, axis=0, raw=False, result_type=None, args=(), **kwargs
    ):  # noqa: PR01, RT01, D200
        """
        Apply a function along an axis of the ``DataFrame``.
        """
        axis = self._get_axis_number(axis)
        query_compiler = super(DataFrame, self).apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs
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
            reduced_index = pandas.Index(["__reduced__"])
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

        axis = self._get_axis_number(axis)
        idx_name = None
        # Drop here indicates whether or not to drop the data column before doing the
        # groupby. The typical pandas behavior is to drop when the data came from this
        # dataframe. When a string, Series directly from this dataframe, or list of
        # strings is passed in, the data used for the groupby is dropped before the
        # groupby takes place.
        drop = False

        if (
            not isinstance(by, (pandas.Series, Series))
            and is_list_like(by)
            and len(by) == 1
        ):
            by = by[0]

        if callable(by):
            by = self.index.map(by)
        elif hashable(by) and not isinstance(by, pandas.Grouper):
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
        elif is_list_like(by):
            # fastpath for multi column groupby
            if axis == 0 and all(
                (
                    (hashable(o) and (o in self))
                    or isinstance(o, Series)
                    or (is_list_like(o) and len(o) == len(self.axes[axis]))
                )
                for o in by
            ):
                # We want to split 'by's into those that belongs to the self (internal_by)
                # and those that doesn't (external_by)
                internal_by, external_by = [], []

                for current_by in by:
                    if hashable(current_by):
                        internal_by.append(current_by)
                    elif isinstance(current_by, Series):
                        if current_by._parent is self:
                            internal_by.append(current_by.name)
                        else:
                            external_by.append(current_by._query_compiler)
                    else:
                        external_by.append(current_by)

                by = internal_by + external_by

                if len(external_by) == 0:
                    by = self[internal_by]._query_compiler

                drop = True
            else:
                mismatch = len(by) != len(self.axes[axis])
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
            squeeze,
            idx_name,
            observed=observed,
            drop=drop,
            dropna=dropna,
        )

    def keys(self):  # noqa: RT01, D200
        """
        Get columns of the ``DataFrame``.
        """
        return self.columns

    def transpose(self, copy=False, *args):  # noqa: PR01, RT01, D200
        """
        Transpose index and columns.
        """
        # FIXME: Judging by pandas docs `*args` serves only compatibility purpose
        # and does not affect the result, we shouldn't pass it to the query compiler.
        return DataFrame(query_compiler=self._query_compiler.transpose(*args))

    T = property(transpose)

    def add(
        self, other, axis="columns", level=None, fill_value=None
    ):  # noqa: PR01, RT01, D200
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

    def append(
        self, other, ignore_index=False, verify_integrity=False, sort=False
    ):  # noqa: PR01, RT01, D200
        """
        Append rows of `other` to the end of caller, returning a new object.
        """
        if sort is False:
            warnings.warn(
                "Due to https://github.com/pandas-dev/pandas/issues/35092, "
                + "Pandas ignores sort=False; Modin correctly does not sort."
            )
        if isinstance(other, (Series, dict)):
            if isinstance(other, dict):
                other = Series(other)
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True"
                    + " or if the Series has a name"
                )
            if other.name is not None:
                # other must have the same index name as self, otherwise
                # index name will be reset
                name = other.name
                # We must transpose here because a Series becomes a new row, and the
                # structure of the query compiler is currently columnar
                other = other._query_compiler.transpose()
                other.index = pandas.Index([name], name=self.index.name)
            else:
                # See note above about transpose
                other = other._query_compiler.transpose()
        elif isinstance(other, list):
            if not all(isinstance(o, BasePandasDataset) for o in other):
                other = DataFrame(pandas.DataFrame(other))._query_compiler
            else:
                other = [obj._query_compiler for obj in other]
        else:
            other = other._query_compiler

        # If ignore_index is False, by definition the Index will be correct.
        # We also do this first to ensure that we don't waste compute/memory.
        if verify_integrity and not ignore_index:
            appended_index = (
                self.index.append(other.index)
                if not isinstance(other, list)
                else self.index.append([o.index for o in other])
            )
            is_valid = next((False for idx in appended_index.duplicated() if idx), True)
            if not is_valid:
                raise ValueError(
                    "Indexes have overlapping values: {}".format(
                        appended_index[appended_index.duplicated()]
                    )
                )

        query_compiler = self._query_compiler.concat(
            0, other, ignore_index=ignore_index, sort=sort
        )
        return DataFrame(query_compiler=query_compiler)

    def assign(self, **kwargs):  # noqa: PR01, RT01, D200
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
    ):  # noqa: PR01, RT01, D200
        """
        Perform column-wise combine with another ``DataFrame``.
        """
        return super(DataFrame, self).combine(
            other, func, fill_value=fill_value, overwrite=overwrite
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
        other = self._validate_other(other, 0, compare_index=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.compare(
                other,
                align_axis=align_axis,
                keep_shape=keep_shape,
                keep_equal=keep_equal,
            )
        )

    def corr(self, method="pearson", min_periods=1):  # noqa: PR01, RT01, D200
        """
        Compute pairwise correlation of columns, excluding NA/null values.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.corr(
                method=method,
                min_periods=min_periods,
            )
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
            pandas.DataFrame.corrwith, other, axis=axis, drop=drop, method=method
        )

    def cov(self, min_periods=None, ddof: Optional[int] = 1):  # noqa: PR01, RT01, D200
        """
        Compute pairwise covariance of columns, excluding NA/null values.
        """
        numeric_df = self.drop(
            columns=[
                i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
            ]
        )

        is_notna = True

        if all(numeric_df.notna().all()):
            if min_periods is not None and min_periods > len(numeric_df):
                result = np.empty((numeric_df.shape[1], numeric_df.shape[1]))
                result.fill(np.nan)
                return numeric_df.__constructor__(result)
            else:
                cols = numeric_df.columns
                idx = cols.copy()
                numeric_df = numeric_df.astype(dtype="float64")
                denom = 1.0 / (len(numeric_df) - ddof)
                means = numeric_df.mean(axis=0)
                result = numeric_df - means
                result = result.T._query_compiler.conj().dot(result._query_compiler)
        else:
            result = numeric_df._query_compiler.cov(min_periods=min_periods)
            is_notna = False

        if is_notna:
            result = numeric_df.__constructor__(
                query_compiler=result, index=idx, columns=cols
            )
            result *= denom
        else:
            result = numeric_df.__constructor__(query_compiler=result)
        return result

    def dot(self, other):  # noqa: PR01, RT01, D200
        """
        Compute the matrix multiplication between the ``DataFrame`` and `other`.
        """
        if isinstance(other, BasePandasDataset):
            common = self.columns.union(other.index)
            if len(common) > len(self.columns) or len(common) > len(other.index):
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

    def eq(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Perform equality comparison of ``DataFrame`` and `other` (binary operator `eq`).
        """
        return self._binary_op(
            "eq", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def equals(self, other):  # noqa: PR01, RT01, D200
        """
        Test whether two objects contain the same elements.
        """
        if isinstance(other, pandas.DataFrame):
            # Copy into a Modin DataFrame to simplify logic below
            other = DataFrame(other)
        return (
            self.index.equals(other.index)
            and self.columns.equals(other.columns)
            and self.eq(other).all().all()
        )

    def _update_var_dicts_in_kwargs(self, expr, kwargs):
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
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        self._update_var_dicts_in_kwargs(expr, kwargs)
        new_query_compiler = self._query_compiler.eval(expr, **kwargs)
        return_type = type(
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .eval(expr, **kwargs)
        ).__name__
        if return_type == type(self).__name__:
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        else:
            if inplace:
                raise ValueError("Cannot operate inplace if there is no assignment")
            return getattr(sys.modules[self.__module__], return_type)(
                query_compiler=new_query_compiler
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
        return super(DataFrame, self)._fillna(
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
    ):  # noqa: PR01, RT01, D200
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
    ):  # pragma: no cover # noqa: PR01, RT01, D200
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
    ):  # pragma: no cover # noqa: PR01, RT01, D200
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

    def ge(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get greater than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._binary_op(
            "ge", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def gt(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get greater than comparison of ``DataFrame`` and `other`, element-wise (binary operator `ge`).
        """
        return self._binary_op(
            "gt", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

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

        def put_str(src, output_len=None, spaces=2):
            src = str(src)
            return src.ljust(output_len if output_len else len(src)) + " " * spaces

        def format_size(num):
            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < 1024.0:
                    return f"{num:3.1f} {x}"
                num /= 1024.0
            return f"{num:3.1f} PB"

        output = []

        type_line = str(type(self))
        index_line = self.index._summary()
        columns = self.columns
        columns_len = len(columns)
        dtypes = self.dtypes
        dtypes_line = f"dtypes: {', '.join(['{}({})'.format(dtype, count) for dtype, count in dtypes.value_counts().items()])}"

        if max_cols is None:
            max_cols = 100

        exceeds_info_cols = columns_len > max_cols

        if buf is None:
            buf = sys.stdout

        if null_counts is None:
            null_counts = not exceeds_info_cols

        if verbose is None:
            verbose = not exceeds_info_cols

        if null_counts and verbose:
            # We're gonna take items from `non_null_count` in a loop, which
            # works kinda slow with `Modin.Series`, that's why we call `_to_pandas()` here
            # that will be faster.
            non_null_count = self.count()._to_pandas()

        if memory_usage is None:
            memory_usage = True

        def get_header(spaces=2):
            output = []
            head_label = " # "
            column_label = "Column"
            null_label = "Non-Null Count"
            dtype_label = "Dtype"
            non_null_label = " non-null"
            delimiter = "-"

            lengths = {}
            lengths["head"] = max(len(head_label), len(pprint_thing(len(columns))))
            lengths["column"] = max(
                len(column_label), max(len(pprint_thing(col)) for col in columns)
            )
            lengths["dtype"] = len(dtype_label)
            dtype_spaces = (
                max(lengths["dtype"], max(len(pprint_thing(dtype)) for dtype in dtypes))
                - lengths["dtype"]
            )

            header = put_str(head_label, lengths["head"]) + put_str(
                column_label, lengths["column"]
            )
            if null_counts:
                lengths["null"] = max(
                    len(null_label),
                    max(len(pprint_thing(x)) for x in non_null_count)
                    + len(non_null_label),
                )
                header += put_str(null_label, lengths["null"])
            header += put_str(dtype_label, lengths["dtype"], spaces=dtype_spaces)

            output.append(header)

            delimiters = put_str(delimiter * lengths["head"]) + put_str(
                delimiter * lengths["column"]
            )
            if null_counts:
                delimiters += put_str(delimiter * lengths["null"])
            delimiters += put_str(delimiter * lengths["dtype"], spaces=dtype_spaces)
            output.append(delimiters)

            return output, lengths

        output.extend([type_line, index_line])

        def verbose_repr(output):
            columns_line = f"Data columns (total {len(columns)} columns):"
            header, lengths = get_header()
            output.extend([columns_line, *header])
            for i, col in enumerate(columns):
                i, col, dtype = map(pprint_thing, [i, col, dtypes[col]])

                to_append = put_str(" {}".format(i), lengths["head"]) + put_str(
                    col, lengths["column"]
                )
                if null_counts:
                    non_null = pprint_thing(non_null_count[col])
                    to_append += put_str(
                        "{} non-null".format(non_null), lengths["null"]
                    )
                to_append += put_str(dtype, lengths["dtype"], spaces=0)
                output.append(to_append)

        def non_verbose_repr(output):
            output.append(columns._summary(name="Columns"))

        if verbose:
            verbose_repr(output)
        else:
            non_verbose_repr(output)

        output.append(dtypes_line)

        if memory_usage:
            deep = memory_usage == "deep"
            mem_usage_bytes = self.memory_usage(index=True, deep=deep).sum()
            mem_line = f"memory usage: {format_size(mem_usage_bytes)}"

            output.append(mem_line)

        output.append("")
        buf.write("\n".join(output))

    def insert(self, loc, column, value, allow_duplicates=False):  # noqa: PR01, D200
        """
        Insert column into ``DataFrame`` at specified location.
        """
        if (
            isinstance(value, (DataFrame, pandas.DataFrame))
            or isinstance(value, np.ndarray)
            and len(value.shape) > 1
        ):
            if value.shape[1] != 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.shape}"
                )
            value = value.squeeze(axis=1)
        if not self._query_compiler.lazy_execution and len(self.index) == 0:
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
            new_query_compiler = DataFrame(
                value, index=new_index, columns=new_columns
            )._query_compiler
        elif len(self.columns) == 0 and loc == 0:
            new_query_compiler = DataFrame(
                data=value, columns=[column], index=self.index
            )._query_compiler
        else:
            if (
                is_list_like(value)
                and not isinstance(value, (pandas.Series, Series))
                and len(value) != len(self.index)
            ):
                raise ValueError(
                    "Length of values ({}) does not match length of index ({})".format(
                        len(value), len(self.index)
                    )
                )
            if not allow_duplicates and column in self.columns:
                raise ValueError(f"cannot insert {column}, already exists")
            if not -len(self.columns) <= loc <= len(self.columns):
                raise IndexError(
                    f"index {loc} is out of bounds for axis 0 with size {len(self.columns)}"
                )
            elif loc < 0:
                raise ValueError("unbounded slice")
            if isinstance(value, Series):
                value = value._query_compiler
            new_query_compiler = self._query_compiler.insert(loc, column, value)

        self._update_inplace(new_query_compiler=new_query_compiler)

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

        def iterrow_builder(s):
            """Return tuple of the given `s` parameter name and the parameter themself."""
            return s.name, s

        partition_iterator = PartitionIterator(self, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self):  # noqa: D200
        """
        Iterate over (column name, ``Series``) pairs.
        """

        def items_builder(s):
            """Return tuple of the given `s` parameter name and the parameter themself."""
            return s.name, s

        partition_iterator = PartitionIterator(self, 1, items_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):  # noqa: RT01, D200
        """
        Iterate over (column name, ``Series``) pairs.
        """
        return self.items()

    def itertuples(self, index=True, name="Pandas"):  # noqa: PR01, D200
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
        self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False
    ):  # noqa: PR01, RT01, D200
        """
        Join columns of another ``DataFrame``.
        """
        if isinstance(other, Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = DataFrame({other.name: other})
        if on is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.join(
                    other._query_compiler,
                    on=on,
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
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
        new_frame = DataFrame(
            query_compiler=self._query_compiler.concat(
                1, [obj._query_compiler for obj in other], join=how, sort=sort
            )
        )
        new_frame.columns = new_columns
        return new_frame

    def le(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get less than or equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `le`).
        """
        return self._binary_op(
            "le", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def lookup(self, row_labels, col_labels):  # noqa: PR01, RT01, D200
        """
        Label-based "fancy indexing" function for ``DataFrame``.
        """
        return self._default_to_pandas(pandas.DataFrame.lookup, row_labels, col_labels)

    def lt(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
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
    ):  # noqa: PR01, RT01, D200
        """
        Unpivot a ``DataFrame`` from wide to long format, optionally leaving identifiers set.
        """
        if id_vars is None:
            id_vars = []
        if not is_list_like(id_vars):
            id_vars = [id_vars]
        if value_vars is None:
            value_vars = self.columns.difference(id_vars)
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

    def memory_usage(self, index=True, deep=False):  # noqa: PR01, RT01, D200
        """
        Return the memory usage of each column in bytes.
        """
        if index:
            result = self._reduce_dimension(
                self._query_compiler.memory_usage(index=False, deep=deep)
            )
            index_value = self.index.memory_usage(deep=deep)
            return Series(index_value, index=["Index"]).append(result)
        return super(DataFrame, self).memory_usage(index=index, deep=deep)

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
            else:
                right = right.to_frame()
        if not isinstance(right, DataFrame):
            raise TypeError(
                f"Can only merge Series or DataFrame objects, a {type(right)} was passed"
            )

        if left_index and right_index:
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
    ):  # noqa: PR01, RT01, D200
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
    ):  # noqa: PR01, RT01, D200
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

    rmul = multiply = mul

    def ne(self, other, axis="columns", level=None):  # noqa: PR01, RT01, D200
        """
        Get not equal comparison of ``DataFrame`` and `other`, element-wise (binary operator `ne`).
        """
        return self._binary_op(
            "ne", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def nlargest(self, n, columns, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in descending order.
        """
        return DataFrame(query_compiler=self._query_compiler.nlargest(n, columns, keep))

    def nsmallest(self, n, columns, keep="first"):  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows ordered by `columns` in ascending order.
        """
        return DataFrame(
            query_compiler=self._query_compiler.nsmallest(
                n=n, columns=columns, keep=keep
            )
        )

    def slice_shift(self, periods=1, axis=0):  # noqa: PR01, RT01, D200
        """
        Equivalent to `shift` without copying data.
        """
        if periods == 0:
            return self.copy()

        if axis == "index" or axis == 0:
            if abs(periods) >= len(self.index):
                return DataFrame(columns=self.columns)
            else:
                new_df = self.iloc[:-periods] if periods > 0 else self.iloc[-periods:]
                new_df.index = (
                    self.index[periods:] if periods > 0 else self.index[:periods]
                )
                return new_df
        else:
            if abs(periods) >= len(self.columns):
                return DataFrame(index=self.index)
            else:
                new_df = (
                    self.iloc[:, :-periods] if periods > 0 else self.iloc[:, -periods:]
                )
                new_df.columns = (
                    self.columns[periods:] if periods > 0 else self.columns[:periods]
                )
                return new_df

    def unstack(self, level=-1, fill_value=None):  # noqa: PR01, RT01, D200
        """
        Pivot a level of the (necessarily hierarchical) index labels.
        """
        if not isinstance(self.index, pandas.MultiIndex) or (
            isinstance(self.index, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.index.nlevels
        ):
            return self._reduce_dimension(
                query_compiler=self._query_compiler.unstack(level, fill_value)
            )
        else:
            return DataFrame(
                query_compiler=self._query_compiler.unstack(level, fill_value)
            )

    def pivot(self, index=None, columns=None, values=None):  # noqa: PR01, RT01, D200
        """
        Return reshaped ``DataFrame`` organized by given index / column values.
        """
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
        observed=False,
        sort=True,
    ):  # noqa: PR01, RT01, D200
        """
        Create a spreadsheet-style pivot table as a ``DataFrame``.
        """
        result = DataFrame(
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
    ):  # noqa: PR01, RT01, D200
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
        axis = self._get_axis_number(axis)
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        if level is not None:
            if (
                not self._query_compiler.has_multiindex(axis=axis)
                and level > 0
                or level < -1
                and level != self.index.name
            ):
                raise ValueError("level > 0 or level < -1 only valid with MultiIndex")
            return self.groupby(level=level, axis=axis, sort=False).prod(
                numeric_only=numeric_only, min_count=min_count
            )

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

    def query(self, expr, inplace=False, **kwargs):  # noqa: PR01, RT01, D200
        """
        Query the columns of a ``DataFrame`` with a boolean expression.
        """
        self._update_var_dicts_in_kwargs(expr, kwargs)
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.query(expr, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

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
    ):  # noqa: PR01, RT01, D200
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
