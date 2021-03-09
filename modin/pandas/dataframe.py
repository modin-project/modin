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
Implement DataFrame public API as Pandas does.

Almost all docstrings for public and magic methods should be inherited from Pandas
for better maintability. So some codes are ignored in pydocstyle check:
    - D101: missing docstring in class
    - D102: missing docstring in public method
    - D105: missing docstring in magic method
Manually add documentation for methods which are not presented in pandas.
"""

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
from pandas._typing import Label, StorageOptions

import itertools
import functools
import numpy as np
import sys
from typing import IO, Optional, Sequence, Tuple, Union, Mapping, Iterator
import warnings

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


@_inherit_docstrings(pandas.DataFrame, excluded=[pandas.DataFrame.__init__])
class DataFrame(BasePandasDataset):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        query_compiler=None,
    ):
        """
        Distributed DataFrame object backed by Pandas dataframes.

        Parameters
        ----------
        data: NumPy ndarray (structured or homogeneous) or dict:
            Dict can contain Series, arrays, constants, or list-like
            objects.
        index: pandas.Index, list
            The row index for this DataFrame.
        columns: pandas.Index
            The column names for this DataFrame, in pandas Index object.
        dtype: Data type to force.
            Only a single dtype is allowed. If None, infer
        copy: bool
            Copy data from inputs. Only affects DataFrame / 2d ndarray input.
        query_compiler: query_compiler
            A query compiler object to manage distributed computation.
        """
        Engine.subscribe(_update_engine)
        if isinstance(data, (DataFrame, Series)):
            self._query_compiler = data._query_compiler.copy()
            if index is not None and any(i not in data.index for i in index):
                raise NotImplementedError(
                    "Passing non-existant columns or index values to constructor not"
                    " yet implemented."
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
                        " yet implemented."
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
        num_rows = pandas.get_option("max_rows") or 60
        num_cols = pandas.get_option("max_columns") or 20

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
        Get the columns for this DataFrame.

        Returns
        -------
        The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_columns(self, new_columns):
        """
        Set the columns for this DataFrame.

        Parameters
        ----------
        new_columns: The new index to set this
        """
        self._query_compiler.columns = new_columns

    columns = property(_get_columns, _set_columns)

    @property
    def ndim(self):
        # DataFrames have an invariant that requires they be 2 dimensions.
        return 2

    def drop_duplicates(
        self, subset=None, keep="first", inplace=False, ignore_index=False
    ):
        return super(DataFrame, self).drop_duplicates(
            subset=subset, keep=keep, inplace=inplace
        )

    @property
    def dtypes(self):
        return self._query_compiler.dtypes

    def duplicated(self, subset=None, keep="first"):
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
    def empty(self):
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def axes(self):
        return [self.index, self.columns]

    @property
    def shape(self):
        return len(self.index), len(self.columns)

    def add_prefix(self, prefix):
        return DataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):
        return DataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(self, func, na_action: Optional[str] = None):
        if not callable(func):
            raise ValueError("'{0}' object is not callable".format(type(func)))
        ErrorMessage.non_verified_udf()
        return DataFrame(query_compiler=self._query_compiler.applymap(func))

    def apply(self, func, axis=0, raw=False, result_type=None, args=(), **kwds):
        axis = self._get_axis_number(axis)
        query_compiler = super(DataFrame, self).apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
        )
        if not isinstance(query_compiler, type(self._query_compiler)):
            return query_compiler
        # This is the simplest way to determine the return type, but there are checks
        # in pandas that verify that some results are created. This is a challenge for
        # empty DataFrames, but fortunately they only happen when the `func` type is
        # a list or a dictionary, which means that the return type won't change from
        # type(self), so we catch that error and use `type(self).__name__` for the return
        # type.
        try:
            if axis == 0:
                init_kwargs = {"index": self.index}
            else:
                init_kwargs = {"columns": self.columns}
            return_type = type(
                getattr(pandas, type(self).__name__)(**init_kwargs).apply(
                    func, axis=axis, raw=raw, result_type=result_type, args=args, **kwds
                )
            ).__name__
        except Exception:
            return_type = type(self).__name__
        if return_type not in ["DataFrame", "Series"]:
            return query_compiler.to_pandas().squeeze()
        else:
            result = getattr(sys.modules[self.__module__], return_type)(
                query_compiler=query_compiler
            )
            if isinstance(result, Series):
                if axis == 0 and result.name == self.index[0] or result.name == 0:
                    result.name = None
                elif axis == 1 and result.name == self.columns[0] or result.name == 0:
                    result.name = None
            return result

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
            if self._query_compiler.has_multiindex(
                axis=axis
            ) and by in self._query_compiler.get_index_names(axis):
                # In this case we pass the string value of the name through to the
                # partitions. This is more efficient than broadcasting the values.
                pass
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

    def keys(self):
        return self.columns

    def transpose(self, copy=False, *args):
        return DataFrame(query_compiler=self._query_compiler.transpose(*args))

    T = property(transpose)

    def add(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "add",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if sort is False:
            warnings.warn(
                "Due to https://github.com/pandas-dev/pandas/issues/35092, "
                "Pandas ignores sort=False; Modin correctly does not sort."
            )
        if isinstance(other, (Series, dict)):
            if isinstance(other, dict):
                other = Series(other)
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True"
                    " or if the Series has a name"
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

    def assign(self, **kwargs):
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
    ):
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

    def combine(self, other, func, fill_value=None, overwrite=True):
        return super(DataFrame, self).combine(
            other, func, fill_value=fill_value, overwrite=overwrite
        )

    def compare(
        self,
        other: "DataFrame",
        align_axis: Union[str, int] = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
    ) -> "DataFrame":
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

    def corr(self, method="pearson", min_periods=1):
        return self.__constructor__(
            query_compiler=self._query_compiler.corr(
                method=method,
                min_periods=min_periods,
            )
        )

    def corrwith(self, other, axis=0, drop=False, method="pearson"):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.corrwith, other, axis=axis, drop=drop, method=method
        )

    def cov(self, min_periods=None, ddof: Optional[int] = 1):
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

    def dot(self, other):
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

    def eq(self, other, axis="columns", level=None):
        return self._binary_op(
            "eq", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def equals(self, other):
        if isinstance(other, pandas.DataFrame):
            # Copy into a Modin DataFrame to simplify logic below
            other = DataFrame(other)
        return (
            self.index.equals(other.index)
            and self.columns.equals(other.columns)
            and self.eq(other).all().all()
        )

    def explode(self, column: Union[str, Tuple], ignore_index: bool = False):
        return self._default_to_pandas(
            pandas.DataFrame.explode, column, ignore_index=ignore_index
        )

    def eval(self, expr, inplace=False, **kwargs):
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
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

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
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
    ):  # pragma: no cover
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
    ):  # pragma: no cover
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

    def ge(self, other, axis="columns", level=None):
        return self._binary_op(
            "ge", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def gt(self, other, axis="columns", level=None):
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
    ):  # pragma: no cover
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
    ):
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

    def insert(self, loc, column, value, allow_duplicates=False):
        if isinstance(value, (DataFrame, pandas.DataFrame)):
            if len(value.columns) != 1:
                raise ValueError("Wrong number of items passed 2, placement implies 1")
            value = value.squeeze(axis=1)

        if not self._query_compiler.lazy_execution and len(self.index) == 0:
            if not hasattr(value, "index"):
                try:
                    value = pandas.Series(value)
                except (TypeError, ValueError, IndexError):
                    raise ValueError(
                        "Cannot insert into a DataFrame with no defined index "
                        "and a value that cannot be converted to a "
                        "Series"
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
                raise ValueError("Length of values does not match length of index")
            if not allow_duplicates and column in self.columns:
                raise ValueError("cannot insert {0}, already exists".format(column))
            if loc > len(self.columns):
                raise IndexError(
                    "index {0} is out of bounds for axis 0 with size {1}".format(
                        loc, len(self.columns)
                    )
                )
            if loc < 0:
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
    ):
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

    def iterrows(self):
        def iterrow_builder(s):
            return s.name, s

        partition_iterator = PartitionIterator(self, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self):
        def items_builder(s):
            return s.name, s

        partition_iterator = PartitionIterator(self, 1, items_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):
        return self.items()

    def itertuples(self, index=True, name="Pandas"):
        def itertuples_builder(s):
            return next(s._to_pandas().to_frame().T.itertuples(index=index, name=name))

        partition_iterator = PartitionIterator(self, 0, itertuples_builder)
        for v in partition_iterator:
            yield v

    def join(self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):
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
            # This constraint carried over from Pandas.
            if on is not None:
                raise ValueError(
                    "Joining multiple DataFrames only supported for joining on index"
                )
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

    def le(self, other, axis="columns", level=None):
        return self._binary_op(
            "le", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def lookup(self, row_labels, col_labels):
        return self._default_to_pandas(pandas.DataFrame.lookup, row_labels, col_labels)

    def lt(self, other, axis="columns", level=None):
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
    ):
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

    def memory_usage(self, index=True, deep=False):
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
    ):
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

    def mod(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "mod",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def mul(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "mul",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    rmul = multiply = mul

    def ne(self, other, axis="columns", level=None):
        return self._binary_op(
            "ne", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def nlargest(self, n, columns, keep="first"):
        return DataFrame(query_compiler=self._query_compiler.nlargest(n, columns, keep))

    def nsmallest(self, n, columns, keep="first"):
        return DataFrame(
            query_compiler=self._query_compiler.nsmallest(
                n=n, columns=columns, keep=keep
            )
        )

    def slice_shift(self, periods=1, axis=0):
        if periods == 0:
            return self.copy()

        if axis == "index" or axis == 0:
            if abs(periods) >= len(self.index):
                return DataFrame(columns=self.columns)
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
            if abs(periods) >= len(self.columns):
                return DataFrame(index=self.index)
            else:
                if periods > 0:
                    new_columns = self.columns.drop(labels=self.columns[:periods])
                    new_df = self.drop(self.columns[-periods:], axis="columns")
                else:
                    new_columns = self.columns.drop(labels=self.columns[periods:])
                    new_df = self.drop(self.columns[:-periods], axis="columns")

                new_df.columns = new_columns
                return new_df

    def unstack(self, level=-1, fill_value=None):
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

    def pivot(self, index=None, columns=None, values=None):
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
    ):
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
    ):
        return self._to_pandas().plot

    def pow(self, other, axis="columns", level=None, fill_value=None):
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

    def query(self, expr, inplace=False, **kwargs):
        ErrorMessage.non_verified_udf()
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
    ):
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
    ):
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

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rfloordiv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rmod(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rmod",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rpow(self, other, axis="columns", level=None, fill_value=None):
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

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rsub",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rtruediv",
            other,
            axis=axis,
            level=level,
            fill_value=fill_value,
            broadcast=isinstance(other, Series),
        )

    rdiv = rtruediv

    def select_dtypes(self, include=None, exclude=None):
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
    ):
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]

        if any(
            isinstance(col, (pandas.Index, Series, np.ndarray, list, Iterator))
            for col in keys
        ):
            # The current implementation cannot mix a list column labels and list like
            # objects.
            if not all(
                isinstance(col, (pandas.Index, Series, np.ndarray, list, Iterator))
                for col in keys
            ):
                return self._default_to_pandas(
                    "set_index",
                    keys,
                    drop=drop,
                    append=append,
                    inplace=inplace,
                    verify_integrity=verify_integrity,
                )
            if inplace:
                frame = self
            else:
                frame = self.copy()
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

    def squeeze(self, axis=None):
        axis = self._get_axis_number(axis) if axis is not None else None
        if axis is None and (len(self.columns) == 1 or len(self.index) == 1):
            return Series(query_compiler=self._query_compiler).squeeze()
        if axis == 1 and len(self.columns) == 1:
            return Series(query_compiler=self._query_compiler)
        if axis == 0 and len(self.index) == 1:
            return Series(query_compiler=self.T._query_compiler)
        else:
            return self.copy()

    def stack(self, level=-1, dropna=True):
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

    def sub(self, other, axis="columns", level=None, fill_value=None):
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
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        axis = self._get_axis_number(axis)
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

    def to_feather(self, path, **kwargs):  # pragma: no cover
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
    ):  # pragma: no cover
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
    ):
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
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_parquet,
            path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            storage_options=storage_options,
            **kwargs,
        )

    def to_period(self, freq=None, axis=0, copy=True):  # pragma: no cover
        return super(DataFrame, self).to_period(freq=freq, axis=axis, copy=copy)

    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
        return self._default_to_pandas(
            pandas.DataFrame.to_records,
            index=index,
            column_dtypes=column_dtypes,
            index_dtypes=index_dtypes,
        )

    def to_stata(
        self,
        path,
        convert_dates=None,
        write_index=True,
        byteorder=None,
        time_stamp=None,
        data_label=None,
        variable_labels=None,
        version=114,
        convert_strl=None,
        compression: Union[str, Mapping[str, str], None] = "infer",
        storage_options: StorageOptions = None,
    ):  # pragma: no cover
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
        )

    def to_timestamp(self, freq=None, how="start", axis=0, copy=True):
        return super(DataFrame, self).to_timestamp(
            freq=freq, how=how, axis=axis, copy=copy
        )

    def truediv(self, other, axis="columns", level=None, fill_value=None):
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
    ):
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

    def value_counts(
        self,
        subset: Optional[Sequence[Label]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
    ):
        return self._default_to_pandas(
            "value_counts",
            subset=subset,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
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

    def xs(self, key, axis=0, level=None, drop_level=True):
        return self._default_to_pandas(
            pandas.DataFrame.xs, key, axis=axis, level=level, drop_level=drop_level
        )

    def _getitem_column(self, key):
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
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key not in _ATTRS_NO_LOOKUP and key in self.columns:
                return self[key]
            raise e

    def __setattr__(self, key, value):
        # We have to check for this first because we have to be able to set
        # _query_compiler before we check if the key is in self
        if key in ["_query_compiler"] or key in self.__dict__:
            pass
        elif key in self and key not in dir(self):
            self.__setitem__(key, value)
        elif isinstance(value, pandas.Series):
            warnings.warn(
                "Modin doesn't allow columns to be created via a new attribute name - see "
                "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access",
                UserWarning,
            )
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return self._setitem_slice(key, value)

        if hashable(key) and key not in self.columns:
            if isinstance(value, Series) and len(self.columns) == 0:
                self._query_compiler = value._query_compiler.copy()
                # Now that the data is appended, we need to update the column name for
                # that column to `key`, otherwise the name could be incorrect. Drop the
                # last column name from the list (the appended value's name and append
                # the new name.
                self.columns = self.columns[:-1].append(pandas.Index([key]))
                return
            elif (
                isinstance(value, (pandas.DataFrame, DataFrame)) and value.shape[1] != 1
            ):
                raise ValueError(
                    "Wrong number of items passed %i, placement implies 1"
                    % value.shape[1]
                )
            elif isinstance(value, np.ndarray) and len(value.shape) > 1:
                if value.shape[1] == 1:
                    # Transform into columnar table and take first column
                    value = value.copy().T[0]
                else:
                    raise ValueError(
                        "Wrong number of items passed %i, placement implies 1"
                        % value.shape[1]
                    )

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

            def setitem_unhashable_key(df):
                # Arrow makes memory-mapped objects immutable, so copy will allow them
                # to be mutable again.
                df = df.copy(True)
                df[key] = value
                return df

            return self._update_inplace(
                self._default_to_pandas(setitem_unhashable_key)._query_compiler
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
            if not isinstance(value, Series):
                value = list(value)

        if not self._query_compiler.lazy_execution and len(self.index) == 0:
            new_self = DataFrame({key: value}, columns=self.columns)
            self._update_inplace(new_self._query_compiler)
        else:
            if isinstance(value, Series):
                value = value._query_compiler
            self._update_inplace(self._query_compiler.setitem(0, key, value))

    def __hash__(self):
        return self._default_to_pandas(pandas.DataFrame.__hash__)

    def __iter__(self):
        return iter(self.columns)

    def __contains__(self, key):
        return self.columns.__contains__(key)

    def __round__(self, decimals=0):
        return self._default_to_pandas(pandas.DataFrame.__round__, decimals=decimals)

    def __delitem__(self, key):
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

    @property
    def attrs(self):
        def attrs(df):
            return df.attrs

        self._default_to_pandas(attrs)

    @property
    def __doc__(self):  # pragma: no cover
        def __doc__(df):
            """Define __name__ attr because properties do not have it."""
            return df.__doc__

        return self._default_to_pandas(__doc__)

    @property
    def style(self):
        def style(df):
            """Define __name__ attr because properties do not have it."""
            return df.style

        return self._default_to_pandas(style)

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
        dataframe
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
        Grabs only numeric columns from frame.

        Parameters
        ----------
        axis: int
            Axis to inspect on having numeric types only.
            If axis is not 0, returns the frame itself.

        Returns
        -------
        DataFrame with numeric data.
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
        Help to check that all the dtypes are the same.

        TODO: add description for parameters.

        Parameters
        ----------
        numeric_only: bool
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
        Raise TypeErrors for sum, prod, and mean where necessary.

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
        return self._query_compiler.to_pandas()

    def _validate_eval_query(self, expr, **kwargs):
        """
        Help to check the arguments to eval() and query().

        Parameters
        ----------
        expr: The expression to evaluate. This string cannot contain any
            Python statements, only Python expressions.
        **kwargs
        """
        if isinstance(expr, str) and expr == "":
            raise ValueError("expr cannot be an empty string")

        if isinstance(expr, str) and "@" in expr:
            ErrorMessage.not_implemented("Local variables not yet supported in eval.")

        if isinstance(expr, str) and "not" in expr:
            if "parser" in kwargs and kwargs["parser"] == "python":
                ErrorMessage.not_implemented(
                    "'Not' nodes are not implemented."
                )  # pragma: no cover

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
        return Series(query_compiler=query_compiler)

    def _set_axis_name(self, name, axis=0, inplace=False):
        """
        Alter the name or names of the axis.

        TODO: add types.

        Parameters
        ----------
        name:
            Name for the Index, or list of names for the MultiIndex
        axis:
            0 or 'index' for the index; 1 or 'columns' for the columns
        inplace:
            Whether to modify `self` directly or return a copy

        Returns
        -------
        Type of caller or None if inplace=True.
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

        Returns
        -------
        datetime
            Series: Series of datetime64 dtype
        """
        return self._reduce_dimension(
            query_compiler=self._query_compiler.to_datetime(**kwargs)
        )

    def _getitem(self, key):
        """
        Get the column specified by key for this DataFrame.

        Parameters
        ----------
        key: the column name.

        Returns
        -------
        A Pandas Series representing the value for the column.
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
        """
        return cls(query_compiler=query_compiler)

    @classmethod
    def _inflate_full(cls, pandas_df):
        """Re-creates the object from previously-serialized disk-storable representation."""
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
