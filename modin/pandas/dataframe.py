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

import pandas
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.common import (
    infer_dtype_from_object,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
)
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.util._validators import validate_bool_kwarg
from pandas.io.formats.printing import pprint_thing
from pandas._libs.lib import no_default
from pandas._typing import Label

import itertools
import functools
import numpy as np
import sys
import os
from typing import Optional, Sequence, Tuple, Union, Mapping
import warnings

from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings, to_pandas, hashable
from .utils import (
    from_pandas,
    from_non_pandas,
)
from .iterator import PartitionIterator
from .series import Series
from .base import BasePandasDataset, _ATTRS_NO_LOOKUP
from .groupby import DataFrameGroupBy


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame, pandas.DataFrame.__init__]
)
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
        """Distributed DataFrame object backed by Pandas dataframes.

        Args:
            data (NumPy ndarray (structured or homogeneous) or dict):
                Dict can contain Series, arrays, constants, or list-like
                objects.
            index (pandas.Index, list, ObjectID): The row index for this
                DataFrame.
            columns (pandas.Index): The column names for this DataFrame, in
                pandas Index object.
            dtype: Data type to force. Only a single dtype is allowed.
                If None, infer
            copy (boolean): Copy data from inputs.
                Only affects DataFrame / 2d ndarray input.
            query_compiler: A query compiler object to manage distributed computation.
        """
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
        """repr function for rendering in Jupyter Notebooks like Pandas
        Dataframes.

        Returns:
            The HTML representation of a Dataframe.
        """
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
        """Get the columns for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_columns(self, new_columns):
        """Set the columns for this DataFrame.

        Args:
            new_columns: The new index to set this
        """
        self._query_compiler.columns = new_columns

    columns = property(_get_columns, _set_columns)

    def _validate_eval_query(self, expr, **kwargs):
        """Helper function to check the arguments to eval() and query()

        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.
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

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this DataFrame.
        """
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
        """Get the dtypes for this DataFrame.

        Returns:
            The dtypes for this DataFrame.
        """
        return self._query_compiler.dtypes

    def duplicated(self, subset=None, keep="first"):
        """
        Return boolean Series denoting duplicate rows, optionally only
        considering certain columns.

        Args:
            subset : column label or sequence of labels, optional
                Only consider certain columns for identifying duplicates, by
                default use all of the columns
            keep : {'first', 'last', False}, default 'first'
                - ``first`` : Mark duplicates as ``True`` except for the
                  first occurrence.
                - ``last`` : Mark duplicates as ``True`` except for the
                  last occurrence.
                - False : Mark all duplicates as ``True``.

        Returns:
            Series
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
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns:
            True if the DataFrame is empty.
            False otherwise.
        """
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def axes(self):
        """Get the axes for the DataFrame.

        Returns:
            The axes for the DataFrame.
        """
        return [self.index, self.columns]

    @property
    def shape(self):
        """Get the size of each of the dimensions in the DataFrame.

        Returns:
            A tuple with the size of each dimension as they appear in axes().
        """
        return len(self.index), len(self.columns)

    def add_prefix(self, prefix):
        """Add a prefix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):
        """Add a suffix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(self, func):
        """Apply a function to a DataFrame elementwise.

        Args:
            func (callable): The function to apply.
        """
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
        """
        Apply a groupby to this DataFrame. See _groupby() remote task.

        Parameters
        ----------
            by: The value to groupby.
            axis: The axis to groupby.
            level: The level of the groupby.
            as_index: Whether or not to store result as index.
            sort: Whether or not to sort the result by the index.
            group_keys: Whether or not to group the keys.
            squeeze: Whether or not to squeeze.
            dropna : bool, default True
                If True, and if group keys contain NA values,
                NA values together with row/column will be dropped.
                If False, NA values will also be treated as the key in groups

        Returns
        -------
            A new DataFrame resulting from the groupby.
        """
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
        elif isinstance(by, str):
            drop = by in self.columns
            idx_name = by
            if (
                self._query_compiler.has_multiindex(axis=axis)
                and by in self.axes[axis].names
                or hasattr(self.axes[axis], "name")
                and self.axes[axis].name == by
            ):
                # In this case we pass the string value of the name through to the
                # partitions. This is more efficient than broadcasting the values.
                pass
            else:
                by = self.__getitem__(by)._query_compiler
        elif isinstance(by, Series):
            drop = by._parent is self
            idx_name = by.name
            by = by._query_compiler
        elif is_list_like(by):
            # fastpath for multi column groupby
            if (
                not isinstance(by, Series)
                and axis == 0
                and all(
                    (
                        (isinstance(o, str) and (o in self))
                        or (isinstance(o, Series) and (o._parent is self))
                    )
                    for o in by
                )
            ):
                # We can just revert Series back to names because the parent is
                # this dataframe:
                by = [o.name if isinstance(o, Series) else o for o in by]
                by = self.__getitem__(by)._query_compiler
                drop = True
            else:
                mismatch = len(by) != len(self.axes[axis])
                if mismatch and all(
                    isinstance(obj, str)
                    and (
                        obj in self
                        or (hasattr(self.index, "names") and obj in self.index.names)
                    )
                    for obj in by
                ):
                    # In the future, we will need to add logic to handle this, but for now
                    # we default to pandas in this case.
                    pass
                elif mismatch and any(
                    isinstance(obj, str) and obj not in self.columns for obj in by
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

    def _reduce_dimension(self, query_compiler):
        return Series(query_compiler=query_compiler)

    def keys(self):
        """Get the info axis for the DataFrame.

        Returns:
            A pandas Index for this DataFrame.
        """
        return self.columns

    def transpose(self, copy=False, *args):
        """Transpose columns and rows for the DataFrame.

        Returns:
            A new DataFrame transposed from this DataFrame.
        """
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
        """Append another DataFrame/list/Series to this one.

        Args:
            other: The object to append to this.
            ignore_index: Ignore the index on appending.
            verify_integrity: Verify the integrity of the index on completion.

        Returns:
            A new DataFrame containing the concatenated values.
        """
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
        """
        Compare to another DataFrame and show the differences.

        Parameters
        ----------
        other : DataFrame
            Object to compare with.

        align_axis : {0 or 'index', 1 or 'columns'}, default 1
            Determine which axis to align the comparison on.

            * 0, or 'index' : Resulting differences are stacked vertically
                with rows drawn alternately from self and other.
            * 1, or 'columns' : Resulting differences are aligned horizontally
                with columns drawn alternately from self and other.

        keep_shape : bool, default False
            If true, all rows and columns are kept.
            Otherwise, only the ones with different values are kept.

        keep_equal : bool, default False
            If true, the result keeps values that are equal.
            Otherwise, equal values are shown as NaNs.

        Returns
        -------
        DataFrame
            DataFrame that shows the differences stacked side by side.

            The resulting index will be a MultiIndex with 'self' and 'other'
            stacked alternately at the inner level.
        """
        return self._default_to_pandas(
            pandas.DataFrame.compare,
            other=other,
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
        )

    def corr(self, method="pearson", min_periods=1):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.

        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result. Currently only available for Pearson
            and Spearman correlation.

        Returns
        -------
        DataFrame
            Correlation matrix.

        Notes
        -----
        Correlation floating point precision may slightly differ from pandas.

        For now pearson method is available only. For other methods defaults to pandas.
        """
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
        """
        Compute pairwise covariance of columns, excluding NA/null values.

        Compute the pairwise covariance among the series of a DataFrame.
        The returned data frame is the `covariance matrix
        <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
        of the DataFrame.

        Both NA and null values are automatically excluded from the
        calculation. (See the note below about bias from missing values.)
        A threshold can be set for the minimum number of
        observations for each value created. Comparisons with observations
        below this threshold will be returned as ``NaN``.

        This method is generally used for the analysis of time series data to
        understand the relationship between different measures
        across time.

        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.
        ddof : int, default 1
            Delta degrees of freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements.

        Returns
        -------
        DataFrame
            The covariance matrix of the series of the DataFrame.

        Notes
        -----
        Covariance floating point precision may slightly differ from pandas.

        If DataFrame contains at least one NA/null value, then defaults to pandas.
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

    def dot(self, other):
        """
        Compute the matrix multiplication between the DataFrame and other.

        This method computes the matrix product between the DataFrame and the
        values of an other Series, DataFrame or a numpy array.

        It can also be called using ``self @ other`` in Python >= 3.5.

        Parameters
        ----------
        other : Series, DataFrame or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        Series or DataFrame
            If other is a Series, return the matrix product between self and
            other as a Series. If other is a DataFrame or a numpy.array, return
            the matrix product of self and other in a DataFrame of a np.array.

        See Also
        --------
        Series.dot: Similar method for Series.

        Notes
        -----
        The dimensions of DataFrame and other must be compatible in order to
        compute the matrix multiplication. In addition, the column names of
        DataFrame and the index of other must contain the same values, as they
        will be aligned prior to the multiplication.

        The dot method for Series computes the inner product, instead of the
        matrix product here.
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

    def eq(self, other, axis="columns", level=None):
        return self._binary_op(
            "eq", other, axis=axis, level=level, broadcast=isinstance(other, Series)
        )

    def equals(self, other):
        """
        Checks if other DataFrame is elementwise equal to the current one

        Returns:
            Boolean: True if equal, otherwise False
        """
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
        """Evaluate a Python expression as a string using various backends.
        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.

            parser: The parser to use to construct the syntax tree from the
                expression. The default of 'pandas' parses code slightly
                different than standard Python. Alternatively, you can parse
                an expression using the 'python' parser to retain strict
                Python semantics. See the enhancing performance documentation
                for more details.

            engine: The engine used to evaluate the expression.

            truediv: Whether to use true division, like in Python >= 3

            local_dict: A dictionary of local variables, taken from locals()
                by default.

            global_dict: A dictionary of global variables, taken from
                globals() by default.

            resolvers: A list of objects implementing the __getitem__ special
                method that you can use to inject an additional collection
                of namespaces to use for variable lookup. For example, this is
                used in the query() method to inject the index and columns
                variables that refer to their respective DataFrame instance
                attributes.

            level: The number of prior stack frames to traverse and add to
                the current scope. Most users will not need to change this
                parameter.

            target: This is the target object for assignment. It is used when
                there is variable assignment in the expression. If so, then
                target must support item assignment with string keys, and if a
                copy is being returned, it must also support .copy().

            inplace: If target is provided, and the expression mutates target,
                whether to modify target inplace. Otherwise, return a copy of
                target with the mutation.
        Returns:
            ndarray, numeric scalar, DataFrame, Series
        """
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
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None
    ):
        """
        Print a concise summary of a DataFrame, which includes the index
        dtype and column dtypes, non-null values and memory usage.

        Parameters
        ----------
            verbose (bool, optional): Whether to print the full summary. Defaults
                to true

            buf (writable buffer): Where to send output. Defaults to sys.stdout

            max_cols (int, optional): When to switch from verbose to truncated
                output. By defualt, this is 100.

            memory_usage (bool, str, optional): Specifies whether the total memory
                usage of the DataFrame elements (including index) should be displayed.
                True always show memory usage. False never shows memory usage. A value
                of 'deep' is equivalent to "True with deep introspection". Memory usage
                is shown in human-readable units (base-2 representation). Without deep
                introspection a memory estimation is made based in column dtype and
                number of rows assuming values consume the same memory amount for
                corresponding dtypes. With deep memory introspection, a real memory
                usage calculation is performed at the cost of computational resources.
                Defaults to True.

            null_counts (bool, optional): Whetehr to show the non-null counts. By
                default, this is shown only when the frame is smaller than 100 columns
                and 1690785 rows. A value of True always shows the counts and False
                never shows the counts.

        Returns
        -------
            Prints the summary of a DataFrame and returns None.
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

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert column into DataFrame at specified location.
        Args:
            loc (int): Insertion index. Must verify 0 <= loc <= len(columns).
            column (hashable object): Label of the inserted column.
            value (int, Series, or array-like): The values to insert.
            allow_duplicates (bool): Whether to allow duplicate column names.
        """
        if isinstance(value, (DataFrame, pandas.DataFrame)):
            if len(value.columns) != 1:
                raise ValueError("Wrong number of items passed 2, placement implies 1")
            value = value.iloc[:, 0]

        if isinstance(value, Series):
            # TODO: Remove broadcast of Series
            value = value._to_pandas()

        if not self._query_compiler.lazy_execution and len(self.index) == 0:
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
                and not isinstance(value, pandas.Series)
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
        """Iterate over DataFrame rows as (index, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the rows of the frame.
        """

        def iterrow_builder(s):
            return s.name, s

        partition_iterator = PartitionIterator(self, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the columns of the frame.
        """

        def items_builder(s):
            return s.name, s

        partition_iterator = PartitionIterator(self, 1, items_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Returns the same thing as .items()

        Returns:
            A generator that iterates over the columns of the frame.
        """
        return self.items()

    def itertuples(self, index=True, name="Pandas"):
        """Iterate over DataFrame rows as namedtuples.

        Args:
            index (boolean, default True): If True, return the index as the
                first element of the tuple.
            name (string, default "Pandas"): The name of the returned
            namedtuples or None to return regular tuples.
        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A tuple representing row data. See args for varying tuples.
        """

        def itertuples_builder(s):
            return next(s._to_pandas().to_frame().T.itertuples(index=index, name=name))

        partition_iterator = PartitionIterator(self, 0, itertuples_builder)
        for v in partition_iterator:
            yield v

    def join(self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):
        """
        Join two or more DataFrames, or a DataFrame with a collection.

        Parameters
        ----------
            other : DataFrame, Series, or list of DataFrame
                Index should be similar to one of the columns in this one.
                If a Series is passed, its name attribute must be set,
                and that will be used as the column name in the resulting joined DataFrame.
            on : str, list of str, or array-like, optional
                Column or index level name(s) in the caller to join on the index in other,
                otherwise joins index-on-index. If multiple values given,
                the other DataFrame must have a MultiIndex. Can pass an array as the join key
                if it is not already contained in the calling DataFrame.
            how : {'left', 'right', 'outer', 'inner'}, Default is 'left'
                How to handle the operation of the two objects.
                - left: use calling frame's index (or column if on is specified)
                - right: use other's index.
                - outer: form union of calling frame's index (or column if on is specified)
                with other's index, and sort it lexicographically.
                - inner: form intersection of calling frame's index (or column if on is specified)
                with other's index, preserving the order of the calling’s one.
            lsuffix : str, default ''
                Suffix to use from left frame's overlapping columns.
            rsuffix : str, default ''
                Suffix to use from right frame's overlapping columns.
            sort : boolean. Default is False
                Order result DataFrame lexicographically by the join key.
                If False, the order of the join key depends on the join type (how keyword).

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the caller and other.
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

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """
        Return the median of the values for the requested axis.

        Parameters
        ----------
            axis : {index (0), columns (1)}
                Axis for the function to be applied on.
            skipna : bool, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            The median of the values for the requested axis
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        if level is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.median(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return self._reduce_dimension(
            self._query_compiler.median(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
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
        """
        Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.

        Parameters
        ----------
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`.
        var_name : scalar
            Name to use for the 'variable' column.
        value_name : scalar, default 'value'
            Name to use for the 'value' column.
        col_level : int or str, optional
            If columns are a MultiIndex then use this level to melt.
        ignore_index : bool, default True
            If True, original index is ignored. If False, the original index is retained.
            Index labels will be repeated as necessary.

        Returns
        -------
        DataFrame
            Unpivoted DataFrame.
        """
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
        """Returns the memory usage of each column in bytes

        Args:
            index (bool): Whether to include the memory usage of the DataFrame's
                index in returned Series. Defaults to True
            deep (bool): If True, introspect the data deeply by interrogating
            objects dtypes for system-level memory consumption. Defaults to False

        Returns:
            A Series where the index are the column names and the values are
            the memory usage of each of the columns in bytes. If `index=true`,
            then the first value of the Series will be 'Index' with its memory usage.
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
    ):
        """
        Merge DataFrame or named Series objects with a database-style join.

        The join is done on columns or indexes. If joining columns on columns,
        the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or
        indexes on a column or columns, the index will be passed on.

        Parameters
        ----------
        right : DataFrame or named Series
            Object to merge with.
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            Type of merge to be performed.
            - left: use only keys from left frame,
              similar to a SQL left outer join; preserve key order.
            - right: use only keys from right frame,
              similar to a SQL right outer join; preserve key order.
            - outer: use union of keys from both frames,
              similar to a SQL full outer join; sort keys lexicographically.
            - inner: use intersection of keys from both frames,
              similar to a SQL inner join; preserve the order of the left keys.
        on : label or list
            Column or index level names to join on.
            These must be found in both DataFrames. If on is None and not merging on indexes
            then this defaults to the intersection of the columns in both DataFrames.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame.
            Can also be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s).
            If it is a MultiIndex, the number of keys in the other DataFrame
            (either the index or a number of columns) must match the number of levels.
        right_index : bool, default False
            Use the index from the right DataFrame as the join key. Same caveats as left_index.
        sort : bool, default False
            Sort the join keys lexicographically in the result DataFrame.
            If False, the order of the join keys depends on the join type (how keyword).
        suffixes : tuple of (str, str), default ('_x', '_y')
            Suffix to apply to overlapping column names in the left and right side, respectively.
            To raise an exception on overlapping columns use (False, False).
        copy : bool, default True
            If False, avoid copy if possible.
        indicator : bool or str, default False
            If True, adds a column to output DataFrame called "_merge" with information
            on the source of each row. If string, column with information on source of each row
            will be added to output DataFrame, and column will be named value of string.
            Information column is Categorical-type and takes on a value of "left_only"
            for observations whose merge key only appears in 'left' DataFrame,
            "right_only" for observations whose merge key only appears in 'right' DataFrame,
            and "both" if the observation’s merge key is found in both.
        validate : str, optional
            If specified, checks if merge is of specified type.
            - 'one_to_one' or '1:1': check if merge keys are unique in both left and right datasets.
            - 'one_to_many' or '1:m': check if merge keys are unique in left dataset.
            - 'many_to_one' or 'm:1': check if merge keys are unique in right dataset.
            - 'many_to_many' or 'm:m': allowed, but does not result in checks.

        Returns
        -------
        DataFrame
             A DataFrame of the two merged objects.
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
        """
        Return the first `n` rows ordered by `columns` in descending order.
        Return the first `n` rows with the largest values in `columns`, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.
        This method is equivalent to
        ``df.sort_values(columns, ascending=False).head(n)``, but more
        performant.
        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:
            - `first` : prioritize the first occurrence(s)
            - `last` : prioritize the last occurrence(s)
            - ``all`` : do not drop any duplicates, even it means
                        selecting more than `n` items.
            .. versionadded:: 0.24.0
        """
        return DataFrame(query_compiler=self._query_compiler.nlargest(n, columns, keep))

    def nsmallest(self, n, columns, keep="first"):
        """
        Return the first `n` rows ordered by `columns` in ascending order.
        Return the first `n` rows with the smallest values in `columns`, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.
        This method is equivalent to
        ``df.sort_values(columns, ascending=True).head(n)``, but more
        performant.
        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.
        keep : {'first', 'last', 'all'}, default 'first'
            Where there are duplicate values:
            - ``first`` : take the first occurrence.
            - ``last`` : take the last occurrence.
            - ``all`` : do not drop any duplicates, even it means
              selecting more than `n` items.
        Returns
        -------
        DataFrame
        """
        return DataFrame(
            query_compiler=self._query_compiler.nsmallest(
                n=n, columns=columns, keep=keep
            )
        )

    def slice_shift(self, periods=1, axis=0):
        """
        Equivalent to `shift` without copying data.
        The shifted data will not include the dropped periods and the
        shifted axis will be smaller than the original.
        Parameters
        ----------
        periods : int
            Number of periods to move, can be positive or negative.
        axis : int or str
            Shift direction.
        Returns
        -------
        shifted : same type as caller
        """
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
        """
        Pivot a level of the (necessarily hierarchical) index labels.
        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.
        If the index is not a MultiIndex, the output will be a Series
        (the analogue of stack when the columns are not a MultiIndex).
        The level involved will automatically get sorted.

        Parameters
        ----------
        level : int, str, or list of these, default -1 (last level)
            Level(s) of index to unstack, can pass level name.
        fill_value : int, str or dict
            Replace NaN with this value if the unstack produces missing values.

        Returns
        -------
        Series or DataFrame
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

    def pivot(self, index=None, columns=None, values=None):
        """
        Return reshaped DataFrame organized by given index / column values.
        Reshape data (produce a "pivot" table) based on column values. Uses
        unique values from specified `index` / `columns` to form axes of the
        resulting DataFrame.
        Parameters
        ----------
        index : str or object, optional
            Column to use to make new frame's index. If None, uses
            existing index.
        columns : str or object
            Column to use to make new frame's columns.
        values : str, object or a list of the previous, optional
            Column(s) to use for populating new frame's values. If not
            specified, all remaining columns will be used and the result will
            have hierarchically indexed columns.
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
        """
        Return the product of the values for the requested axis.

        Parameters
        ----------
            axis : {index (0), columns (1)}
                Axis for the function to be applied on.
            skipna : bool, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            min_count : int, default 0
                The required number of valid values to perform the operation.
                If fewer than min_count non-NA values are present the result will be NA.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            The product of the values for the requested axis.
        """
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

        data = self._validate_dtypes_sum_prod_mean(axis, numeric_only, ignore_axis=True)
        if level is not None:
            return data.__constructor__(
                query_compiler=data._query_compiler.prod_min_count(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
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
        """Queries the Dataframe with a boolean expression

        Returns:
            A new DataFrame if inplace=False
        """
        ErrorMessage.non_verified_udf()
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.query(expr, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

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
        """Alters axes labels.

        Args:
            mapper, index, columns: Transformations to apply to the axis's
                values.
            axis: Axis to target with mapper.
            copy: Also copy underlying data.
            inplace: Whether to return a new DataFrame.
            level: Only rename a specific level of a MultiIndex.

        Returns:
            If inplace is False, a new DataFrame with the updated axes.
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
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):
        """
        Replace values given in `to_replace` with `value`.

        Values of the DaraFrame are replaced with other values dynamically.
        This differs from updating with .loc or .iloc, which require
        you to specify a location to update with some value.

        Parameters
        ----------
        to_replace : str, regex, list, dict, Series, int, float, or None
            How to find the values that will be replaced.
        value : scalar, dict, list, str, regex, default None
            Value to replace any values matching `to_replace` with.
            For a DataFrame a dict of values can be used to specify which
            value to use for each column (columns not in the dict will not be
            filled). Regular expressions, strings and lists or dicts of such
            objects are also allowed.
        inplace : bool, default False
            If True, in place. Note: this will modify any
            other views on this object (e.g. a column from a DataFrame).
            Returns the caller if this is True.
        limit : int, default None
            Maximum size gap to forward or backward fill.
        regex : bool or same types as `to_replace`, default False
            Whether to interpret `to_replace` and/or `value` as regular
            expressions. If this is ``True`` then `to_replace` *must* be a
            string. Alternatively, this could be a regular expression or a
            list, dict, or array of regular expressions in which case
            `to_replace` must be ``None``.
        method : {{'pad', 'ffill', 'bfill', `None`}}
            The method to use when for replacement, when `to_replace` is a
            scalar, list or tuple and `value` is ``None``.

        Returns
        -------
        DataFrame
            Object after replacement.
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

    def _set_axis_name(self, name, axis=0, inplace=False):
        """Alter the name or names of the axis.

        Args:
            name: Name for the Index, or list of names for the MultiIndex
            axis: 0 or 'index' for the index; 1 or 'columns' for the columns
            inplace: Whether to modify `self` directly or return a copy

        Returns:
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

    def sem(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        """
        Return unbiased standard error of the mean over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
            axis : {index (0), columns (1)}
            skipna : bool, default True
                Exclude NA/null values. If an entire row/column is NA,
                the result will be NA.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            ddof : int, default 1
                Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
                where N represents the number of elements.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.

        Returns
        -------
            Series or DataFrame (if level specified)
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        if level is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.sem(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    ddof=ddof,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return self._reduce_dimension(
            self._query_compiler.sem(
                axis=axis,
                skipna=skipna,
                level=level,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ):
        """Set the DataFrame index using one or more existing columns.

        Args:
            keys: column label or list of column labels / arrays.
            drop (boolean): Delete columns to be used as the new index.
            append (boolean): Whether to append columns to existing index.
            inplace (boolean): Modify the DataFrame in place.
            verify_integrity (boolean): Check the new index for duplicates.
                Otherwise defer the check until necessary. Setting to False
                will improve the performance of this method

        Returns:
            If inplace is set to false returns a new DataFrame, otherwise None.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]
        if inplace:
            frame = self
        else:
            frame = self.copy()

        arrays = []
        names = []
        if append:
            names = [x for x in self.index.names]
            if self._query_compiler.has_multiindex():
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)
        to_remove = []
        for col in keys:
            if isinstance(col, pandas.MultiIndex):
                # append all but the last column so we don't have to modify
                # the end of this loop
                for n in range(col.nlevels - 1):
                    arrays.append(col._get_level_values(n))

                level = col._get_level_values(col.nlevels - 1)
                names.extend(col.names)
            elif isinstance(col, pandas.Series):
                level = col._values
                names.append(col.name)
            elif isinstance(col, pandas.Index):
                level = col
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray, pandas.Index)):
                level = col
                names.append(None)
            else:
                level = frame[col]._to_pandas()._values
                names.append(col)
                if drop:
                    to_remove.append(col)
            arrays.append(level)
        index = ensure_index_from_sequences(arrays, names)

        if verify_integrity and not index.is_unique:
            duplicates = index.get_duplicates()
            raise ValueError("Index has duplicate keys: %s" % duplicates)

        for c in to_remove:
            del frame[c]
        # clear up memory usage
        index._cleanup()
        frame.index = index

        if not inplace:
            return frame

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """
        Return unbiased skew over requested axis. Normalized by N-1

        Parameters
        ----------
            axis : {index (0), columns (1)}
                Axis for the function to be applied on.
            skipna : boolean, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical),
                count along a particular level, collapsing into a Series.
            numeric_only : boolean, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            Unbiased skew over requested axis.
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        if level is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.skew(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return self._reduce_dimension(
            self._query_compiler.skew(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    @property
    def sparse(self):
        return self._default_to_pandas(pandas.DataFrame.sparse)

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

    def std(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        """
        Return sample standard deviation over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument.

        Parameters
        ----------
            axis : {index (0), columns (1)}
                The axis to take the std on.
            skipna : bool, default True
                Exclude NA/null values. If an entire row/column is NA, the result will be NA.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            ddof : int, default 1
                Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
                where N represents the number of elements.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            The sample standard deviation.
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        if level is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.std(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    ddof=ddof,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return self._reduce_dimension(
            self._query_compiler.std(
                axis=axis,
                skipna=skipna,
                level=level,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def stack(self, level=-1, dropna=True):
        """
        Stack the prescribed level(s) from columns to index.
        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to the current
        DataFrame. The new inner-most levels are created by pivoting the
        columns of the current dataframe:
          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        Parameters
        ----------
        level : int, str, list, default -1
            Level(s) to stack from the column axis onto the index
            axis, defined as one index or label, or a list of indices
            or labels.
        dropna : bool, default True
            Whether to drop rows in the resulting Frame/Series with
            missing values. Stacking a column level onto the index
            axis can create combinations of index and column values
            that are missing from the original dataframe. See Examples
            section.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.
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
        """
        Return the sum of the values for the requested axis.

        Parameters
        ----------
            axis : {index (0), columns (1)}
                Axis for the function to be applied on.
            skipna : bool, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            min_count : int, default 0
                The required number of valid values to perform the operation.
                If fewer than min_count non-NA values are present the result will be NA.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            The sum of the values for the requested axis
        """
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
            return data.__constructor__(
                query_compiler=data._query_compiler.sum_min_count(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
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
        path,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        **kwargs,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_parquet,
            path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
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
        """
        Modify in place using non-NA values from another DataFrame.

        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or object coercible into a DataFrame
            Should have at least one matching index/column label
            with the original DataFrame. If a Series is passed,
            its name attribute must be set, and that will be
            used as the column name to align with the original DataFrame.
        join : {'left'}, default 'left'
            Only left join is implemented, keeping the index and columns of the
            original object.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys:

            * True: overwrite original DataFrame's values
              with values from `other`.
            * False: only update values that are NA in
              the original DataFrame.

        filter_func : callable(1d-array) -> bool 1d-array, optional
            Can choose to replace values other than NA. Return True for values
            that should be updated.
        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise a ValueError if the DataFrame and `other`
            both contain non-NA data in the same place.

        Returns
        -------
        None : method directly changes calling object

        Raises
        ------
        ValueError
            * When `errors='raise'` and there's overlapping non-NA data.
            * When `errors` is not either `'ignore'` or `'raise'`
        NotImplementedError
            * If `join != 'left'`
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

    def var(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        """
        Return unbiased variance over requested axis.

        Normalized by N-1 by default. This can be changed using the ddof argument

        Parameters
        ----------
            axis : {index (0), columns (1)}
                The axis to take the variance on.
            skipna : bool, default True
                Exclude NA/null values. If an entire row/column is NA, the result will be NA.
            level : int or level name, default None
                If the axis is a MultiIndex (hierarchical), count along a particular level,
                collapsing into a Series.
            ddof : int, default 1
                Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
                where N represents the number of elements.
            numeric_only : bool, default None
                Include only float, int, boolean columns. If None, will attempt to use everything,
                then use only numeric data. Not implemented for Series.
            **kwargs
                Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series or DataFrame (if level specified)
            The unbiased variance.
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)
        if level is not None:
            return self.__constructor__(
                query_compiler=self._query_compiler.var(
                    axis=axis,
                    skipna=skipna,
                    level=level,
                    ddof=ddof,
                    numeric_only=numeric_only,
                    **kwargs,
                )
            )
        return self._reduce_dimension(
            self._query_compiler.var(
                axis=axis,
                skipna=skipna,
                level=level,
                ddof=ddof,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def value_counts(
        self,
        subset: Optional[Sequence[Label]] = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
    ):
        """
         Return a Series containing counts of unique rows in the DataFrame.

        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.

        Returns
        -------
        Series
        """
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
        """Replaces values not meeting condition with values in other.

        Args:
            cond: A condition to be met, can be callable, array-like or a
                DataFrame.
            other: A value or DataFrame of values to use for setting this.
            inplace: Whether or not to operate inplace.
            axis: The axis to apply over. Only valid when a Series is passed
                as other.
            level: The MultiLevel index level to apply over.
            errors: Whether or not to raise errors. Does nothing in Pandas.
            try_cast: Try to cast the result back to the input type.

        Returns:
            A new DataFrame with the replaced values.
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

    def xs(self, key, axis=0, level=None, drop_level=True):
        return self._default_to_pandas(
            pandas.DataFrame.xs, key, axis=axis, level=level, drop_level=drop_level
        )

    def _getitem(self, key):
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
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
        """After regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
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
        if hashable(key) and key not in self.columns:
            # Handle new column case first
            if isinstance(value, Series):
                if len(self.columns) == 0:
                    self._query_compiler = value._query_compiler.copy()
                else:
                    self._create_or_update_from_compiler(
                        self._query_compiler.concat(1, value._query_compiler),
                        inplace=True,
                    )
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

        if not isinstance(key, str):

            if isinstance(key, DataFrame) or isinstance(key, np.ndarray):
                if isinstance(key, np.ndarray):
                    if key.shape != self.shape:
                        raise ValueError("Array must be same shape as DataFrame")
                    key = DataFrame(key, columns=self.columns)
                return self.mask(key, value, inplace=True)

            def setitem_without_string_columns(df):
                # Arrow makes memory-mapped objects immutable, so copy will allow them
                # to be mutable again.
                df = df.copy(True)
                df[key] = value
                return df

            return self._update_inplace(
                self._default_to_pandas(setitem_without_string_columns)._query_compiler
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
        """Iterate over the columns

        Returns:
            An Iterator over the columns of the DataFrame.
        """
        return iter(self.columns)

    def __contains__(self, key):
        """Searches columns for specific key

        Args:
            key : The column name

        Returns:
            Returns a boolean if the specified key exists as a column name
        """
        return self.columns.__contains__(key)

    def __round__(self, decimals=0):
        return self._default_to_pandas(pandas.DataFrame.__round__, decimals=decimals)

    def __setstate__(self, state):
        return self._default_to_pandas(pandas.DataFrame.__setstate__, state)

    def __delitem__(self, key):
        """Delete a column by key. `del a[key]` for example.
           Operation happens in place.

           Notes: This operation happen on row and column partition
                  simultaneously. No rebuild.
        Args:
            key: key to delete
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
    __div__ = div
    __rdiv__ = rdiv

    @property
    def attrs(self):
        def attrs(df):
            return df.attrs

        self._default_to_pandas(attrs)

    @property
    def __doc__(self):  # pragma: no cover
        def __doc__(df):
            """Defined because properties do not have a __name__"""
            return df.__doc__

        return self._default_to_pandas(__doc__)

    @property
    def style(self):
        def style(df):
            """Defined because properties do not have a __name__"""
            return df.style

        return self._default_to_pandas(style)

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _validate_dtypes(self, numeric_only=False):
        """Helper method to check that all the dtypes are the same"""
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
        # Pandas ignores `numeric_only` if `axis` is 1, but we do have to drop
        # non-numeric columns if `axis` is 0.
        if numeric_only and axis == 0:
            return self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
        else:
            return self

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        """Raises TypeErrors for sum, prod, and mean where necessary"""
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
        # Pandas ignores `numeric_only` if `axis` is 1, but we do have to drop
        # non-numeric columns if `axis` is 0.
        if numeric_only and axis == 0:
            return self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
        else:
            return self

    def _to_pandas(self):
        return self._query_compiler.to_pandas()


if os.environ.get("MODIN_EXPERIMENTAL", "").title() == "True":
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(DataFrame, "make_dataframe_wrapper")
