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
from pandas.core.common import is_bool_indexer
from pandas.core.indexing import check_bool_indexer
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
)
from pandas.core.base import DataError
from collections.abc import Iterable, Container
from typing import List, Hashable
import warnings


from modin.backends.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import try_cast_to_pandas, wrap_udf_function, hashable
from modin.data_management.functions import (
    FoldFunction,
    MapFunction,
    MapReduceFunction,
    ReductionFunction,
    BinaryFunction,
    GroupbyReduceFunction,
    groupby_reduce_functions,
)


def _get_axis(axis):
    if axis == 0:
        return lambda self: self._modin_frame.index
    else:
        return lambda self: self._modin_frame.columns


def _set_axis(axis):
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


def _str_map(func_name):
    def str_op_builder(df, *args, **kwargs):
        str_s = df.squeeze(axis=1).str
        return getattr(pandas.Series.str, func_name)(str_s, *args, **kwargs).to_frame()

    return str_op_builder


def _dt_prop_map(property_name):
    """
    Create a function that call property of property `dt` of the series.

    Parameters
    ----------
    property_name
        The property of `dt`, which will be applied.

    Returns
    -------
        A callable function to be applied in the partitions

    Notes
    -----
    This applies non-callable properties of `Series.dt`.
    """

    def dt_op_builder(df, *args, **kwargs):
        prop_val = getattr(df.squeeze(axis=1).dt, property_name)
        if isinstance(prop_val, pandas.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, pandas.DataFrame):
            return prop_val
        else:
            return pandas.DataFrame([prop_val])

    return dt_op_builder


def _dt_func_map(func_name):
    """
    Create a function that call method of property `dt` of the series.

    Parameters
    ----------
    func_name
        The method of `dt`, which will be applied.

    Returns
    -------
        A callable function to be applied in the partitions

    Notes
    -----
    This applies callable methods of `Series.dt`.
    """

    def dt_op_builder(df, *args, **kwargs):
        dt_s = df.squeeze(axis=1).dt
        return pandas.DataFrame(
            getattr(pandas.Series.dt, func_name)(dt_s, *args, **kwargs)
        )

    return dt_op_builder


def copy_df_for_func(func, display_name: str = None):
    """
    Create a function that copies the dataframe, likely because `func` is inplace.

    Parameters
    ----------
    func : callable
        The function, usually updates a dataframe inplace.
    display_name : str, optional
        The function's name, which is displayed by progress bar.

    Returns
    -------
    callable
        A callable function to be applied in the partitions
    """

    def caller(df, *args, **kwargs):
        df = df.copy()
        func(df, *args, **kwargs)
        return df

    if display_name is not None:
        caller.__name__ = display_name
    return caller


class PandasQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
    with a Pandas backend. This logic is specific to Pandas."""

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """
        Default to pandas behavior.

        Parameters
        ----------
        pandas_op : callable
            The operation to apply, must be compatible pandas DataFrame call
        args
            The arguments for the `pandas_op`
        kwargs
            The keyword arguments for the `pandas_op`

        Returns
        -------
        PandasQueryCompiler
            The result of the `pandas_op`, converted back to PandasQueryCompiler

        Notes
        -----
        This operation takes a distributed object and converts it directly to pandas.
        """
        op_name = getattr(pandas_op, "__name__", str(pandas_op))
        ErrorMessage.default_to_pandas(op_name)
        args = (a.to_pandas() if isinstance(a, type(self)) else a for a in args)
        kwargs = {
            k: v.to_pandas if isinstance(v, type(self)) else v
            for k, v in kwargs.items()
        }

        result = pandas_op(self.to_pandas(), *args, **kwargs)
        if isinstance(result, pandas.Series):
            if result.name is None:
                result.name = "__reduced__"
            result = result.to_frame()
        if isinstance(result, pandas.DataFrame):
            return self.from_pandas(result, type(self._modin_frame))
        else:
            return result

    def finalize(self):
        self._modin_frame.finalize()

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._modin_frame.dtypes

    # END Index, columns, and dtypes objects

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._modin_frame.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._modin_frame.add_suffix(suffix, axis))

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(self._modin_frame.copy())

    # END Copy

    # Append/Concat/Join (Not Merge)
    # The append/concat/join operations should ideally never trigger remote
    # compute. These operations should only ever be manipulations of the
    # metadata of the resulting object. It should just be a simple matter of
    # appending the other object's blocks and adding np.nan columns for the new
    # columns, if needed. If new columns are added, some compute may be
    # required, though it can be delayed.
    #
    # Currently this computation is not delayed, and it may make a copy of the
    # DataFrame in memory. This can be problematic and should be fixed in the
    # future. TODO (devin-petersohn): Delay reindexing

    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(o, type(self)) for o in other
        ), "Different Manager objects are being used. This is not allowed"
        sort = kwargs.get("sort", None)
        if sort is None:
            sort = False
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        other_modin_frame = [o._modin_frame for o in other]
        new_modin_frame = self._modin_frame._concat(axis, other_modin_frame, join, sort)
        result = self.__constructor__(new_modin_frame)
        if ignore_index:
            if axis == 0:
                return result.reset_index(drop=True)
            else:
                result.columns = pandas.RangeIndex(len(result.columns))
                return result
        return result

    # END Append/Concat/Join

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object."""
        # TODO create a way to clean up this object.
        return

    # END Data Management Methods

    # To NumPy
    def to_numpy(self, **kwargs):
        """
        Converts Modin DataFrame to NumPy array.

        Returns
        -------
            NumPy array of the QueryCompiler.
        """
        arr = self._modin_frame.to_numpy(**kwargs)
        ErrorMessage.catch_bugs_and_request_email(
            len(arr) != len(self.index) or len(arr[0]) != len(self.columns)
        )
        return arr

    # END To NumPy

    # Binary operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    add = BinaryFunction.register(pandas.DataFrame.add)
    combine = BinaryFunction.register(pandas.DataFrame.combine)
    combine_first = BinaryFunction.register(pandas.DataFrame.combine_first)
    eq = BinaryFunction.register(pandas.DataFrame.eq)
    floordiv = BinaryFunction.register(pandas.DataFrame.floordiv)
    ge = BinaryFunction.register(pandas.DataFrame.ge)
    gt = BinaryFunction.register(pandas.DataFrame.gt)
    le = BinaryFunction.register(pandas.DataFrame.le)
    lt = BinaryFunction.register(pandas.DataFrame.lt)
    mod = BinaryFunction.register(pandas.DataFrame.mod)
    mul = BinaryFunction.register(pandas.DataFrame.mul)
    ne = BinaryFunction.register(pandas.DataFrame.ne)
    pow = BinaryFunction.register(pandas.DataFrame.pow)
    rfloordiv = BinaryFunction.register(pandas.DataFrame.rfloordiv)
    rmod = BinaryFunction.register(pandas.DataFrame.rmod)
    rpow = BinaryFunction.register(pandas.DataFrame.rpow)
    rsub = BinaryFunction.register(pandas.DataFrame.rsub)
    rtruediv = BinaryFunction.register(pandas.DataFrame.rtruediv)
    sub = BinaryFunction.register(pandas.DataFrame.sub)
    truediv = BinaryFunction.register(pandas.DataFrame.truediv)
    __and__ = BinaryFunction.register(pandas.DataFrame.__and__)
    __or__ = BinaryFunction.register(pandas.DataFrame.__or__)
    __rand__ = BinaryFunction.register(pandas.DataFrame.__rand__)
    __ror__ = BinaryFunction.register(pandas.DataFrame.__ror__)
    __rxor__ = BinaryFunction.register(pandas.DataFrame.__rxor__)
    __xor__ = BinaryFunction.register(pandas.DataFrame.__xor__)
    df_update = BinaryFunction.register(
        copy_df_for_func(pandas.DataFrame.update, display_name="update"),
        join_type="left",
    )
    series_update = BinaryFunction.register(
        copy_df_for_func(
            lambda x, y: pandas.Series.update(x.squeeze(axis=1), y.squeeze(axis=1)),
            display_name="update",
        ),
        join_type="left",
    )

    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """

        assert isinstance(
            cond, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"
        if isinstance(other, type(self)):
            # Note: Currently we are doing this with two maps across the entire
            # data. This can be done with a single map, but it will take a
            # modification in the `BlockPartition` class.
            # If this were in one pass it would be ~2x faster.
            # TODO (devin-petersohn) rewrite this to take one pass.
            def where_builder_first_pass(cond, other, **kwargs):
                return cond.where(cond, other, **kwargs)

            first_pass = cond._modin_frame._binary_op(
                where_builder_first_pass, other._modin_frame, join_type="left"
            )

            def where_builder_second_pass(df, new_other, **kwargs):
                return df.where(new_other.eq(True), new_other, **kwargs)

            new_modin_frame = self._modin_frame._binary_op(
                where_builder_second_pass, first_pass, join_type="left"
            )
        # This will be a Series of scalars to be applied based on the condition
        # dataframe.
        else:

            def where_builder_series(df, cond):
                return df.where(cond, other, **kwargs)

            new_modin_frame = self._modin_frame._binary_op(
                where_builder_series, cond._modin_frame, join_type="left"
            )
        return self.__constructor__(new_modin_frame)

    def merge(self, right, **kwargs):
        """
        Merge DataFrame or named Series objects with a database-style join.

        Parameters
        ----------
        right : PandasQueryCompiler
            The query compiler of the right DataFrame to merge with.

        Returns
        -------
        PandasQueryCompiler
            A new query compiler that contains result of the merge.

        Notes
        -----
        See pd.merge or pd.DataFrame.merge for more info on kwargs.
        """
        how = kwargs.get("how", "inner")
        on = kwargs.get("on", None)
        left_on = kwargs.get("left_on", None)
        right_on = kwargs.get("right_on", None)
        left_index = kwargs.get("left_index", False)
        right_index = kwargs.get("right_index", False)
        sort = kwargs.get("sort", False)

        if how in ["left", "inner"] and left_index is False and right_index is False:
            right = right.to_pandas()

            kwargs["sort"] = False

            def map_func(left, right=right, kwargs=kwargs):
                return pandas.merge(left, right, **kwargs)

            new_self = self.__constructor__(
                self._modin_frame._apply_full_axis(1, map_func)
            )
            is_reset_index = True
            if left_on and right_on:
                left_on = left_on if is_list_like(left_on) else [left_on]
                right_on = right_on if is_list_like(right_on) else [right_on]
                is_reset_index = (
                    False
                    if any(o in new_self.index.names for o in left_on)
                    and any(o in right.index.names for o in right_on)
                    else True
                )
                if sort:
                    new_self = (
                        new_self.sort_rows_by_column_values(left_on.append(right_on))
                        if is_reset_index
                        else new_self.sort_index(axis=0, level=left_on.append(right_on))
                    )
            if on:
                on = on if is_list_like(on) else [on]
                is_reset_index = not any(
                    o in new_self.index.names and o in right.index.names for o in on
                )
                if sort:
                    new_self = (
                        new_self.sort_rows_by_column_values(on)
                        if is_reset_index
                        else new_self.sort_index(axis=0, level=on)
                    )
            return new_self.reset_index(drop=True) if is_reset_index else new_self
        else:
            return self.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    def join(self, right, **kwargs):
        """
        Join columns of another DataFrame.

        Parameters
        ----------
        right : BaseQueryCompiler
            The query compiler of the right DataFrame to join with.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler that contains result of the join.

        Notes
        -----
        See pd.DataFrame.join for more info on kwargs.
        """
        on = kwargs.get("on", None)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)

        if how in ["left", "inner"]:
            right = right.to_pandas()

            def map_func(left, right=right, kwargs=kwargs):
                return pandas.DataFrame.join(left, right, **kwargs)

            new_self = self.__constructor__(
                self._modin_frame._apply_full_axis(1, map_func)
            )
            return new_self.sort_rows_by_column_values(on) if sort else new_self
        else:
            return self.default_to_pandas(pandas.DataFrame.join, right, **kwargs)

    # END Inter-Data operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manager.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            A new QueryCompiler with updated data and new index.
        """
        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.reindex(labels=labels, axis=axis, **kwargs),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            A new QueryCompiler with updated data and reset index.
        """
        drop = kwargs.get("drop", False)
        level = kwargs.get("level", None)
        new_index = None
        if level is not None:
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [self.index._get_level_number(lev) for lev in level]
            uniq_sorted_level = sorted(set(level))
            if len(uniq_sorted_level) < self.index.nlevels:
                # We handle this by separately computing the index. We could just
                # put the labels into the data and pull them back out, but that is
                # expensive.
                new_index = (
                    self.index.droplevel(uniq_sorted_level)
                    if len(level) < self.index.nlevels
                    else pandas.RangeIndex(len(self.index))
                )
        else:
            uniq_sorted_level = list(range(self.index.nlevels))

        if not drop:
            if len(uniq_sorted_level) < self.index.nlevels:
                # These are the index levels that will remain after the reset_index
                keep_levels = [
                    i for i in range(self.index.nlevels) if i not in uniq_sorted_level
                ]
                new_copy = self.copy()
                # Change the index to have only the levels that will be inserted
                # into the data. We will replace the old levels later.
                new_copy.index = self.index.droplevel(keep_levels)
                new_copy.index.names = [
                    "level_{}".format(level_value)
                    if new_copy.index.names[level_index] is None
                    else new_copy.index.names[level_index]
                    for level_index, level_value in enumerate(uniq_sorted_level)
                ]
                new_modin_frame = new_copy._modin_frame.from_labels()
                # Replace the levels that will remain as a part of the index.
                new_modin_frame.index = new_index
            else:
                new_modin_frame = self._modin_frame.from_labels()
            if isinstance(new_modin_frame.columns, pandas.MultiIndex):
                # Fix col_level and col_fill in generated column names because from_labels works with assumption
                # that col_level and col_fill are not specified but it expands tuples in level names.
                col_level = kwargs.get("col_level", 0)
                col_fill = kwargs.get("col_fill", "")
                if col_level != 0 or col_fill != "":
                    # Modify generated column names if col_level and col_fil have values different from default.
                    levels_names_list = [
                        f"level_{level_index}" if level_name is None else level_name
                        for level_index, level_name in enumerate(self.index.names)
                    ]
                    if col_fill is None:
                        # Initialize col_fill if it is None.
                        # This is some weird undocumented Pandas behavior to take first
                        # element of the last column name.
                        last_col_name = levels_names_list[uniq_sorted_level[-1]]
                        last_col_name = (
                            list(last_col_name)
                            if isinstance(last_col_name, tuple)
                            else [last_col_name]
                        )
                        if len(last_col_name) not in (1, self.columns.nlevels):
                            raise ValueError(
                                "col_fill=None is incompatible "
                                f"with incomplete column name {last_col_name}"
                            )
                        col_fill = last_col_name[0]
                    columns_list = new_modin_frame.columns.tolist()
                    for level_index, level_value in enumerate(uniq_sorted_level):
                        level_name = levels_names_list[level_value]
                        # Expand tuples into separate items and fill the rest with col_fill
                        top_level = [col_fill] * col_level
                        middle_level = (
                            list(level_name)
                            if isinstance(level_name, tuple)
                            else [level_name]
                        )
                        bottom_level = [col_fill] * (
                            self.columns.nlevels - (col_level + len(middle_level))
                        )
                        item = tuple(top_level + middle_level + bottom_level)
                        if len(item) > self.columns.nlevels:
                            raise ValueError(
                                "Item must have length equal to number of levels."
                            )
                        columns_list[level_index] = item
                    new_modin_frame.columns = pandas.MultiIndex.from_tuples(
                        columns_list, names=self.columns.names
                    )
            new_self = self.__constructor__(new_modin_frame)
        else:
            new_self = self.copy()
            new_self.index = (
                pandas.RangeIndex(len(new_self.index))
                if new_index is None
                else new_index
            )
        return new_self

    def set_index_from_columns(
        self, keys: List[Hashable], drop: bool = True, append: bool = False
    ):
        """Create new row labels from a list of columns.

        Parameters
        ----------
        keys : list of hashable
            The list of column names that will become the new index.
        drop : boolean
            Whether or not to drop the columns provided in the `keys` argument
        append : boolean
            Whether or not to add the columns in `keys` as new levels appended to the
            existing index.

        Returns
        -------
        PandasQueryCompiler
            A new QueryCompiler with updated index.
        """
        new_modin_frame = self._modin_frame.to_labels(keys)
        if append:
            arrays = []
            # Appending keeps the original order of the index levels, then appends the
            # new index objects.
            names = list(self.index.names)
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)

            # Add the names in the correct order.
            names.extend(new_modin_frame.index.names)
            if isinstance(new_modin_frame.index, pandas.MultiIndex):
                for i in range(new_modin_frame.index.nlevels):
                    arrays.append(new_modin_frame.index._get_level_values(i))
            else:
                arrays.append(new_modin_frame.index)
            new_modin_frame.index = ensure_index_from_sequences(arrays, names)
        if not drop:
            # The algebraic operator for this operation always drops the column, but we
            # can copy the data in this object and just use the index from the result of
            # the query compiler call.
            result = self._modin_frame.copy()
            result.index = new_modin_frame.index
        else:
            result = new_modin_frame
        return self.__constructor__(result)

    # END Reindex/reset_index

    # Transpose
    # For transpose, we aren't going to immediately copy everything. Since the
    # actual transpose operation is very fast, we will just do it before any
    # operation that gets called on the transposed data. See _prepare_method
    # for how the transpose is applied.
    #
    # Our invariants assume that the blocks are transposed, but not the
    # data inside. Sometimes we have to reverse this transposition of blocks
    # for simplicity of implementation.

    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        # Switch the index and columns and transpose the data within the blocks.
        return self.__constructor__(self._modin_frame.transpose())

    def columnarize(self):
        """
        Transposes this QueryCompiler if it has a single row but multiple columns.

        This method should be called for QueryCompilers representing a Series object,
        i.e. self.is_series_like() should be True.

        Returns
        -------
        PandasQueryCompiler
            Transposed new QueryCompiler or self.
        """
        if len(self.columns) != 1 or (
            len(self.index) == 1 and self.index[0] == "__reduced__"
        ):
            return self.transpose()
        return self

    def is_series_like(self):
        """Return True if QueryCompiler has a single column or row"""
        return len(self.columns) == 1 or len(self.index) == 1

    # END Transpose

    # MapReduce operations

    def is_monotonic_decreasing(self):
        def is_monotonic_decreasing(df):
            return pandas.DataFrame([df.squeeze(axis=1).is_monotonic_decreasing])

        return self.default_to_pandas(is_monotonic_decreasing)

    def is_monotonic_increasing(self):
        def is_monotonic_increasing(df):
            return pandas.DataFrame([df.squeeze(axis=1).is_monotonic_increasing])

        return self.default_to_pandas(is_monotonic_increasing)

    count = MapReduceFunction.register(pandas.DataFrame.count, pandas.DataFrame.sum)
    sum = MapReduceFunction.register(pandas.DataFrame.sum)
    prod = MapReduceFunction.register(pandas.DataFrame.prod)
    any = MapReduceFunction.register(pandas.DataFrame.any, pandas.DataFrame.any)
    all = MapReduceFunction.register(pandas.DataFrame.all, pandas.DataFrame.all)
    memory_usage = MapReduceFunction.register(
        pandas.DataFrame.memory_usage,
        lambda x, *args, **kwargs: pandas.DataFrame.sum(x),
        axis=0,
    )

    def max(self, axis, **kwargs):
        def map_func(df, **kwargs):
            return pandas.DataFrame.max(df, **kwargs)

        def reduce_func(df, **kwargs):
            if kwargs.get("numeric_only", False):
                kwargs = kwargs.copy()
                kwargs["numeric_only"] = False
            return pandas.DataFrame.max(df, **kwargs)

        return MapReduceFunction.register(map_func, reduce_func)(
            self, axis=axis, **kwargs
        )

    def min(self, axis, **kwargs):
        def map_func(df, **kwargs):
            return pandas.DataFrame.min(df, **kwargs)

        def reduce_func(df, **kwargs):
            if kwargs.get("numeric_only", False):
                kwargs = kwargs.copy()
                kwargs["numeric_only"] = False
            return pandas.DataFrame.min(df, **kwargs)

        return MapReduceFunction.register(map_func, reduce_func)(
            self, axis=axis, **kwargs
        )

    def mean(self, axis, **kwargs):
        if kwargs.get("level") is not None:
            return self.default_to_pandas(pandas.DataFrame.mean, axis=axis, **kwargs)

        skipna = kwargs.get("skipna", True)

        # TODO-FIX: this function may work incorrectly with user-defined "numeric" values.
        # Since `count(numeric_only=True)` discards all unknown "numeric" types, we can get incorrect
        # divisor inside the reduce function.
        def map_fn(df, **kwargs):
            result = pandas.DataFrame(
                {
                    "sum": df.sum(axis=axis, skipna=skipna),
                    "count": df.count(axis=axis, numeric_only=True),
                }
            )
            return result if axis else result.T

        def reduce_fn(df, **kwargs):
            sum_cols = df["sum"] if axis else df.loc["sum"]
            count_cols = df["count"] if axis else df.loc["count"]

            if not isinstance(sum_cols, pandas.Series):
                # If we got `NaN` as the result of the sum in any axis partition,
                # then we must consider the whole sum as `NaN`, so setting `skipna=False`
                sum_cols = sum_cols.sum(axis=axis, skipna=False)
                count_cols = count_cols.sum(axis=axis, skipna=False)
            return sum_cols / count_cols

        return MapReduceFunction.register(
            map_fn,
            reduce_fn,
        )(self, axis=axis, **kwargs)

    def value_counts(self, **kwargs):
        """
        Return a QueryCompiler of Series containing counts of unique values.

        Returns
        -------
        PandasQueryCompiler
        """

        def value_counts(df):
            return df.squeeze(axis=1).value_counts(**kwargs).to_frame()

        return self.default_to_pandas(value_counts)

    # END MapReduce operations

    # Reduction operations
    idxmax = ReductionFunction.register(pandas.DataFrame.idxmax)
    idxmin = ReductionFunction.register(pandas.DataFrame.idxmin)
    median = ReductionFunction.register(pandas.DataFrame.median)
    nunique = ReductionFunction.register(pandas.DataFrame.nunique)
    skew = ReductionFunction.register(pandas.DataFrame.skew)
    kurt = ReductionFunction.register(pandas.DataFrame.kurt)
    sem = ReductionFunction.register(pandas.DataFrame.sem)
    std = ReductionFunction.register(pandas.DataFrame.std)
    var = ReductionFunction.register(pandas.DataFrame.var)
    sum_min_count = ReductionFunction.register(pandas.DataFrame.sum)
    prod_min_count = ReductionFunction.register(pandas.DataFrame.prod)
    quantile_for_single_value = ReductionFunction.register(pandas.DataFrame.quantile)
    mad = ReductionFunction.register(pandas.DataFrame.mad)
    to_datetime = ReductionFunction.register(
        lambda df, *args, **kwargs: pandas.to_datetime(
            df.squeeze(axis=1), *args, **kwargs
        ),
        axis=1,
    )

    # END Reduction operations

    def _resample_func(
        self, resample_args, func_name, new_columns=None, df_op=None, *args, **kwargs
    ):
        def map_func(df, resample_args=resample_args):
            if df_op is not None:
                df = df_op(df)
            resampled_val = df.resample(*resample_args)
            op = getattr(pandas.core.resample.Resampler, func_name)
            if callable(op):
                try:
                    # This will happen with Arrow buffer read-only errors. We don't want to copy
                    # all the time, so this will try to fast-path the code first.
                    val = op(resampled_val, *args, **kwargs)
                except (ValueError):
                    resampled_val = df.copy().resample(*resample_args)
                    val = op(resampled_val, *args, **kwargs)
            else:
                val = getattr(resampled_val, func_name)

            if isinstance(val, pandas.Series):
                return val.to_frame()
            else:
                return val

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis=0, func=map_func, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def resample_get_group(self, resample_args, name, obj):
        return self._resample_func(resample_args, "get_group", name=name, obj=obj)

    def resample_app_ser(self, resample_args, func, *args, **kwargs):
        return self._resample_func(
            resample_args,
            "apply",
            df_op=lambda df: df.squeeze(axis=1),
            func=func,
            *args,
            **kwargs,
        )

    def resample_app_df(self, resample_args, func, *args, **kwargs):
        return self._resample_func(resample_args, "apply", func=func, *args, **kwargs)

    def resample_agg_ser(self, resample_args, func, *args, **kwargs):
        return self._resample_func(
            resample_args,
            "aggregate",
            df_op=lambda df: df.squeeze(axis=1),
            func=func,
            *args,
            **kwargs,
        )

    def resample_agg_df(self, resample_args, func, *args, **kwargs):
        return self._resample_func(
            resample_args, "aggregate", func=func, *args, **kwargs
        )

    def resample_transform(self, resample_args, arg, *args, **kwargs):
        return self._resample_func(resample_args, "transform", arg=arg, *args, **kwargs)

    def resample_pipe(self, resample_args, func, *args, **kwargs):
        return self._resample_func(resample_args, "pipe", func=func, *args, **kwargs)

    def resample_ffill(self, resample_args, limit):
        return self._resample_func(resample_args, "ffill", limit=limit)

    def resample_backfill(self, resample_args, limit):
        return self._resample_func(resample_args, "backfill", limit=limit)

    def resample_bfill(self, resample_args, limit):
        return self._resample_func(resample_args, "bfill", limit=limit)

    def resample_pad(self, resample_args, limit):
        return self._resample_func(resample_args, "pad", limit=limit)

    def resample_nearest(self, resample_args, limit):
        return self._resample_func(resample_args, "nearest", limit=limit)

    def resample_fillna(self, resample_args, method, limit):
        return self._resample_func(resample_args, "fillna", method=method, limit=limit)

    def resample_asfreq(self, resample_args, fill_value):
        return self._resample_func(resample_args, "asfreq", fill_value=fill_value)

    def resample_interpolate(
        self,
        resample_args,
        method,
        axis,
        limit,
        inplace,
        limit_direction,
        limit_area,
        downcast,
        **kwargs,
    ):
        return self._resample_func(
            resample_args,
            "interpolate",
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    def resample_count(self, resample_args):
        return self._resample_func(resample_args, "count")

    def resample_nunique(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "nunique", _method=_method, *args, **kwargs
        )

    def resample_first(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "first", _method=_method, *args, **kwargs
        )

    def resample_last(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "last", _method=_method, *args, **kwargs
        )

    def resample_max(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "max", _method=_method, *args, **kwargs
        )

    def resample_mean(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "median", _method=_method, *args, **kwargs
        )

    def resample_median(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "median", _method=_method, *args, **kwargs
        )

    def resample_min(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "min", _method=_method, *args, **kwargs
        )

    def resample_ohlc_ser(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args,
            "ohlc",
            df_op=lambda df: df.squeeze(axis=1),
            _method=_method,
            *args,
            **kwargs,
        )

    def resample_ohlc_df(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "ohlc", _method=_method, *args, **kwargs
        )

    def resample_prod(self, resample_args, _method, min_count, *args, **kwargs):
        return self._resample_func(
            resample_args, "prod", _method=_method, min_count=min_count, *args, **kwargs
        )

    def resample_size(self, resample_args):
        return self._resample_func(resample_args, "size", new_columns=["__reduced__"])

    def resample_sem(self, resample_args, _method, *args, **kwargs):
        return self._resample_func(
            resample_args, "sem", _method=_method, *args, **kwargs
        )

    def resample_std(self, resample_args, ddof, *args, **kwargs):
        return self._resample_func(resample_args, "std", ddof=ddof, *args, **kwargs)

    def resample_sum(self, resample_args, _method, min_count, *args, **kwargs):
        return self._resample_func(
            resample_args, "sum", _method=_method, min_count=min_count, *args, **kwargs
        )

    def resample_var(self, resample_args, ddof, *args, **kwargs):
        return self._resample_func(resample_args, "var", ddof=ddof, *args, **kwargs)

    def resample_quantile(self, resample_args, q, **kwargs):
        return self._resample_func(resample_args, "quantile", q=q, **kwargs)

    window_mean = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).mean(*args, **kwargs)
        )
    )
    window_sum = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).sum(*args, **kwargs)
        )
    )
    window_var = FoldFunction.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).var(ddof=ddof, *args, **kwargs)
        )
    )
    window_std = FoldFunction.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).std(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_count = FoldFunction.register(
        lambda df, rolling_args: pandas.DataFrame(df.rolling(*rolling_args).count())
    )
    rolling_sum = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).sum(*args, **kwargs)
        )
    )
    rolling_mean = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).mean(*args, **kwargs)
        )
    )
    rolling_median = FoldFunction.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).median(**kwargs)
        )
    )
    rolling_var = FoldFunction.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).var(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_std = FoldFunction.register(
        lambda df, rolling_args, ddof, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).std(ddof=ddof, *args, **kwargs)
        )
    )
    rolling_min = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).min(*args, **kwargs)
        )
    )
    rolling_max = FoldFunction.register(
        lambda df, rolling_args, *args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).max(*args, **kwargs)
        )
    )
    rolling_skew = FoldFunction.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).skew(**kwargs)
        )
    )
    rolling_kurt = FoldFunction.register(
        lambda df, rolling_args, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).kurt(**kwargs)
        )
    )
    rolling_apply = FoldFunction.register(
        lambda df, rolling_args, func, raw, engine, engine_kwargs, args, kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).apply(
                func=func,
                raw=raw,
                engine=engine,
                engine_kwargs=engine_kwargs,
                args=args,
                kwargs=kwargs,
            )
        )
    )
    rolling_quantile = FoldFunction.register(
        lambda df, rolling_args, quantile, interpolation, **kwargs: pandas.DataFrame(
            df.rolling(*rolling_args).quantile(
                quantile=quantile, interpolation=interpolation, **kwargs
            )
        )
    )

    def rolling_corr(self, rolling_args, other, pairwise, *args, **kwargs):
        if len(self.columns) > 1:
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.rolling(df, *rolling_args).corr(
                    other=other, pairwise=pairwise, *args, **kwargs
                )
            )
        else:
            return FoldFunction.register(
                lambda df: pandas.DataFrame(
                    df.rolling(*rolling_args).corr(
                        other=other, pairwise=pairwise, *args, **kwargs
                    )
                )
            )(self)

    def rolling_cov(self, rolling_args, other, pairwise, ddof, **kwargs):
        if len(self.columns) > 1:
            return self.default_to_pandas(
                lambda df: pandas.DataFrame.rolling(df, *rolling_args).cov(
                    other=other, pairwise=pairwise, ddof=ddof, **kwargs
                )
            )
        else:
            return FoldFunction.register(
                lambda df: pandas.DataFrame(
                    df.rolling(*rolling_args).cov(
                        other=other, pairwise=pairwise, ddof=ddof, **kwargs
                    )
                )
            )(self)

    def rolling_aggregate(self, rolling_args, func, *args, **kwargs):
        new_modin_frame = self._modin_frame._apply_full_axis(
            0,
            lambda df: pandas.DataFrame(
                df.rolling(*rolling_args).aggregate(func=func, *args, **kwargs)
            ),
            new_index=self.index,
        )
        return self.__constructor__(new_modin_frame)

    def unstack(self, level, fill_value):
        if not isinstance(self.index, pandas.MultiIndex) or (
            isinstance(self.index, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.index.nlevels
        ):
            axis = 1
            new_columns = ["__reduced__"]
            need_reindex = True
        else:
            axis = 0
            new_columns = None
            need_reindex = False

        def map_func(df):
            return pandas.DataFrame(df.unstack(level=level, fill_value=fill_value))

        def is_tree_like_or_1d(calc_index, valid_index):
            if not isinstance(calc_index, pandas.MultiIndex):
                return True
            actual_len = 1
            for lvl in calc_index.levels:
                actual_len *= len(lvl)
            return len(self.index) * len(self.columns) == actual_len * len(valid_index)

        is_tree_like_or_1d_index = is_tree_like_or_1d(self.index, self.columns)
        is_tree_like_or_1d_cols = is_tree_like_or_1d(self.columns, self.index)

        is_all_multi_list = False
        if (
            isinstance(self.index, pandas.MultiIndex)
            and isinstance(self.columns, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.index.nlevels
            and is_tree_like_or_1d_index
            and is_tree_like_or_1d_cols
        ):
            is_all_multi_list = True
            real_cols_bkp = self.columns
            obj = self.copy()
            obj.columns = np.arange(len(obj.columns))
        else:
            obj = self

        new_modin_frame = obj._modin_frame._apply_full_axis(
            axis, map_func, new_columns=new_columns
        )
        result = self.__constructor__(new_modin_frame)

        def compute_index(index, columns, consider_index=True, consider_columns=True):
            def get_unique_level_values(index):
                return [
                    index.get_level_values(lvl).unique()
                    for lvl in np.arange(index.nlevels)
                ]

            new_index = (
                get_unique_level_values(index)
                if consider_index
                else index
                if isinstance(index, list)
                else [index]
            )

            new_columns = (
                get_unique_level_values(columns) if consider_columns else [columns]
            )
            return pandas.MultiIndex.from_product([*new_columns, *new_index])

        if is_all_multi_list and is_tree_like_or_1d_index and is_tree_like_or_1d_cols:
            result = result.sort_index()
            index_level_values = [lvl for lvl in obj.index.levels]

            result.index = compute_index(
                index_level_values, real_cols_bkp, consider_index=False
            )
            return result

        if need_reindex:
            if is_tree_like_or_1d_index and is_tree_like_or_1d_cols:
                is_recompute_index = isinstance(self.index, pandas.MultiIndex)
                is_recompute_columns = not is_recompute_index and isinstance(
                    self.columns, pandas.MultiIndex
                )
                new_index = compute_index(
                    self.index, self.columns, is_recompute_index, is_recompute_columns
                )
            elif is_tree_like_or_1d_index != is_tree_like_or_1d_cols:
                if isinstance(self.columns, pandas.MultiIndex) or not isinstance(
                    self.index, pandas.MultiIndex
                ):
                    return result
                else:
                    index = (
                        self.index.sortlevel()[0]
                        if is_tree_like_or_1d_index
                        and not is_tree_like_or_1d_cols
                        and isinstance(self.index, pandas.MultiIndex)
                        else self.index
                    )
                    index = pandas.MultiIndex.from_tuples(
                        list(index) * len(self.columns)
                    )
                    columns = self.columns.repeat(len(self.index))
                    index_levels = [
                        index.get_level_values(i) for i in range(index.nlevels)
                    ]
                    new_index = pandas.MultiIndex.from_arrays(
                        [columns] + index_levels,
                        names=self.columns.names + self.index.names,
                    )
            else:
                return result
            result = result.reindex(0, new_index)
        return result

    def stack(self, level, dropna):
        if not isinstance(self.columns, pandas.MultiIndex) or (
            isinstance(self.columns, pandas.MultiIndex)
            and is_list_like(level)
            and len(level) == self.columns.nlevels
        ):
            new_columns = ["__reduced__"]
        else:
            new_columns = None

        new_modin_frame = self._modin_frame._apply_full_axis(
            1,
            lambda df: pandas.DataFrame(df.stack(level=level, dropna=dropna)),
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    abs = MapFunction.register(pandas.DataFrame.abs, dtypes="copy")
    applymap = MapFunction.register(pandas.DataFrame.applymap)
    conj = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.conj(df))
    )
    invert = MapFunction.register(pandas.DataFrame.__invert__)
    isin = MapFunction.register(pandas.DataFrame.isin, dtypes=np.bool)
    isna = MapFunction.register(pandas.DataFrame.isna, dtypes=np.bool)
    _isfinite = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.isfinite(df))
    )
    negative = MapFunction.register(pandas.DataFrame.__neg__)
    notna = MapFunction.register(pandas.DataFrame.notna, dtypes=np.bool)
    round = MapFunction.register(pandas.DataFrame.round)
    replace = MapFunction.register(pandas.DataFrame.replace)
    series_view = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            df.squeeze(axis=1).view(*args, **kwargs)
        )
    )
    to_numeric = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            pandas.to_numeric(df.squeeze(axis=1), *args, **kwargs)
        )
    )

    def repeat(self, repeats):
        def map_fn(df):
            return pandas.DataFrame(df.squeeze(axis=1).repeat(repeats))

        if isinstance(repeats, int) or (is_list_like(repeats) and len(repeats) == 1):
            return MapFunction.register(map_fn, validate_index=True)(self)
        else:
            return self.__constructor__(self._modin_frame._apply_full_axis(0, map_fn))

    # END Map partitions operations

    # String map partitions operations

    str_capitalize = MapFunction.register(_str_map("capitalize"), dtypes="copy")
    str_center = MapFunction.register(_str_map("center"), dtypes="copy")
    str_contains = MapFunction.register(_str_map("contains"), dtypes=np.bool)
    str_count = MapFunction.register(_str_map("count"), dtypes=int)
    str_endswith = MapFunction.register(_str_map("endswith"), dtypes=np.bool)
    str_find = MapFunction.register(_str_map("find"), dtypes="copy")
    str_findall = MapFunction.register(_str_map("findall"), dtypes="copy")
    str_get = MapFunction.register(_str_map("get"), dtypes="copy")
    str_index = MapFunction.register(_str_map("index"), dtypes="copy")
    str_isalnum = MapFunction.register(_str_map("isalnum"), dtypes=np.bool)
    str_isalpha = MapFunction.register(_str_map("isalpha"), dtypes=np.bool)
    str_isdecimal = MapFunction.register(_str_map("isdecimal"), dtypes=np.bool)
    str_isdigit = MapFunction.register(_str_map("isdigit"), dtypes=np.bool)
    str_islower = MapFunction.register(_str_map("islower"), dtypes=np.bool)
    str_isnumeric = MapFunction.register(_str_map("isnumeric"), dtypes=np.bool)
    str_isspace = MapFunction.register(_str_map("isspace"), dtypes=np.bool)
    str_istitle = MapFunction.register(_str_map("istitle"), dtypes=np.bool)
    str_isupper = MapFunction.register(_str_map("isupper"), dtypes=np.bool)
    str_join = MapFunction.register(_str_map("join"), dtypes="copy")
    str_len = MapFunction.register(_str_map("len"), dtypes=int)
    str_ljust = MapFunction.register(_str_map("ljust"), dtypes="copy")
    str_lower = MapFunction.register(_str_map("lower"), dtypes="copy")
    str_lstrip = MapFunction.register(_str_map("lstrip"), dtypes="copy")
    str_match = MapFunction.register(_str_map("match"), dtypes="copy")
    str_normalize = MapFunction.register(_str_map("normalize"), dtypes="copy")
    str_pad = MapFunction.register(_str_map("pad"), dtypes="copy")
    str_partition = MapFunction.register(_str_map("partition"), dtypes="copy")
    str_repeat = MapFunction.register(_str_map("repeat"), dtypes="copy")
    str_replace = MapFunction.register(_str_map("replace"), dtypes="copy")
    str_rfind = MapFunction.register(_str_map("rfind"), dtypes="copy")
    str_rindex = MapFunction.register(_str_map("rindex"), dtypes="copy")
    str_rjust = MapFunction.register(_str_map("rjust"), dtypes="copy")
    str_rpartition = MapFunction.register(_str_map("rpartition"), dtypes="copy")
    str_rsplit = MapFunction.register(_str_map("rsplit"), dtypes="copy")
    str_rstrip = MapFunction.register(_str_map("rstrip"), dtypes="copy")
    str_slice = MapFunction.register(_str_map("slice"), dtypes="copy")
    str_slice_replace = MapFunction.register(_str_map("slice_replace"), dtypes="copy")
    str_split = MapFunction.register(_str_map("split"), dtypes="copy")
    str_startswith = MapFunction.register(_str_map("startswith"), dtypes=np.bool)
    str_strip = MapFunction.register(_str_map("strip"), dtypes="copy")
    str_swapcase = MapFunction.register(_str_map("swapcase"), dtypes="copy")
    str_title = MapFunction.register(_str_map("title"), dtypes="copy")
    str_translate = MapFunction.register(_str_map("translate"), dtypes="copy")
    str_upper = MapFunction.register(_str_map("upper"), dtypes="copy")
    str_wrap = MapFunction.register(_str_map("wrap"), dtypes="copy")
    str_zfill = MapFunction.register(_str_map("zfill"), dtypes="copy")

    # END String map partitions operations

    def unique(self):
        """Return unique values of Series object.

        Returns
        -------
        ndarray
            The unique values returned as a NumPy array.
        """
        new_modin_frame = self._modin_frame._apply_full_axis(
            0,
            lambda x: x.squeeze(axis=1).unique(),
            new_columns=self.columns,
        )
        return self.__constructor__(new_modin_frame)

    def searchsorted(self, **kwargs):
        """
        Return a QueryCompiler with value/values indicies, which they should be inserted
        to maintain order of the passed Series.

        Returns
        -------
        PandasQueryCompiler
        """

        def searchsorted(df):
            result = df.squeeze(axis=1).searchsorted(**kwargs)
            if not is_list_like(result):
                result = [result]
            return pandas.DataFrame(result)

        return self.default_to_pandas(searchsorted)

    # Dt map partitions operations

    dt_date = MapFunction.register(_dt_prop_map("date"))
    dt_time = MapFunction.register(_dt_prop_map("time"))
    dt_timetz = MapFunction.register(_dt_prop_map("timetz"))
    dt_year = MapFunction.register(_dt_prop_map("year"))
    dt_month = MapFunction.register(_dt_prop_map("month"))
    dt_day = MapFunction.register(_dt_prop_map("day"))
    dt_hour = MapFunction.register(_dt_prop_map("hour"))
    dt_minute = MapFunction.register(_dt_prop_map("minute"))
    dt_second = MapFunction.register(_dt_prop_map("second"))
    dt_microsecond = MapFunction.register(_dt_prop_map("microsecond"))
    dt_nanosecond = MapFunction.register(_dt_prop_map("nanosecond"))
    dt_week = MapFunction.register(_dt_prop_map("week"))
    dt_weekofyear = MapFunction.register(_dt_prop_map("weekofyear"))
    dt_dayofweek = MapFunction.register(_dt_prop_map("dayofweek"))
    dt_weekday = MapFunction.register(_dt_prop_map("weekday"))
    dt_dayofyear = MapFunction.register(_dt_prop_map("dayofyear"))
    dt_quarter = MapFunction.register(_dt_prop_map("quarter"))
    dt_is_month_start = MapFunction.register(_dt_prop_map("is_month_start"))
    dt_is_month_end = MapFunction.register(_dt_prop_map("is_month_end"))
    dt_is_quarter_start = MapFunction.register(_dt_prop_map("is_quarter_start"))
    dt_is_quarter_end = MapFunction.register(_dt_prop_map("is_quarter_end"))
    dt_is_year_start = MapFunction.register(_dt_prop_map("is_year_start"))
    dt_is_year_end = MapFunction.register(_dt_prop_map("is_year_end"))
    dt_is_leap_year = MapFunction.register(_dt_prop_map("is_leap_year"))
    dt_daysinmonth = MapFunction.register(_dt_prop_map("daysinmonth"))
    dt_days_in_month = MapFunction.register(_dt_prop_map("days_in_month"))

    def dt_tz(self):
        def datetime_tz(df):
            return pandas.DataFrame([df.squeeze(axis=1).dt.tz])

        return self.default_to_pandas(datetime_tz)

    def dt_freq(self):
        def datetime_freq(df):
            return pandas.DataFrame([df.squeeze(axis=1).dt.freq])

        return self.default_to_pandas(datetime_freq)

    dt_to_period = MapFunction.register(_dt_func_map("to_period"))
    dt_to_pydatetime = MapFunction.register(_dt_func_map("to_pydatetime"))
    dt_tz_localize = MapFunction.register(_dt_func_map("tz_localize"))
    dt_tz_convert = MapFunction.register(_dt_func_map("tz_convert"))
    dt_normalize = MapFunction.register(_dt_func_map("normalize"))
    dt_strftime = MapFunction.register(_dt_func_map("strftime"))
    dt_round = MapFunction.register(_dt_func_map("round"))
    dt_floor = MapFunction.register(_dt_func_map("floor"))
    dt_ceil = MapFunction.register(_dt_func_map("ceil"))
    dt_month_name = MapFunction.register(_dt_func_map("month_name"))
    dt_day_name = MapFunction.register(_dt_func_map("day_name"))
    dt_to_pytimedelta = MapFunction.register(_dt_func_map("to_pytimedelta"))
    dt_total_seconds = MapFunction.register(_dt_func_map("total_seconds"))
    dt_seconds = MapFunction.register(_dt_prop_map("seconds"))
    dt_days = MapFunction.register(_dt_prop_map("days"))
    dt_microseconds = MapFunction.register(_dt_prop_map("microseconds"))
    dt_nanoseconds = MapFunction.register(_dt_prop_map("nanoseconds"))
    dt_components = MapFunction.register(
        _dt_prop_map("components"), validate_columns=True
    )
    dt_qyear = MapFunction.register(_dt_prop_map("qyear"))
    dt_start_time = MapFunction.register(_dt_prop_map("start_time"))
    dt_end_time = MapFunction.register(_dt_prop_map("end_time"))
    dt_to_timestamp = MapFunction.register(_dt_func_map("to_timestamp"))

    # END Dt map partitions operations

    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return self.__constructor__(self._modin_frame.astype(col_dtypes))

    # Column/Row partitions reduce operations

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def first_valid_index_builder(df):
            return df.set_axis(
                pandas.RangeIndex(len(df.index)), axis="index", inplace=False
            ).apply(lambda df: df.first_valid_index())

        # We get the minimum from each column, then take the min of that to get
        # first_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(
                self._modin_frame._fold_reduce(0, first_valid_index_builder)
            )
            .min(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def last_valid_index_builder(df):
            return df.set_axis(
                pandas.RangeIndex(len(df.index)), axis="index", inplace=False
            ).apply(lambda df: df.last_valid_index())

        # We get the maximum from each column, then take the max of that to get
        # last_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(
                self._modin_frame._fold_reduce(0, last_valid_index_builder)
            )
            .max(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    # END Column/Row partitions reduce operations

    # Column/Row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # This will return a new QueryCompiler object which the front end will handle.

    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        # Use pandas to calculate the correct columns
        empty_df = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .describe(**kwargs)
        )
        new_index = empty_df.index

        # Note: `describe` convert timestamp type to object type
        # which results in the loss of two values in index: `first` and `last`
        # for empty DataFrame.
        datetime_is_numeric = kwargs.get("datetime_is_numeric") or False
        if not any(map(is_numeric_dtype, empty_df.dtypes)) and not datetime_is_numeric:
            for col_name in empty_df.dtypes.index:
                # if previosly type of `col_name` was datetime or timedelta
                if is_datetime_or_timedelta_dtype(self.dtypes[col_name]):
                    new_index = pandas.Index(
                        empty_df.index.to_list() + ["first"] + ["last"]
                    )
                    break

        def describe_builder(df, internal_indices=[]):
            return df.iloc[:, internal_indices].describe(**kwargs)

        return self.__constructor__(
            self._modin_frame._apply_full_axis_select_indices(
                0,
                describe_builder,
                empty_df.columns,
                new_index=new_index,
                new_columns=empty_df.columns,
            )
        )

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    cummax = FoldFunction.register(pandas.DataFrame.cummax)
    cummin = FoldFunction.register(pandas.DataFrame.cummin)
    cumsum = FoldFunction.register(pandas.DataFrame.cumsum)
    cumprod = FoldFunction.register(pandas.DataFrame.cumprod)
    diff = FoldFunction.register(pandas.DataFrame.diff)

    def clip(self, lower, upper, **kwargs):
        kwargs["upper"] = upper
        kwargs["lower"] = lower
        axis = kwargs.get("axis", 0)
        if is_list_like(lower) or is_list_like(upper):
            new_modin_frame = self._modin_frame._fold(
                axis, lambda df: df.clip(**kwargs)
            )
        else:
            new_modin_frame = self._modin_frame._map(lambda df: df.clip(**kwargs))
        return self.__constructor__(new_modin_frame)

    def corr(self, method="pearson", min_periods=1):
        if method == "pearson":
            numeric_self = self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
            return numeric_self._nancorr(min_periods=min_periods)
        else:
            return super().corr(method=method, min_periods=min_periods)

    def cov(self, min_periods=None):
        return self._nancorr(min_periods=min_periods, cov=True)

    def _nancorr(self, min_periods=1, cov=False):
        """
        Compute either pairwise covariance or pairwise correlation of columns,
        considering NA/null values the same like pandas does.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations required per pair of columns
            to have a valid result.
        cov : boolean, default False
            Either covariance or correlation should be computed.

        Returns
        -------
        PandasQueryCompiler
            The covariance or correlation matrix of the series of the DataFrame.
        """
        other = self.to_numpy()
        other_mask = self._isfinite().to_numpy()
        n_cols = other.shape[1]

        if min_periods is None:
            min_periods = 1

        def map_func(df):
            df = df.to_numpy()
            n_rows = df.shape[0]
            df_mask = np.isfinite(df)

            result = np.empty((n_rows, n_cols), dtype=np.float64)

            for i in range(n_rows):
                df_ith_row = df[i]
                df_ith_mask = df_mask[i]

                for j in range(n_cols):
                    other_jth_col = other[:, j]

                    valid = df_ith_mask & other_mask[:, j]

                    vx = df_ith_row[valid]
                    vy = other_jth_col[valid]

                    nobs = len(vx)

                    if nobs < min_periods:
                        result[i, j] = np.nan
                    else:
                        vx = vx - vx.mean()
                        vy = vy - vy.mean()
                        sumxy = (vx * vy).sum()
                        sumxx = (vx * vx).sum()
                        sumyy = (vy * vy).sum()

                        denom = (nobs - 1.0) if cov else np.sqrt(sumxx * sumyy)
                        if denom != 0:
                            result[i, j] = sumxy / denom
                        else:
                            result[i, j] = np.nan

            return pandas.DataFrame(result)

        columns = self.columns
        index = columns.copy()
        transponed_self = self.transpose()
        new_modin_frame = transponed_self._modin_frame._apply_full_axis(
            1, map_func, new_index=index, new_columns=columns
        )
        return transponed_self.__constructor__(new_modin_frame)

    def dot(self, other, squeeze_self=None, squeeze_other=None):
        """
        Computes the matrix multiplication of self and other.

        Parameters
        ----------
            other : PandasQueryCompiler or NumPy array
                The other query compiler or NumPy array to matrix multiply with self.
            squeeze_self : boolean
                The flag to squeeze self.
            squeeze_other : boolean
                The flag to squeeze other (this flag is applied if other is query compiler).

        Returns
        -------
        PandasQueryCompiler
            A new query compiler that contains result of the matrix multiply.
        """
        if isinstance(other, PandasQueryCompiler):
            other = (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )

        def map_func(df, other=other, squeeze_self=squeeze_self):
            result = df.squeeze(axis=1).dot(other) if squeeze_self else df.dot(other)
            if is_list_like(result):
                return pandas.DataFrame(result)
            else:
                return pandas.DataFrame([result])

        num_cols = other.shape[1] if len(other.shape) > 1 else 1
        if len(self.columns) == 1:
            new_index = (
                ["__reduced__"]
                if (len(self.index) == 1 or squeeze_self) and num_cols == 1
                else None
            )
            new_columns = ["__reduced__"] if squeeze_self and num_cols == 1 else None
            axis = 0
        else:
            new_index = self.index
            new_columns = ["__reduced__"] if num_cols == 1 else None
            axis = 1

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, map_func, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def _nsort(self, n, columns=None, keep="first", sort_type="nsmallest"):
        def map_func(df, n=n, keep=keep, columns=columns):
            if columns is None:
                return pandas.DataFrame(
                    getattr(pandas.Series, sort_type)(
                        df.squeeze(axis=1), n=n, keep=keep
                    )
                )
            return getattr(pandas.DataFrame, sort_type)(
                df, n=n, columns=columns, keep=keep
            )

        if columns is None:
            new_columns = ["__reduced__"]
        else:
            new_columns = self.columns

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis=0, func=map_func, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def nsmallest(self, *args, **kwargs):
        return self._nsort(sort_type="nsmallest", *args, **kwargs)

    def nlargest(self, *args, **kwargs):
        return self._nsort(sort_type="nlargest", *args, **kwargs)

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        empty_eval = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .eval(expr, inplace=False, **kwargs)
        )
        if isinstance(empty_eval, pandas.Series):
            new_columns = (
                [empty_eval.name] if empty_eval.name is not None else ["__reduced__"]
            )
        else:
            new_columns = empty_eval.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            1,
            lambda df: pandas.DataFrame(df.eval(expr, inplace=False, **kwargs)),
            new_index=self.index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        axis = kwargs.get("axis", 0)

        def mode_builder(df):
            result = pandas.DataFrame(df.mode(**kwargs))
            # We return a dataframe with the same shape as the input to ensure
            # that all the partitions will be the same shape
            if axis == 0 and len(df) != len(result):
                # Pad rows
                result = result.reindex(index=pandas.RangeIndex(len(df.index)))
            elif axis == 1 and len(df.columns) != len(result.columns):
                # Pad columns
                result = result.reindex(columns=pandas.RangeIndex(len(df.columns)))
            return pandas.DataFrame(result)

        if axis == 0:
            new_index = pandas.RangeIndex(len(self.index))
            new_columns = self.columns
        else:
            new_index = self.index
            new_columns = pandas.RangeIndex(len(self.columns))
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, mode_builder, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame).dropna(axis=axis, how="all")

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
        axis = kwargs.get("axis", 0)
        value = kwargs.get("value")
        method = kwargs.get("method", None)
        limit = kwargs.get("limit", None)
        full_axis = method is not None or limit is not None
        if isinstance(value, dict):
            kwargs.pop("value")

            def fillna(df):
                func_dict = {c: value[c] for c in value if c in df.columns}
                return df.fillna(value=func_dict, **kwargs)

        else:

            def fillna(df):
                return df.fillna(**kwargs)

        if full_axis:
            new_modin_frame = self._modin_frame._fold(axis, fillna)
        else:
            new_modin_frame = self._modin_frame._map(fillna)
        return self.__constructor__(new_modin_frame)

    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            QueryCompiler containing quantiles of original QueryCompiler along an axis.
        """
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q")
        numeric_only = kwargs.get("numeric_only", True)
        assert isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list))

        if numeric_only:
            new_columns = self._modin_frame._numeric_columns()
        else:
            new_columns = [
                col
                for col, dtype in zip(self.columns, self.dtypes)
                if (is_numeric_dtype(dtype) or is_datetime_or_timedelta_dtype(dtype))
            ]
        if axis == 1:
            query_compiler = self.getitem_column_array(new_columns)
            new_columns = self.index
        else:
            query_compiler = self

        def quantile_builder(df, **kwargs):
            result = df.quantile(**kwargs)
            return result.T if kwargs.get("axis", 0) == 1 else result

        # This took a long time to debug, so here is the rundown of why this is needed.
        # Previously, we were operating on select indices, but that was broken. We were
        # not correctly setting the columns/index. Because of how we compute `to_pandas`
        # and because of the static nature of the index for `axis=1` it is easier to
        # just handle this as the transpose (see `quantile_builder` above for the
        # transpose within the partition) than it is to completely rework other
        # internal methods. Basically we are returning the transpose of the object for
        # correctness and cleanliness of the code.
        if axis == 1:
            q_index = new_columns
            new_columns = pandas.Float64Index(q)
        else:
            q_index = pandas.Float64Index(q)
        new_modin_frame = query_compiler._modin_frame._apply_full_axis(
            axis,
            lambda df: quantile_builder(df, **kwargs),
            new_index=q_index,
            new_columns=new_columns,
            dtypes=np.float64,
        )
        result = self.__constructor__(new_modin_frame)
        return result.transpose() if axis == 1 else result

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """

        def query_builder(df, **kwargs):
            return df.query(expr, inplace=False, **kwargs)

        return self.__constructor__(
            self._modin_frame.filter_full_axis(1, query_builder)
        )

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.rank(**kwargs),
            new_index=self.index,
            new_columns=self.columns if not numeric_only else None,
            dtypes=np.float64,
        )
        return self.__constructor__(new_modin_frame)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        axis = kwargs.pop("axis", 0)
        level = kwargs.pop("level", None)
        sort_remaining = kwargs.pop("sort_remaining", True)
        kwargs["inplace"] = False

        if level is not None or self.has_multiindex(axis=axis):
            return self.default_to_pandas(
                pandas.DataFrame.sort_index,
                axis=axis,
                level=level,
                sort_remaining=sort_remaining,
                **kwargs,
            )

        # sort_index can have ascending be None and behaves as if it is False.
        # sort_values cannot have ascending be None. Thus, the following logic is to
        # convert the ascending argument to one that works with sort_values
        ascending = kwargs.pop("ascending", True)
        if ascending is None:
            ascending = False
        kwargs["ascending"] = ascending
        if axis:
            new_columns = pandas.Series(self.columns).sort_values(**kwargs)
            new_index = self.index
        else:
            new_index = pandas.Series(self.index).sort_values(**kwargs)
            new_columns = self.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.sort_index(
                axis=axis, level=level, sort_remaining=sort_remaining, **kwargs
            ),
            new_index,
            new_columns,
            dtypes="copy" if axis == 0 else None,
        )
        return self.__constructor__(new_modin_frame)

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
        ignore_index=True,
    ):
        ErrorMessage.missmatch_with_pandas(
            operation="melt", message="Order of rows could be different from pandas"
        )

        if var_name is None:
            var_name = "variable"

        def _convert_to_list(x):
            if is_list_like(x):
                x = [*x]
            elif x is not None:
                x = [x]
            else:
                x = []
            return x

        id_vars, value_vars = map(_convert_to_list, [id_vars, value_vars])

        if len(value_vars) == 0:
            value_vars = self.columns.drop(id_vars)

        if len(id_vars) != 0:
            to_broadcast = self.getitem_column_array(id_vars)._modin_frame
        else:
            to_broadcast = None

        def applyier(df, internal_indices, other=[], internal_other_indices=[]):
            if len(other):
                other = pandas.concat(other, axis=1)
                columns_to_add = other.columns.difference(df.columns)
                df = pandas.concat([df, other[columns_to_add]], axis=1)
            return df.melt(
                id_vars=id_vars,
                value_vars=df.columns[internal_indices],
                var_name=var_name,
                value_name=value_name,
                col_level=col_level,
            )

        # we have no able to calculate correct indices here, so making it `dummy_index`
        inconsistent_frame = self._modin_frame.broadcast_apply_select_indices(
            axis=0,
            apply_indices=value_vars,
            func=applyier,
            other=to_broadcast,
            new_index=["dummy_index"] * len(id_vars),
            new_columns=["dummy_index"] * len(id_vars),
        )
        # after applying `melt` for selected indices we will get partitions like this:
        #     id_vars   vars   value |     id_vars   vars   value
        #  0      foo   col3       1 |  0      foo   col5       a    so stacking it into
        #  1      fiz   col3       2 |  1      fiz   col5       b    `new_parts` to get
        #  2      bar   col3       3 |  2      bar   col5       c    correct answer
        #  3      zoo   col3       4 |  3      zoo   col5       d
        new_parts = np.array(
            [np.array([x]) for x in np.concatenate(inconsistent_frame._partitions.T)]
        )
        new_index = pandas.RangeIndex(len(self.index) * len(value_vars))
        new_modin_frame = self._modin_frame.__constructor__(
            new_parts,
            index=new_index,
            columns=id_vars + [var_name, value_name],
        )
        result = self.__constructor__(new_modin_frame)
        # this assigment needs to propagate correct indices into partitions
        result.index = new_index
        return result

    # END Map across rows/columns

    # __getitem__ methods
    def getitem_array(self, key):
        """
        Get column or row data specified by key.

        Parameters
        ----------
        key : PandasQueryCompiler, numpy.ndarray, pandas.Index or list
            Target numeric indices or labels by which to retrieve data.

        Returns
        -------
        PandasQueryCompiler
            A new Query Compiler.
        """
        # TODO: dont convert to pandas for array indexing
        if isinstance(key, type(self)):
            key = key.to_pandas().squeeze(axis=1)
        if is_bool_indexer(key):
            if isinstance(key, pandas.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    "Item wrong length {} instead of {}.".format(
                        len(key), len(self.index)
                    )
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pandas.RangeIndex(len(self.index))[key]
            if len(key):
                return self.getitem_row_array(key)
            else:
                return self.from_pandas(
                    pandas.DataFrame(columns=self.columns), type(self._modin_frame)
                )
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return self.getitem_column_array(key)

    def getitem_column_array(self, key, numeric=False):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.
            numeric: A boolean representing whether or not the key passed in represents
                the numeric index or the named index.

        Returns:
            A new QueryCompiler.
        """
        # Convert to list for type checking
        if numeric:
            new_modin_frame = self._modin_frame.mask(col_numeric_idx=key)
        else:
            new_modin_frame = self._modin_frame.mask(col_indices=key)
        return self.__constructor__(new_modin_frame)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.mask(row_numeric_idx=key))

    def setitem(self, axis, key, value):
        """Set the column defined by `key` to the `value` provided.

        Args:
            key: The column name to set.
            value: The value to set the column to.

        Returns:
             A new QueryCompiler
        """
        return self._setitem(axis=axis, key=key, value=value, how=None)

    def _setitem(self, axis, key, value, how="inner"):
        def setitem_builder(df, internal_indices=[]):
            df = df.copy()
            if len(internal_indices) == 1:
                if axis == 0:
                    df[df.columns[internal_indices[0]]] = value
                else:
                    df.iloc[internal_indices[0]] = value
            else:
                if axis == 0:
                    df[df.columns[internal_indices]] = value
                else:
                    df.iloc[internal_indices] = value
            return df

        if isinstance(value, type(self)):
            value.columns = [key]
            if axis == 1:
                value = value.transpose()
            idx = self.get_axis(axis ^ 1).get_indexer_for([key])[0]
            return self.insert_item(axis ^ 1, idx, value, how, replace=True)

        # TODO: rework by passing list-like values to `_apply_select_indices`
        # as an item to distribute
        if is_list_like(value):
            new_modin_frame = self._modin_frame._apply_full_axis_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        else:
            new_modin_frame = self._modin_frame._apply_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        return self.__constructor__(new_modin_frame)

    # END __getitem__ methods

    # Drop/Dropna
    # This will change the shape of the resulting data.
    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.

        Return:
            a new QueryCompiler
        """

        return self.__constructor__(
            self._modin_frame.filter_full_axis(
                kwargs.get("axis", 0) ^ 1,
                lambda df: pandas.DataFrame.dropna(df, **kwargs),
            )
        )

    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
        if index is not None:
            # The unique here is to avoid duplicating rows with the same name
            index = np.sort(
                self.index.get_indexer_for(self.index[~self.index.isin(index)].unique())
            )
        if columns is not None:
            # The unique here is to avoid duplicating columns with the same name
            columns = np.sort(
                self.columns.get_indexer_for(
                    self.columns[~self.columns.isin(columns)].unique()
                )
            )
        new_modin_frame = self._modin_frame.mask(
            row_numeric_idx=index, col_numeric_idx=columns
        )
        return self.__constructor__(new_modin_frame)

    # END Drop/Dropna

    # Insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    def insert(self, loc, column, value):
        """Insert new column data.

        Args:
            loc: Insertion index.
            column: Column labels to insert.
            value: Dtype object values to insert.

        Returns:
            A new PandasQueryCompiler with new data inserted.
        """

        if isinstance(value, type(self)):
            value.columns = [column]
            return self.insert_item(axis=1, loc=loc, value=value, how=None)

        if is_list_like(value):
            value = list(value)
        else:
            value = [value] * len(self.index)

        def insert(df, internal_indices=[]):
            internal_idx = int(internal_indices[0])
            df.insert(internal_idx, column, value)
            return df

        # TODO: rework by passing list-like values to `_apply_select_indices`
        # as an item to distribute
        new_modin_frame = self._modin_frame._apply_full_axis_select_indices(
            0,
            insert,
            numeric_indices=[loc],
            keep_remaining=True,
            new_index=self.index,
            new_columns=self.columns.insert(loc, column),
        )
        return self.__constructor__(new_modin_frame)

    # END Insert

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        """Apply func across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        # if any of args contain modin object, we should
        # convert it to pandas
        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)
        if isinstance(func, str):
            return self._apply_text_func_elementwise(func, axis, *args, **kwargs)
        elif callable(func):
            return self._callable_func(func, axis, *args, **kwargs)
        elif isinstance(func, dict):
            return self._dict_func(func, axis, *args, **kwargs)
        elif is_list_like(func):
            return self._list_like_func(func, axis, *args, **kwargs)
        else:
            pass

    def _apply_text_func_elementwise(self, func, axis, *args, **kwargs):
        """Apply func passed as str across given axis in elementwise manner.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        assert isinstance(func, str)
        kwargs["axis"] = axis
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, lambda df: df.apply(func, *args, **kwargs)
        )
        return self.__constructor__(new_modin_frame)

    def _dict_func(self, func, axis, *args, **kwargs):
        """Apply function to certain indices across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        def dict_apply_builder(df, func_dict={}):
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            return pandas.DataFrame(df.apply(func_dict, *args, **kwargs))

        func = {k: wrap_udf_function(v) if callable(v) else v for k, v in func.items()}
        return self.__constructor__(
            self._modin_frame._apply_full_axis_select_indices(
                axis, dict_apply_builder, func, keep_remaining=False
            )
        )

    def _list_like_func(self, func, axis, *args, **kwargs):
        """
        Apply list-like function across given axis.

        Parameters
        ----------
            func : list-like
                The function to apply.
            axis : 0 or 1 (0 - index, 1 - columns)
                Target axis to apply the function along.

        Returns
        -------
        PandasQueryCompiler
            A new QueryCompiler.
        """
        # When the function is list-like, the function names become the index/columns
        new_index = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 0
            else self.index
        )
        new_columns = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 1
            else self.columns
        )
        func = [wrap_udf_function(f) if callable(f) else f for f in func]
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def _callable_func(self, func, axis, *args, **kwargs):
        """Apply callable functions across given axis.

        Args:
            func: The functions to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        func = wrap_udf_function(func)
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, lambda df: df.apply(func, axis=axis, *args, **kwargs)
        )
        return self.__constructor__(new_modin_frame)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    groupby_all = GroupbyReduceFunction.register("all")
    groupby_any = GroupbyReduceFunction.register("any")
    groupby_count = GroupbyReduceFunction.register("count")
    groupby_max = GroupbyReduceFunction.register("max")
    groupby_min = GroupbyReduceFunction.register("min")
    groupby_prod = GroupbyReduceFunction.register("prod")
    groupby_size = GroupbyReduceFunction.register("size", method="size")
    groupby_sum = GroupbyReduceFunction.register("sum")

    def _groupby_dict_reduce(
        self, by, axis, agg_func, agg_args, agg_kwargs, groupby_kwargs, drop=False
    ):
        map_dict = {}
        reduce_dict = {}
        rename_columns = any(
            not isinstance(fn, str) and isinstance(fn, Iterable)
            for fn in agg_func.values()
        )
        for col, col_funcs in agg_func.items():
            if not rename_columns:
                map_dict[col], reduce_dict[col] = groupby_reduce_functions[col_funcs]
                continue

            if isinstance(col_funcs, str):
                col_funcs = [col_funcs]

            map_fns = []
            for i, fn in enumerate(col_funcs):
                if not isinstance(fn, str) and isinstance(fn, Iterable):
                    new_col_name, func = fn
                elif isinstance(fn, str):
                    new_col_name, func = fn, fn
                else:
                    raise TypeError

                map_fns.append((new_col_name, groupby_reduce_functions[func][0]))
                reduced_col_name = (
                    (*col, new_col_name)
                    if isinstance(col, tuple)
                    else (col, new_col_name)
                )
                reduce_dict[reduced_col_name] = groupby_reduce_functions[func][1]
            map_dict[col] = map_fns
        return GroupbyReduceFunction.register(map_dict, reduce_dict)(
            query_compiler=self,
            by=by,
            axis=axis,
            groupby_args=groupby_kwargs,
            map_args=agg_kwargs,
            reduce_args=agg_kwargs,
            numeric_only=False,
            drop=drop,
        )

    def groupby_agg(
        self,
        by,
        is_multi_by,
        axis,
        agg_func,
        agg_args,
        agg_kwargs,
        groupby_kwargs,
        drop=False,
    ):
        def is_reduce_fn(fn, deep_level=0):
            if not isinstance(fn, str) and isinstance(fn, Container):
                # `deep_level` parameter specifies the number of nested containers that was met:
                # - if it's 0, then we're outside of container, `fn` could be either function name
                #   or container of function names/renamers.
                # - if it's 1, then we're inside container of function names/renamers. `fn` must be
                #   either function name or renamer (renamer is some container which length == 2,
                #   the first element is the new column name and the second is the function name).
                assert deep_level == 0 or (
                    deep_level > 0 and len(fn) == 2
                ), f"Got the renamer with incorrect length, expected 2 got {len(fn)}."
                return (
                    all(is_reduce_fn(f, deep_level + 1) for f in fn)
                    if deep_level == 0
                    else is_reduce_fn(fn[1], deep_level + 1)
                )
            return isinstance(fn, str) and fn in groupby_reduce_functions

        if isinstance(agg_func, dict) and all(
            is_reduce_fn(x) for x in agg_func.values()
        ):
            return self._groupby_dict_reduce(
                by, axis, agg_func, agg_args, agg_kwargs, groupby_kwargs, drop
            )

        if callable(agg_func):
            agg_func = wrap_udf_function(agg_func)

        # since we're going to modify `groupby_kwargs` dict in a `groupby_agg_builder`,
        # we want to copy it to not propagate these changes into source dict, in case
        # of unsuccessful end of function
        groupby_kwargs = groupby_kwargs.copy()

        as_index = groupby_kwargs.get("as_index", True)
        if isinstance(by, type(self)):
            # `drop` parameter indicates whether or not 'by' data came
            # from the `self` frame:
            # True: 'by' data came from the `self`
            # False: external 'by' data
            if drop:
                internal_by = by.columns
                by = [by]
            else:
                internal_by = []
                by = [by]
        else:
            if not isinstance(by, list):
                by = [by]
            internal_by = [o for o in by if hashable(o) and o in self.columns]
            internal_qc = (
                [self.getitem_column_array(internal_by)] if len(internal_by) else []
            )

            by = internal_qc + by[len(internal_by) :]

        broadcastable_by = [o._modin_frame for o in by if isinstance(o, type(self))]
        not_broadcastable_by = [o for o in by if not isinstance(o, type(self))]

        def groupby_agg_builder(df, by=None, drop=False, partition_idx=None):
            # Set `as_index` to True to track the metadata of the grouping object
            # It is used to make sure that between phases we are constructing the
            # right index and placing columns in the correct order.
            groupby_kwargs["as_index"] = True

            internal_by_cols = pandas.Index([])
            missmatched_cols = pandas.Index([])
            if by is not None:
                internal_by_df = by[internal_by]

                if isinstance(internal_by_df, pandas.Series):
                    internal_by_df = internal_by_df.to_frame()

                missmatched_cols = internal_by_df.columns.difference(df.columns)
                df = pandas.concat(
                    [df, internal_by_df[missmatched_cols]],
                    axis=1,
                    copy=False,
                )
                internal_by_cols = internal_by_df.columns

                external_by = by.columns.difference(internal_by)
                external_by_df = by[external_by].squeeze(axis=1)

                if isinstance(external_by_df, pandas.DataFrame):
                    external_by_cols = [o for _, o in external_by_df.iteritems()]
                else:
                    external_by_cols = [external_by_df]

                by = internal_by_cols.tolist() + external_by_cols

            else:
                by = []

            by += not_broadcastable_by

            def compute_groupby(df, drop=False, partition_idx=0):
                grouped_df = df.groupby(by=by, axis=axis, **groupby_kwargs)
                try:
                    if isinstance(agg_func, dict):
                        # Filter our keys that don't exist in this partition. This happens when some columns
                        # from this original dataframe didn't end up in every partition.
                        partition_dict = {
                            k: v for k, v in agg_func.items() if k in df.columns
                        }
                        result = grouped_df.agg(partition_dict)
                    else:
                        result = agg_func(grouped_df, **agg_kwargs)
                # This happens when the partition is filled with non-numeric data and a
                # numeric operation is done. We need to build the index here to avoid
                # issues with extracting the index.
                except (DataError, TypeError):
                    result = pandas.DataFrame(index=grouped_df.size().index)
                if isinstance(result, pandas.Series):
                    result = result.to_frame(
                        result.name if result.name is not None else "__reduced__"
                    )

                result_cols = result.columns
                result.drop(columns=missmatched_cols, inplace=True, errors="ignore")

                if not as_index:
                    keep_index_levels = len(by) > 1 and any(
                        isinstance(x, pandas.CategoricalDtype)
                        for x in df[internal_by_cols].dtypes
                    )

                    if internal_by_cols.nlevels != result_cols.nlevels:
                        cols_to_insert = (
                            pandas.Index([])
                            if keep_index_levels
                            else internal_by_cols.copy()
                        )
                    else:
                        cols_to_insert = (
                            internal_by_cols.intersection(result_cols)
                            if keep_index_levels
                            else internal_by_cols.difference(result_cols)
                        )

                    if keep_index_levels:
                        result.drop(
                            columns=cols_to_insert, inplace=True, errors="ignore"
                        )

                    drop = True
                    if partition_idx == 0:
                        drop = False
                        if not keep_index_levels:
                            lvls_to_drop = [
                                i
                                for i, name in enumerate(result.index.names)
                                if name not in cols_to_insert
                            ]
                            if len(lvls_to_drop) == result.index.nlevels:
                                drop = True
                            else:
                                result.index = result.index.droplevel(lvls_to_drop)

                    if (
                        not isinstance(result.index, pandas.MultiIndex)
                        and result.index.name is None
                    ):
                        drop = True

                    result.reset_index(drop=drop, inplace=True)

                new_index_names = [
                    None
                    if isinstance(name, str) and name.startswith("__reduced__")
                    else name
                    for name in result.index.names
                ]

                cols_to_drop = (
                    result.columns[result.columns.str.match(r"__reduced__.*", na=False)]
                    if hasattr(result.columns, "str")
                    else []
                )

                result.index.names = new_index_names

                # Not dropping columns if result is Series
                if len(result.columns) > 1:
                    result.drop(columns=cols_to_drop, inplace=True)

                return result

            try:
                return compute_groupby(df, drop, partition_idx)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except (ValueError, KeyError):
                return compute_groupby(df.copy(), drop, partition_idx)

        apply_indices = list(agg_func.keys()) if isinstance(agg_func, dict) else None

        new_modin_frame = self._modin_frame.broadcast_apply_full_axis(
            axis=axis,
            func=lambda df, by=None, partition_idx=None: groupby_agg_builder(
                df, by, drop, partition_idx
            ),
            other=broadcastable_by,
            apply_indices=apply_indices,
            enumerate_partitions=True,
        )
        result = self.__constructor__(new_modin_frame)

        # that means that exception in `compute_groupby` was raised
        # in every partition, so we also should raise it
        if len(result.columns) == 0 and len(self.columns) != 0:
            # determening type of raised exception by applying `aggfunc`
            # to empty DataFrame
            try:
                pandas.DataFrame(index=[1], columns=[1]).agg(agg_func) if isinstance(
                    agg_func, dict
                ) else agg_func(
                    pandas.DataFrame(index=[1], columns=[1]).groupby(level=0),
                    **agg_kwargs,
                )
            except Exception as e:
                raise type(e)("No numeric types to aggregate.")

        return result

    # END Manual Partitioning methods

    def pivot(self, index, columns, values):
        from pandas.core.reshape.pivot import _convert_by

        def __convert_by(by):
            if isinstance(by, pandas.Index):
                by = list(by)
            by = _convert_by(by)
            if (
                len(by) > 0
                and (not is_list_like(by[0]) or isinstance(by[0], tuple))
                and not all([key in self.columns for key in by])
            ):
                by = [by]
            return by

        index, columns, values = map(__convert_by, [index, columns, values])
        is_custom_index = (
            len(index) == 1
            and is_list_like(index[0])
            and not isinstance(index[0], tuple)
        )

        if is_custom_index or len(index) == 0:
            to_reindex = columns
        else:
            to_reindex = index + columns

        if len(values) != 0:
            obj = self.getitem_column_array(to_reindex + values)
        else:
            obj = self

        if is_custom_index:
            obj.index = index

        reindexed = self.__constructor__(
            obj._modin_frame._apply_full_axis(
                1,
                lambda df: df.set_index(to_reindex, append=(len(to_reindex) == 1)),
                new_columns=obj.columns.drop(to_reindex),
            )
        )

        unstacked = reindexed.unstack(level=columns, fill_value=None)
        if len(reindexed.columns) == 1 and unstacked.columns.nlevels > 1:
            unstacked.columns = unstacked.columns.droplevel(0)

        return unstacked

    def pivot_table(
        self,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
    ):
        ErrorMessage.missmatch_with_pandas(
            operation="pivot_table",
            message="Order of columns could be different from pandas",
        )

        from pandas.core.reshape.pivot import _convert_by

        def __convert_by(by):
            if isinstance(by, pandas.Index):
                return list(by)
            return _convert_by(by)

        index, columns, values = map(__convert_by, [index, columns, values])

        unique_keys = np.unique(index + columns)
        unique_values = np.unique(values)

        if len(values):
            to_group = self.getitem_column_array(unique_values)
        else:
            to_group = self.drop(columns=unique_keys)

        keys_columns = self.getitem_column_array(unique_keys)

        def applyier(df, other):
            concated = pandas.concat([df, other], axis=1, copy=False)
            result = concated.pivot_table(
                index=index,
                values=values if len(values) > 0 else None,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=fill_value,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
            )

            # in that case Pandas transposes the result of `pivot_table`,
            # transposing it back to be consistent with column axis values along
            # different partitions
            if len(index) == 0 and len(columns) > 0:
                result = result.T

            return result

        result = self.__constructor__(
            to_group._modin_frame.broadcast_apply_full_axis(
                axis=0, func=applyier, other=keys_columns._modin_frame
            )
        )

        # transposing the result again, to be consistent with Pandas result
        if len(index) == 0 and len(columns) > 0:
            result = result.transpose()

        if len(values) == 0:
            values = self.columns.drop(unique_keys)

        # if only one value is specified, removing level that maps
        # columns from `values` to the actual values
        if len(index) > 0 and len(values) == 1 and result.columns.nlevels > 1:
            result.columns = result.columns.droplevel(int(margins))

        return result

    # Get_dummies
    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """
        # `columns` as None does not mean all columns, by default it means only
        # non-numeric columns.
        if columns is None:
            columns = [c for c in self.columns if not is_numeric_dtype(self.dtypes[c])]
            # If we aren't computing any dummies, there is no need for any
            # remote compute.
            if len(columns) == 0:
                return self.copy()
        elif not is_list_like(columns):
            columns = [columns]

        # In some cases, we are mapping across all of the data. It is more
        # efficient if we are mapping over all of the data to do it this way
        # than it would be to reuse the code for specific columns.
        if len(columns) == len(self.columns):
            new_modin_frame = self._modin_frame._apply_full_axis(
                0, lambda df: pandas.get_dummies(df, **kwargs), new_index=self.index
            )
            untouched_frame = None
        else:
            new_modin_frame = self._modin_frame.mask(
                col_indices=columns
            )._apply_full_axis(
                0, lambda df: pandas.get_dummies(df, **kwargs), new_index=self.index
            )
            untouched_frame = self.drop(columns=columns)
        # If we mapped over all the data we are done. If not, we need to
        # prepend the `new_modin_frame` with the raw data from the columns that were
        # not selected.
        if len(columns) != len(self.columns):
            new_modin_frame = untouched_frame._modin_frame._concat(
                1, [new_modin_frame], how="left", sort=False
            )
        return self.__constructor__(new_modin_frame)

    # END Get_dummies

    # Indexing
    def view(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.mask(row_numeric_idx=index, col_numeric_idx=columns)
        )

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            partition.iloc[row_internal_indices, col_internal_indices] = item
            return partition

        new_modin_frame = self._modin_frame._apply_select_indices(
            axis=None,
            func=iloc_mut,
            row_indices=row_numeric_index,
            col_indices=col_numeric_index,
            new_index=self.index,
            new_columns=self.columns,
            keep_remaining=True,
            item_to_distribute=broadcasted_items,
        )
        return self.__constructor__(new_modin_frame)

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        """Reorder the rows based on the lexicographic order of the given columns.

        Parameters
        ----------
        columns : scalar or list of scalar
            The column or columns to sort by
        ascending : bool
            Sort in ascending order (True) or descending order (False)

        Returns
        -------
        PandasQueryCompiler
            A new query compiler that contains result of the sort
        """
        na_position = kwargs.get("na_position", "last")
        kind = kwargs.get("kind", "quicksort")
        if not is_list_like(columns):
            columns = [columns]
        # Currently, sort_values will just reindex based on the sorted values.
        # TODO create a more efficient way to sort
        ErrorMessage.default_to_pandas("sort_values")
        broadcast_value_dict = {
            col: self.getitem_column_array([col]).to_pandas().squeeze(axis=1)
            for col in columns
        }
        # Index may contain duplicates
        broadcast_values1 = pandas.DataFrame(broadcast_value_dict, index=self.index)
        # Index without duplicates
        broadcast_values2 = pandas.DataFrame(broadcast_value_dict)
        broadcast_values2 = broadcast_values2.reset_index(drop=True)
        # Index may contain duplicates
        new_index1 = broadcast_values1.sort_values(
            by=columns,
            axis=0,
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        ).index
        # Index without duplicates
        new_index2 = broadcast_values2.sort_values(
            by=columns,
            axis=0,
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        ).index

        result = self.reset_index(drop=True).reindex(axis=0, labels=new_index2)
        result.index = new_index1
        return result

    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        """Reorder the columns based on the lexicographic order of the given rows.

        Parameters
        ----------
        rows : scalar or list of scalar
            The row or rows to sort by
        ascending : bool
            Sort in ascending order (True) or descending order (False)

        Returns
        -------
        PandasQueryCompiler
            A new query compiler that contains result of the sort
        """
        na_position = kwargs.get("na_position", "last")
        kind = kwargs.get("kind", "quicksort")
        if not is_list_like(rows):
            rows = [rows]
        ErrorMessage.default_to_pandas("sort_values")
        broadcast_value_list = [
            self.getitem_row_array([row]).to_pandas() for row in rows
        ]
        index_builder = list(zip(broadcast_value_list, rows))
        broadcast_values = pandas.concat(
            [row for row, idx in index_builder], copy=False
        )
        broadcast_values.columns = self.columns
        new_columns = broadcast_values.sort_values(
            by=rows,
            axis=1,
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        ).columns
        return self.reindex(axis=1, labels=new_columns)

    # Cat operations
    def cat_codes(self):
        return self.default_to_pandas(lambda df: df[df.columns[0]].cat.codes)

    # END Cat operations

    def compare(self, other, **kwargs):
        return self.__constructor__(
            self._modin_frame.broadcast_apply_full_axis(
                0,
                lambda l, r: pandas.DataFrame.compare(l, r, **kwargs),
                other._modin_frame,
            )
        )
