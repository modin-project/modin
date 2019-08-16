from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas

from pandas.compat import string_types
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
)
from pandas.core.base import DataError

from modin.backends.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.data_management.functions import MapFunction


def _get_axis(axis):
    if axis == 0:
        return lambda self: self._data_obj.index
    else:
        return lambda self: self._data_obj.columns


def _set_axis(axis):
    if axis == 0:
        def set_idx(self, idx):
            self._data_obj.index = idx

        return set_idx
    if axis == 1:
        def set_cols(self, cols):
            self._data_obj.columns = cols

        return set_cols


def _str_map(func):
    def str_op_builder(df, *args, **kwargs):
        str_series = df.squeeze().str
        return func(str_series, *args, **kwargs).to_frame()
    return str_op_builder


class PandasQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
        with a Pandas backend. This logic is specific to Pandas."""

    def __init__(self, data_object):
        self._data_obj = data_object

    def to_pandas(self):
        return self._data_obj.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._data_obj.dtypes

    def compute_index(self, axis, data_object, compute_diff=True):
        """Computes the index after a number of rows have been removed.

        Note: In order for this to be used properly, the indexes must not be
            changed before you compute this.

        Args:
            axis: The axis to extract the index from.
            data_object: The new data object to extract the index from.
            compute_diff: True to use `self` to compute the index from self
                rather than data_object. This is used when the dimension of the
                index may have changed, but the deleted rows/columns are
                unknown.

        Returns:
            A new pandas.Index object.
        """

        def pandas_index_extraction(df, axis):
            if not axis:
                return df.index
            else:
                try:
                    return df.columns
                except AttributeError:
                    return pandas.Index([])

        index_obj = self.index if not axis else self.columns
        old_blocks = self.data if compute_diff else None
        new_indices = data_object.get_indices(
            axis=axis,
            index_func=lambda df: pandas_index_extraction(df, axis),
            old_blocks=old_blocks,
        )
        return index_obj[new_indices] if compute_diff else new_indices

    # END Index, columns, and dtypes objects

    def numeric_function_clean_dataframe(self, axis):
        """Preprocesses numeric functions to clean dataframe and pick numeric indices.

        Args:
            axis: '0' if columns and '1' if rows.

        Returns:
            Tuple with return value(if any), indices to apply func to & cleaned Manager.
        """
        result = None
        query_compiler = self
        # If no numeric columns and over columns, then return empty Series
        if not axis and len(self.index) == 0:
            result = pandas.Series(dtype=np.int64)

        nonnumeric = [
            col
            for col, dtype in zip(self.columns, self.dtypes)
            if not is_numeric_dtype(dtype)
        ]
        if len(nonnumeric) == len(self.columns):
            # If over rows and no numeric columns, return this
            if axis:
                result = pandas.Series([np.nan for _ in self.index])
            else:
                result = pandas.Series([0 for _ in self.index])
        else:
            query_compiler = self.drop(columns=nonnumeric)
        return result, query_compiler

    # END Internal methods

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._data_obj.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._data_obj.add_suffix(suffix, axis))

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(self._data_obj.copy())

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
        assert all(isinstance(o, type(self)) for o in other), "Different Manager objects are being used. This is not allowed"
        sort = kwargs.get("sort", None)
        if sort is None:
            sort = False
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        other_data = [o._data_obj for o in other]
        new_data = self._data_obj._concat(axis, other_data, join, sort)
        if ignore_index:
            new_data.index = pandas.RangeIndex(len(self.index) + sum(len(o.index) for o in other))
        return self.__constructor__(new_data)

    # END Append/Concat/Join

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
        # TODO create a way to clean up this object.
        return

    # END Data Management Methods

    # To NumPy
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy Array.

        Returns:
            NumPy Array of the QueryCompiler.
        """
        arr = self._data_obj.to_numpy()
        ErrorMessage.catch_bugs_and_request_email(
            len(arr) != len(self.index) or len(arr[0]) != len(self.columns)
        )
        return arr

    # END To NumPy

    # Inter-Data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.
    def _inter_manager_operations(self, other, how_to_join, func):
        """Inter-data operations (e.g. add, sub).

        Args:
            other: The other Manager for the operation.
            how_to_join: The type of join to join to make (e.g. right, outer).

        Returns:
            New QueryCompiler with new data and index.
        """

        new_data = self._data_obj._binary_op(func, other)
        return self.__constructor__(new_data)

    def _inter_df_op_handler(self, func, other, **kwargs):
        """Helper method for inter-manager and scalar operations.

        Args:
            func: The function to use on the Manager/scalar.
            other: The other Manager/scalar.

        Returns:
            New QueryCompiler with new data and index.
        """
        axis = kwargs.get("axis", 0)
        if isinstance(other, type(self)):
            return self.__constructor__(self._data_obj._binary_op(
                lambda x, y: func(x, y, **kwargs), other._data_obj
            ))
        else:
            if isinstance(other, (list, np.ndarray, pandas.Series)):
                if axis == 1 and isinstance(other, pandas.Series):
                    new_columns = self.columns.join(other.index, how="outer")
                else:
                    new_columns = self.columns
                new_data = self._data_obj._apply_full_axis(axis, lambda df: func(df, other, **kwargs), new_index=self.index, new_columns=new_columns)
            else:
                new_data = self._data_obj._map_partitions(lambda df: func(df, other, **kwargs))
            return self.__constructor__(new_data)

    def binary_op(self, op, other, **kwargs):
        """Perform an operation between two objects.

        Note: The list of operations is as follows:
            - add
            - eq
            - floordiv
            - ge
            - gt
            - le
            - lt
            - mod
            - mul
            - ne
            - pow
            - rfloordiv
            - rmod
            - rpow
            - rsub
            - rtruediv
            - sub
            - truediv
            - __and__
            - __or__
            - __xor__
        Args:
            op: The operation. See list of operations above
            other: The object to operate against.

        Returns:
            A new QueryCompiler object.
        """
        func = getattr(pandas.DataFrame, op)
        return self._inter_df_op_handler(func, other, **kwargs)

    def clip(self, lower, upper, **kwargs):
        kwargs["upper"] = upper
        kwargs["lower"] = lower
        axis = kwargs.get("axis", 0)
        if is_list_like(lower) or is_list_like(upper):
            new_data = self._data_obj._map_across_full_axis(axis, lambda df: df.clip(**kwargs))
        else:
            new_data = self._data_obj._map_partitions(lambda df: df.clip(**kwargs))
        return self.__constructor__(new_data)

    def update(self, other, **kwargs):
        """Uses other manager to update corresponding values in this manager.

        Args:
            other: The other manager.

        Returns:
            New QueryCompiler with updated data and index.
        """
        assert isinstance(
            other, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"

        def update_builder(df, other, **kwargs):
            # This is because of a requirement in Arrow
            df = df.copy()
            df.update(other, **kwargs)
            return df

        return self._inter_df_op_handler(update_builder, other, **kwargs)

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

            def where_builder_second_pass(df, new_other, **kwargs):
                return df.where(new_other.eq(True), new_other, **kwargs)

            first_pass = cond._inter_manager_operations(
                other, "left", where_builder_first_pass
            )
            final_pass = self._inter_manager_operations(
                first_pass, "left", where_builder_second_pass
            )
            return self.__constructor__(final_pass.data, self.index, self.columns)
        else:
            axis = kwargs.get("axis", 0)
            # Rather than serializing and passing in the index/columns, we will
            # just change this index to match the internal index.
            if isinstance(other, pandas.Series):
                other.index = pandas.RangeIndex(len(other.index))

            def where_builder_series(df, cond):
                if axis == 0:
                    df.index = pandas.RangeIndex(len(df.index))
                    cond.index = pandas.RangeIndex(len(cond.index))
                else:
                    df.columns = pandas.RangeIndex(len(df.columns))
                    cond.columns = pandas.RangeIndex(len(cond.columns))
                return df.where(cond, other, **kwargs)

            reindexed_self, reindexed_cond, a = self.copartition(
                axis, cond, "left", False
            )
            # Unwrap from list given by `copartition`
            reindexed_cond = reindexed_cond[0]
            new_data = reindexed_self.inter_data_operation(
                axis, lambda l, r: where_builder_series(l, r), reindexed_cond
            )
            return self.__constructor__(new_data, self.index, self.columns)

    # END Inter-Data operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manger.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            A new QueryCompiler with updated data and new index.
        """
        # To reindex, we need a function that will be shipped to each of the
        # partitions.
        def reindex_builer(df, axis, old_labels, new_labels, **kwargs):
            if axis:
                while len(df.columns) < len(old_labels):
                    df[len(df.columns)] = np.nan
                df.columns = old_labels
                new_df = df.reindex(columns=new_labels, **kwargs)
                # reset the internal columns back to a RangeIndex
                new_df.columns = pandas.RangeIndex(len(new_df.columns))
                return new_df
            else:
                while len(df.index) < len(old_labels):
                    df.loc[len(df.index)] = np.nan
                df.index = old_labels
                new_df = df.reindex(index=new_labels, **kwargs)
                # reset the internal index back to a RangeIndex
                new_df.reset_index(inplace=True, drop=True)
                return new_df

        old_labels = self.columns if axis else self.index
        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns
        func = self._prepare_method(
            lambda df: reindex_builer(df, axis, old_labels, labels, **kwargs)
        )
        # The reindex can just be mapped over the axis we are modifying. This
        # is for simplicity in implementation. We specify num_splits here
        # because if we are repartitioning we should (in the future).
        # Additionally this operation is often followed by an operation that
        # assumes identical partitioning. Internally, we *may* change the
        # partitioning during a map across a full axis.
        new_data = self._map_across_full_axis(axis, func)
        return self.__constructor__(new_data, new_index, new_columns)

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            A new QueryCompiler with updated data and reset index.
        """
        drop = kwargs.get("drop", False)
        new_index = pandas.RangeIndex(len(self.index))
        if not drop:
            if isinstance(self.index, pandas.MultiIndex):
                # TODO (devin-petersohn) ensure partitioning is properly aligned
                new_column_names = pandas.Index(self.index.names)
                new_columns = new_column_names.append(self.columns)
                index_data = pandas.DataFrame(list(zip(*self.index))).T
                result = self.data.from_pandas(index_data).concat(1, self.data)
                return self.__constructor__(result, new_index, new_columns)
            else:
                new_column_name = (
                    self.index.name
                    if self.index.name is not None
                    else "index"
                    if "index" not in self.columns
                    else "level_0"
                )
                new_columns = self.columns.insert(0, new_column_name)
                result = self.insert(0, new_column_name, self.index)
                return self.__constructor__(result.data, new_index, new_columns)
        else:
            # The copies here are to ensure that we do not give references to
            # this object for the purposes of updates.
            return self.__constructor__(
                self.data.copy(), new_index, self.columns.copy(), self._dtype_cache
            )

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
        return self.__constructor__(self._data_obj.transpose())

    # END Transpose

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # This will return a new QueryCompiler, which will be handled in the front end.

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            A new QueryCompiler object containing counts of non-NaN objects from each
            column or row.
        """
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_reduce(axis, lambda df: df.count(**kwargs), lambda df: df.sum(**kwargs))

    def dot(self, other):
        """Computes the matrix multiplication of self and other.

        Args:
            other: The other query compiler or other array-like to matrix
            multiply with self.

        Returns:
            Returns the result of the matrix multiply.
        """

        def map_func(df, other=other):
            if isinstance(other, pandas.DataFrame):
                other = other.squeeze()
            result = df.squeeze().dot(other)
            if is_list_like(result):
                return pandas.DataFrame(result)
            else:
                return pandas.DataFrame([result])

        if isinstance(other, BaseQueryCompiler):
            if len(self.columns) > 1 and len(other.columns) == 1:
                # If self is DataFrame and other is a series, we take the transpose
                # to copartition along the columns.
                new_self = self
                other = other.transpose()
                axis = 1
                new_index = self.index
            elif len(self.columns) == 1 and len(other.columns) > 1:
                # If self is series and other is a Dataframe, we take the transpose
                # to copartition along the columns.
                new_self = self.transpose()
                axis = 1
                new_index = self.index
            elif len(self.columns) == 1 and len(other.columns) == 1:
                # If both are series, then we copartition along the rows.
                new_self = self
                axis = 0
                new_index = ["__reduce__"]
            new_self, list_of_others, _ = new_self.copartition(
                axis, other, "left", False
            )
            other = list_of_others[0]
            reduce_func = self._build_mapreduce_func(
                pandas.DataFrame.sum, axis=axis, skipna=False
            )
            new_data = new_self.groupby_reduce(axis, other, map_func, reduce_func)
        else:
            if len(self.columns) == 1:
                axis = 0
                new_index = ["__reduce__"]
            else:
                axis = 1
                new_index = self.index
            new_data = self.data.map_across_full_axis(axis, map_func)
        return self.__constructor__(new_data, index=new_index, columns=["__reduced__"])

    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            A new QueryCompiler object with the maximum values from each column or row.
        """
        return self._data_obj._full_reduce(kwargs.get("axis", 0), lambda df: df.max(**kwargs))

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            A new QueryCompiler object containing the mean from each numerical column or
            row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        sums = self.sum(**kwargs)
        counts = self.count(axis=axis, numeric_only=kwargs.get("numeric_only", None))
        return sums.binary_op("truediv", counts, axis=axis)

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            A new QueryCompiler object with the minimum value from each column or row.
        """
        return self._data_obj._full_reduce(kwargs.get("axis", 0), lambda df: df.min(**kwargs))

    def _process_sum_prod(self, func, **kwargs):
        """Calculates the sum or product of the DataFrame.

        Args:
            func: Pandas func to apply to DataFrame.
            ignore_axis: Whether to ignore axis when raising TypeError
        Return:
            A new QueryCompiler object with sum or prod of the object.
        """
        axis = kwargs.get("axis", 0)
        min_count = kwargs.get("min_count", 0)
        if min_count <= 1:
            return self.__constructor__(self._data_obj._full_reduce(axis, lambda df: func(df, **kwargs)))
        else:
            return self.__constructor__(self._data_obj._full_axis_reduce(axis, lambda df: func(df, **kwargs)))

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            A new QueryCompiler object with the product of each numerical column or row.
        """
        return self._process_sum_prod(pandas.DataFrame.prod, **kwargs)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            A new QueryCompiler object with the sum of each numerical column or row.
        """
        return self._process_sum_prod(pandas.DataFrame.sum, **kwargs)

    def _process_all_any(self, func, **kwargs):
        """Calculates if any or all the values are true.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        axis = kwargs.get("axis", 0)
        axis = 0 if axis is None else axis
        kwargs["axis"] = axis
        return self.__constructor__(self._data_obj._full_reduce(axis, lambda df: func(df, **kwargs)))

    def all(self, **kwargs):
        """Returns whether all the elements are true, potentially over an axis.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        return self._process_all_any(lambda df, **kwargs: df.all(**kwargs), **kwargs)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        return self._process_all_any(lambda df, **kwargs: df.any(**kwargs), **kwargs)

    def memory_usage(self, axis=0, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            A new QueryCompiler object containing the memory usage of each column.
        """

        def memory_usage_builder(df):
            # We have to manually change the orientation of the data within the
            # partitions because memory_usage does not take in an axis argument
            # and always does it along columns.
            if axis:
                df = df.T
            result = df.memory_usage(**kwargs)
            return result

        def sum_memory_usage(df):
            return df.sum(axis=axis)

        return self._data_obj._full_reduce(axis, memory_usage_builder, sum_memory_usage)

    # END Full Reduce operations

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    abs = MapFunction.register(pandas.DataFrame.abs, dtypes="copy")
    applymap = MapFunction.register(pandas.DataFrame.applymap)
    invert = MapFunction.register(pandas.DataFrame.__invert__)
    isin = MapFunction.register(pandas.DataFrame.isin, dtypes=np.bool)
    isna = MapFunction.register(pandas.DataFrame.isna, dtypes=np.bool)
    negative = MapFunction.register(pandas.DataFrame.__neg__)
    notna = MapFunction.register(pandas.DataFrame.notna, dtypes=np.bool)
    round = MapFunction.register(pandas.DataFrame.round)

    # END Map partitions operations

    # This is here to shorten the call to pandas
    str_ops = pandas.Series.str

    str_capitalize = MapFunction.register(_str_map(str_ops.capitalize), dtypes="copy")
    str_center = MapFunction.register(_str_map(str_ops.center), dtypes="copy")
    str_contains = MapFunction.register(_str_map(str_ops.contains), dtypes=np.bool)
    str_count = MapFunction.register(_str_map(str_ops.count), dtypes=int)
    str_endswith = MapFunction.register(_str_map(str_ops.endswith), dtypes=np.bool)
    str_find = MapFunction.register(_str_map(str_ops.find), dtypes="copy")
    str_findall = MapFunction.register(_str_map(str_ops.findall), dtypes="copy")
    str_get = MapFunction.register(_str_map(str_ops.get), dtypes="copy")
    str_index = MapFunction.register(_str_map(str_ops.index), dtypes="copy")
    str_isalnum = MapFunction.register(_str_map(str_ops.isalnum), dtypes=np.bool)
    str_isalpha = MapFunction.register(_str_map(str_ops.isalpha), dtypes=np.bool)
    str_isdecimal = MapFunction.register(_str_map(str_ops.isdecimal), dtypes=np.bool)
    str_isdigit = MapFunction.register(_str_map(str_ops.isdigit), dtypes=np.bool)
    str_islower = MapFunction.register(_str_map(str_ops.islower), dtypes=np.bool)
    str_isnumeric = MapFunction.register(_str_map(str_ops.isnumeric), dtypes=np.bool)
    str_isspace = MapFunction.register(_str_map(str_ops.isspace), dtypes=np.bool)
    str_istitle = MapFunction.register(_str_map(str_ops.istitle), dtypes=np.bool)
    str_isupper = MapFunction.register(_str_map(str_ops.isupper), dtypes=np.bool)
    str_join = MapFunction.register(_str_map(str_ops.join), dtypes="copy")
    str_len = MapFunction.register(_str_map(str_ops.len), dtypes=int)
    str_ljust = MapFunction.register(_str_map(str_ops.ljust), dtypes="copy")
    str_lower = MapFunction.register(_str_map(str_ops.lower), dtypes="copy")
    str_lstrip = MapFunction.register(_str_map(str_ops.lstrip), dtypes="copy")
    str_match = MapFunction.register(_str_map(str_ops.match), dtypes="copy")
    str_normalize = MapFunction.register(_str_map(str_ops.normalize), dtypes="copy")
    str_pad = MapFunction.register(_str_map(str_ops.pad), dtypes="copy")
    str_partition = MapFunction.register(_str_map(str_ops.partition), dtypes="copy")
    str_repeat = MapFunction.register(_str_map(str_ops.repeat), dtypes="copy")
    str_replace = MapFunction.register(_str_map(str_ops.replace), dtypes="copy")
    str_rfind = MapFunction.register(_str_map(str_ops.rfind), dtypes="copy")
    str_rindex = MapFunction.register(_str_map(str_ops.rindex), dtypes="copy")
    str_rjust = MapFunction.register(_str_map(str_ops.rjust), dtypes="copy")
    str_rpartition = MapFunction.register(_str_map(str_ops.rpartition), dtypes="copy")
    str_rsplit = MapFunction.register(_str_map(str_ops.rsplit), dtypes="copy")
    str_rstrip = MapFunction.register(_str_map(str_ops.rstrip), dtypes="copy")
    str_slice = MapFunction.register(_str_map(str_ops.slice), dtypes="copy")
    str_slice_replace = MapFunction.register(_str_map(str_ops.slice_replace), dtypes="copy")
    str_split = MapFunction.register(_str_map(str_ops.split), dtypes="copy")
    str_startswith = MapFunction.register(_str_map(str_ops.startswith), dtypes=np.bool)
    str_strip = MapFunction.register(_str_map(str_ops.strip), dtypes="copy")
    str_swapcase = MapFunction.register(_str_map(str_ops.swapcase), dtypes="copy")
    str_translate = MapFunction.register(_str_map(str_ops.translate), dtypes="copy")
    str_upper = MapFunction.register(_str_map(str_ops.upper), dtypes="copy")
    str_wrap = MapFunction.register(_str_map(str_ops.wrap), dtypes="copy")
    str_zfill = MapFunction.register(_str_map(str_ops.zfill), dtypes="copy")

    # END String map partitions operations

    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return self.__constructor__(self._data_obj.astype(col_dtypes))

    # Column/Row partitions reduce operations

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        def first_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.first_valid_index())

        # We get the minimum from each column, then take the min of that to get
        # first_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = self.__constructor__(self._data_obj._full_axis_reduce(0, first_valid_index_builder)).min(axis=1).to_pandas().squeeze()
        return self.index[first_result]

    def idxmax(self, **kwargs):
        """Returns the first occurrence of the maximum over requested axis.

        Returns:
            A new QueryCompiler object containing the maximum of each column or axis.
        """
        axis = kwargs.get("axis", 0)
        return self.__constructor__(self._data_obj._full_axis_reduce(axis, lambda df: df.idxmax(**kwargs)))

    def idxmin(self, **kwargs):
        """Returns the first occurrence of the minimum over requested axis.

        Returns:
            A new QueryCompiler object containing the minimum of each column or axis.
        """
        axis = kwargs.get("axis", 0)
        return self.__constructor__(self._data_obj._full_axis_reduce(axis, lambda df: df.idxmin(**kwargs)))

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def last_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.last_valid_index())

        # We get the maximum from each column, then take the max of that to get
        # last_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = self.__constructor__(self._data_obj._full_axis_reduce(0, last_valid_index_builder)).max(axis=1).to_pandas().squeeze()
        return self.index[first_result]

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            A new QueryCompiler object containing the median of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_axis_reduce(axis, lambda df: df.median(**kwargs))

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            A new QueryCompiler object of ints indexed by column or index names.
        """
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_axis_reduce(axis, lambda df: df.nunique(**kwargs))

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            A new QueryCompiler object containing the quantile of each column or row.
        """
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q", 0.5)
        assert type(q) is float

        def quantile_builder(df):
            try:
                return pandas.DataFrame.quantile(df, **kwargs)
            except ValueError:
                return pandas.Series()

        result = self._data_obj._full_axis_reduce(axis, quantile_builder)
        if axis == 0:
            result.index = [q]
        else:
            result.columns = [q]
        return self.__constructor__(result)

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            A new QueryCompiler object containing the skew of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_axis_reduce(axis, lambda df: df.skew(**kwargs))

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            A new QueryCompiler object containing the standard deviation of each column
            or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_axis_reduce(axis, lambda df: df.std(**kwargs))

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            A new QueryCompiler object containing the variance of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        return self._data_obj._full_axis_reduce(axis, lambda df: df.var(**kwargs))

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

        def describe_builder(df, internal_indices=[]):
            return df.iloc[:, internal_indices].describe(**kwargs)

        return self.__constructor__(self._data_obj._apply_full_axis_select_indices(0, describe_builder,
                                                                                               empty_df.columns,
                                                                                               new_idx=empty_df.index))

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    def _cumulative_builder(self, func, **kwargs):
        axis = kwargs.get("axis", 0)
        new_data = self._data_obj._map_across_full_axis(axis, lambda df: func(df, **kwargs))
        return self.__constructor__(new_data)

    def cummax(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummax, **kwargs)

    def cummin(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummin, **kwargs)

    def cumsum(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumsum, **kwargs)

    def cumprod(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumprod, **kwargs)

    def diff(self, **kwargs):
        axis = kwargs.get("axis", 0)
        return self.__constructor__(self._data_obj._map_across_full_axis(axis, lambda df: df.diff(**kwargs)))

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        empty_eval = pandas.DataFrame(columns=self.columns).astype(self.dtypes).eval(expr, inplace=False, **kwargs)
        expect_series = isinstance(empty_eval, pandas.Series)

        def eval_builder(df, **kwargs):
            return pandas.DataFrame(df.eval(expr, inplace=False, **kwargs))

        new_data = self._data_obj._apply_full_axis(1, eval_builder, new_index=self.index, new_columns=[empty_eval.name] if expect_series else empty_eval.columns)
        return self.__constructor__(new_data)

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        axis = kwargs.get("axis", 0)

        def mode_builder(df, **kwargs):
            result = df.mode(**kwargs)
            # We return a dataframe with the same shape as the input to ensure
            # that all the partitions will be the same shape
            if not axis and len(df) != len(result):
                # Pad rows
                result = result.reindex(index=df.index)
            elif axis and len(df.columns) != len(result.columns):
                # Pad columns
                result = result.reindex(columns=df.columns)
            return pandas.DataFrame(result)

        new_data = self._data_obj._map_across_full_axis(axis, mode_builder)
        if axis == 0:
            new_data.index = pandas.RangeIndex(len(self.index))
        else:
            new_data.columns = pandas.RangeIndex(len(self.columns))
        return self.__constructor__(new_data).dropna(axis=axis, how="all")

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
            raise NotImplementedError("FIXME")
            kwargs.pop("value")

            if axis == 0:
                index = self.columns
            else:
                index = self.index
            value = {
                idx: value[key] for key in value for idx in index.get_indexer_for([key])
            }

            def fillna_dict_builder(df, func_dict={}):
                # We do this to ensure that no matter the state of the columns we get
                # the correct ones.
                func_dict = {df.columns[idx]: func_dict[idx] for idx in func_dict}
                return df.fillna(value=func_dict, **kwargs)

            if full_axis:
                new_data = self.data.apply_func_to_select_indices_along_full_axis(
                    axis, fillna_dict_builder, value, keep_remaining=True
                )
            else:
                new_data = self.data.apply_func_to_select_indices(
                    axis, fillna_dict_builder, value, keep_remaining=True
                )
        else:
            if full_axis:
                new_data = self._data_obj._map_across_full_axis(axis, lambda df: df.fillna(**kwargs))
            else:
                new_data = self._data_obj._map_partitions(lambda df: df.fillna(**kwargs))
        return self.__constructor__(new_data)

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
            new_columns = self._data_obj._numeric_columns()
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
        new_data = query_compiler._data_obj._apply_full_axis(axis, lambda df: quantile_builder(df, **kwargs), new_index=q_index, new_columns=new_columns, new_dtypes=np.float64)
        result = self.__constructor__(new_data)
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

        return self.__constructor__(self._data_obj._apply_full_axis(1, query_builder, new_columns=self.columns))

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        new_data = self._data_obj._apply_full_axis(axis, lambda df: df.rank(**kwargs), new_index=self.index, new_columns=self.columns if not numeric_only else None, dtypes=np.float64)
        return self.__constructor__(new_data)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        axis = kwargs.pop("axis", 0)
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
        new_data = self._data_obj._apply_full_axis(axis, lambda df: df.sort_index(axis=axis, **kwargs), new_index, new_columns, dtypes="copy" if axis == 0 else None)
        return self.__constructor__(new_data)

    # END Map across rows/columns

    # Head/Tail/Front/Back
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
        """
        return self.__constructor__(self._data_obj.head(n))

    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        return self.__constructor__(self._data_obj.tail(n))

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
        """
        return self.__constructor__(self._data_obj.front(n))

    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the last n columns of the original QueryCompiler.
        """
        return self.__constructor__(self._data_obj.back(n))

    # End Head/Tail/Front/Back

    # __getitem__ methods
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
            new_data = self._data_obj.mask(col_numeric_idx=key)
        else:
            new_data = self._data_obj.mask(col_indices=key)
        return self.__constructor__(new_data)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new QueryCompiler.
        """
        return self.__constructor__(self._data_obj.mask(row_numeric_idx=key))

    def setitem(self, axis, key, value):
        """Set the column defined by `key` to the `value` provided.

        Args:
            key: The column name to set.
            value: The value to set the column to.

        Returns:
             A new QueryCompiler
        """

        def setitem(df, internal_indices=[]):
            def _setitem():
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

            try:
                _setitem()
            except ValueError:
                # TODO: This is a workaround for a pyarrow serialization issue
                df = df.copy()
                _setitem()
            return df

        if is_list_like(value):
            new_data = self._data_obj._apply_full_axis_select_indices(
                axis, setitem, [key], new_idx=self.index, keep_remaining=True
            )
        else:
            new_data = self._data_obj._apply_select_indices(
                axis, setitem, [key], new_idx=self.index, keep_remaining=True
            )
        return self.__constructor__(new_data)

    # END __getitem__ methods

    # Drop/Dropna
    # This will change the shape of the resulting data.
    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.
        Return:
            a new QueryCompiler
        """
        axis = kwargs.get("axis", 0)
        subset = kwargs.get("subset", None)
        thresh = kwargs.get("thresh", None)
        how = kwargs.get("how", "any")
        # We need to subset the axis that we care about with `subset`. This
        # will be used to determine the number of values that are NA.
        if subset is not None:
            if not axis:
                compute_na = self.getitem_column_array(subset)
            else:
                compute_na = self.getitem_row_array(self.index.get_indexer_for(subset))
        else:
            compute_na = self

        if not isinstance(axis, list):
            axis = [axis]
        # We are building this dictionary first to determine which columns
        # and rows to drop. This way we do not drop some columns before we
        # know which rows need to be dropped.
        if thresh is not None:
            # Count the number of NA values and specify which are higher than
            # thresh.
            drop_values = {
                ax ^ 1: compute_na.isna().sum(axis=ax ^ 1).to_pandas().squeeze()
                > thresh
                for ax in axis
            }
        else:
            drop_values = {
                ax
                ^ 1: getattr(compute_na.isna(), how)(axis=ax ^ 1).to_pandas().squeeze()
                for ax in axis
            }

        if 0 not in drop_values:
            drop_values[0] = None

        if 1 not in drop_values:
            drop_values[1] = None

            rm_from_index = (
                [obj for obj in compute_na.index[drop_values[1]]]
                if drop_values[1] is not None
                else None
            )
            rm_from_columns = (
                [obj for obj in compute_na.columns[drop_values[0]]]
                if drop_values[0] is not None
                else None
            )
        else:
            rm_from_index = (
                compute_na.index[drop_values[1]] if drop_values[1] is not None else None
            )
            rm_from_columns = (
                compute_na.columns[drop_values[0]]
                if drop_values[0] is not None
                else None
            )

        return self.drop(index=rm_from_index, columns=rm_from_columns)

    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
        if index is not None:
            index = self.index[~self.index.isin(index)]
        if columns is not None:
            columns = self.columns[~self.columns.isin(columns)]
        new_data = self._data_obj.mask(row_indices=index, col_indices=columns)
        return self.__constructor__(new_data)

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
        if is_list_like(value):
            # TODO make work with another querycompiler object as `value`.
            # This will require aligning the indices with a `reindex` and ensuring that
            # the data is partitioned identically.
            if isinstance(value, pandas.Series):
                value = value.reindex(self.index)
            value = list(value)

        def insert(df, internal_indices=[]):
            internal_idx = int(internal_indices[0])
            old_index = df.index
            df.index = pandas.RangeIndex(len(df.index))
            df.insert(internal_idx, internal_idx, value, allow_duplicates=True)
            df.columns = pandas.RangeIndex(len(df.columns))
            df.index = old_index
            return df

        new_data = self.data.apply_func_to_select_indices_along_full_axis(
            0, insert, loc, keep_remaining=True
        )
        new_columns = self.columns.insert(loc, column)
        return self.__constructor__(new_data, self.index, new_columns)

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
        if callable(func):
            return self._callable_func(func, axis, *args, **kwargs)
        elif isinstance(func, dict):
            return self._dict_func(func, axis, *args, **kwargs)
        elif is_list_like(func):
            return self._list_like_func(func, axis, *args, **kwargs)
        else:
            pass

    def _dict_func(self, func, axis, *args, **kwargs):
        """Apply function to certain indices across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        raise NotImplementedError("FIXME")
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        def dict_apply_builder(df, func_dict={}):
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            return pandas.DataFrame(df.apply(func_dict, *args, **kwargs))

        return self.__constructor__(self._data_obj._apply_full_axis_select_indices(
            axis, dict_apply_builder, func, keep_remaining=False
        ))

    def _list_like_func(self, func, axis, *args, **kwargs):
        """Apply list-like function across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        # When the function is list-like, the function names become the index/columns
        new_index = (
            [f if isinstance(f, string_types) else f.__name__ for f in func]
            if axis == 0
            else self.index
        )
        new_columns = (
            [f if isinstance(f, string_types) else f.__name__ for f in func]
            if axis == 1
            else self.columns
        )
        new_data = self._data_obj._apply_full_axis(axis, lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)), new_index=new_index, new_columns=new_columns)
        return self.__constructor__(new_data)

    def _callable_func(self, func, axis, *args, **kwargs):
        """Apply callable functions across given axis.

        Args:
            func: The functions to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        if isinstance(pandas.DataFrame().apply(func), pandas.Series):
            new_data = self._data_obj._full_axis_reduce(axis, lambda df: df.apply(func, axis=axis, *args, **kwargs))
        else:
            new_data = self._data_obj._apply_full_axis(axis, lambda df: df.apply(func, axis=axis, *args, **kwargs))
        return self.__constructor__(new_data)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    def groupby_reduce(
        self,
        by,
        axis,
        groupby_args,
        map_func,
        map_args,
        reduce_func=None,
        reduce_args=None,
        numeric_only=True,
    ):
        def _map(df, other):
            return map_func(
                df.groupby(by=other.squeeze(), axis=axis, **groupby_args), **map_args
            ).reset_index(drop=False)

        if reduce_func is not None:

            def _reduce(df):
                return reduce_func(
                    df.groupby(by=df.columns[0], axis=axis, **groupby_args),
                    **reduce_args
                )

        else:

            def _reduce(df):
                return map_func(
                    df.groupby(by=df.columns[0], axis=axis, **groupby_args), **map_args
                )

        new_data = self.data.groupby_reduce(axis, by.data, _map, _reduce)
        if axis == 0:
            new_columns = (
                self.columns if not numeric_only else self._numeric_columns(True)
            )
            new_index = self.compute_index(axis, new_data, False)
        else:
            new_columns = self.compute_index(axis, new_data, False)
            new_index = self.index
        return self.__constructor__(new_data, new_index, new_columns)

    def groupby_agg(self, by, axis, agg_func, groupby_args, agg_args):

        remote_index = self.index if not axis else self.columns

        def groupby_agg_builder(df):
            if not axis:
                df.index = remote_index
                df.columns = pandas.RangeIndex(len(df.columns))
                # We need to be careful that our internal index doesn't overlap with the
                # groupby values, otherwise we return an incorrect result. We
                # temporarily modify the columns so that we don't run into correctness
                # issues.
                if all(b in df for b in by):
                    df = df.add_prefix("_")
            else:
                df.columns = remote_index
                df.index = pandas.RangeIndex(len(df.index))

            def compute_groupby(df):
                grouped_df = df.groupby(by=by, axis=axis, **groupby_args)
                try:
                    result = agg_func(grouped_df, **agg_args)
                    # This will set things back if we changed them (see above).
                    if axis == 0 and not is_numeric_dtype(result.columns.dtype):
                        result.columns = [int(col[1:]) for col in result]
                # This happens when the partition is filled with non-numeric data and a
                # numeric operation is done. We need to build the index here to avoid issues
                # with extracting the index.
                except DataError:
                    result = pandas.DataFrame(index=grouped_df.size().index)
                return result

            try:
                return compute_groupby(df)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                return compute_groupby(df.copy())

        func_prepared = self._prepare_method(lambda df: groupby_agg_builder(df))
        result_data = self._map_across_full_axis(axis, func_prepared)
        if axis == 0:
            index = self.compute_index(0, result_data, False)
            columns = self.compute_index(1, result_data, True)
        else:
            index = self.compute_index(0, result_data, True)
            columns = self.compute_index(1, result_data, False)
        return self.__constructor__(result_data)

    # END Manual Partitioning methods

    # Get_dummies
    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """
        cls = type(self)
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

        # We have to do one of two things in order to ensure the final columns
        # are correct. Our first option is to map over the data and assign the
        # columns in a separate pass. That is what we have chosen to do here.
        # This is not as efficient, but it requires less information from the
        # lower layers and does not break any of our internal requirements. The
        # second option is that we assign the columns as a part of the
        # `get_dummies` call. This requires knowledge of the length of each
        # partition, and breaks some of our assumptions and separation of
        # concerns.
        def set_columns(df, columns):
            df.columns = columns
            return df

        set_cols = self.columns
        columns_applied = self._map_across_full_axis(
            1, lambda df: set_columns(df, set_cols)
        )
        # In some cases, we are mapping across all of the data. It is more
        # efficient if we are mapping over all of the data to do it this way
        # than it would be to reuse the code for specific columns.
        if len(columns) == len(self.columns):

            def get_dummies_builder(df):
                if df is not None:
                    if not df.empty:
                        return pandas.get_dummies(df, **kwargs)
                    else:
                        return pandas.DataFrame([])

            func = self._prepare_method(lambda df: get_dummies_builder(df))
            new_data = columns_applied.map_across_full_axis(0, func)
            untouched_data = None
        else:

            def get_dummies_builder(df, internal_indices=[]):
                return pandas.get_dummies(
                    df.iloc[:, internal_indices], columns=None, **kwargs
                )

            numeric_indices = list(self.columns.get_indexer_for(columns))
            new_data = columns_applied.apply_func_to_select_indices_along_full_axis(
                0, get_dummies_builder, numeric_indices, keep_remaining=False
            )
            untouched_data = self.drop(columns=columns)
        # Since we set the columns in the beginning, we can just extract them
        # here. There is fortunately no required extra steps for a correct
        # column index.
        final_columns = self.compute_index(1, new_data, False)
        # If we mapped over all the data we are done. If not, we need to
        # prepend the `new_data` with the raw data from the columns that were
        # not selected.
        if len(columns) != len(self.columns):
            new_data = untouched_data.data.concat(1, new_data)
            final_columns = untouched_data.columns.append(pandas.Index(final_columns))
        return cls(new_data, self.index, final_columns)

    # END Get_dummies

    # Indexing
    def view(self, index=None, columns=None):
        return self.__constructor__(self._data_obj.mask(row_numeric_idx=index, col_numeric_idx=columns))

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            partition.iloc[row_internal_indices, col_internal_indices] = item
            return partition

        new_data = self._data_obj._apply_select_indices(
            axis=None,
            func=iloc_mut,
            row_indices=row_numeric_index,
            col_indices=col_numeric_index,
            keep_remaining=True,
            item_to_distribute=broadcasted_items,
        )
        return self.__constructor__(new_data)

    def global_idx_to_numeric_idx(self, axis, indices):
        """
        Note: this function involves making copies of the index in memory.

        Args:
            axis: Axis to extract indices.
            indices: Indices to convert to numerical.

        Returns:
            An Index object.
        """
        assert axis in ["row", "col", "columns"]
        if axis == "row":
            return pandas.Index(
                pandas.Series(np.arange(len(self.index)), index=self.index)
                .loc[indices]
                .values
            )
        elif axis in ["col", "columns"]:
            return pandas.Index(
                pandas.Series(np.arange(len(self.columns)), index=self.columns)
                .loc[indices]
                .values
            )

    def enlarge_partitions(self, new_row_labels=None, new_col_labels=None):
        new_data = self.data.enlarge_partitions(
            len(new_row_labels), len(new_col_labels)
        )
        concated_index = (
            self.index.append(type(self.index)(new_row_labels))
            if new_row_labels
            else self.index
        )
        concated_columns = (
            self.columns.append(type(self.columns)(new_col_labels))
            if new_col_labels
            else self.columns
        )
        return self.__constructor__(new_data, concated_index, concated_columns)


class PandasQueryCompilerView(PandasQueryCompiler):
    """
    This class represent a view of the PandasQueryCompiler

    In particular, the following constraints are broken:
    - (len(self.index), len(self.columns)) != self.data.shape

    Note:
        The constraint will be satisfied when we get the data
    """

    def __init__(
        self,
        data_object,
        index_map_series=None,
        columns_map_series=None,
    ):
        """
        Args:
            index_map_series: a Pandas Series Object mapping user-facing index to
                numeric index.
            columns_map_series: a Pandas Series Object mapping user-facing index to
                numeric index.
        """
        assert index_map_series is not None
        assert columns_map_series is not None
        self.index_map = index_map_series
        self.columns_map = columns_map_series
        PandasQueryCompiler.__init__(self, data_object)

    @property
    def __constructor__(self):
        """Return parent object when getting the constructor."""
        return PandasQueryCompiler

    def _get_data(self):
        """Perform the map step

        Returns:
            A BaseFrameManager object.
        """
        masked_data = self.parent_data.mask(
            row_indices=self.index_map.values, col_indices=self.columns_map.values
        )
        return masked_data

    def _set_data(self, new_data):
        """Note this setter will be called by the
            `super(PandasQueryCompiler).__init__` function
        """
        self.parent_data = new_data

    _data_obj = property(_get_data, _set_data)

    def global_idx_to_numeric_idx(self, axis, indices):
        assert axis in ["row", "col", "columns"]
        if axis == "row":
            return self.index_map.loc[indices].index
        elif axis in ["col", "columns"]:
            return self.columns_map.loc[indices].index
