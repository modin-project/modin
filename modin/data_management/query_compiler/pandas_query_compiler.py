from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas

from pandas.compat import string_types
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
    is_bool_dtype,
)
from pandas.core.index import _ensure_index
from pandas.core.base import DataError

from modin.error_message import ErrorMessage
from modin.engines.base.block_partitions import BaseBlockPartitions


class PandasQueryCompiler(object):
    """This class implements the logic necessary for operating on partitions
        with a Pandas backend. This logic is specific to Pandas."""

    def __init__(
        self,
        block_partitions_object: BaseBlockPartitions,
        index: pandas.Index,
        columns: pandas.Index,
        dtypes=None,
    ):
        assert isinstance(block_partitions_object, BaseBlockPartitions)
        self.data = block_partitions_object
        self.index = index
        self.columns = columns
        if dtypes is not None:
            self._dtype_cache = dtypes

    def __constructor__(self, block_paritions_object, index, columns, dtypes=None):
        """By default, constructor method will invoke an init"""
        return type(self)(block_paritions_object, index, columns, dtypes)

    # Index, columns and dtypes objects
    _dtype_cache = None

    def _get_dtype(self):
        if self._dtype_cache is None:
            map_func = self._prepare_method(lambda df: df.dtypes)

            def dtype_builder(df):
                return df.apply(lambda row: find_common_type(row.values), axis=0)

            self._dtype_cache = self.full_reduce(0, map_func, dtype_builder)
            self._dtype_cache.index = self.columns
        elif not self._dtype_cache.index.equals(self.columns):
            self._dtype_cache.index = self.columns
        return self._dtype_cache

    def _set_dtype(self, dtypes):
        self._dtype_cache = dtypes

    dtypes = property(_get_dtype, _set_dtype)

    # These objects are currently not distributed.
    _index_cache = None
    _columns_cache = None

    def _get_index(self):
        return self._index_cache

    def _get_columns(self):
        return self._columns_cache

    def _validate_set_axis(self, new_labels, old_labels):
        new_labels = _ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
            )
        return new_labels

    def _set_index(self, new_index):
        if self._index_cache is None:
            self._index_cache = _ensure_index(new_index)
        else:
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index

    def _set_columns(self, new_columns):
        if self._columns_cache is None:
            self._columns_cache = _ensure_index(new_columns)
        else:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    # END Index, columns, and dtypes objects

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

    # END Index and columns objects

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!
    def _prepare_method(self, pandas_func, **kwargs):
        """Prepares methods given various metadata.
        Args:
            pandas_func: The function to prepare.

        Returns
            Helper function which handles potential transpose.
        """
        if self._is_transposed:

            def helper(df, internal_indices=[]):
                return pandas_func(df.T, **kwargs)

        else:

            def helper(df, internal_indices=[]):
                return pandas_func(df, **kwargs)

        return helper

    def numeric_columns(self, include_bool=True):
        """Returns the numeric columns of the Manager.

        Returns:
            List of index names.
        """
        columns = []
        for col, dtype in zip(self.columns, self.dtypes):
            if is_numeric_dtype(dtype) and (
                include_bool or (not include_bool and dtype != np.bool_)
            ):
                columns.append(col)
        return columns

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
    def add_prefix(self, prefix):
        new_column_names = self.columns.map(lambda x: str(prefix) + str(x))
        new_dtype_cache = self._dtype_cache.copy()
        if new_dtype_cache is not None:
            new_dtype_cache.index = new_column_names
        return self.__constructor__(
            self.data, self.index, new_column_names, new_dtype_cache
        )

    def add_suffix(self, suffix):
        new_column_names = self.columns.map(lambda x: str(x) + str(suffix))
        new_dtype_cache = self._dtype_cache.copy()
        if new_dtype_cache is not None:
            new_dtype_cache.index = new_column_names
        return self.__constructor__(
            self.data, self.index, new_column_names, new_dtype_cache
        )

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(
            self.data.copy(), self.index.copy(), self.columns.copy(), self._dtype_cache
        )

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
    def _join_index_objects(self, axis, other_index, how, sort=True):
        """Joins a pair of index objects (columns or rows) by a given strategy.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other_index: The other_index to join on.
            how: The type of join to join to make (e.g. right, left).

        Returns:
            Joined indices.
        """
        if isinstance(other_index, list):
            joined_obj = self.columns if not axis else self.index
            # TODO: revisit for performance
            for obj in other_index:
                joined_obj = joined_obj.join(obj, how=how)

            return joined_obj
        if not axis:
            return self.columns.join(other_index, how=how, sort=sort)
        else:
            return self.index.join(other_index, how=how, sort=sort)

    def join(self, other, **kwargs):
        """Joins a list or two objects together

        Args:
            other: The other object(s) to join on.

        Returns:
            Joined objects.
        """
        if not isinstance(other, list):
            other = [other]
        return self._join_list_of_managers(other, **kwargs)

    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        return self._append_list_of_managers(other, axis, **kwargs)

    def _append_list_of_managers(self, others, axis, **kwargs):
        if not isinstance(others, list):
            others = [others]
        assert all(
            isinstance(other, type(self)) for other in others
        ), "Different Manager objects are being used. This is not allowed"

        sort = kwargs.get("sort", None)
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        new_self, to_append, joined_axis = self.copartition(
            axis ^ 1, others, join, sort, force_repartition=True
        )
        new_data = new_self.concat(axis, to_append)

        if axis == 0:
            # The indices will be appended to form the final index.
            # If `ignore_index` is true, we create a RangeIndex that is the
            # length of all of the index objects combined. This is the same
            # behavior as pandas.
            new_index = (
                self.index.append([other.index for other in others])
                if not ignore_index
                else pandas.RangeIndex(
                    len(self.index) + sum(len(other.index) for other in others)
                )
            )
            return self.__constructor__(new_data, new_index, joined_axis)
        else:
            # The columns will be appended to form the final columns.
            new_columns = self.columns.append([other.columns for other in others])
            return self.__constructor__(new_data, joined_axis, new_columns)

    def _join_list_of_managers(self, others, **kwargs):
        assert isinstance(
            others, list
        ), "This method is for lists of DataManager objects only"
        assert all(
            isinstance(other, type(self)) for other in others
        ), "Different Manager objects are being used. This is not allowed"
        # Uses join's default value (though should not revert to default)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)
        lsuffix = kwargs.get("lsuffix", "")
        rsuffix = kwargs.get("rsuffix", "")
        new_self, to_join, joined_index = self.copartition(
            0, others, how, sort, force_repartition=True
        )
        new_data = new_self.concat(1, to_join)
        # This stage is to efficiently get the resulting columns, including the
        # suffixes.
        if len(others) == 1:
            others_proxy = pandas.DataFrame(columns=others[0].columns)
        else:
            others_proxy = [pandas.DataFrame(columns=other.columns) for other in others]
        self_proxy = pandas.DataFrame(columns=self.columns)
        new_columns = self_proxy.join(
            others_proxy, lsuffix=lsuffix, rsuffix=rsuffix
        ).columns
        return self.__constructor__(new_data, joined_index, new_columns)

    # END Append/Concat/Join

    # Copartition
    def copartition(self, axis, other, how_to_join, sort, force_repartition=False):
        """Copartition two QueryCompiler objects.

        Args:
            axis: The axis to copartition along.
            other: The other Query Compiler(s) to copartition against.
            how_to_join: How to manage joining the index object ("left", "right", etc.)
            sort: Whether or not to sort the joined index.
            force_repartition: Whether or not to force the repartitioning. By default,
                this method will skip repartitioning if it is possible. This is because
                reindexing is extremely inefficient. Because this method is used to
                `join` or `append`, it is vital that the internal indices match.

        Returns:
            A tuple (left query compiler, right query compiler list, joined index).
        """
        if isinstance(other, type(self)):
            other = [other]

        index_obj = (
            [o.index for o in other] if axis == 0 else [o.columns for o in other]
        )
        joined_index = self._join_index_objects(
            axis ^ 1, index_obj, how_to_join, sort=sort
        )
        # We have to set these because otherwise when we perform the functions it may
        # end up serializing this entire object.
        left_old_idx = self.index if axis == 0 else self.columns
        right_old_idxes = index_obj

        # Start with this and we'll repartition the first time, and then not again.
        reindexed_self = self.data
        reindexed_other_list = []

        def compute_reindex(old_idx):
            """Create a function based on the old index and axis.

            Args:
                old_idx: The old index/columns

            Returns:
                A function that will be run in each partition.
            """

            def reindex_partition(df):
                if axis == 0:
                    df.index = old_idx
                    new_df = df.reindex(index=joined_index)
                    new_df.index = pandas.RangeIndex(len(new_df.index))
                else:
                    df.columns = old_idx
                    new_df = df.reindex(columns=joined_index)
                    new_df.columns = pandas.RangeIndex(len(new_df.columns))
                return new_df

            return reindex_partition

        for i in range(len(other)):

            # If the indices are equal we can skip partitioning so long as we are not
            # forced to repartition. See note above about `force_repartition`.
            if i != 0 or (left_old_idx.equals(joined_index) and not force_repartition):
                reindex_left = None
            else:
                reindex_left = compute_reindex(left_old_idx)
            if right_old_idxes[i].equals(joined_index) and not force_repartition:
                reindex_right = None
            else:
                reindex_right = compute_reindex(right_old_idxes[i])

            reindexed_self, reindexed_other = reindexed_self.copartition_datasets(
                axis, other[i].data, reindex_left, reindex_right
            )
            reindexed_other_list.append(reindexed_other)
        return reindexed_self, reindexed_other_list, joined_index

    # Inter-Data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.
    def inter_manager_operations(self, other, how_to_join, func):
        """Inter-data operations (e.g. add, sub).

        Args:
            other: The other Manager for the operation.
            how_to_join: The type of join to join to make (e.g. right, outer).

        Returns:
            New DataManager with new data and index.
        """
        reindexed_self, reindexed_other_list, joined_index = self.copartition(
            0, other, how_to_join, False
        )
        # unwrap list returned by `copartition`.
        reindexed_other = reindexed_other_list[0]
        new_columns = self._join_index_objects(
            0, other.columns, how_to_join, sort=False
        )
        # THere is an interesting serialization anomaly that happens if we do
        # not use the columns in `inter_data_op_builder` from here (e.g. if we
        # pass them in). Passing them in can cause problems, so we will just
        # use them from here.
        self_cols = self.columns
        other_cols = other.columns

        def inter_data_op_builder(left, right, func):
            left.columns = self_cols
            right.columns = other_cols
            # We reset here to make sure that the internal indexes match. We aligned
            # them in the previous step, so this step is to prevent mismatches.
            left.index = pandas.RangeIndex(len(left.index))
            right.index = pandas.RangeIndex(len(right.index))
            result = func(left, right)
            result.columns = pandas.RangeIndex(len(result.columns))
            return result

        new_data = reindexed_self.inter_data_operation(
            1, lambda l, r: inter_data_op_builder(l, r, func), reindexed_other
        )
        return self.__constructor__(new_data, joined_index, new_columns)

    def _inter_df_op_handler(self, func, other, **kwargs):
        """Helper method for inter-manager and scalar operations.

        Args:
            func: The function to use on the Manager/scalar.
            other: The other Manager/scalar.

        Returns:
            New DataManager with new data and index.
        """
        axis = kwargs.get("axis", 0)
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0

        if isinstance(other, type(self)):
            return self.inter_manager_operations(
                other, "outer", lambda x, y: func(x, y, **kwargs)
            )
        else:
            return self.scalar_operations(
                axis, other, lambda df: func(df, other, **kwargs)
            )

    def add(self, other, **kwargs):
        """Adds this manager with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with added data and new index.
        """
        func = pandas.DataFrame.add
        return self._inter_df_op_handler(func, other, **kwargs)

    def div(self, other, **kwargs):
        """Divides this manager with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with divided data and new index.
        """
        func = pandas.DataFrame.div
        return self._inter_df_op_handler(func, other, **kwargs)

    def eq(self, other, **kwargs):
        """Compares equality (==) with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.eq
        return self._inter_df_op_handler(func, other, **kwargs)

    def floordiv(self, other, **kwargs):
        """Floordivs this manager with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with floordiv-ed data and index.
        """
        func = pandas.DataFrame.floordiv
        return self._inter_df_op_handler(func, other, **kwargs)

    def ge(self, other, **kwargs):
        """Compares this manager >= than other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.ge
        return self._inter_df_op_handler(func, other, **kwargs)

    def gt(self, other, **kwargs):
        """Compares this manager > than other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.gt
        return self._inter_df_op_handler(func, other, **kwargs)

    def le(self, other, **kwargs):
        """Compares this manager < than other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.le
        return self._inter_df_op_handler(func, other, **kwargs)

    def lt(self, other, **kwargs):
        """Compares this manager <= than other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.lt
        return self._inter_df_op_handler(func, other, **kwargs)

    def mod(self, other, **kwargs):
        """Mods this manager against other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with mod-ed data and index.
        """
        func = pandas.DataFrame.mod
        return self._inter_df_op_handler(func, other, **kwargs)

    def mul(self, other, **kwargs):
        """Multiplies this manager against other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with multiplied data and index.
        """
        func = pandas.DataFrame.mul
        return self._inter_df_op_handler(func, other, **kwargs)

    def ne(self, other, **kwargs):
        """Compares this manager != to other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with compared data and index.
        """
        func = pandas.DataFrame.ne
        return self._inter_df_op_handler(func, other, **kwargs)

    def pow(self, other, **kwargs):
        """Exponential power of this manager to other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with pow-ed data and index.
        """
        func = pandas.DataFrame.pow
        return self._inter_df_op_handler(func, other, **kwargs)

    def rdiv(self, other, **kwargs):
        """Divides other object (manager or scalar) with this manager.

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with divided data and new index.
        """
        func = pandas.DataFrame.rdiv
        return self._inter_df_op_handler(func, other, **kwargs)

    def rfloordiv(self, other, **kwargs):
        """Floordivs this manager with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with floordiv-ed data and index.
        """
        func = pandas.DataFrame.rfloordiv
        return self._inter_df_op_handler(func, other, **kwargs)

    def rmod(self, other, **kwargs):
        """Mods this manager with other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with mod data and index.
        """
        func = pandas.DataFrame.rmod
        return self._inter_df_op_handler(func, other, **kwargs)

    def rpow(self, other, **kwargs):
        """Exponential power of other object (manager or scalar) to this manager.

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with pow-ed data and new index.
        """
        func = pandas.DataFrame.rpow
        return self._inter_df_op_handler(func, other, **kwargs)

    def rsub(self, other, **kwargs):
        """Subtracts other object (manager or scalar) from this manager.

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with subtracted data and new index.
        """
        func = pandas.DataFrame.rsub
        return self._inter_df_op_handler(func, other, **kwargs)

    def sub(self, other, **kwargs):
        """Subtracts this manager from other object (manager or scalar).

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with subtracted data and new index.
        """
        func = pandas.DataFrame.sub
        return self._inter_df_op_handler(func, other, **kwargs)

    def truediv(self, other, **kwargs):
        """Divides this manager with other object (manager or scalar).
           Functionally same as div

        Args:
            other: The other object (manager or scalar).

        Returns:
            New DataManager with divided data and new index.
        """
        func = pandas.DataFrame.truediv
        return self._inter_df_op_handler(func, other, **kwargs)

    def clip(self, lower, upper, **kwargs):
        kwargs["upper"] = upper
        kwargs["lower"] = lower
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.clip, **kwargs)
        if is_list_like(lower) or is_list_like(upper):
            df = self.map_across_full_axis(axis, func)
            return self.__constructor__(df, self.index, self.columns)
        return self.scalar_operations(axis, lower or upper, func)

    def update(self, other, **kwargs):
        """Uses other manager to update corresponding values in this manager.

        Args:
            other: The other manager.

        Returns:
            New DataManager with updated data and index.
        """
        assert isinstance(
            other, type(self)
        ), "Must have the same DataManager subclass to perform this operation"

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
            New DataManager with updated data and index.
        """

        assert isinstance(
            cond, type(self)
        ), "Must have the same DataManager subclass to perform this operation"
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

            first_pass = cond.inter_manager_operations(
                other, "left", where_builder_first_pass
            )
            final_pass = self.inter_manager_operations(
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

    # Single Manager scalar operations (e.g. add to scalar, list of scalars)
    def scalar_operations(self, axis, scalar, func):
        """Handler for mapping scalar operations across a Manager.

        Args:
            axis: The axis index object to execute the function on.
            scalar: The scalar value to map.
            func: The function to use on the Manager with the scalar.

        Returns:
            New DataManager with updated data and new index.
        """
        if isinstance(scalar, (list, np.ndarray, pandas.Series)):
            new_data = self.map_across_full_axis(axis, func)
            return self.__constructor__(new_data, self.index, self.columns)
        else:
            return self.map_partitions(func)

    # END Single Manager scalar operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manger.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            New DataManager with updated data and new index.
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
        new_data = self.map_across_full_axis(axis, func)
        return self.__constructor__(new_data, new_index, new_columns)

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            New DataManager with updated data and reset index.
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
                new_column_name = "index" if "index" not in self.columns else "level_0"
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
    #
    # _is_transposed, 0 for False or non-transposed, 1 for True or transposed.
    _is_transposed = 0

    def transpose(self, *args, **kwargs):
        """Transposes this DataManager.

        Returns:
            Transposed new DataManager.
        """
        new_data = self.data.transpose(*args, **kwargs)
        # Switch the index and columns and transpose the
        new_manager = self.__constructor__(new_data, self.columns, self.index)
        # It is possible that this is already transposed
        new_manager._is_transposed = self._is_transposed ^ 1
        return new_manager

    # END Transpose

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def full_reduce(self, axis, map_func, reduce_func=None, numeric_only=False):
        """Apply function that will reduce the data to a Pandas Series.

        Args:
            axis: 0 for columns and 1 for rows. Default is 0.
            map_func: Callable function to map the dataframe.
            reduce_func: Callable function to reduce the dataframe. If none,
                then apply map_func twice.
            numeric_only: Apply only over the numeric rows.

        Return:
            Returns Pandas Series containing the results from map_func and reduce_func.
        """
        if numeric_only:
            result, query_compiler = self.numeric_function_clean_dataframe(axis)
            if result is not None:
                return result
        else:
            query_compiler = self
        if reduce_func is None:
            reduce_func = map_func
        # The XOR here will ensure that we reduce over the correct axis that
        # exists on the internal partitions. We flip the axis
        mapped_parts = query_compiler.data.map_across_blocks(map_func).partitions
        if reduce_func is None:
            reduce_func = map_func

        if reduce_func is None:
            reduce_func = map_func
        # For now we return a pandas.Series until ours gets implemented.
        # We have to build the intermediate frame based on the axis passed,
        # thus axis=axis and axis=axis ^ 1
        #
        # This currently requires special treatment because of the intermediate
        # DataFrame. The individual partitions return Series objects, and those
        # cannot be concatenated the correct way without casting them as
        # DataFrames.
        full_frame = pandas.concat(
            [
                pandas.concat(
                    [pandas.DataFrame(part.get()).T for part in row_of_parts],
                    axis=axis ^ 1,
                )
                for row_of_parts in mapped_parts
            ],
            axis=axis,
        )

        # Transpose because operations where axis == 1 assume that the
        # operation is performed across the other axis
        if axis == 1:
            full_frame = full_frame.T
        result = reduce_func(full_frame)

        if result.shape == (0,):
            return result
        elif not axis:
            result.index = query_compiler.columns
        else:
            result.index = query_compiler.index
        return result

    def _process_min_max(self, func, **kwargs):
        """Calculates the min or max of the DataFrame.

        Return:
           Pandas series containing the min or max values from each column or
           row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)

        def min_max_builder(df, **kwargs):
            if not df.empty:
                return func(df, **kwargs)

        map_func = self._prepare_method(min_max_builder, **kwargs)
        return self.full_reduce(axis, map_func, numeric_only=numeric_only)

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            Pandas series containing counts of non-NaN objects from each column or row.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = kwargs.get("numeric_only", False)
        map_func = self._prepare_method(pandas.DataFrame.count, **kwargs)
        reduce_func = self._prepare_method(pandas.DataFrame.sum, **kwargs)
        return self.full_reduce(axis, map_func, reduce_func, numeric_only)

    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            Pandas series with the maximum values from each column or row.
        """
        return self._process_min_max(pandas.DataFrame.max, **kwargs)

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            Pandas series containing the mean from each numerical column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        sums = self.sum(**kwargs)
        counts = self.count(axis=axis, numeric_only=kwargs.get("numeric_only", None))
        try:
            # If we need to drop any columns, it will throw a TypeError
            return sums.divide(counts)
        # In the case that a TypeError is thrown, we need to iterate through, similar to
        # how pandas does and do the division only on things that can be divided.
        # NOTE: We will only hit this condition if numeric_only is not True.
        except TypeError:

            def can_divide(l, r):
                try:
                    pandas.Series([l]).divide(r)
                except TypeError:
                    return False
                return True

            # Iterate through the sums to check that we can divide them. If not, then
            # drop the record. This matches pandas behavior.
            return pandas.Series(
                {
                    idx: sums[idx] / counts[idx]
                    for idx in sums.index
                    if can_divide(sums[idx], counts[idx])
                }
            )

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            Pandas series with the minimum value from each column or row.
        """
        return self._process_min_max(pandas.DataFrame.min, **kwargs)

    def _process_sum_prod(self, func, **kwargs):
        """Calculates the sum or product of the DataFrame.

        Args:
            func: Pandas func to apply to DataFrame.
            ignore_axis: Whether to ignore axis when raising TypeError
        Return:
            Pandas Series with sum or prod of DataFrame.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = kwargs.get("numeric_only", None) if not axis else True
        min_count = kwargs.get("min_count", 0)
        reduce_index = self.columns if axis else self.index

        if numeric_only:
            result, query_compiler = self.numeric_function_clean_dataframe(axis)
        else:
            query_compiler = self
        new_index = query_compiler.index if axis else query_compiler.columns

        def sum_prod_builder(df, **kwargs):
            if not df.empty:
                return func(df, **kwargs)
            else:
                return pandas.DataFrame([])

        map_func = self._prepare_method(sum_prod_builder, **kwargs)

        if min_count <= 1:
            return self.full_reduce(axis, map_func, numeric_only=numeric_only)
        elif min_count > len(reduce_index):
            return pandas.Series(
                [np.nan] * len(new_index), index=new_index, dtype=np.dtype("object")
            )
        else:
            return self.full_axis_reduce(map_func, axis)

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        return self._process_sum_prod(pandas.DataFrame.prod, **kwargs)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        return self._process_sum_prod(pandas.DataFrame.sum, **kwargs)

    # END Full Reduce operations

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    def map_partitions(self, func, new_dtypes=None):
        return self.__constructor__(
            self.data.map_across_blocks(func), self.index, self.columns, new_dtypes
        )

    def abs(self):
        func = self._prepare_method(pandas.DataFrame.abs)
        return self.map_partitions(func, new_dtypes=self.dtypes.copy())

    def applymap(self, func):
        remote_func = self._prepare_method(pandas.DataFrame.applymap, func=func)
        return self.map_partitions(remote_func)

    def isin(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.isin, **kwargs)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def isna(self):
        func = self._prepare_method(pandas.DataFrame.isna)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def isnull(self):
        func = self._prepare_method(pandas.DataFrame.isnull)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def negative(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.__neg__, **kwargs)
        return self.map_partitions(func)

    def notna(self):
        func = self._prepare_method(pandas.DataFrame.notna)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def notnull(self):
        func = self._prepare_method(pandas.DataFrame.notnull)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self.map_partitions(func, new_dtypes=new_dtypes)

    def round(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.round, **kwargs)
        return self.map_partitions(func, new_dtypes=self._dtype_cache)

    # END Map partitions operations

    # Map partitions across select indices
    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        # Group indices to update by dtype for less map operations
        dtype_indices = {}
        columns = col_dtypes.keys()
        numeric_indices = list(self.columns.get_indexer_for(columns))
        # Create Series for the updated dtypes
        new_dtypes = self.dtypes.copy()
        for i, column in enumerate(columns):
            dtype = col_dtypes[column]
            if dtype != self.dtypes[column]:
                # Only add dtype only if different
                if dtype in dtype_indices.keys():
                    dtype_indices[dtype].append(numeric_indices[i])
                else:
                    dtype_indices[dtype] = [numeric_indices[i]]
                # Update the new dtype series to the proper pandas dtype
                new_dtype = np.dtype(dtype)
                if dtype != np.int32 and new_dtype == np.int32:
                    new_dtype = np.dtype("int64")
                elif dtype != np.float32 and new_dtype == np.float32:
                    new_dtype = np.dtype("float64")
                new_dtypes[column] = new_dtype
        # Update partitions for each dtype that is updated
        new_data = self.data
        for dtype in dtype_indices.keys():

            def astype(df, internal_indices=[]):
                block_dtypes = {}
                for ind in internal_indices:
                    block_dtypes[df.columns[ind]] = dtype
                return df.astype(block_dtypes)

            new_data = new_data.apply_func_to_select_indices(
                0, astype, dtype_indices[dtype], keep_remaining=True
            )

        return self.__constructor__(new_data, self.index, self.columns, new_dtypes)

    # END Map partitions across select indices

    # Column/Row partitions reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def full_axis_reduce(self, func, axis, alternate_index=None):
        """Applies map that reduce Manager to series but require knowledge of full axis.

        Args:
            func: Function to reduce the Manager by. This function takes in a Manager.
            axis: axis to apply the function to.
            alternate_index: If the resulting series should have an index
                different from the current query_compiler's index or columns.

        Return:
            Pandas series containing the reduced data.
        """
        # We XOR with axis because if we are doing an operation over the columns
        # (i.e. along the rows), we want to take the transpose so that the
        # results from the same parition will be concated together first.
        # We need this here because if the operations is over the columns,
        # map_across_full_axis does not transpose the result before returning.
        result = self.data.map_across_full_axis(axis, func).to_pandas(
            self._is_transposed ^ axis
        )
        if result.empty:
            return result
        if not axis:
            result.index = (
                alternate_index if alternate_index is not None else self.columns
            )
        else:
            result.index = (
                alternate_index if alternate_index is not None else self.index
            )
        return result

    def all(self, **kwargs):
        """Returns whether all the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        return self._process_all_any(lambda df, **kwargs: df.all(**kwargs), **kwargs)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        return self._process_all_any(lambda df, **kwargs: df.any(**kwargs), **kwargs)

    def _process_all_any(self, func, **kwargs):
        """Calculates if any or all the values are true.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        axis = kwargs.get("axis", 0)
        axis_none = True if axis is None else False
        axis = 0 if axis is None else axis
        kwargs["axis"] = axis
        bool_only = kwargs.get("bool_only", None)
        kwargs["bool_only"] = False if bool_only is None else bool_only

        not_bool_col = []
        numeric_col_count = 0
        for col, dtype in zip(self.columns, self.dtypes):
            if not is_bool_dtype(dtype):
                not_bool_col.append(col)
            numeric_col_count += 1 if is_numeric_dtype(dtype) else 0

        if bool_only:
            if axis == 0 and not axis_none and len(not_bool_col) == len(self.columns):
                return pandas.Series(dtype=bool)
            if len(not_bool_col) == len(self.columns):
                query_compiler = self
            else:
                query_compiler = self.drop(columns=not_bool_col)
        else:
            if (
                bool_only is False
                and axis_none
                and len(not_bool_col) == len(self.columns)
                and numeric_col_count != len(self.columns)
            ):
                if func == pandas.DataFrame.all:
                    return self.getitem_single_key(self.columns[-1])[self.index[-1]]
                elif func == pandas.DataFrame.any:
                    return self.getitem_single_key(self.columns[0])[self.index[0]]
            query_compiler = self

        builder_func = query_compiler._prepare_method(func, **kwargs)
        result = query_compiler.full_axis_reduce(builder_func, axis)
        if axis_none:
            return func(result)
        else:
            return result

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        # It may be possible to incrementally check each partition, but this
        # computation is fairly cheap.
        def first_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.first_valid_index())

        func = self._prepare_method(first_valid_index_builder)
        # We get the minimum from each column, then take the min of that to get
        # first_valid_index.
        first_result = self.full_axis_reduce(func, 0)

        return self.index[first_result.min()]

    def _post_process_idx_ops(self, axis, intermediate_result):
        """Converts internal index to external index.

        Args:
            axis: 0 for columns and 1 for rows. Defaults to 0.
            intermediate_result: Internal index of self.data.

        Returns:
            External index of the intermediate_result.
        """
        index = self.index if not axis else self.columns
        result = intermediate_result.apply(lambda x: index[x])
        return result

    def idxmax(self, **kwargs):
        """Returns the first occurance of the maximum over requested axis.

        Returns:
            Series containing the maximum of each column or axis.
        """
        # The reason for the special treatment with idxmax/min is because we
        # need to communicate the row number back here.
        def idxmax_builder(df, **kwargs):
            df.index = pandas.RangeIndex(len(df.index))
            return df.idxmax(**kwargs)

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(idxmax_builder, **kwargs)
        max_result = self.full_axis_reduce(func, axis)
        # Because our internal partitions don't track the external index, we
        # have to do a conversion.
        return self._post_process_idx_ops(axis, max_result)

    def idxmin(self, **kwargs):
        """Returns the first occurance of the minimum over requested axis.

        Returns:
            Series containing the minimum of each column or axis.
        """
        # The reason for the special treatment with idxmax/min is because we
        # need to communicate the row number back here.
        def idxmin_builder(df, **kwargs):
            df.index = pandas.RangeIndex(len(df.index))
            return df.idxmin(**kwargs)

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(idxmin_builder, **kwargs)
        min_result = self.full_axis_reduce(func, axis)
        # Because our internal partitions don't track the external index, we
        # have to do a conversion.
        return self._post_process_idx_ops(axis, min_result)

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def last_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.last_valid_index())

        func = self._prepare_method(last_valid_index_builder)
        # We get the maximum from each column, then take the max of that to get
        # last_valid_index.
        first_result = self.full_axis_reduce(func, 0)

        return self.index[first_result.max()]

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            Series containing the median of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        result, query_compiler = self.numeric_function_clean_dataframe(axis)
        if result is not None:
            return result
        func = self._prepare_method(pandas.DataFrame.median, **kwargs)
        return query_compiler.full_axis_reduce(func, axis)

    def memory_usage(self, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            Series containing the memory usage of each column.
        """

        def memory_usage_builder(df, **kwargs):
            return df.memory_usage(index=False, deep=deep)

        deep = kwargs.get("deep", False)
        func = self._prepare_method(memory_usage_builder, **kwargs)
        return self.full_axis_reduce(func, 0)

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            Series of ints indexed by column or index names.
        """
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.nunique, **kwargs)
        return self.full_axis_reduce(func, axis)

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            Series containing the quantile of each column or row.
        """
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q", 0.5)
        numeric_only = kwargs.get("numeric_only", True)
        assert type(q) is float
        if numeric_only:
            result, query_compiler = self.numeric_function_clean_dataframe(axis)
            if result is not None:
                return result
        else:
            query_compiler = self

        def quantile_builder(df, **kwargs):
            try:
                return pandas.DataFrame.quantile(df, **kwargs)
            except ValueError:
                return pandas.Series()

        func = self._prepare_method(quantile_builder, **kwargs)
        result = query_compiler.full_axis_reduce(func, axis)
        result.name = q
        return result

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            Series containing the skew of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        result, query_compiler = self.numeric_function_clean_dataframe(axis)
        if result is not None:
            return result
        func = self._prepare_method(pandas.DataFrame.skew, **kwargs)
        return query_compiler.full_axis_reduce(func, axis)

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            Series containing the standard deviation of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        result, query_compiler = self.numeric_function_clean_dataframe(axis)
        if result is not None:
            return result
        func = self._prepare_method(pandas.DataFrame.std, **kwargs)
        return query_compiler.full_axis_reduce(func, axis)

    def to_datetime(self, **kwargs):
        """Converts the Manager to a Series of DateTime objects.

        Returns:
            Series of DateTime objects.
        """
        columns = self.columns

        def to_datetime_builder(df, **kwargs):
            df.columns = columns
            return pandas.to_datetime(df, **kwargs)

        func = self._prepare_method(to_datetime_builder, **kwargs)
        return self.full_axis_reduce(func, 1)

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            Series containing the variance of each column or row.
        """
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        result, query_compiler = self.numeric_function_clean_dataframe(axis)
        if result is not None:
            return result
        func = query_compiler._prepare_method(pandas.DataFrame.var, **kwargs)
        return query_compiler.full_axis_reduce(func, axis)

    # END Column/Row partitions reduce operations

    # Column/Row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def full_axis_reduce_along_select_indices(
        self, func, axis, index, pandas_result=True
    ):
        """Reduce Manger along select indices using function that needs full axis.

        Args:
            func: Callable that reduces Manager to Series using full knowledge of an
                axis.
            axis: 0 for columns and 1 for rows. Defaults to 0.
            index: Index of the resulting series.
            pandas_result: Return the result as a Pandas Series instead of raw data.

        Returns:
            Either a Pandas Series with index or BaseBlockPartitions object.
        """
        # Convert indices to numeric indices
        old_index = self.index if axis else self.columns
        numeric_indices = [i for i, name in enumerate(old_index) if name in index]
        result = self.data.apply_func_to_select_indices_along_full_axis(
            axis, func, numeric_indices
        )
        if pandas_result:
            result = result.to_pandas(self._is_transposed)
            result.index = index
        return result

    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        # Only describe numeric if there are numeric columns
        # Otherwise, describe all
        new_columns = self.numeric_columns(include_bool=False)
        if len(new_columns) != 0:
            numeric = True
            exclude = kwargs.get("exclude", None)
            include = kwargs.get("include", None)
            # This is done to check against the default dtypes with 'in'.
            # We don't change `include` in kwargs, so we can just use this for the
            # check.
            if include is None:
                include = []
            default_excludes = [np.timedelta64, np.datetime64, np.object, np.bool]
            add_to_excludes = [e for e in default_excludes if e not in include]
            if is_list_like(exclude):
                exclude.append(add_to_excludes)
            else:
                exclude = add_to_excludes
            kwargs["exclude"] = exclude
        else:
            numeric = False
            # If only timedelta and datetime objects, only do the timedelta
            # columns
            if all(
                (
                    dtype
                    for dtype in self.dtypes
                    if dtype == np.datetime64 or dtype == np.timedelta64
                )
            ):
                new_columns = [
                    self.columns[i]
                    for i in range(len(self.columns))
                    if self.dtypes[i] != np.dtype("datetime64[ns]")
                ]
            else:
                # Describe all columns
                new_columns = self.columns

        def describe_builder(df, **kwargs):
            try:
                return pandas.DataFrame.describe(df, **kwargs)
            except ValueError:
                return pandas.DataFrame(index=df.index)

        # Apply describe and update indices, columns, and dtypes
        func = self._prepare_method(describe_builder, **kwargs)
        new_data = self.full_axis_reduce_along_select_indices(
            func, 0, new_columns, False
        )
        new_index = self.compute_index(0, new_data, False)
        if numeric:
            new_dtypes = pandas.Series(
                [np.float64 for _ in new_columns], index=new_columns
            )
        else:
            new_dtypes = pandas.Series(
                [np.object for _ in new_columns], index=new_columns
            )
        return self.__constructor__(new_data, new_index, new_columns, new_dtypes)

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def map_across_full_axis(self, axis, func):
        return self.data.map_across_full_axis(axis, func)

    def _cumulative_builder(self, func, **kwargs):
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(func, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        return self.__constructor__(
            new_data, self.index, self.columns, self._dtype_cache
        )

    def cumsum(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumsum, **kwargs)

    def cummax(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummax, **kwargs)

    def cummin(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cummin, **kwargs)

    def cumprod(self, **kwargs):
        return self._cumulative_builder(pandas.DataFrame.cumprod, **kwargs)

    def diff(self, **kwargs):

        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.diff, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        return self.__constructor__(new_data, self.index, self.columns)

    def dropna(self, **kwargs):
        """Returns a new DataManager with null values dropped along given axis.
        Return:
            a new DataManager
        """
        axis = kwargs.get("axis", 0)
        subset = kwargs.get("subset")
        thresh = kwargs.get("thresh")
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
                ax ^ 1: compute_na.isna().sum(axis=ax ^ 1) > thresh for ax in axis
            }
        else:
            drop_values = {
                ax ^ 1: getattr(compute_na.isna(), how)(axis=ax ^ 1) for ax in axis
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

    def eval(self, expr, **kwargs):
        """Returns a new DataManager with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new PandasDataManager with new columns after applying expr.
        """
        inplace = kwargs.get("inplace", False)

        columns = self.index if self._is_transposed else self.columns
        index = self.columns if self._is_transposed else self.index

        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        columns_copy = pandas.DataFrame(columns=self.columns)
        columns_copy = columns_copy.eval(expr, inplace=False, **kwargs)
        expect_series = isinstance(columns_copy, pandas.Series)
        # if there is no assignment, then we simply save the results
        # in the first column
        if expect_series:
            if inplace:
                raise ValueError("Cannot operate inplace if there is no assignment")
            else:
                expr = "{0} = {1}".format(columns[0], expr)

        def eval_builder(df, **kwargs):
            df.columns = columns
            result = df.eval(expr, inplace=False, **kwargs)
            result.columns = pandas.RangeIndex(0, len(result.columns))
            return result

        func = self._prepare_method(eval_builder, **kwargs)
        new_data = self.map_across_full_axis(1, func)

        if expect_series:
            result = new_data.to_pandas()[0]
            result.name = columns_copy.name
            result.index = index
            return result
        else:
            columns = columns_copy.columns
            return self.__constructor__(new_data, self.index, columns)

    def mode(self, **kwargs):
        """Returns a new DataManager with modes calculated for each label along given axis.

        Returns:
            A new PandasDataManager with modes calculated.
        """
        axis = kwargs.get("axis", 0)

        def mode_builder(df, **kwargs):
            result = df.mode(**kwargs)
            # We return a dataframe with the same shape as the input to ensure
            # that all the partitions will be the same shape
            if not axis and len(df) != len(result):
                # Pad columns
                append_values = pandas.DataFrame(
                    columns=result.columns, index=range(len(result), len(df))
                )
                result = pandas.concat([result, append_values], ignore_index=True)
            elif axis and len(df.columns) != len(result.columns):
                # Pad rows
                append_vals = pandas.DataFrame(
                    columns=range(len(result.columns), len(df.columns)),
                    index=result.index,
                )
                result = pandas.concat([result, append_vals], axis=1)
            return result

        func = self._prepare_method(mode_builder, **kwargs)
        new_data = self.map_across_full_axis(axis, func)

        new_index = pandas.RangeIndex(len(self.index)) if not axis else self.index
        new_columns = self.columns if not axis else pandas.RangeIndex(len(self.columns))
        return self.__constructor__(
            new_data, new_index, new_columns, self._dtype_cache
        ).dropna(axis=axis, how="all")

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new PandasDataManager with null values filled.
        """
        axis = kwargs.get("axis", 0)
        value = kwargs.get("value")
        if isinstance(value, dict):
            value = kwargs.pop("value")

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

            new_data = self.data.apply_func_to_select_indices(
                axis, fillna_dict_builder, value, keep_remaining=True
            )
            return self.__constructor__(new_data, self.index, self.columns)
        else:
            func = self._prepare_method(pandas.DataFrame.fillna, **kwargs)
            new_data = self.map_across_full_axis(axis, func)
            return self.__constructor__(new_data, self.index, self.columns)

    def query(self, expr, **kwargs):
        """Query columns of the DataManager with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            DataManager containing the rows where the boolean expression is satisfied.
        """
        columns = self.columns

        def query_builder(df, **kwargs):
            # This is required because of an Arrow limitation
            # TODO revisit for Arrow error
            df = df.copy()
            df.index = pandas.RangeIndex(len(df))
            df.columns = columns
            df.query(expr, inplace=True, **kwargs)
            df.columns = pandas.RangeIndex(len(df.columns))
            return df

        func = self._prepare_method(query_builder, **kwargs)
        new_data = self.map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self.compute_index(0, new_data, True)

        return self.__constructor__(new_data, new_index, self.columns, self.dtypes)

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            DataManager containing the ranks of the values along an axis.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        func = self._prepare_method(pandas.DataFrame.rank, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        # Since we assume no knowledge of internal state, we get the columns
        # from the internal partitions.
        if numeric_only:
            new_columns = self.compute_index(1, new_data, True)
        else:
            new_columns = self.columns
        new_dtypes = pandas.Series([np.float64 for _ in new_columns], index=new_columns)
        return self.__constructor__(new_data, self.index, new_columns, new_dtypes)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            DataManager containing the data sorted by columns or indices.
        """
        axis = kwargs.pop("axis", 0)
        index = self.columns if axis else self.index

        # sort_index can have ascending be None and behaves as if it is False.
        # sort_values cannot have ascending be None. Thus, the following logic is to
        # convert the ascending argument to one that works with sort_values
        ascending = kwargs.pop("ascending", True)
        if ascending is None:
            ascending = False
        kwargs["ascending"] = ascending

        def sort_index_builder(df, **kwargs):
            if axis:
                df.columns = index
            else:
                df.index = index
            return df.sort_index(axis=axis, **kwargs)

        func = self._prepare_method(sort_index_builder, **kwargs)
        new_data = self.map_across_full_axis(axis, func)
        if axis:
            new_columns = pandas.Series(self.columns).sort_values(**kwargs)
            new_index = self.index
        else:
            new_index = pandas.Series(self.index).sort_values(**kwargs)
            new_columns = self.columns
        return self.__constructor__(
            new_data, new_index, new_columns, self.dtypes.copy()
        )

    # END Map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def map_across_full_axis_select_indices(
        self, axis, func, indices, keep_remaining=False
    ):
        """Maps function to select indices along full axis.

        Args:
            axis: 0 for columns and 1 for rows.
            func: Callable mapping function over the BlockParitions.
            indices: indices along axis to map over.
            keep_remaining: True if keep indices where function was not applied.

        Returns:
            BaseBlockPartitions containing the result of mapping func over axis on indices.
        """
        return self.data.apply_func_to_select_indices_along_full_axis(
            axis, func, indices, keep_remaining
        )

    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            DataManager containing quantiles of original DataManager along an axis.
        """
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q")
        numeric_only = kwargs.get("numeric_only", True)
        assert isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list))

        if numeric_only:
            new_columns = self.numeric_columns()
        else:
            new_columns = [
                col
                for col, dtype in zip(self.columns, self.dtypes)
                if (is_numeric_dtype(dtype) or is_datetime_or_timedelta_dtype(dtype))
            ]
        if axis:
            # If along rows, then drop the nonnumeric columns, record the index, and
            # take transpose. We have to do this because if we don't, the result is all
            # in one column for some reason.
            nonnumeric = [
                col
                for col, dtype in zip(self.columns, self.dtypes)
                if not is_numeric_dtype(dtype)
            ]
            query_compiler = self.drop(columns=nonnumeric)
            new_columns = query_compiler.index
            numeric_indices = list(query_compiler.index.get_indexer_for(new_columns))
            query_compiler = query_compiler.transpose()
            kwargs.pop("axis")
        else:
            query_compiler = self
            numeric_indices = list(self.columns.get_indexer_for(new_columns))

        def quantile_builder(df, internal_indices=[], **kwargs):
            return pandas.DataFrame.quantile(df, **kwargs)

        func = self._prepare_method(quantile_builder, **kwargs)
        q_index = pandas.Float64Index(q)
        new_data = query_compiler.map_across_full_axis_select_indices(
            0, func, numeric_indices
        )
        return self.__constructor__(new_data, q_index, new_columns)

    # END Map across rows/columns

    # Head/Tail/Front/Back
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            DataManager containing the first n rows of the original DataManager.
        """
        # We grab the front if it is transposed and flag as transposed so that
        # we are not physically updating the data from this manager. This
        # allows the implementation to stay modular and reduces data copying.
        if n < 0:
            n = max(0, len(self.index) + n)
        if self._is_transposed:
            # Transpose the blocks back to their original orientation first to
            # ensure that we extract the correct data on each node. The index
            # on a transposed manager is already set to the correct value, so
            # we need to only take the head of that instead of re-transposing.
            result = self.__constructor__(
                self.data.transpose().take(1, n).transpose(),
                self.index[:n],
                self.columns,
                self._dtype_cache,
            )
            result._is_transposed = True
        else:
            result = self.__constructor__(
                self.data.take(0, n), self.index[:n], self.columns, self._dtype_cache
            )
        return result

    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            DataManager containing the last n rows of the original DataManager.
        """
        # See head for an explanation of the transposed behavior
        if n < 0:
            n = max(0, len(self.index) + n)
        if n == 0:
            index = pandas.Index([])
        else:
            index = self.index[-n:]
        if self._is_transposed:
            result = self.__constructor__(
                self.data.transpose().take(1, -n).transpose(),
                index,
                self.columns,
                self._dtype_cache,
            )
            result._is_transposed = True
        else:
            result = self.__constructor__(
                self.data.take(0, -n), index, self.columns, self._dtype_cache
            )

        return result

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            DataManager containing the first n columns of the original DataManager.
        """
        new_dtypes = (
            self._dtype_cache if self._dtype_cache is None else self._dtype_cache[:n]
        )
        # See head for an explanation of the transposed behavior
        if self._is_transposed:
            result = self.__constructor__(
                self.data.transpose().take(0, n).transpose(),
                self.index,
                self.columns[:n],
                new_dtypes,
            )
            result._is_transposed = True
        else:
            result = self.__constructor__(
                self.data.take(1, n), self.index, self.columns[:n], new_dtypes
            )
        return result

    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            DataManager containing the last n columns of the original DataManager.
        """
        new_dtypes = (
            self._dtype_cache if self._dtype_cache is None else self._dtype_cache[-n:]
        )
        # See head for an explanation of the transposed behavior
        if self._is_transposed:
            result = self.__constructor__(
                self.data.transpose().take(0, -n).transpose(),
                self.index,
                self.columns[-n:],
                new_dtypes,
            )
            result._is_transposed = True
        else:
            result = self.__constructor__(
                self.data.take(1, -n), self.index, self.columns[-n:], new_dtypes
            )
        return result

    # End Head/Tail/Front/Back

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
        # TODO create a way to clean up this object.
        return

    # END Data Management Methods

    # To/From Pandas
    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the DataManager.
        """
        df = self.data.to_pandas(is_transposed=self._is_transposed)
        if df.empty:
            dtype_dict = {
                col_name: pandas.Series(dtype=self.dtypes[col_name])
                for col_name in self.columns
            }
            df = pandas.DataFrame(dtype_dict, self.index)
        else:
            ErrorMessage.catch_bugs_and_request_email(
                len(df.index) != len(self.index) or len(df.columns) != len(self.columns)
            )
            df.index = self.index
            df.columns = self.columns
        return df

    @classmethod
    def from_pandas(cls, df, block_partitions_cls):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Args:
            cls: DataManger object to convert the DataFrame to.
            df: Pandas DataFrame object.
            block_partitions_cls: BlockParitions object to store partitions

        Returns:
            Returns DataManager containing data from the Pandas DataFrame.
        """
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes
        new_data = block_partitions_cls.from_pandas(df)
        return cls(new_data, new_index, new_columns, dtypes=new_dtypes)

    # __getitem__ methods
    def getitem_single_key(self, key):
        """Get item for a single target index.

        Args:
            key: Target index by which to retrieve data.

        Returns:
            A new PandasDataManager.
        """
        new_data = self.getitem_column_array([key])
        if len(self.columns.get_indexer_for([key])) > 1:
            return new_data
        else:
            # This is the case that we are returning a single Series.
            # We do this post processing because everything is treated a a list
            # from here on, and that will result in a DataFrame.
            return new_data.to_pandas()[key]

    def getitem_column_array(self, key):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.

        Returns:
            A new PandasDataManager.
        """
        # Convert to list for type checking
        numeric_indices = list(self.columns.get_indexer_for(key))

        # Internal indices is left blank and the internal
        # `apply_func_to_select_indices` will do the conversion and pass it in.
        def getitem(df, internal_indices=[]):
            return df.iloc[:, internal_indices]

        result = self.data.apply_func_to_select_indices(
            0, getitem, numeric_indices, keep_remaining=False
        )
        # We can't just set the columns to key here because there may be
        # multiple instances of a key.
        new_columns = self.columns[numeric_indices]
        new_dtypes = self.dtypes[numeric_indices]
        return self.__constructor__(result, self.index, new_columns, new_dtypes)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new PandasDataManager.
        """
        # Convert to list for type checking
        key = list(key)

        def getitem(df, internal_indices=[]):
            return df.iloc[internal_indices]

        result = self.data.apply_func_to_select_indices(
            1, getitem, key, keep_remaining=False
        )
        # We can't just set the index to key here because there may be multiple
        # instances of a key.
        new_index = self.index[key]
        return self.__constructor__(result, new_index, self.columns, self._dtype_cache)

    # END __getitem__ methods

    # __delitem__ and drop
    # These will change the shape of the resulting data.
    def delitem(self, key):
        return self.drop(columns=[key])

    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new PandasDataManager.
        """
        if index is None:
            new_data = self.data
            new_index = self.index
        else:

            def delitem(df, internal_indices=[]):
                return df.drop(index=df.index[internal_indices])

            numeric_indices = list(self.index.get_indexer_for(index))
            new_data = self.data.apply_func_to_select_indices(
                1, delitem, numeric_indices, keep_remaining=True
            )
            # We can't use self.index.drop with duplicate keys because in Pandas
            # it throws an error.
            new_index = self.index[~self.index.isin(index)]
        if columns is None:
            new_columns = self.columns
            new_dtypes = self.dtypes
        else:

            def delitem(df, internal_indices=[]):
                return df.drop(columns=df.columns[internal_indices])

            numeric_indices = list(self.columns.get_indexer_for(columns))
            new_data = new_data.apply_func_to_select_indices(
                0, delitem, numeric_indices, keep_remaining=True
            )

            new_columns = self.columns[~self.columns.isin(columns)]
            new_dtypes = self.dtypes.drop(columns)
        return self.__constructor__(new_data, new_index, new_columns, new_dtypes)

    # END __delitem__ and drop

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
            from modin.pandas.series import SeriesView

            if isinstance(value, (pandas.Series, SeriesView)):
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

    def _post_process_apply(self, result_data, axis, try_scale=True):
        """Recompute the index after applying function.

        Args:
            result_data: a BaseBlockPartitions object.
            axis: Target axis along which function was applied.

        Returns:
            A new PandasQueryCompiler.
        """
        if try_scale:
            try:
                internal_index = self.compute_index(0, result_data, True)
            except IndexError:
                internal_index = self.compute_index(0, result_data, False)
            try:
                internal_columns = self.compute_index(1, result_data, True)
            except IndexError:
                internal_columns = self.compute_index(1, result_data, False)
        else:
            internal_index = self.compute_index(0, result_data, False)
            internal_columns = self.compute_index(1, result_data, False)
        if not axis:
            index = internal_index
            # We check if the two columns are the same length because if
            # they are the same length, `self.columns` is the correct index.
            # However, if the operation resulted in a different number of columns,
            # we must use the derived columns from `self.compute_index()`.
            if len(internal_columns) != len(self.columns):
                columns = internal_columns
            else:
                columns = self.columns
        else:
            columns = internal_columns
            # See above explanation for checking the lengths of columns
            if len(internal_index) != len(self.index):
                index = internal_index
            else:
                index = self.index
        # `apply` and `aggregate` can return a Series or a DataFrame object,
        # and since we need to handle each of those differently, we have to add
        # this logic here.
        if len(columns) == 0:
            series_result = result_data.to_pandas(False)
            if not axis and len(series_result) == len(self.columns):
                index = self.columns
            elif axis and len(series_result) == len(self.index):
                index = self.index
            series_result.index = index
            return series_result
        return self.__constructor__(result_data, index, columns)

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

        if axis == 0:
            index = self.columns
        else:
            index = self.index
        func = {idx: func[key] for key in func for idx in index.get_indexer_for([key])}

        def dict_apply_builder(df, func_dict={}):
            return df.apply(func_dict, *args, **kwargs)

        result_data = self.data.apply_func_to_select_indices_along_full_axis(
            axis, dict_apply_builder, func, keep_remaining=False
        )
        full_result = self._post_process_apply(result_data, axis)
        # The columns can get weird because we did not broadcast them to the
        # partitions and we do not have any guarantee that they are correct
        # until here. Fortunately, the keys of the function will tell us what
        # the columns are.
        if isinstance(full_result, pandas.Series):
            full_result.index = [self.columns[idx] for idx in func]
        return full_result

    def _list_like_func(self, func, axis, *args, **kwargs):
        """Apply list-like function across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        func_prepared = self._prepare_method(lambda df: df.apply(func, *args, **kwargs))
        new_data = self.map_across_full_axis(axis, func_prepared)
        # When the function is list-like, the function names become the index
        new_index = [f if isinstance(f, string_types) else f.__name__ for f in func]
        return self.__constructor__(new_data, new_index, self.columns)

    def _callable_func(self, func, axis, *args, **kwargs):
        """Apply callable functions across given axis.

        Args:
            func: The functions to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """

        def callable_apply_builder(df, func, axis, index, *args, **kwargs):
            if not axis:
                df.index = index
                df.columns = pandas.RangeIndex(len(df.columns))
            else:
                df.columns = index
                df.index = pandas.RangeIndex(len(df.index))

            result = df.apply(func, axis=axis, *args, **kwargs)
            return result

        index = self.index if not axis else self.columns
        func_prepared = self._prepare_method(
            lambda df: callable_apply_builder(df, func, axis, index, *args, **kwargs)
        )
        result_data = self.map_across_full_axis(axis, func_prepared)
        return self._post_process_apply(result_data, axis)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.
    def _manual_repartition(self, axis, repartition_func, **kwargs):
        """This method applies all manual partitioning functions.

        Args:
            axis: The axis to shuffle data along.
            repartition_func: The function used to repartition data.

        Returns:
            A `BaseBlockPartitions` object.
        """
        func = self._prepare_method(repartition_func, **kwargs)
        return self.data.manual_shuffle(axis, func)

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
        result_data = self.map_across_full_axis(axis, func_prepared)
        if axis == 0:
            index = self.compute_index(0, result_data, False)
            columns = self.compute_index(1, result_data, True)
        else:
            index = self.compute_index(0, result_data, True)
            columns = self.compute_index(1, result_data, False)
        # If the result is a Series, this is how `compute_index` returns the columns.
        if len(columns) == 0 and len(index) != 0:
            return self._post_process_apply(result_data, axis, try_scale=True)
        else:
            return self.__constructor__(result_data, index, columns)

    # END Manual Partitioning methods

    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new PandasDataManager.
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
        columns_applied = self.map_across_full_axis(
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

    # Indexing
    def view(self, index=None, columns=None):
        index_map_series = pandas.Series(np.arange(len(self.index)), index=self.index)
        column_map_series = pandas.Series(
            np.arange(len(self.columns)), index=self.columns
        )
        if index is not None:
            index_map_series = index_map_series.reindex(index)
        if columns is not None:
            column_map_series = column_map_series.reindex(columns)
        return PandasQueryCompilerView(
            self.data,
            index_map_series.index,
            column_map_series.index,
            self.dtypes,
            index_map_series,
            column_map_series,
        )

    def squeeze(self, ndim=0, axis=None):
        to_squeeze = self.data.to_pandas()
        # This is the case for 1xN or Nx1 DF - Need to call squeeze
        if ndim == 1:
            if axis is None:
                axis = 0 if self.data.shape[1] > 1 else 1
            squeezed = pandas.Series(to_squeeze.squeeze(axis))
            scaler_axis = self.columns if axis else self.index
            non_scaler_axis = self.index if axis else self.columns
            squeezed.name = scaler_axis[0]
            squeezed.index = non_scaler_axis
            return squeezed
        # This is the case for a 1x1 DF - We don't need to squeeze
        else:
            return to_squeeze.values[0][0]

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            partition.iloc[row_internal_indices, col_internal_indices] = item
            return partition

        mutated_blk_partitions = self.data.apply_func_to_indices_both_axis(
            func=iloc_mut,
            row_indices=row_numeric_index,
            col_indices=col_numeric_index,
            mutate=True,
            item_to_distribute=broadcasted_items,
        )
        self.data = mutated_blk_partitions

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
    This class represent a view of the PandasDataManager

    In particular, the following constraints are broken:
    - (len(self.index), len(self.columns)) != self.data.shape

    Note:
        The constraint will be satisfied when we get the data
    """

    def __init__(
        self,
        block_partitions_object: BaseBlockPartitions,
        index: pandas.Index,
        columns: pandas.Index,
        dtypes=None,
        index_map_series: pandas.Series = None,
        columns_map_series: pandas.Series = None,
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
        assert index.equals(index_map_series.index)
        assert columns.equals(columns_map_series.index)

        self.index_map = index_map_series
        self.columns_map = columns_map_series

        PandasQueryCompiler.__init__(
            self, block_partitions_object, index, columns, dtypes
        )

    _dtype_cache = None

    def _set_dtype(self, dtypes):
        self._dtype_cache = dtypes

    def _get_dtype(self):
        """Override the parent on this to avoid getting the wrong dtypes"""
        if self._dtype_cache is None:
            map_func = self._prepare_method(lambda df: df.dtypes)

            def dtype_builder(df):
                return df.apply(lambda row: find_common_type(row.values), axis=0)

            self._dtype_cache = self.full_reduce(0, map_func, dtype_builder)
            self._dtype_cache.index = self.columns
        elif not self._dtype_cache.index.equals(self.columns):
            self._dtype_cache = self._dtype_cache.reindex(self.columns)
        return self._dtype_cache

    dtypes = property(_get_dtype, _set_dtype)

    def __constructor__(
        self,
        block_partitions_object: BaseBlockPartitions,
        index: pandas.Index,
        columns: pandas.Index,
        dtypes=None,
    ):
        return PandasQueryCompiler(block_partitions_object, index, columns, dtypes)

    def _get_data(self) -> BaseBlockPartitions:
        """Perform the map step

        Returns:
            A BaseBlockPartitions object.
        """

        def iloc(partition, row_internal_indices, col_internal_indices):
            return partition.iloc[row_internal_indices, col_internal_indices]

        masked_data = self.parent_data.apply_func_to_indices_both_axis(
            func=iloc,
            row_indices=self.index_map.values,
            col_indices=self.columns_map.values,
            lazy=False,
            keep_remaining=False,
        )
        return masked_data

    def _set_data(self, new_data):
        """Note this setter will be called by the
            `super(PandasDataManagerView).__init__` function
        """
        self.parent_data = new_data

    data = property(_get_data, _set_data)

    def global_idx_to_numeric_idx(self, axis, indices):
        assert axis in ["row", "col", "columns"]
        if axis == "row":
            return self.index_map.loc[indices].index
        elif axis in ["col", "columns"]:
            return self.columns_map.loc[indices].index
