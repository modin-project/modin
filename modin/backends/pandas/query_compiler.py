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
)
from pandas.core.index import ensure_index
from pandas.core.base import DataError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.error_message import ErrorMessage
from modin.backends.base.query_compiler import BaseQueryCompiler


class PandasQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
        with a Pandas backend. This logic is specific to Pandas."""

    def __init__(
        self, block_partitions_object, index, columns, dtypes=None, is_transposed=False
    ):
        assert isinstance(block_partitions_object, BaseFrameManager)
        self.data = block_partitions_object
        self.index = index
        self.columns = columns
        if dtypes is not None:
            self._dtype_cache = dtypes
        self._is_transposed = int(is_transposed)

    # Index, columns and dtypes objects
    _dtype_cache = None

    def _get_dtype(self):
        if self._dtype_cache is None:

            def dtype_builder(df):
                return df.apply(lambda row: find_common_type(row.values), axis=0)

            map_func = self._prepare_method(
                self._build_mapreduce_func(lambda df: df.dtypes)
            )
            reduce_func = self._build_mapreduce_func(dtype_builder)
            # For now we will use a pandas Series for the dtypes.
            if len(self.columns) > 0:
                self._dtype_cache = (
                    self._full_reduce(0, map_func, reduce_func).to_pandas().iloc[0]
                )
            else:
                self._dtype_cache = pandas.Series([])
            # reset name to None because we use "__reduced__" internally
            self._dtype_cache.name = None
        return self._dtype_cache

    dtypes = property(_get_dtype)

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

    def _validate_set_axis(self, new_labels, old_labels):
        new_labels = ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
            )
        return new_labels

    _index_cache = None
    _columns_cache = None

    def _get_index(self):
        return self._index_cache

    def _get_columns(self):
        return self._columns_cache

    def _set_index(self, new_index):
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index

    def _set_columns(self, new_columns):
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)
    # END Index, columns, and dtypes objects

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
                if len(internal_indices) > 0:
                    return pandas_func(
                        df.T, internal_indices=internal_indices, **kwargs
                    )
                return pandas_func(df.T, **kwargs)

        else:

            def helper(df, internal_indices=[]):
                if len(internal_indices) > 0:
                    return pandas_func(df, internal_indices=internal_indices, **kwargs)
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
    def add_prefix(self, prefix, axis=1):
        if axis == 1:
            new_columns = self.columns.map(lambda x: str(prefix) + str(x))
            if self._dtype_cache is not None:
                new_dtype_cache = self._dtype_cache.copy()
                new_dtype_cache.index = new_columns
            else:
                new_dtype_cache = None
            new_index = self.index
        else:
            new_index = self.index.map(lambda x: str(prefix) + str(x))
            new_columns = self.columns
            new_dtype_cache = self._dtype_cache
        return self.__constructor__(
            self.data, new_index, new_columns, new_dtype_cache, self._is_transposed
        )

    def add_suffix(self, suffix, axis=1):
        if axis == 1:
            new_columns = self.columns.map(lambda x: str(x) + str(suffix))
            if self._dtype_cache is not None:
                new_dtype_cache = self._dtype_cache.copy()
                new_dtype_cache.index = new_columns
            else:
                new_dtype_cache = None
            new_index = self.index
        else:
            new_index = self.index.map(lambda x: str(x) + str(suffix))
            new_columns = self.columns
            new_dtype_cache = self._dtype_cache
        return self.__constructor__(
            self.data, new_index, new_columns, new_dtype_cache, self._is_transposed
        )

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(
            self.data.copy(),
            self.index.copy(),
            self.columns.copy(),
            self._dtype_cache,
            self._is_transposed,
        )

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
        """Joins a list or two objects together.

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
        if self._is_transposed:
            # If others are transposed, we handle that behavior correctly in
            # `copartition`, but it is not handled correctly in the case that `self` is
            # transposed.
            return (
                self.transpose()
                ._append_list_of_managers(
                    [o.transpose() for o in others], axis ^ 1, **kwargs
                )
                .transpose()
            )
        assert all(
            isinstance(other, type(self)) for other in others
        ), "Different Manager objects are being used. This is not allowed"

        sort = kwargs.get("sort", None)
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        new_self, to_append, joined_axis = self.copartition(
            axis ^ 1,
            others,
            join,
            sort,
            force_repartition=any(obj._is_transposed for obj in [self] + others),
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
        ), "This method is for lists of QueryCompiler objects only"
        assert all(
            isinstance(other, type(self)) for other in others
        ), "Different Manager objects are being used. This is not allowed"
        # Uses join's default value (though should not revert to default)
        how = kwargs.get("how", "left")
        sort = kwargs.get("sort", False)
        lsuffix = kwargs.get("lsuffix", "")
        rsuffix = kwargs.get("rsuffix", "")
        new_self, to_join, joined_index = self.copartition(
            0,
            others,
            how,
            sort,
            force_repartition=any(obj._is_transposed for obj in [self] + others),
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
                reindex_left = self._prepare_method(compute_reindex(left_old_idx))
            if right_old_idxes[i].equals(joined_index) and not force_repartition:
                reindex_right = None
            else:
                reindex_right = compute_reindex(right_old_idxes[i])
            reindexed_self, reindexed_other = reindexed_self.copartition_datasets(
                axis,
                other[i].data,
                reindex_left,
                reindex_right,
                other[i]._is_transposed,
            )
            reindexed_other_list.append(reindexed_other)
        return reindexed_self, reindexed_other_list, joined_index

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
            Pandas DataFrame of the QueryCompiler.
        """
        df = self.data.to_pandas(is_transposed=self._is_transposed)
        if df.empty:
            if len(self.columns) != 0:
                df = pandas.DataFrame(columns=self.columns).astype(self.dtypes)
            else:
                df = pandas.DataFrame(columns=self.columns, index=self.index)
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
            Returns QueryCompiler containing data from the Pandas DataFrame.
        """
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes
        new_data = block_partitions_cls.from_pandas(df)
        return cls(new_data, new_index, new_columns, dtypes=new_dtypes)

    # END To/From Pandas

    # To NumPy
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy Array.

        Returns:
            NumPy Array of the QueryCompiler.
        """
        arr = self.data.to_numpy(is_transposed=self._is_transposed)
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
        reindexed_self, reindexed_other_list, joined_index = self.copartition(
            0, other, how_to_join, sort=False
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
            New QueryCompiler with new data and index.
        """
        axis = kwargs.get("axis", 0)
        axis = pandas.DataFrame()._get_axis_number(axis) if axis is not None else 0
        if isinstance(other, type(self)):
            # If this QueryCompiler is transposed, copartition can sometimes fail to
            # properly co-locate the data. It does not fail if other is transposed, so
            # if this object is transposed, we will transpose both and do the operation,
            # then transpose at the end.
            if self._is_transposed:
                return (
                    self.transpose()
                    ._inter_manager_operations(
                        other.transpose(), "outer", lambda x, y: func(x, y, **kwargs)
                    )
                    .transpose()
                )
            return self._inter_manager_operations(
                other, "outer", lambda x, y: func(x, y, **kwargs)
            )
        else:
            return self._scalar_operations(
                axis, other, lambda df: func(df, other, **kwargs)
            )

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
        func = self._prepare_method(pandas.DataFrame.clip, **kwargs)
        if is_list_like(lower) or is_list_like(upper):
            df = self._map_across_full_axis(axis, func)
            return self.__constructor__(df, self.index, self.columns)
        return self._scalar_operations(axis, lower or upper, func)

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

    # Single Manager scalar operations (e.g. add to scalar, list of scalars)
    def _scalar_operations(self, axis, scalar, func):
        """Handler for mapping scalar operations across a Manager.

        Args:
            axis: The axis index object to execute the function on.
            scalar: The scalar value to map.
            func: The function to use on the Manager with the scalar.

        Returns:
            A new QueryCompiler with updated data and new index.
        """
        if isinstance(scalar, (list, np.ndarray, pandas.Series)):
            new_index = self.index if axis == 0 else self.columns

            def list_like_op(df):
                if axis == 0:
                    df.index = new_index
                else:
                    df.columns = new_index
                return func(df)

            new_data = self._map_across_full_axis(
                axis, self._prepare_method(list_like_op)
            )
            if axis == 1 and isinstance(scalar, pandas.Series):
                new_columns = self.columns.union(
                    [label for label in scalar.index if label not in self.columns]
                )
            else:
                new_columns = self.columns
            return self.__constructor__(new_data, self.index, new_columns)
        else:
            return self._map_partitions(self._prepare_method(func))

    # END Single Manager scalar operations

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manger.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            A new QueryCompiler with updated data and new index.
        """
        if self._is_transposed:
            return (
                self.transpose()
                .reindex(axis=axis ^ 1, labels=labels, **kwargs)
                .transpose()
            )

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
        new_data = self.data.transpose(*args, **kwargs)
        # Switch the index and columns and transpose the data within the blocks.
        new_manager = self.__constructor__(
            new_data, self.columns, self.index, is_transposed=self._is_transposed ^ 1
        )
        return new_manager

    # END Transpose

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # This will return a new QueryCompiler, which will be handled in the front end.
    def _full_reduce(self, axis, map_func, reduce_func=None):
        """Apply function that will reduce the data to a Pandas Series.

        Args:
            axis: 0 for columns and 1 for rows. Default is 0.
            map_func: Callable function to map the dataframe.
            reduce_func: Callable function to reduce the dataframe. If none,
                then apply map_func twice.

        Return:
            A new QueryCompiler object containing the results from map_func and
            reduce_func.
        """
        if reduce_func is None:
            reduce_func = map_func

        mapped_parts = self.data.map_across_blocks(map_func)
        full_frame = mapped_parts.map_across_full_axis(axis, reduce_func)
        if axis == 0:
            columns = self.columns
            return self.__constructor__(
                full_frame, index=["__reduced__"], columns=columns
            )
        else:
            index = self.index
            return self.__constructor__(
                full_frame, index=index, columns=["__reduced__"]
            )

    def _build_mapreduce_func(self, func, **kwargs):
        def _map_reduce_func(df):
            series_result = func(df, **kwargs)
            if kwargs.get("axis", 0) == 0 and isinstance(series_result, pandas.Series):
                # In the case of axis=0, we need to keep the shape of the data
                # consistent with what we have done. In the case of a reduction, the
                # data for axis=0 should be a single value for each column. By
                # transposing the data after we convert to a DataFrame, we ensure that
                # the columns of the result line up with the columns from the data.
                # axis=1 does not have this requirement because the index already will
                # line up with the index of the data based on how pandas creates a
                # DataFrame from a Series.
                return pandas.DataFrame(series_result).T
            return pandas.DataFrame(series_result)

        return _map_reduce_func

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            A new QueryCompiler object containing counts of non-NaN objects from each
            column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().count(**kwargs)
        axis = kwargs.get("axis", 0)
        map_func = self._build_mapreduce_func(pandas.DataFrame.count, **kwargs)
        reduce_func = self._build_mapreduce_func(pandas.DataFrame.sum, **kwargs)
        return self._full_reduce(axis, map_func, reduce_func)

    def dot(self, other):
        """Computes the matrix multiplication of self and other.

        Args:
            other: The other query compiler or other array-like to matrix
            multiply with self.

        Returns:
            Returns the result of the matrix multiply.
        """
        if self._is_transposed:
            return self.transpose().dot(other).transpose()

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
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().max(**kwargs)
        mapreduce_func = self._build_mapreduce_func(pandas.DataFrame.max, **kwargs)
        return self._full_reduce(kwargs.get("axis", 0), mapreduce_func)

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            A new QueryCompiler object containing the mean from each numerical column or
            row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().mean(**kwargs)
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        sums = self.sum(**kwargs)
        counts = self.count(axis=axis, numeric_only=kwargs.get("numeric_only", None))
        if sums._is_transposed and counts._is_transposed:
            sums = sums.transpose()
            counts = counts.transpose()
        result = sums.binary_op("truediv", counts, axis=axis)
        return result.transpose() if axis == 0 else result

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            A new QueryCompiler object with the minimum value from each column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().min(**kwargs)
        mapreduce_func = self._build_mapreduce_func(pandas.DataFrame.min, **kwargs)
        return self._full_reduce(kwargs.get("axis", 0), mapreduce_func)

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

        def sum_prod_builder(df, **kwargs):
            return func(df, **kwargs)

        builder_func = self._build_mapreduce_func(sum_prod_builder, **kwargs)
        if min_count <= 1:
            return self._full_reduce(axis, builder_func)
        else:
            return self._full_axis_reduce(axis, builder_func)

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            A new QueryCompiler object with the product of each numerical column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().prod(**kwargs)
        return self._process_sum_prod(pandas.DataFrame.prod, **kwargs)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            A new QueryCompiler object with the sum of each numerical column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().sum(**kwargs)
        return self._process_sum_prod(pandas.DataFrame.sum, **kwargs)

    def _process_all_any(self, func, **kwargs):
        """Calculates if any or all the values are true.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        axis = kwargs.get("axis", 0)
        axis = 0 if axis is None else axis
        kwargs["axis"] = axis
        builder_func = self._build_mapreduce_func(func, **kwargs)
        return self._full_reduce(axis, builder_func)

    def all(self, **kwargs):
        """Returns whether all the elements are true, potentially over an axis.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        if self._is_transposed:
            # Pandas ignores on axis=1
            kwargs["bool_only"] = False
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().all(**kwargs)
        return self._process_all_any(lambda df, **kwargs: df.all(**kwargs), **kwargs)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            A new QueryCompiler object containing boolean values or boolean.
        """
        if self._is_transposed:
            if kwargs.get("axis", 0) == 1:
                # Pandas ignores on axis=1
                kwargs["bool_only"] = False
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().any(**kwargs)
        return self._process_all_any(lambda df, **kwargs: df.any(**kwargs), **kwargs)

    # END Full Reduce operations

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    def _map_partitions(self, func, new_dtypes=None):
        return self.__constructor__(
            self.data.map_across_blocks(func), self.index, self.columns, new_dtypes
        )

    def abs(self):
        func = self._prepare_method(pandas.DataFrame.abs)
        return self._map_partitions(func, new_dtypes=self.dtypes.copy())

    def applymap(self, func):
        remote_func = self._prepare_method(pandas.DataFrame.applymap, func=func)
        return self._map_partitions(remote_func)

    def invert(self):
        remote_func = self._prepare_method(pandas.DataFrame.__invert__)
        return self._map_partitions(remote_func)

    def isin(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.isin, **kwargs)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self._map_partitions(func, new_dtypes=new_dtypes)

    def isna(self):
        func = self._prepare_method(pandas.DataFrame.isna)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self._map_partitions(func, new_dtypes=new_dtypes)

    def memory_usage(self, axis=0, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            A new QueryCompiler object containing the memory usage of each column.
        """
        if self._is_transposed:
            return self.transpose().memory_usage(axis=1, **kwargs)

        def memory_usage_builder(df, **kwargs):
            axis = kwargs.pop("axis")
            # We have to manually change the orientation of the data within the
            # partitions because memory_usage does not take in an axis argument
            # and always does it along columns.
            if axis:
                df = df.T
            result = df.memory_usage(**kwargs)
            return result

        def sum_memory_usage(df, **kwargs):
            axis = kwargs.pop("axis")
            return df.sum(axis=axis)

        # Even though memory_usage does not take in an axis argument, we have to
        # pass in an axis kwargs for _build_mapreduce_func to properly arrange
        # the results.
        map_func = self._build_mapreduce_func(memory_usage_builder, axis=axis, **kwargs)
        reduce_func = self._build_mapreduce_func(sum_memory_usage, axis=axis, **kwargs)
        return self._full_reduce(axis, map_func, reduce_func)

    def negative(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.__neg__, **kwargs)
        return self._map_partitions(func)

    def notna(self):
        func = self._prepare_method(pandas.DataFrame.notna)
        new_dtypes = pandas.Series(
            [np.dtype("bool") for _ in self.columns], index=self.columns
        )
        return self._map_partitions(func, new_dtypes=new_dtypes)

    def round(self, **kwargs):
        func = self._prepare_method(pandas.DataFrame.round, **kwargs)
        return self._map_partitions(func, new_dtypes=self._dtype_cache)

    # END Map partitions operations

    # String map partition operations
    def _str_map_partitions(self, func, new_dtypes=None, **kwargs):
        def str_op_builder(df, **kwargs):
            str_series = df.squeeze().str
            return func(str_series, **kwargs).to_frame()

        builder_func = self._prepare_method(str_op_builder, **kwargs)
        return self._map_partitions(builder_func, new_dtypes=new_dtypes)

    def str_split(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.split, new_dtypes=self.dtypes, **kwargs
        )

    def str_rsplit(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.rsplit, new_dtypes=self.dtypes, **kwargs
        )

    def str_get(self, i):
        return self._str_map_partitions(
            pandas.Series.str.get, new_dtypes=self.dtypes, i=i
        )

    def str_join(self, sep):
        return self._str_map_partitions(
            pandas.Series.str.join, new_dtypes=self.dtypes, sep=sep
        )

    def str_contains(self, pat, **kwargs):
        kwargs["pat"] = pat
        new_dtypes = pandas.Series([bool])
        return self._str_map_partitions(
            pandas.Series.str.contains, new_dtypes=new_dtypes, **kwargs
        )

    def str_replace(self, pat, repl, **kwargs):
        kwargs["pat"] = pat
        kwargs["repl"] = repl
        return self._str_map_partitions(
            pandas.Series.str.replace, new_dtypes=self.dtypes, **kwargs
        )

    def str_repeats(self, repeats):
        return self._str_map_partitions(
            pandas.Series.str.repeats, new_dtypes=self.dtypes, repeats=repeats
        )

    def str_pad(self, width, **kwargs):
        kwargs["width"] = width
        return self._str_map_partitions(
            pandas.Series.str.pad, new_dtypes=self.dtypes, **kwargs
        )

    def str_center(self, width, **kwargs):
        kwargs["width"] = width
        return self._str_map_partitions(
            pandas.Series.str.center, new_dtypes=self.dtypes, **kwargs
        )

    def str_ljust(self, width, **kwargs):
        kwargs["width"] = width
        return self._str_map_partitions(
            pandas.Series.str.ljust, new_dtypes=self.dtypes, **kwargs
        )

    def str_rjust(self, width, **kwargs):
        kwargs["width"] = width
        return self._str_map_partitions(
            pandas.Series.str.rjust, new_dtypes=self.dtypes, **kwargs
        )

    def str_zfill(self, width):
        return self._str_map_partitions(
            pandas.Series.str.zfill, new_dtypes=self.dtypes, width=width
        )

    def str_wrap(self, width, **kwargs):
        kwargs["width"] = width
        return self._str_map_partitions(
            pandas.Series.str.wrap, new_dtypes=self.dtypes, **kwargs
        )

    def str_slice(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.slice, new_dtypes=self.dtypes, **kwargs
        )

    def str_slice_replace(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.slice_replace, new_dtypes=self.dtypes, **kwargs
        )

    def str_count(self, pat, **kwargs):
        kwargs["pat"] = pat
        new_dtypes = pandas.Series([int])
        # We have to pass in a lambda because pandas.Series.str.count does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.count(**kwargs), new_dtypes=new_dtypes
        )

    def str_startswith(self, pat, **kwargs):
        kwargs["pat"] = pat
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.startswith does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.startswith(**kwargs), new_dtypes=new_dtypes
        )

    def str_endswith(self, pat, **kwargs):
        kwargs["pat"] = pat
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.endswith does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.endswith(**kwargs), new_dtypes=new_dtypes
        )

    def str_findall(self, pat, **kwargs):
        kwargs["pat"] = pat
        # We have to pass in a lambda because pandas.Series.str.findall does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.findall(**kwargs), new_dtypes=self.dtypes
        )

    def str_match(self, pat, **kwargs):
        kwargs["pat"] = pat
        return self._str_map_partitions(
            pandas.Series.str.match, new_dtypes=self.dtypes, **kwargs
        )

    def str_len(self):
        new_dtypes = pandas.Series([int])
        return self._str_map_partitions(pandas.Series.str.len, new_dtypes=new_dtypes)

    def str_strip(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.strip, new_dtypes=self.dtypes, **kwargs
        )

    def str_rstrip(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.rstrip, new_dtypes=self.dtypes, **kwargs
        )

    def str_lstrip(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.lstrip, new_dtypes=self.dtypes, **kwargs
        )

    def str_partition(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.partition, new_dtypes=self.dtypes, **kwargs
        )

    def str_rpartition(self, **kwargs):
        return self._str_map_partitions(
            pandas.Series.str.rpartition, new_dtypes=self.dtypes, **kwargs
        )

    def str_lower(self):
        # We have to pass in a lambda because pandas.Series.str.lower does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.lower(), new_dtypes=self.dtypes
        )

    def str_upper(self):
        # We have to pass in a lambda because pandas.Series.str.upper does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.upper(), new_dtypes=self.dtypes
        )

    def str_find(self, sub, **kwargs):
        kwargs["sub"] = sub
        return self._str_map_partitions(
            pandas.Series.str.find, new_dtypes=self.dtypes, **kwargs
        )

    def str_rfind(self, sub, **kwargs):
        kwargs["sub"] = sub
        return self._str_map_partitions(
            pandas.Series.str.rfind, new_dtypes=self.dtypes, **kwargs
        )

    def str_index(self, sub, **kwargs):
        kwargs["sub"] = sub
        return self._str_map_partitions(
            pandas.Series.str.index, new_dtypes=self.dtypes, **kwargs
        )

    def str_rindex(self, sub, **kwargs):
        kwargs["sub"] = sub
        return self._str_map_partitions(
            pandas.Series.str.rindex, new_dtypes=self.dtypes, **kwargs
        )

    def str_capitalize(self):
        # We have to pass in a lambda because pandas.Series.str.capitalize does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.capitalize(), new_dtypes=self.dtypes
        )

    def str_swapcase(self):
        # We have to pass in a lambda because pandas.Series.str.swapcase does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.swapcase(), new_dtypes=self.dtypes
        )

    def str_normalize(self, form):
        return self._str_map_partitions(
            pandas.Series.str.normalize, new_dtypes=self.dtypes, form=form
        )

    def str_translate(self, table, **kwargs):
        kwargs["table"] = table
        return self._str_map_partitions(
            pandas.Series.str.translate, new_dtypes=self.dtypes, **kwargs
        )

    def str_isalnum(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isalnum does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isalnum(), new_dtypes=new_dtypes
        )

    def str_isalpha(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isalpha does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isalpha(), new_dtypes=new_dtypes
        )

    def str_isdigit(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isdigit does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isdigit(), new_dtypes=new_dtypes
        )

    def str_isspace(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isspace does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isspace(), new_dtypes=new_dtypes
        )

    def str_islower(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.islower does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.islower(), new_dtypes=new_dtypes
        )

    def str_isupper(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isupper does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isupper(), new_dtypes=new_dtypes
        )

    def str_istitle(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.istitle does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.istitle(), new_dtypes=new_dtypes
        )

    def str_isnumeric(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isnumeric does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isnumeric(), new_dtypes=new_dtypes
        )

    def str_isdecimal(self):
        new_dtypes = pandas.Series([bool])
        # We have to pass in a lambda because pandas.Series.str.isdecimal does not exist for python2
        return self._str_map_partitions(
            lambda str_series: str_series.isdecimal(), new_dtypes=new_dtypes
        )

    # END String map partitions operations

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
            if (
                not isinstance(dtype, type(self.dtypes[column]))
                or dtype != self.dtypes[column]
            ):
                # Only add dtype only if different
                if dtype in dtype_indices.keys():
                    dtype_indices[dtype].append(numeric_indices[i])
                else:
                    dtype_indices[dtype] = [numeric_indices[i]]
                # Update the new dtype series to the proper pandas dtype
                try:
                    new_dtype = np.dtype(dtype)
                except TypeError:
                    new_dtype = dtype
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
    # This will return a new QueryCompiler object which the font end will handle.
    def _full_axis_reduce(self, axis, func, alternate_index=None):
        """Applies map that reduce Manager to series but require knowledge of full axis.

        Args:
            func: Function to reduce the Manager by. This function takes in a Manager.
            axis: axis to apply the function to.
            alternate_index: If the resulting series should have an index
                different from the current query_compiler's index or columns.

        Return:
            Pandas series containing the reduced data.
        """
        result = self.data.map_across_full_axis(axis, func)
        if axis == 0:
            columns = alternate_index if alternate_index is not None else self.columns
            return self.__constructor__(result, index=["__reduced__"], columns=columns)
        else:
            index = alternate_index if alternate_index is not None else self.index
            return self.__constructor__(result, index=index, columns=["__reduced__"])

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

        func = self._build_mapreduce_func(first_valid_index_builder)
        # We get the minimum from each column, then take the min of that to get
        # first_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = self._full_axis_reduce(0, func).min(axis=1).to_pandas().squeeze()
        return self.index[first_result]

    def idxmax(self, **kwargs):
        """Returns the first occurrence of the maximum over requested axis.

        Returns:
            A new QueryCompiler object containing the maximum of each column or axis.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().idxmax(**kwargs)

        axis = kwargs.get("axis", 0)
        index = self.index if axis == 0 else self.columns

        def idxmax_builder(df, **kwargs):
            if axis == 0:
                df.index = index
            else:
                df.columns = index
            return df.idxmax(**kwargs)

        func = self._build_mapreduce_func(idxmax_builder, **kwargs)
        return self._full_axis_reduce(axis, func)

    def idxmin(self, **kwargs):
        """Returns the first occurrence of the minimum over requested axis.

        Returns:
            A new QueryCompiler object containing the minimum of each column or axis.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().idxmin(**kwargs)

        axis = kwargs.get("axis", 0)
        index = self.index if axis == 0 else self.columns

        def idxmin_builder(df, **kwargs):
            if axis == 0:
                df.index = index
            else:
                df.columns = index
            return df.idxmin(**kwargs)

        func = self._build_mapreduce_func(idxmin_builder, **kwargs)
        return self._full_axis_reduce(axis, func)

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def last_valid_index_builder(df):
            df.index = pandas.RangeIndex(len(df.index))
            return df.apply(lambda df: df.last_valid_index())

        func = self._build_mapreduce_func(last_valid_index_builder)
        # We get the maximum from each column, then take the max of that to get
        # last_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = self._full_axis_reduce(0, func).max(axis=1).to_pandas().squeeze()
        return self.index[first_result]

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            A new QueryCompiler object containing the median of each column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().median(**kwargs)
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._build_mapreduce_func(pandas.DataFrame.median, **kwargs)
        return self._full_axis_reduce(axis, func)

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            A new QueryCompiler object of ints indexed by column or index names.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().nunique(**kwargs)
        axis = kwargs.get("axis", 0)
        func = self._build_mapreduce_func(pandas.DataFrame.nunique, **kwargs)
        return self._full_axis_reduce(axis, func)

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            A new QueryCompiler object containing the quantile of each column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().quantile_for_single_value(**kwargs)
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q", 0.5)
        assert type(q) is float

        def quantile_builder(df, **kwargs):
            try:
                return pandas.DataFrame.quantile(df, **kwargs)
            except ValueError:
                return pandas.Series()

        func = self._build_mapreduce_func(quantile_builder, **kwargs)
        result = self._full_axis_reduce(axis, func)
        if axis == 0:
            result.index = [q]
        else:
            result.columns = [q]
        return result

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            A new QueryCompiler object containing the skew of each column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().skew(**kwargs)
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._build_mapreduce_func(pandas.DataFrame.skew, **kwargs)
        return self._full_axis_reduce(axis, func)

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            A new QueryCompiler object containing the standard deviation of each column
            or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().std(**kwargs)
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._build_mapreduce_func(pandas.DataFrame.std, **kwargs)
        return self._full_axis_reduce(axis, func)

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            A new QueryCompiler object containing the variance of each column or row.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().var(**kwargs)
        # Pandas default is 0 (though not mentioned in docs)
        axis = kwargs.get("axis", 0)
        func = self._build_mapreduce_func(pandas.DataFrame.var, **kwargs)
        return self._full_axis_reduce(axis, func)

    # END Column/Row partitions reduce operations

    # Column/Row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # This will return a new QueryCompiler object which the front end will handle.
    def _full_axis_reduce_along_select_indices(self, func, axis, index):
        """Reduce Manger along select indices using function that needs full axis.

        Args:
            func: Callable that reduces the dimension of the object and requires full
                knowledge of the entire axis.
            axis: 0 for columns and 1 for rows. Defaults to 0.
            index: Index of the resulting QueryCompiler.

        Returns:
            A new QueryCompiler object with index or BaseFrameManager object.
        """
        # Convert indices to numeric indices
        old_index = self.index if axis else self.columns
        numeric_indices = [i for i, name in enumerate(old_index) if name in index]
        result = self.data.apply_func_to_select_indices_along_full_axis(
            axis, func, numeric_indices
        )
        return result

    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        # Use pandas to calculate the correct columns
        new_columns = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .describe(**kwargs)
            .columns
        )

        def describe_builder(df, internal_indices=[], **kwargs):
            return df.iloc[:, internal_indices].describe(**kwargs)

        # Apply describe and update indices, columns, and dtypes
        func = self._prepare_method(describe_builder, **kwargs)
        new_data = self._full_axis_reduce_along_select_indices(func, 0, new_columns)
        new_index = self.compute_index(0, new_data, False)
        return self.__constructor__(new_data, new_index, new_columns)

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def _map_across_full_axis(self, axis, func):
        return self.data.map_across_full_axis(axis, func)

    def _cumulative_builder(self, func, **kwargs):
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(func, **kwargs)
        new_data = self._map_across_full_axis(axis, func)
        return self.__constructor__(
            new_data, self.index, self.columns, self._dtype_cache
        )

    def cummax(self, **kwargs):
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().cummax(**kwargs).transpose()
        return self._cumulative_builder(pandas.DataFrame.cummax, **kwargs)

    def cummin(self, **kwargs):
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().cummin(**kwargs).transpose()
        return self._cumulative_builder(pandas.DataFrame.cummin, **kwargs)

    def cumsum(self, **kwargs):
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().cumsum(**kwargs).transpose()
        return self._cumulative_builder(pandas.DataFrame.cumsum, **kwargs)

    def cumprod(self, **kwargs):
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().cumprod(**kwargs).transpose()
        return self._cumulative_builder(pandas.DataFrame.cumprod, **kwargs)

    def diff(self, **kwargs):
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().diff(**kwargs).transpose()
        axis = kwargs.get("axis", 0)
        func = self._prepare_method(pandas.DataFrame.diff, **kwargs)
        new_data = self._map_across_full_axis(axis, func)
        return self.__constructor__(new_data, self.index, self.columns)

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        columns = self.index if self._is_transposed else self.columns
        index = self.columns if self._is_transposed else self.index

        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        columns_copy = pandas.DataFrame(columns=self.columns)
        columns_copy = columns_copy.eval(expr, inplace=False, **kwargs)
        expect_series = isinstance(columns_copy, pandas.Series)

        def eval_builder(df, **kwargs):
            # pop the `axis` parameter because it was needed to build the mapreduce
            # function but it is not a parameter used by `eval`.
            kwargs.pop("axis", None)
            df.columns = columns
            result = df.eval(expr, inplace=False, **kwargs)
            return result

        func = self._build_mapreduce_func(eval_builder, axis=1, **kwargs)
        new_data = self._map_across_full_axis(1, func)

        if expect_series:
            new_columns = [columns_copy.name]
            new_index = index
        else:
            new_columns = columns_copy.columns
            new_index = self.index
        return self.__constructor__(new_data, new_index, new_columns)

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
            return pandas.DataFrame(result)

        func = self._prepare_method(mode_builder, **kwargs)
        new_data = self._map_across_full_axis(axis, func)

        new_index = pandas.RangeIndex(len(self.index)) if not axis else self.index
        new_columns = self.columns if not axis else pandas.RangeIndex(len(self.columns))
        new_dtypes = self._dtype_cache
        if new_dtypes is not None:
            new_dtypes.index = new_columns
        return self.__constructor__(
            new_data, new_index, new_columns, new_dtypes
        ).dropna(axis=axis, how="all")

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

            if full_axis:
                new_data = self.data.apply_func_to_select_indices_along_full_axis(
                    axis, fillna_dict_builder, value, keep_remaining=True
                )
            else:
                new_data = self.data.apply_func_to_select_indices(
                    axis, fillna_dict_builder, value, keep_remaining=True
                )
            return self.__constructor__(new_data, self.index, self.columns)
        else:
            func = self._prepare_method(pandas.DataFrame.fillna, **kwargs)
            if full_axis:
                new_data = self._map_across_full_axis(axis, func)
                return self.__constructor__(new_data, self.index, self.columns)
            else:
                return self._map_partitions(func)

    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            QueryCompiler containing quantiles of original QueryCompiler along an axis.
        """
        if self._is_transposed:
            kwargs["axis"] = kwargs.get("axis", 0) ^ 1
            return self.transpose().quantile_for_list_of_values(**kwargs)
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
        else:
            query_compiler = self

        def quantile_builder(df, **kwargs):
            result = df.quantile(**kwargs)
            return result.T if axis == 1 else result

        func = query_compiler._prepare_method(quantile_builder, **kwargs)
        q_index = pandas.Float64Index(q)
        new_data = query_compiler._map_across_full_axis(axis, func)

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
        result = self.__constructor__(new_data, q_index, new_columns)
        return result.transpose() if axis == 1 else result

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
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
        new_data = self._map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self.compute_index(0, new_data, True)

        return self.__constructor__(new_data, new_index, self.columns, self.dtypes)

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        func = self._prepare_method(pandas.DataFrame.rank, **kwargs)
        new_data = self._map_across_full_axis(axis, func)
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
            QueryCompiler containing the data sorted by columns or indices.
        """
        axis = kwargs.pop("axis", 0)
        if self._is_transposed:
            return self.transpose().sort_index(axis=axis ^ 1, **kwargs).transpose()
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
        new_data = self._map_across_full_axis(axis, func)
        if axis:
            new_columns = pandas.Series(self.columns).sort_values(**kwargs)
            new_index = self.index
        else:
            new_index = pandas.Series(self.index).sort_values(**kwargs)
            new_columns = self.columns
        return self.__constructor__(
            new_data, new_index, new_columns, self.dtypes.copy(), self._is_transposed
        )

    # END Map across rows/columns

    # Head/Tail/Front/Back
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
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
                self._is_transposed,
            )
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
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        # See head for an explanation of the transposed behavior
        if n < 0:
            n = max(0, len(self.index) + n)
        if self._is_transposed:
            result = self.__constructor__(
                self.data.transpose().take(1, -n).transpose(),
                self.index[-n:],
                self.columns,
                self._dtype_cache,
                self._is_transposed,
            )
        else:
            result = self.__constructor__(
                self.data.take(0, -n), self.index[-n:], self.columns, self._dtype_cache
            )
        return result

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
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
                self._is_transposed,
            )
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
            QueryCompiler containing the last n columns of the original QueryCompiler.
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
                self._is_transposed,
            )
        else:
            result = self.__constructor__(
                self.data.take(1, -n), self.index, self.columns[-n:], new_dtypes
            )
        return result

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
        if self._is_transposed:
            return (
                self.transpose()
                .getitem_row_array(self.columns.get_indexer_for(key))
                .transpose()
            )
        # Convert to list for type checking
        if not numeric:
            numeric_indices = self.columns.get_indexer_for(key)
        else:
            numeric_indices = key
        result = self.data.mask(col_indices=numeric_indices)
        # We can't just set the columns to key here because there may be
        # multiple instances of a key.
        new_columns = self.columns[numeric_indices]
        if self._dtype_cache is not None:
            new_dtypes = self.dtypes[numeric_indices]
        else:
            new_dtypes = None
        return self.__constructor__(result, self.index, new_columns, new_dtypes)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new QueryCompiler.
        """
        if self._is_transposed:
            return self.transpose().getitem_column_array(key, numeric=True).transpose()
        result = self.data.mask(row_indices=key)
        # We can't just set the index to key here because there may be multiple
        # instances of a key.
        new_index = self.index[key]
        return self.__constructor__(result, new_index, self.columns, self._dtype_cache)

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

        if axis == 0:
            numeric_indices = list(self.columns.get_indexer_for([key]))
        else:
            numeric_indices = list(self.index.get_indexer_for([key]))
        prepared_func = self._prepare_method(setitem)
        if is_list_like(value):
            new_data = self.data.apply_func_to_select_indices_along_full_axis(
                axis, prepared_func, numeric_indices, keep_remaining=True
            )
        else:
            new_data = self.data.apply_func_to_select_indices(
                axis, prepared_func, numeric_indices, keep_remaining=True
            )
        return self.__constructor__(new_data, self.index, self.columns)

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
        if index is None and columns is None:
            return self.copy()
        if self._is_transposed:
            return self.transpose().drop(index=columns, columns=index).transpose()
        if index is None:
            new_index = self.index
            idx_numeric_indices = None
        else:
            idx_numeric_indices = pandas.RangeIndex(len(self.index)).drop(
                self.index.get_indexer_for(index)
            )
            new_index = self.index[~self.index.isin(index)]
        if columns is None:
            new_columns = self.columns
            new_dtypes = self._dtype_cache
            col_numeric_indices = None
        else:
            col_numeric_indices = pandas.RangeIndex(len(self.columns)).drop(
                self.columns.get_indexer_for(columns)
            )
            new_columns = self.columns[~self.columns.isin(columns)]
            if self._dtype_cache is not None:
                new_dtypes = self.dtypes.drop(columns)
            else:
                new_dtypes = None
        new_data = self.data.mask(
            row_indices=idx_numeric_indices, col_indices=col_numeric_indices
        )
        return self.__constructor__(new_data, new_index, new_columns, new_dtypes)

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

    def _post_process_apply(self, result_data, axis, try_scale=True):
        """Recompute the index after applying function.

        Args:
            result_data: a BaseFrameManager object.
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
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            return pandas.DataFrame(df.apply(func_dict, *args, **kwargs))

        result_data = self.data.apply_func_to_select_indices_along_full_axis(
            axis, dict_apply_builder, func, keep_remaining=False
        )
        full_result = self._post_process_apply(result_data, axis)
        return full_result

    def _list_like_func(self, func, axis, *args, **kwargs):
        """Apply list-like function across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        func_prepared = self._prepare_method(
            lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs))
        )
        new_data = self._map_across_full_axis(axis, func_prepared)
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
        return self.__constructor__(new_data, new_index, new_columns)

    def _callable_func(self, func, axis, *args, **kwargs):
        """Apply callable functions across given axis.

        Args:
            func: The functions to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """

        def callable_apply_builder(df, axis=0):
            if not axis:
                df.index = index
                df.columns = pandas.RangeIndex(len(df.columns))
            else:
                df.columns = index
                df.index = pandas.RangeIndex(len(df.index))
            result = df.apply(func, axis=axis, *args, **kwargs)
            return result

        index = self.index if not axis else self.columns
        func_prepared = self._build_mapreduce_func(callable_apply_builder, axis=axis)
        result_data = self._map_across_full_axis(axis, func_prepared)
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
            A `BaseFrameManager` object.
        """
        func = self._prepare_method(repartition_func, **kwargs)
        return self.data.manual_shuffle(axis, func)

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
                self.columns if not numeric_only else self.numeric_columns(True)
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
        # If the result is a Series, this is how `compute_index` returns the columns.
        if len(columns) == 0 and len(index) != 0:
            return self._post_process_apply(result_data, axis, try_scale=True)
        else:
            return self.__constructor__(result_data, index, columns)

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
        if self._is_transposed:
            return self.transpose().view(columns=index, index=columns)
        index_map_series = pandas.Series(np.arange(len(self.index)), index=self.index)
        column_map_series = pandas.Series(
            np.arange(len(self.columns)), index=self.columns
        )
        if index is not None:
            index_map_series = index_map_series.iloc[index]
        if columns is not None:
            column_map_series = column_map_series.iloc[columns]
        return PandasQueryCompilerView(
            self.data,
            index_map_series.index,
            column_map_series.index,
            self._dtype_cache,
            index_map_series,
            column_map_series,
        )

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
    This class represent a view of the PandasQueryCompiler

    In particular, the following constraints are broken:
    - (len(self.index), len(self.columns)) != self.data.shape

    Note:
        The constraint will be satisfied when we get the data
    """

    def __init__(
        self,
        block_partitions_object,
        index,
        columns,
        dtypes=None,
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
        assert index.equals(index_map_series.index)
        assert columns.equals(columns_map_series.index)

        self.index_map = index_map_series
        self.columns_map = columns_map_series

        PandasQueryCompiler.__init__(
            self, block_partitions_object, index, columns, dtypes
        )

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

    data = property(_get_data, _set_data)

    def global_idx_to_numeric_idx(self, axis, indices):
        assert axis in ["row", "col", "columns"]
        if axis == "row":
            return self.index_map.loc[indices].index
        elif axis in ["col", "columns"]:
            return self.columns_map.loc[indices].index
