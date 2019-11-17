import numpy as np
import pandas

from modin.error_message import ErrorMessage
from modin.data_management.utils import compute_chunksize


class BaseFrameManager(object):
    # Partition class is the class to use for storing each partition. It must
    # extend the `BaseFramePartition` class.
    _partition_class = None
    # Column partitions class is the class to use to create the column partitions.
    _column_partitions_class = None
    # Row partitions class is the class to use to create the row partitions.
    _row_partition_class = None

    @classmethod
    def preprocess_func(cls, map_func):
        """Preprocess a function to be applied to `BaseFramePartition` objects.

        Note: If your `BaseFramePartition` objects assume that a function provided
            is serialized or wrapped or in some other format, this is the place
            to add that logic. It is possible that this can also just return
            `map_func` if the `apply` method of the `BaseFramePartition` object
            you are using does not require any modification to a given
            function.

        Args:
            map_func: The function to be preprocessed.

        Returns
            The preprocessed version of the `map_func` provided. Note: This
            does not require any specific format, only that the
            `BaseFramePartition.apply` method will recognize it (For the subclass
            being used).
        """
        return cls._partition_class.preprocess_func(map_func)

    # END Abstract Methods

    @classmethod
    def column_partitions(cls, partitions):
        """A list of `BaseFrameAxisPartition` objects.

        Note: Each value in this list will be an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns a list of `BaseFrameAxisPartition` objects.
        """
        return [cls._column_partitions_class(col) for col in partitions.T]

    @classmethod
    def row_partitions(cls, partitions):
        """A list of `BaseFrameAxisPartition` objects, represents column partitions.

        Note: Each value in this list will an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns a list of `BaseFrameAxisPartition` objects.
        """
        return [cls._row_partition_class(row) for row in partitions]

    @classmethod
    def groupby_reduce(cls, axis, partitions, by, map_func, reduce_func):
        by_parts = np.squeeze(by)
        if len(by_parts.shape) == 0:
            by_parts = np.array([by_parts.item()])
        [obj.drain_call_queue() for obj in by_parts]
        new_partitions = np.array(
            [
                [
                    part.apply(
                        map_func,
                        other=by_parts[col_idx].get()
                        if axis
                        else by_parts[row_idx].get(),
                    )
                    for col_idx, part in enumerate(partitions[row_idx])
                ]
                for row_idx in range(len(partitions))
            ]
        )
        return cls.map_axis_partitions(axis, new_partitions, reduce_func)

    @classmethod
    def map_partitions(cls, partitions, map_func):
        """Applies `map_func` to every partition.

        Args:
            map_func: The function to apply.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [part.apply(preprocessed_map_func) for part in row_of_parts]
                for row_of_parts in partitions
            ]
        )

    @classmethod
    def lazy_map_partitions(cls, partitions, map_func):
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [part.add_to_apply_calls(preprocessed_map_func) for part in row]
                for row in partitions
            ]
        )

    @classmethod
    def map_axis_partitions(cls, axis, partitions, map_func, keep_partitioning=False):
        """Applies `map_func` to every partition.

        Note: This method should be used in the case that `map_func` relies on
            some global information about the axis.

        Args:
            axis: The axis to perform the map across (0 - index, 1 - columns).
            map_func: The function to apply.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        if keep_partitioning:
            num_splits = len(partitions) if axis == 0 else len(partitions.T)
        else:
            num_splits = cls._compute_num_partitions()
        preprocessed_map_func = cls.preprocess_func(map_func)
        partitions = (
            cls.column_partitions(partitions)
            if not axis
            else cls.row_partitions(partitions)
        )
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        result_blocks = np.array(
            [
                part.apply(preprocessed_map_func, num_splits=num_splits)
                for part in partitions
            ]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D numpy array to return
        # the structure to the correct order.
        return result_blocks.T if not axis else result_blocks

    @classmethod
    def take(cls, axis, partitions, block_lengths, n):
        """Take the first (or last) n rows or columns from the blocks

        Note: Axis = 0 will be equivalent to `head` or `tail`
              Axis = 1 will be equivalent to `front` or `back`

        Args:
            axis: The axis to extract (0 for extracting rows, 1 for extracting columns)
            n: The number of rows or columns to extract, negative denotes to extract
                from the bottom of the object

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        # These are the partitions that we will extract over
        if not axis:
            partitions = partitions
            bin_lengths = block_lengths
        else:
            partitions = partitions.T
            bin_lengths = block_lengths
        if n < 0:
            length_bins = np.cumsum(bin_lengths[::-1])
            n *= -1
            idx = int(np.digitize(n, length_bins))
            if idx > 0:
                remaining = int(n - length_bins[idx - 1])
            else:
                remaining = n
            # In this case, we require no remote compute. This is much faster.
            if remaining == 0:
                result = partitions[-idx:]
            else:
                # Reverse for ease of iteration and then re-reverse at the end
                partitions = partitions[::-1]
                # We build this iloc to avoid creating a bunch of helper methods.
                # This code creates slice objects to be passed to `iloc` to grab
                # the last n rows or columns depending on axis.
                slice_obj = (
                    slice(-remaining, None)
                    if axis == 0
                    else (slice(None), slice(-remaining, None))
                )
                func = cls.preprocess_func(lambda df: df.iloc[slice_obj])
                # We use idx + 1 here because the loop is not inclusive, and we
                # need to iterate through idx.
                result = np.array(
                    [
                        partitions[i]
                        if i != idx
                        else [obj.apply(func) for obj in partitions[i]]
                        for i in range(idx + 1)
                    ]
                )[::-1]
        else:
            length_bins = np.cumsum(bin_lengths)
            idx = int(np.digitize(n, length_bins))
            if idx > 0:
                remaining = int(n - length_bins[idx - 1])
            else:
                remaining = n
            # In this case, we require no remote compute. This is much faster.
            if remaining == 0:
                result = partitions[:idx]
            else:
                # We build this iloc to avoid creating a bunch of helper methods.
                # This code creates slice objects to be passed to `iloc` to grab
                # the first n rows or columns depending on axis.
                slice_obj = (
                    slice(remaining) if axis == 0 else (slice(None), slice(remaining))
                )
                func = cls.preprocess_func(lambda df: df.iloc[slice_obj])
                # See note above about idx + 1
                result = np.array(
                    [
                        partitions[i]
                        if i != idx
                        else [obj.apply(func) for obj in partitions[i]]
                        for i in range(idx + 1)
                    ]
                )
        return result.T if axis else result

    @classmethod
    def concat(cls, axis, left_parts, right_parts):
        """Concatenate the blocks with another set of blocks.

        Note: Assumes that the blocks are already the same shape on the
            dimension being concatenated. A ValueError will be thrown if this
            condition is not met.

        Args:
            axis: The axis to concatenate to.
            right_parts: the other blocks to be concatenated. This is a
                BaseFrameManager object.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if type(right_parts) is list:
            return np.concatenate([left_parts] + right_parts, axis=axis)
        else:
            return np.append(left_parts, right_parts, axis=axis)

    @classmethod
    def to_pandas(cls, partitions):
        """Convert this object into a Pandas DataFrame from the partitions.

        Returns:
            A Pandas DataFrame
        """
        retrieved_objects = [[obj.to_pandas() for obj in part] for part in partitions]
        if all(
            isinstance(part, pandas.Series) for row in retrieved_objects for part in row
        ):
            axis = 0
        elif all(
            isinstance(part, pandas.DataFrame)
            for row in retrieved_objects
            for part in row
        ):
            axis = 1
        else:
            ErrorMessage.catch_bugs_and_request_email(True)
        df_rows = [
            pandas.concat([part for part in row], axis=axis)
            for row in retrieved_objects
            if not all(part.empty for part in row)
        ]
        if len(df_rows) == 0:
            return pandas.DataFrame()
        else:
            return pandas.concat(df_rows)

    @classmethod
    def to_numpy(cls, partitions):
        """Convert this object into a NumPy Array from the partitions.

        Returns:
            A NumPy Array
        """
        return np.block([[block.to_numpy() for block in row] for row in partitions])

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        num_splits = cls._compute_num_partitions()
        put_func = cls._partition_class.put
        row_chunksize, col_chunksize = compute_chunksize(df, num_splits)
        parts = [
            [
                put_func(df.iloc[i : i + row_chunksize, j : j + col_chunksize].copy())
                for j in range(0, len(df.columns), col_chunksize)
            ]
            for i in range(0, len(df), row_chunksize)
        ]
        if not return_dims:
            return np.array(parts)
        else:
            row_lengths = [
                row_chunksize
                if i + row_chunksize < len(df)
                else len(df) % row_chunksize or row_chunksize
                for i in range(0, len(df), row_chunksize)
            ]
            col_widths = [
                col_chunksize
                if i + col_chunksize < len(df.columns)
                else len(df.columns) % col_chunksize or col_chunksize
                for i in range(0, len(df.columns), col_chunksize)
            ]
            return np.array(parts), row_lengths, col_widths

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """This gets the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.

        Returns:
            A Pandas Index object.
        """
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            new_idx = (
                [idx.apply(func).get() for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func).get() for idx in partitions[0]]
                if len(partitions)
                else []
            )
        # TODO FIX INFORMATION LEAK!!!!1!!1!!
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def _compute_num_partitions(cls):
        """Currently, this method returns the default. In the future it will
            estimate the optimal number of partitions.

        :return:
        """
        from modin.pandas import DEFAULT_NPARTITIONS

        return DEFAULT_NPARTITIONS

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """Applies a function to a list of remote partitions.

        Note: The main use for this is to preprocess the func.

        Args:
            func: The func to apply
            partitions: The list of partitions

        Returns:
            A list of BaseFramePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    @classmethod
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """Applies a function to select indices.

        Note: Your internal function must take a kwarg `internal_indices` for
            this to work correctly. This prevents information leakage of the
            internal index to the external representation.

        Args:
            axis: The axis to apply the func over.
            func: The function to apply to these indices.
            indices: The indices to apply the function to.
            keep_remaining: Whether or not to keep the other partitions.
                Some operations may want to drop the remaining partitions and
                keep only the results.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        if not axis:
            partitions_for_apply = partitions.T
        else:
            partitions_for_apply = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[o_idx],
                            func_dict={
                                i_idx: dict_func[i_idx]
                                for i_idx in list_to_apply
                                if i_idx >= 0
                            },
                        )
                        for o_idx, list_to_apply in indices
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_func[idx] for idx in indices[i] if idx >= 0
                            },
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # We are passing internal indices in here. In order for func to
                # actually be able to use this information, it must be able to take in
                # the internal indices. This might mean an iloc in the case of Pandas
                # or some other way to index into the internal representation.
                result = np.array(
                    [
                        cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[idx],
                            internal_indices=list_to_apply,
                        )
                        for idx, list_to_apply in indices
                    ]
                )
            else:
                # The difference here is that we modify a subset and return the
                # remaining (non-updated) blocks in their original position.
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            func, partitions_for_apply[i], internal_indices=indices[i]
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        return result.T if not axis else result

    @classmethod
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """Applies a function to a select subset of full columns/rows.

        Note: This should be used when you need to apply a function that relies
            on some global information for the entire column/row, but only need
            to apply a function to a subset.

        Important: For your func to operate directly on the indices provided,
            it must use `internal_indices` as a keyword argument.

        Args:
            axis: The axis to apply the function over (0 - rows, 1 - columns)
            func: The function to apply.
            indices: The global indices to apply the func to.
            keep_remaining: Whether or not to keep the other partitions.
                Some operations may want to drop the remaining partitions and
                keep only the results.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        preprocessed_func = cls.preprocess_func(func)
        # Since we might be keeping the remaining blocks that are not modified,
        # we have to also keep the block_partitions object in the correct
        # direction (transpose for columns).
        if not axis:
            partitions_for_apply = cls.column_partitions(partitions)
            partitions_for_remaining = partitions.T
        else:
            partitions_for_apply = cls.row_partitions(partitions)
            partitions_for_remaining = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        partitions_for_apply[i].apply(
                            preprocessed_func,
                            func_dict={idx: dict_func[idx] for idx in indices[i]},
                        )
                        for i in indices
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            preprocessed_func,
                            partitions_for_apply[i],
                            func_dict={idx: dict_func[idx] for idx in indices[i]},
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=indices[i]
                        )
                        for i in indices
                    ]
                )
            else:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in indices
                        else partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=indices[i]
                        )
                        for i in range(len(partitions_for_remaining))
                    ]
                )
        return result.T if not axis else result

    @classmethod
    def apply_func_to_indices_both_axis(
        cls,
        partitions,
        func,
        row_partitions_list,
        col_partitions_list,
        item_to_distribute=None,
    ):
        """
        Apply a function to along both axis

        Important: For your func to operate directly on the indices provided,
            it must use `row_internal_indices, col_internal_indices` as keyword
            arguments.
        """
        partition_copy = partitions.copy()
        row_position_counter = 0
        for row_idx, row_values in enumerate(row_partitions_list):
            row_blk_idx, row_internal_idx = row_values
            col_position_counter = 0
            for col_idx, col_values in enumerate(col_partitions_list):
                col_blk_idx, col_internal_idx = col_values
                remote_part = partition_copy[row_blk_idx, col_blk_idx]

                if item_to_distribute is not None:
                    item = item_to_distribute[
                        row_position_counter : row_position_counter
                        + len(row_internal_idx),
                        col_position_counter : col_position_counter
                        + len(col_internal_idx),
                    ]
                    item = {"item": item}
                else:
                    item = {}
                block_result = remote_part.add_to_apply_calls(
                    func,
                    row_internal_indices=row_internal_idx,
                    col_internal_indices=col_internal_idx,
                    **item
                )
                partition_copy[row_blk_idx, col_blk_idx] = block_result
                col_position_counter += len(col_internal_idx)
            row_position_counter += len(row_internal_idx)
        return partition_copy

    @classmethod
    def binary_operation(cls, axis, left, func, right):
        """Apply a function that requires two BaseFrameManager objects.

        Args:
            axis: The axis to apply the function over (0 - rows, 1 - columns)
            func: The function to apply
            other: The other BaseFrameManager object to apply func to.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        if axis:
            left_partitions = cls.row_partitions(left)
            right_partitions = cls.row_partitions(right)
        else:
            left_partitions = cls.column_partitions(left)
            right_partitions = cls.column_partitions(right)
        func = cls.preprocess_func(func)
        result = np.array(
            [
                left_partitions[i].apply(
                    func,
                    num_splits=cls._compute_num_partitions(),
                    other_axis_partition=right_partitions[i],
                )
                for i in range(len(left_partitions))
            ]
        )
        return result if axis else result.T
