from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import numpy as np
import pandas

from modin.data_management.partitioning.utils import (
    compute_chunksize,
    _get_nan_block_id,
)


class BaseBlockPartitions(object):
    """Abstract Class that manages a set of `BaseRemotePartition` objects, and
        structures them into a 2D numpy array. This object will interact with
        each of these objects through the `BaseRemotePartition` API.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.
    def __init__(self, partitions):
        """Init must accept a parameter `partitions` that is a 2D numpy array
            of type `_partition_class` (defined below). This method will be
            called from a factory.

        Args:
            partitions: A 2D numpy array of the type defined in
                `_partition_class`.
        """
        raise NotImplementedError("Must be implemented in children classes")

    # Partition class is the class to use for storing each partition. It must
    # extend the `BaseRemotePartition` class.
    _partition_class = None
    # Column partitions class is the class to use to create the column partitions.
    _column_partitions_class = None
    # Row partitions class is the class to use to create row partitions.
    _row_partition_class = None
    # Whether or not we have already filtered out the empty partitions.
    _filtered_empties = False

    def _get_partitions(self):
        if not self._filtered_empties:
            self._partitions_cache = np.array(
                [
                    [
                        self._partitions_cache[i][j]
                        for j in range(len(self._partitions_cache[i]))
                        if self.block_lengths[i] != 0 or self.block_widths[j] != 0
                    ]
                    for i in range(len(self._partitions_cache))
                ]
            )
            self._remove_empty_blocks()
            self._filtered_empties = True
        return self._partitions_cache

    def _set_partitions(self, new_partitions):
        self._filtered_empties = False
        self._partitions_cache = new_partitions

    partitions = property(_get_partitions, _set_partitions)

    def preprocess_func(self, map_func):
        """Preprocess a function to be applied to `BaseRemotePartition` objects.

        Note: If your `BaseRemotePartition` objects assume that a function provided
            is serialized or wrapped or in some other format, this is the place
            to add that logic. It is possible that this can also just return
            `map_func` if the `apply` method of the `BaseRemotePartition` object
            you are using does not require any modification to a given
            function.

        Args:
            map_func: The function to be preprocessed.

        Returns
            The preprocessed version of the `map_func` provided. Note: This
            does not require any specific format, only that the
            `BaseRemotePartition.apply` method will recognize it (For the subclass
            being used).
        """
        return self._partition_class.preprocess_func(map_func)

    # END Abstract Methods

    @property
    def column_partitions(self):
        """A list of `BaseAxisPartition` objects.

        Note: Each value in this list will be an `BaseAxisPartition` object.
            `BaseAxisPartition` is located in the `base_remote_partition.py` file.

        Returns a list of `BaseAxisPartition` objects.
        """
        return [self._column_partitions_class(col) for col in self.partitions.T]

    @property
    def row_partitions(self):
        """A list of `BaseAxisPartition` objects, represents column partitions.

        Note: Each value in this list will an `BaseAxisPartition` object.
            `BaseAxisPartition` is located in the `base_remote_partition.py` file.

        Returns a list of `BaseAxisPartition` objects.
        """
        return [self._row_partition_class(row) for row in self.partitions]

    # Lengths of the blocks
    _lengths_cache = None

    # These are set up as properties so that we only use them when we need
    # them. We also do not want to trigger this computation on object creation.
    @property
    def block_lengths(self):
        """Gets the lengths of the blocks.

        Note: This works with the property structure `_lengths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._lengths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same length in a
            # row of blocks.
            self._lengths_cache = (
                [obj.length().get() for obj in self._partitions_cache.T[0]]
                if len(self._partitions_cache.T) > 0
                else []
            )
        return self._lengths_cache

    # Widths of the blocks
    _widths_cache = None

    @property
    def block_widths(self):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._widths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same width in a
            # column of blocks.
            self._widths_cache = (
                [obj.width().get() for obj in self._partitions_cache[0]]
                if len(self._partitions_cache) > 0
                else []
            )
        return self._widths_cache

    def _remove_empty_blocks(self):
        if self._widths_cache is not None:
            self._widths_cache = [width for width in self._widths_cache if width != 0]
        if self._lengths_cache is not None:
            self._lengths_cache = [
                length for length in self._lengths_cache if length != 0
            ]

    @property
    def shape(self) -> Tuple[int, int]:
        return int(np.sum(self.block_lengths)), int(np.sum(self.block_widths))

    def full_reduce(self, map_func, reduce_func, axis):
        """Perform a full reduce on the data.

        Note: This follows the 2-phase reduce paradigm, where each partition
            performs a local reduction (map_func), then partitions are brought
            together and the final reduction occurs.
        Args:
            map_func: The function that will be performed on all partitions.
                This is the local reduction on each partition.
            reduce_func: The final reduction function. This can differ from the
                `map_func`
            axis: The axis to perform this operation along
                (0 - index, 1 - columns)
        Returns:
            A Pandas Series
        """
        mapped_parts = self.map_across_blocks(map_func).partitions
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
        return reduce_func(full_frame)

    def map_across_blocks(self, map_func):
        """Applies `map_func` to every partition.

        Args:
            map_func: The function to apply.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        # For the subclasses, because we never return this abstract type
        cls = type(self)

        preprocessed_map_func = self.preprocess_func(map_func)
        new_partitions = np.array(
            [
                [part.apply(preprocessed_map_func) for part in row_of_parts]
                for row_of_parts in self.partitions
            ]
        )
        return cls(new_partitions)

    def lazy_map_across_blocks(self, map_func, kwargs):
        cls = type(self)
        preprocessed_map_func = self.preprocess_func(map_func)
        new_partitions = np.array(
            [
                [
                    part.add_to_apply_calls(preprocessed_map_func, kwargs)
                    for part in row_of_parts
                ]
                for row_of_parts in self.partitions
            ]
        )
        return cls(new_partitions)

    def map_across_full_axis(self, axis, map_func):
        """Applies `map_func` to every partition.

        Note: This method should be used in the case that `map_func` relies on
            some global information about the axis.

        Args:
            axis: The axis to perform the map across (0 - index, 1 - columns).
            map_func: The function to apply.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        num_splits = cls._compute_num_partitions()
        preprocessed_map_func = self.preprocess_func(map_func)
        partitions = self.column_partitions if not axis else self.row_partitions
        result_blocks = np.array(
            [part.apply(preprocessed_map_func, num_splits) for part in partitions]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D numpy array to return
        # the structure to the correct order.
        return cls(result_blocks.T) if not axis else cls(result_blocks)

    def take(self, axis, n):
        """Take the first (or last) n rows or columns from the blocks

        Note: Axis = 0 will be equivalent to `head` or `tail`
              Axis = 1 will be equivalent to `front` or `back`

        Args:
            axis: The axis to extract (0 for extracting rows, 1 for extracting columns)
            n: The number of rows or columns to extract, negative denotes to extract
                from the bottom of the object

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        # These are the partitions that we will extract over
        if not axis:
            partitions = self.partitions
            bin_lengths = self.block_lengths
        else:
            partitions = self.partitions.T
            bin_lengths = self.block_widths
        if n < 0:
            reversed_bins = bin_lengths
            reversed_bins.reverse()
            length_bins = np.cumsum(reversed_bins)
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
                func = self.preprocess_func(lambda df: df.iloc[slice_obj])
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
                func = self.preprocess_func(lambda df: df.iloc[slice_obj])
                # See note above about idx + 1
                result = np.array(
                    [
                        partitions[i]
                        if i != idx
                        else [obj.apply(func) for obj in partitions[i]]
                        for i in range(idx + 1)
                    ]
                )
        return cls(result.T) if axis else cls(result)

    def concat(self, axis, other_blocks):
        """Concatenate the blocks with another set of blocks.

        Note: Assumes that the blocks are already the same shape on the
            dimension being concatenated. A ValueError will be thrown if this
            condition is not met.

        Args:
            axis: The axis to concatenate to.
            other_blocks: the other blocks to be concatenated. This is a
                BaseBlockPartitions object.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        if type(other_blocks) is list:
            other_blocks = [blocks.partitions for blocks in other_blocks]
            return cls(np.concatenate([self.partitions] + other_blocks, axis=axis))
        else:
            return cls(np.append(self.partitions, other_blocks.partitions, axis=axis))

    def copy(self):
        """Create a copy of this object.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        return cls(self.partitions.copy())

    def transpose(self, *args, **kwargs):
        """Transpose the blocks stored in this object.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        return cls(self.partitions.T)

    def to_pandas(self, is_transposed=False):
        """Convert this object into a Pandas DataFrame from the partitions.

        Args:
            is_transposed: A flag for telling this object that the external
                representation is transposed, but not the internal.

        Returns:
            A Pandas DataFrame
        """
        # In the case this is transposed, it is easier to just temporarily
        # transpose back then transpose after the conversion. The performance
        # is the same as if we individually transposed the blocks and
        # concatenated them, but the code is much smaller.
        if is_transposed:
            return self.transpose().to_pandas(False).T
        else:
            retrieved_objects = [
                [obj.to_pandas() for obj in part] for part in self.partitions
            ]
            if all(
                isinstance(part, pandas.Series)
                for row in retrieved_objects
                for part in row
            ):
                axis = 0
            elif all(
                isinstance(part, pandas.DataFrame)
                for row in retrieved_objects
                for part in row
            ):
                axis = 1
            else:
                raise ValueError(
                    "Some partitions contain Series and some contain DataFrames"
                )
            df_rows = [
                pandas.concat([part for part in row], axis=axis)
                for row in retrieved_objects
            ]
            if len(df_rows) == 0:
                return pandas.DataFrame()
            else:
                return pandas.concat(df_rows)

    @classmethod
    def from_pandas(cls, df):
        num_splits = cls._compute_num_partitions()
        put_func = cls._partition_class.put
        row_chunksize = max(1, compute_chunksize(len(df), num_splits))
        col_chunksize = max(1, compute_chunksize(len(df.columns), num_splits))

        # Each chunk must have a RangeIndex that spans its length and width
        # according to our invariant.
        def chunk_builder(i, j):
            chunk = df.iloc[i : i + row_chunksize, j : j + col_chunksize]
            chunk.index = pandas.RangeIndex(len(chunk.index))
            chunk.columns = pandas.RangeIndex(len(chunk.columns))
            return put_func(chunk)

        parts = [
            [chunk_builder(i, j) for j in range(0, len(df.columns), col_chunksize)]
            for i in range(0, len(df), row_chunksize)
        ]
        return cls(np.array(parts))

    def get_indices(self, axis=0, index_func=None, old_blocks=None):
        """This gets the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.
            old_blocks: An optional previous object that this object was
                created from. This is used to compute the correct offsets.

        Returns:
            A Pandas Index object.
        """
        assert callable(index_func), "Must tell this function how to extract index"
        if axis == 0:
            func = self.preprocess_func(index_func)
            # We grab the first column of blocks and extract the indices
            new_indices = [idx.apply(func).get() for idx in self.partitions.T[0]]
            # This is important because sometimes we have resized the data. The new
            # sizes will not be valid if we are trying to compute the index on a
            # new object that has a different length.
            if old_blocks is not None:
                cumulative_block_lengths = np.array(old_blocks.block_lengths).cumsum()
            else:
                cumulative_block_lengths = np.array(self.block_lengths).cumsum()
        else:
            func = self.preprocess_func(index_func)
            new_indices = [idx.apply(func).get() for idx in self.partitions[0]]

            if old_blocks is not None:
                cumulative_block_lengths = np.array(old_blocks.block_widths).cumsum()
            else:
                cumulative_block_lengths = np.array(self.block_widths).cumsum()
        full_indices = new_indices[0]
        if old_blocks is not None:
            for i in range(len(new_indices)):
                # If the length is 0 there is nothing to append.
                if i == 0 or len(new_indices[i]) == 0:
                    continue
                # The try-except here is intended to catch issues where we are
                # trying to get a string index out of the internal index.
                try:
                    append_val = new_indices[i] + cumulative_block_lengths[i - 1]
                except TypeError:
                    append_val = new_indices[i]

                full_indices = full_indices.append(append_val)
        else:
            full_indices = full_indices.append(new_indices[1:])
        return full_indices

    @classmethod
    def _compute_num_partitions(cls):
        """Currently, this method returns the default. In the future it will
            estimate the optimal number of partitions.

        :return:
        """
        from ....pandas import DEFAULT_NPARTITIONS

        return DEFAULT_NPARTITIONS

    # Extracting rows/columns
    def _get_blocks_containing_index(self, axis, index):
        """Convert a global index to a block index and local index.

        Note: This method is primarily used to convert a global index into a
            partition index (along the axis provided) and local index (useful
            for `iloc` or similar operations.

        Args:
            axis: The axis along which to get the indices
                (0 - columns, 1 - rows)
            index: The global index to convert.

        Returns:
            A tuple containing (block index and internal index).
        """
        if not axis:
            cumulative_column_widths = np.array(self.block_widths).cumsum()
            block_idx = int(np.digitize(index, cumulative_column_widths))
            if block_idx == len(cumulative_column_widths):
                block_idx -= 1
            # Compute the internal index based on the previous lengths. This
            # is a global index, so we must subtract the lengths first.
            internal_idx = (
                index
                if not block_idx
                else index - cumulative_column_widths[block_idx - 1]
            )
            return block_idx, internal_idx
        else:
            cumulative_row_lengths = np.array(self.block_lengths).cumsum()
            block_idx = int(np.digitize(index, cumulative_row_lengths))
            # See note above about internal index
            internal_idx = (
                index
                if not block_idx
                else index - cumulative_row_lengths[block_idx - 1]
            )
            return block_idx, internal_idx

    def _get_dict_of_block_index(self, axis, indices):
        """Convert indices to a dict of block index to internal index mapping.

        Note: See `_get_blocks_containing_index` for primary usage. This method
            accepts a list of indices rather than just a single value, and uses
            `_get_blocks_containing_index`.

        Args:
            axis: The axis along which to get the indices
                (0 - columns, 1 - rows)
            indices: A list of global indices to convert.

        Returns
            A dictionary of {block index: list of local indices}.
        """
        # Get the internal index and create a dictionary so we only have to
        # travel to each partition once.
        all_partitions_and_idx = [
            self._get_blocks_containing_index(axis, i) for i in indices
        ]
        partitions_dict = {}
        for part_idx, internal_idx in all_partitions_and_idx:
            if part_idx not in partitions_dict:
                partitions_dict[part_idx] = [internal_idx]
            else:
                partitions_dict[part_idx].append(internal_idx)
        return partitions_dict

    def _apply_func_to_list_of_partitions(self, func, partitions, **kwargs):
        """Applies a function to a list of remote partitions.

        Note: The main use for this is to preprocess the func.

        Args:
            func: The func to apply
            partitions: The list of partitions

        Returns:
            A list of BaseRemotePartition objects.
        """
        preprocessed_func = self.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    def apply_func_to_select_indices(self, axis, func, indices, keep_remaining=False):
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
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(indices, dict):
            dict_indices = indices
            indices = list(indices.keys())
        else:
            dict_indices = None
        if not isinstance(indices, list):
            indices = [indices]
        partitions_dict = self._get_dict_of_block_index(axis, indices)
        if not axis:
            partitions_for_apply = self.partitions.T
        else:
            partitions_for_apply = self.partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_indices is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_indices[idx] for idx in partitions_dict[i]
                            },
                        )
                        for i in partitions_dict
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in partitions_dict
                        else self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_indices[i] for idx in partitions_dict[i]
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
                        self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            internal_indices=partitions_dict[i],
                        )
                        for i in partitions_dict
                    ]
                )
            else:
                # The difference here is that we modify a subset and return the
                # remaining (non-updated) blocks in their original position.
                result = np.array(
                    [
                        partitions_for_apply[i]
                        if i not in partitions_dict
                        else self._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            internal_indices=partitions_dict[i],
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        return cls(result.T) if not axis else cls(result)

    def apply_func_to_select_indices_along_full_axis(
        self, axis, func, indices, keep_remaining=False
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
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        if isinstance(indices, dict):
            dict_indices = indices
            indices = list(indices.keys())
        else:
            dict_indices = None
        if not isinstance(indices, list):
            indices = [indices]
        partitions_dict = self._get_dict_of_block_index(axis, indices)
        preprocessed_func = self.preprocess_func(func)
        # Since we might be keeping the remaining blocks that are not modified,
        # we have to also keep the block_partitions object in the correct
        # direction (transpose for columns).
        if not axis:
            partitions_for_apply = self.column_partitions
            partitions_for_remaining = self.partitions.T
        else:
            partitions_for_apply = self.row_partitions
            partitions_for_remaining = self.partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_indices is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        partitions_for_apply[i].apply(
                            preprocessed_func,
                            func_dict={
                                idx: dict_indices[idx] for idx in partitions_dict[i]
                            },
                        )
                        for i in partitions_dict
                    ]
                )
            else:
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in partitions_dict
                        else self._apply_func_to_list_of_partitions(
                            preprocessed_func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_indices[idx] for idx in partitions_dict[i]
                            },
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
                            preprocessed_func, internal_indices=partitions_dict[i]
                        )
                        for i in partitions_dict
                    ]
                )
            else:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        partitions_for_remaining[i]
                        if i not in partitions_dict
                        else partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=partitions_dict[i]
                        )
                        for i in range(len(partitions_for_remaining))
                    ]
                )
        return cls(result.T) if not axis else cls(result)

    def apply_func_to_indices_both_axis(
        self,
        func,
        row_indices,
        col_indices,
        lazy=False,
        keep_remaining=True,
        mutate=False,
        item_to_distribute=None,
    ):
        """
        Apply a function to along both axis

        Important: For your func to operate directly on the indices provided,
            it must use `row_internal_indices, col_internal_indices` as keyword
            arguments.
        """
        cls = type(self)

        if not mutate:
            partition_copy = self.partitions.copy()
        else:
            partition_copy = self.partitions

        operation_mask = np.full(self.partitions.shape, False)
        row_position_counter = 0
        for row_blk_idx, row_internal_idx in self._get_dict_of_block_index(
            1, row_indices
        ).items():
            col_position_counter = 0
            for col_blk_idx, col_internal_idx in self._get_dict_of_block_index(
                0, col_indices
            ).items():
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

                if lazy:
                    result = remote_part.add_to_apply_calls(
                        func,
                        row_internal_indices=row_internal_idx,
                        col_internal_indices=col_internal_idx,
                        **item
                    )
                else:
                    result = remote_part.apply(
                        func,
                        row_internal_indices=row_internal_idx,
                        col_internal_indices=col_internal_idx,
                        **item
                    )
                partition_copy[row_blk_idx, col_blk_idx] = result
                operation_mask[row_blk_idx, col_blk_idx] = True
                col_position_counter += len(col_internal_idx)

            row_position_counter += len(row_internal_idx)

        column_idx = np.where(np.any(operation_mask, axis=0))[0]
        row_idx = np.where(np.any(operation_mask, axis=1))[0]
        if not keep_remaining:
            partition_copy = partition_copy[row_idx][:, column_idx]
        return cls(partition_copy)

    def inter_data_operation(self, axis, func, other):
        """Apply a function that requires two BaseBlockPartitions objects.

        Args:
            axis: The axis to apply the function over (0 - rows, 1 - columns)
            func: The function to apply
            other: The other BaseBlockPartitions object to apply func to.

        Returns:
            A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)
        if axis:
            partitions = self.row_partitions
            other_partitions = other.row_partitions
        else:
            partitions = self.column_partitions
            other_partitions = other.column_partitions
        func = self.preprocess_func(func)
        result = np.array(
            [
                partitions[i].apply(
                    func,
                    num_splits=cls._compute_num_partitions(),
                    other_axis_partition=other_partitions[i],
                )
                for i in range(len(partitions))
            ]
        )
        return cls(result) if axis else cls(result.T)

    def manual_shuffle(self, axis, shuffle_func):
        """Shuffle the partitions based on the `shuffle_func`.

        Args:
            axis:
            shuffle_func:

        Returns:
             A new BaseBlockPartitions object, the type of object that called this.
        """
        cls = type(self)

        if axis:
            partitions = self.row_partitions
        else:
            partitions = self.column_partitions
        func = self.preprocess_func(shuffle_func)
        result = np.array(
            [
                part.shuffle(func, num_splits=cls._compute_num_partitions())
                for part in partitions
            ]
        )
        return cls(result) if axis else cls(result.T)

    def __getitem__(self, key):
        cls = type(self)
        return cls(self.partitions[key])

    def __len__(self):
        return sum(self.block_lengths)

    def enlarge_partitions(self, n_rows=None, n_cols=None):
        data = self.partitions
        block_partitions_cls = type(self)

        if n_rows:
            n_cols_lst = self.block_widths
            nan_oids_lst = [
                self._partition_class(
                    _get_nan_block_id(self._partition_class, n_rows, n_cols_)
                )
                for n_cols_ in n_cols_lst
            ]
            new_chunk = block_partitions_cls(np.array([nan_oids_lst]))
            data = self.concat(axis=0, other_blocks=new_chunk)

        if n_cols:
            n_rows_lst = self.block_lengths
            nan_oids_lst = [
                self._partition_class(
                    _get_nan_block_id(self._partition_class, n_rows_, n_cols)
                )
                for n_rows_ in n_rows_lst
            ]
            new_chunk = block_partitions_cls(np.array([nan_oids_lst]).T)
            data = self.concat(axis=1, other_blocks=new_chunk)
        return data
