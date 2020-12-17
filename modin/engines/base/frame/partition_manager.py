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

from abc import ABC
import numpy as np
import pandas

from modin.error_message import ErrorMessage
from modin.data_management.utils import compute_chunksize
from pandas.api.types import union_categoricals


class BaseFrameManager(ABC):
    """Partition class is the class to use for storing each partition. It must extend the `BaseFramePartition` class.

    It is the base class for managing the dataframe data layout and operators.
    """

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
        -------
            The preprocessed version of the `map_func` provided. Note: This
            does not require any specific format, only that the
            `BaseFramePartition.apply` method will recognize it (For the subclass
            being used).
        """
        return cls._partition_class.preprocess_func(map_func)

    # END Abstract Methods

    @classmethod
    def column_partitions(cls, partitions):
        """List of `BaseFrameAxisPartition` objects.

        Note: Each value in this list will be an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns
        -------
            a list of `BaseFrameAxisPartition` objects.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [
            cls._column_partitions_class(col) for frame in partitions for col in frame.T
        ]

    @classmethod
    def row_partitions(cls, partitions):
        """List of `BaseFrameAxisPartition` objects, represents column partitions.

        Note: Each value in this list will an `BaseFrameAxisPartition` object.
            `BaseFrameAxisPartition` is located in `axis_partition.py`.

        Returns
        -------
            a list of `BaseFrameAxisPartition` objects.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [cls._row_partition_class(row) for frame in partitions for row in frame]

    @classmethod
    def axis_partition(cls, partitions, axis):
        """Logically partition along either the columns or the rows."""
        return (
            cls.column_partitions(partitions)
            if not axis
            else cls.row_partitions(partitions)
        )

    @classmethod
    def groupby_reduce(
        cls, axis, partitions, by, map_func, reduce_func, apply_indices=None
    ):
        """
        Groupby data using the map_func provided along the axis over the partitions then reduce using reduce_func.

        Parameters
        ----------
            axis: int,
                Axis to groupby over.
            partitions: numpy 2D array,
                Partitions of the ModinFrame to groupby.
            by: numpy 2D array (optional),
                Partitions of 'by' to broadcast.
            map_func: callable,
                Map function.
            reduce_func: callable,
                Reduce function.
            apply_indices : list of ints (optional),
                Indices of `axis ^ 1` to apply function over.

        Returns
        -------
            Partitions with applied groupby.
        """
        if apply_indices is not None:
            partitions = (
                partitions[apply_indices] if axis else partitions[:, apply_indices]
            )

        if by is not None:
            mapped_partitions = cls.broadcast_apply(
                axis, map_func, left=partitions, right=by, other_name="other"
            )
        else:
            mapped_partitions = cls.map_partitions(partitions, map_func)
        return cls.map_axis_partitions(
            axis, mapped_partitions, reduce_func, enumerate_partitions=True
        )

    @classmethod
    def broadcast_apply_select_indices(
        cls,
        axis,
        apply_func,
        left,
        right,
        left_indices,
        right_indices,
        keep_remaining=False,
    ):
        """
        Broadcast the right partitions to left and apply a function to selected indices.

        Note: Your internal function must take this kwargs:
            [`internal_indices`, `other`, `internal_other_indices`] to work correctly

        Parameters
        ----------
            axis : The axis to apply and broadcast over.
            apply_func : The function to apply.
            left : The left partitions.
            right : The right partitions.
            left_indices : indices to apply function.
            right_indices : dictianary of indices of right partitions that
                you want to bring at specified left partition, for example that dict
                {key: {key1: [0, 1], key2: [5]}} means, that in left[key] you want to
                broadcast [right[key1], right[key2]] partitions and internal indices
                for `right` must be [[0, 1], [5]]
            keep_remaining : Whether or not to keep the other partitions.
                Some operations may want to drop the remaining partitions and
                keep only the results.

        Returns
        -------
            A new `np.array` of partition objects.
        """
        if not axis:
            partitions_for_apply = left.T
            right = right.T
        else:
            partitions_for_apply = left
            right = right

        [obj.drain_call_queue() for row in right for obj in row]

        def get_partitions(index):
            must_grab = right_indices[index]
            partitions_list = np.array([right[i] for i in must_grab.keys()])
            indices_list = list(must_grab.values())
            return {"other": partitions_list, "internal_other_indices": indices_list}

        new_partitions = np.array(
            [
                partitions_for_apply[i]
                if i not in left_indices
                else cls._apply_func_to_list_of_partitions_broadcast(
                    apply_func,
                    partitions_for_apply[i],
                    internal_indices=left_indices[i],
                    **get_partitions(i),
                )
                for i in range(len(partitions_for_apply))
                if i in left_indices or keep_remaining
            ]
        )
        if not axis:
            new_partitions = new_partitions.T
        return new_partitions

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
        """Broadcast the right partitions to left and apply a function.

        Note: This will often be overridden by implementations. It materializes the
            entire partitions of the right and applies them to the left through `apply`.

        Parameters
        ----------
            axis: The axis to apply and broadcast over.
            apply_func: The function to apply.
            left: The left partitions.
            right: The right partitions.
            other_name: Name of key-value argument for `apply_func` that
                obtains `right`. (optional, by default it's `"r"`)

        Returns
        -------
            A new `np.array` of partition objects.
        """
        [obj.drain_call_queue() for row in right for obj in row]
        new_right = np.empty(shape=right.shape[axis], dtype=object)

        if axis:
            right = right.T

        for i in range(len(right)):
            new_right[i] = pandas.concat(
                [right[i][j].get() for j in range(len(right[i]))], axis=axis ^ 1
            )
        right = new_right.T if axis else new_right

        new_partitions = np.array(
            [
                [
                    part.apply(
                        apply_func,
                        **{other_name: right[col_idx] if axis else right[row_idx]},
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )

        return new_partitions

    @classmethod
    def broadcast_axis_partitions(
        cls,
        axis,
        apply_func,
        left,
        right,
        keep_partitioning=False,
        apply_indices=None,
        enumerate_partitions=False,
        lengths=None,
    ):
        """
        Broadcast the right partitions to left and apply a function along full axis.

        Parameters
        ----------
        axis : The axis to apply and broadcast over.
        apply_func : The function to apply.
        left : The left partitions.
        right : The right partitions.
        keep_partitioning : boolean. Default is False
            The flag to keep partitions for Modin Frame.
        apply_indices : list of ints (optional),
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool (optional, default False),
            Whether or not to pass partition index into `apply_func`.
            Note that `apply_func` must be able to obtain `partition_idx` kwarg.
        lengths : list(int), default None
            The list of lengths to shuffle the object.

        Returns
        -------
        A new `np.array` of partition objects.
        """
        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        if keep_partitioning:
            num_splits = len(left) if axis == 0 else len(left.T)
        elif lengths:
            num_splits = len(lengths)
        else:
            num_splits = cls._compute_num_partitions()
        preprocessed_map_func = cls.preprocess_func(apply_func)
        left_partitions = cls.axis_partition(left, axis)
        right_partitions = None if right is None else cls.axis_partition(right, axis)
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        kw = {
            "num_splits": num_splits,
            "other_axis_partition": right_partitions,
        }
        if lengths:
            kw["_lengths"] = lengths
            kw["manual_partition"] = True

        if apply_indices is None:
            apply_indices = np.arange(len(left_partitions))

        result_blocks = np.array(
            [
                left_partitions[i].apply(
                    preprocessed_map_func,
                    **kw,
                    **({"partition_idx": idx} if enumerate_partitions else {}),
                )
                for idx, i in enumerate(apply_indices)
            ]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D NumPy array to return
        # the structure to the correct order.
        return result_blocks.T if not axis else result_blocks

    @classmethod
    def map_partitions(cls, partitions, map_func):
        """Apply `map_func` to every partition.

        Parameters
        ----------
        map_func: callable
           The function to apply.

        Returns
        -------
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
        """
        Apply `map_func` to every partition lazily.

        Parameters
        ----------
        map_func: callable
           The function to apply.

        Returns
        -------
            A new BaseFrameManager object, the type of object that called this.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [part.add_to_apply_calls(preprocessed_map_func) for part in row]
                for row in partitions
            ]
        )

    @classmethod
    def map_axis_partitions(
        cls,
        axis,
        partitions,
        map_func,
        keep_partitioning=False,
        lengths=None,
        enumerate_partitions=False,
    ):
        """
        Apply `map_func` to every partition.

        Parameters
        ----------
        axis : 0 or 1
            The axis to perform the map across (0 - index, 1 - columns).
        partitions : NumPy array
            The partitions of Modin Frame.
        map_func : callable
            The function to apply.
        keep_partitioning : bool. Default is False
            The flag to keep partitions for Modin Frame.
        lengths : list(int)
            The list of lengths to shuffle the object.
        enumerate_partitions : bool (optional, default False),
            Whether or not to pass partition index into `map_func`.
            Note that `map_func` must be able to obtain `partition_idx` kwarg.

        Returns
        -------
        NumPy array
            An array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case that `map_func` relies on
        some global information about the axis.
        """
        return cls.broadcast_axis_partitions(
            axis=axis,
            left=partitions,
            apply_func=map_func,
            keep_partitioning=keep_partitioning,
            right=None,
            lengths=lengths,
            enumerate_partitions=enumerate_partitions,
        )

    @classmethod
    def simple_shuffle(cls, axis, partitions, map_func, lengths):
        """
        Shuffle data using `lengths` via `map_func`.

        Parameters
        ----------
            axis : 0 or 1
                The axis to perform the map across (0 - index, 1 - columns).
            partitions : NumPy array
                The partitions of Modin Frame.
            map_func : callable
                The function to apply.
            lengths : list(int)
                The list of lengths to shuffle the object

        Returns
        -------
        NumPy array
            An array of new partitions for a Modin Frame.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        partitions = cls.axis_partition(partitions, axis)
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        result_blocks = np.array(
            [
                part.apply(
                    preprocessed_map_func, _lengths=lengths, manual_partition=True
                )
                for part in partitions
            ]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D NumPy array to return
        # the structure to the correct order.
        return result_blocks.T if not axis else result_blocks

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

        Returns
        -------
            A new BaseFrameManager object, the type of object that called this.
        """
        if type(right_parts) is list:
            # `np.array` with partitions of empty ModinFrame has a shape (0,)
            # but `np.concatenate` can concatenate arrays only if its shapes at
            # specified axis are equals, so filtering empty frames to avoid concat error
            right_parts = [o for o in right_parts if o.size != 0]
            to_concat = (
                [left_parts] + right_parts if left_parts.size != 0 else right_parts
            )
            return (
                np.concatenate(to_concat, axis=axis) if len(to_concat) else left_parts
            )
        else:
            return np.append(left_parts, right_parts, axis=axis)

    @classmethod
    def concatenate(cls, dfs):
        """
        Concatenate Pandas DataFrames with saving 'category' dtype.

        Parameters
        ----------
            dfs: list of DataFrames

        Returns
        -------
            A Pandas DataFrame
        """
        categoricals_columns = set.intersection(
            *[set(df.select_dtypes("category").columns.tolist()) for df in dfs]
        )

        for col in categoricals_columns:
            uc = union_categoricals([df[col] for df in dfs])
            for df in dfs:
                df[col] = pandas.Categorical(df[col], categories=uc.categories)

        return pandas.concat(dfs)

    @classmethod
    def to_pandas(cls, partitions):
        """Convert this object into a Pandas DataFrame from the partitions.

        Returns
        -------
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
            return cls.concatenate(df_rows)

    @classmethod
    def to_numpy(cls, partitions, **kwargs):
        """
        Convert this object into a NumPy array from the partitions.

        Returns
        -------
            A NumPy array
        """
        return np.block(
            [[block.to_numpy(**kwargs) for block in row] for row in partitions]
        )

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        """Return the partitions from Pandas DataFrame."""
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
    def from_arrow(cls, at, return_dims=False):
        """Return the partitions from Apache Arrow (PyArrow)."""
        return cls.from_pandas(at.to_pandas(), return_dims=return_dims)

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """Get the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.

        Returns
        -------
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
        """Retrieve the default number of partitions currently. Will estimate the optimal no. of partitions in future.

        Returns
        -------
            Number of partitions.
        """
        from modin.pandas import DEFAULT_NPARTITIONS

        return DEFAULT_NPARTITIONS

    @classmethod
    def _apply_func_to_list_of_partitions_broadcast(
        cls, func, partitions, other, **kwargs
    ):
        preprocessed_func = cls.preprocess_func(func)
        return [
            obj.apply(preprocessed_func, other=[o.get() for o in broadcasted], **kwargs)
            for obj, broadcasted in zip(partitions, other.T)
        ]

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """Apply a function to a list of remote partitions.

        Note: The main use for this is to preprocess the func.

        Args:
            func: The func to apply
            partitions: The list of partitions

        Returns
        -------
            A list of BaseFramePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    @classmethod
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """Apply a function to select indices.

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

        Returns
        -------
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
                        for o_idx, list_to_apply in indices.items()
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
                        for idx, list_to_apply in indices.items()
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
        """Apply a function to a select subset of full columns/rows.

        Note: This should be used when you need to apply a function that relies
            on some global information for the entire column/row, but only need
            to apply a function to a subset.

        Important: For your func to operate directly on the indices provided,
            it must use `internal_indices` as a keyword argument.

        Parameters
        ----------
        axis: int
            The axis to apply the function over (0 - rows, 1 - columns)
        func: callable
            The function to apply.
        indices: list-like
            The global indices to apply the func to.
        keep_remaining: boolean
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
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
        if not keep_remaining:
            selected_partitions = partitions.T if not axis else partitions
            selected_partitions = np.array([selected_partitions[i] for i in indices])
            selected_partitions = (
                selected_partitions.T if not axis else selected_partitions
            )
        else:
            selected_partitions = partitions
        if not axis:
            partitions_for_apply = cls.column_partitions(selected_partitions)
            partitions_for_remaining = partitions.T
        else:
            partitions_for_apply = cls.row_partitions(selected_partitions)
            partitions_for_remaining = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        part.apply(
                            preprocessed_func,
                            func_dict={idx: dict_func[idx] for idx in indices[i]},
                        )
                        for i, part in zip(indices, partitions_for_apply)
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
                        part.apply(preprocessed_func, internal_indices=indices[i])
                        for i, part in zip(indices, partitions_for_apply)
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
        Apply a function to along both axis.

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
                    **item,
                )
                partition_copy[row_blk_idx, col_blk_idx] = block_result
                col_position_counter += len(col_internal_idx)
            row_position_counter += len(row_internal_idx)
        return partition_copy

    @classmethod
    def binary_operation(cls, axis, left, func, right):
        """
        Apply a function that requires two BasePandasFrame objects.

        Parameters
        ----------
            axis : int
                The axis to apply the function over (0 - rows, 1 - columns)
            left : NumPy array
                The partitions of left Modin Frame
            func : callable
                The function to apply
            right : NumPy array
                The partitions of right Modin Frame.

        Returns
        -------
        NumPy array
            A new BasePandasFrame object, the type of object that called this.
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
