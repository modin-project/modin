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
import cudf

from modin.error_message import ErrorMessage
from modin.data_management.utils import compute_chunksize

from pandas.api.types import union_categoricals

import numpy as np

from .axis_partition import (
    cuDFOnRayFrameColumnPartition,
    cuDFOnRayFrameRowPartition,
)
from .partition import cuDFOnRayFramePartition
from modin.error_message import ErrorMessage

import ray


@ray.remote(num_cpus=1, num_gpus=0.5)
def func(df, other, apply_func):
    return apply_func(ray.get(df.get.remote()), ray.get(other.get.remote()))

class cuDFOnRayFrameManager(object):

    _partition_class = cuDFOnRayFramePartition
    _column_partitions_class = cuDFOnRayFrameColumnPartition
    _row_partition_class = cuDFOnRayFrameRowPartition

    @classmethod
    def _get_gpu_managers(cls):
        from modin.pandas import GPU_MANAGERS
        return GPU_MANAGERS

    @classmethod
    def _compute_num_row_partitions(cls):
        from modin.config import NPartitions
        return NPartitions.get()

    @classmethod
    def _compute_num_col_partitions(cls):
        # How are column partition defined?
        from modin.config import NPartitions
        return NPartitions.get()

    @classmethod
    def preprocess_func(cls, map_func):
        return cls._partition_class.preprocess_func(map_func)

    @classmethod
    def row_partitions(cls, partitions):
        return [cls._row_partition_class(row) for row in partitions]

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            new_idx = [
                idx.apply_result_not_dataframe(func)
                for idx in partitions.T[0]
            ] if len(partitions.T) else []
        else:
            new_idx = [
                idx.apply_result_not_dataframe(func)
                for idx in partitions[0]
            ] if len(partitions) else []

        new_idx = ray.get(new_idx)
        returned_idx = new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx
        return returned_idx

    # FIXME (kvu35): Enable work load balancing
    @classmethod
    def map_axis_partitions(
        cls,
        axis,
        partitions,
        map_func,
        keep_partitioning=True,
        persistent=False
    ):
        # if keep_partitioning:
        #     num_splits = len(partitions) if axis == 0 else len(partitions.T)
        # else:
        #     num_splits = cls._compute_num_partitions()
        preprocessed_map_func = cls.preprocess_func(map_func)
        axis_partitions = (
            cls.column_partitions(partitions)
            if axis == 0
            else cls.row_partitions(partitions)
        )
        result_blocks = np.array(
            [
                [
                    axis_partition.reduce(preprocessed_map_func)
                    for axis_partition in axis_partitions
                ]
            ]
        )
        return result_blocks if axis == 0 else result_blocks.T

    @classmethod
    def groupby_reduce(cls, axis, partitions, by_partitions, map_func, reduce_func):
        map_func = ray.put(map_func)
        assert partitions.shape[0] == by_partitions.shape[0]

        partitions_shape = partitions.shape

        num_rows = partitions.shape[0]
        num_cols = partitions.shape[1]

        keys = []
        gpu_managers = []

        for i in range(num_rows):
            for j in range(num_cols):
                partition = partitions[i, j]
                partition_key = partition.get_key()
                partition_gpu_manager = partition.get_gpu_manager()
                by_partition = list(by_partitions[i, :])
                by_partition_object_id = [by.get_object_id() for by in by_partition]
                key = partition_gpu_manager.apply_with_key_oid_list.remote(
                    partition_key, by_partition_object_id, map_func
                )
                gpu_managers.append(partition_gpu_manager)
                keys.append(key)

        keys = ray.get(keys)
        new_partitions = [
            cuDFOnRayFramePartition(gpu_manager, key)
            for gpu_manager, key in zip(gpu_managers, keys)
        ]
        new_partitions = np.array(new_partitions).reshape(partitions_shape)
        return cls.map_axis_partitions(axis, new_partitions, reduce_func)

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        gpu_managers = cls._get_gpu_managers()
        put_func = cls._partition_class.put
        if not df.empty:
            num_cols = cls._compute_num_col_partitions()
            num_rows = cls._compute_num_row_partitions()
            row_size = len(df.index)
            col_size = len(df.columns)
            col_chunksize = int(np.ceil(col_size/num_cols))
            row_chunksize = int(np.ceil(row_size/num_rows))
        else:
            # For empty dataframes, we just need a single partition that is placed in any GPU
            row_size = max(1, len(df.index))
            col_size = max(1, len(df.columns))
            col_chunksize = col_size
            row_chunksize = row_size
            num_cols = 1
            num_rows = 1
        gpu_idx = 0
        keys = []
        for i in range(0, row_size, row_chunksize):
            for j in range(0, col_size, col_chunksize):
                keys.append(
                    put_func(
                        gpu_managers[gpu_idx],
                        df.iloc[i: i + row_chunksize, j: j + col_chunksize]
                    )
                )
                gpu_idx = gpu_idx + 1
        keys = ray.get(keys)
        partitions = [
            cuDFOnRayFramePartition(gpu_manager, key)
            for gpu_manager, key in zip(gpu_managers, keys)
        ]
        partitions = [[partition] for partition in partitions]
        partitions = np.array(partitions, dtype=object).reshape((num_rows, num_cols))
        if not return_dims:
            return partitions
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
            return partitions, row_lengths, col_widths

    @classmethod
    def to_pandas(cls, partitions):
        pandas_partitions = [[ray.get(partition.to_pandas()) for partition in row] for row in partitions]
        if all(
                isinstance(partition, pandas.Series)
                for row in pandas_partitions
                for partition in row
        ):
            axis = 0
        elif all(
                isinstance(partition, pandas.DataFrame)
                for row in pandas_partitions
                for partition in row
        ):
            axis = 1
        else:
            ErrorMessage.catch_bugs_and_request_email(True)
        df_rows = [
            pandas.concat([part for part in row], axis=axis)
            for row in pandas_partitions
            if not all(partition.empty for partition in row)
        ]
        if len(df_rows) == 0:
            return pandas.DataFrame()
        else:
            return cls.concatenate(df_rows)

    @classmethod
    def isna(cls, partitions):

        put_func = cls._partition_class.put

        cudf_parts = [
            ray.get([part.isna.remote() for part in row_of_parts])
            for row_of_parts in partitions
        ]

        parts = [
            [put_func(part) for part in row_of_parts]
            for row_of_parts in cudf_parts
        ]

        return parts

    @classmethod
    def map_partitions(cls, partitions, map_func, persistent=True):
        """Applies `map_func` to every partition.

        Args:
            map_func: The function to apply.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        keys_and_gpus = np.array(
            [
                [
                    [
                        part.get_gpu_manager().apply.remote(part.key, map_func),
                        part.get_gpu_manager()
                    ]
                    for part in row_of_parts
                ]
                for row_of_parts in partitions
            ]
        )
        num_rows, num_cols = keys_and_gpus.shape[:-1]
        keys = ray.get(list(keys_and_gpus[:,:,0].flatten()))
        gpu_managers = keys_and_gpus[:,:,1].flatten()
        return np.array([
            [cuDFOnRayFramePartition(gpu_manager, key)]
            for gpu_manager, key in zip(gpu_managers, keys)
        ], dtype=object).reshape(num_rows, num_cols)

    @classmethod
    def preprocess_func(cls, map_func):
        return cls._partition_class.preprocess_func(map_func)

    # END Abstract Methods

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right):
        """Broadcast the right partitions to left and apply a function.

        Note: This will often be overridden by implementations. It materializes the
            entire partitions of the right and applies them to the left through `apply`.

        Args:
            axis: The axis to apply and broadcast over.
            apply_func: The function to apply.
            left: The left partitions.
            right: The right partitions.

        Returns:
            A new `np.array` of partition objects.
        """
        if right.shape == (1, 1):
            right_parts = right[0]
        else:
            right_parts = np.squeeze(right)

        [obj.drain_call_queue() for obj in right_parts]
        return np.array(
            [
                [
                    part.apply(
                        apply_func,
                        r=right_parts[col_idx].get()
                        if axis
                        else right_parts[row_idx].get(),
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )

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
    def concatenate(cls, dfs):
        """
        Concatenate Pandas DataFrames with saving 'category' dtype

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
    def to_numpy(cls, partitions):
        oids = np.array(
            [
                [
                    block.to_numpy()
                    for block in row
                ]
                for row in partitions
            ]
        )
        num_rows, num_cols = oids.shape
        blocks = ray.get(list(oids.flatten()))
        return np.block(
            [
                [
                    blocks[i * j + j]
                    for j in range(num_cols)
                ]
                for i in range(num_rows)
            ]
        )

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
        """Applies a function to a list of remote partitions.

        Note: The main use for this is to preprocess the func.

        Args:
            func: The func to apply
            partitions: The list of partitions

        Returns:
            A list of BaseFramePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        keys_and_gpus = np.array(
            [
                [
                    obj.apply(preprocessed_func, **kwargs),
                    obj.get_gpu_manager(),
                ]
                for obj in partitions
            ]
        )
        keys = list(keys_and_gpus[:,0].flatten())
        gpus = list(keys_and_gpus[:,1].flatten())
        return [
            cls._partition_class(gpu, key) for gpu, key in zip(gpus, keys)
        ]

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
        Apply a function that requires two cuDFOnRayFramePartition objects.

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
            A new cuDFOnRayFramePartition object, the type of object that called this.
        """
        # TODO: Make this in-place. Now, it is creating a new df
        # TODO: Make it work given axis. Now, it only supports row.

        # Idea: Assuming that each DF is distribtued in the same fashion, apply on each gpu
        # manager a binary func with the dataframe keys of each partition

        func = cls.preprocess_func(func)
        left_flat = left.flatten()
        right_flat = right.flatten()
        def perfom_binary_function(left, right, func):
            result_gpu_manager = left.get_gpu_manager()
            if result_gpu_manager == right.get_gpu_manager():
                result_key = result_gpu_manager.apply_with_two_keys.remote(
                        left.get_key(),
                        right.get_key(),
                        func)
                return result_gpu_manager, result_key
            else:
                raise Exception("The underlying partition on this df not in the same device")


        result = [
            perfom_binary_function(left_flat[i], right_flat[i], func) for i in range(len(left_flat))
        ]
        result = np.array([*map(lambda x: [cuDFOnRayFramePartition(x[0], ray.get(x[1]))], result)])
        return result

    @classmethod
    def join_operation(cls, left, right, func):
        """
        Apply join operation across cuDFOnRayFramePartition objects. Currently,
        the implementation is a brute force with a twist.

        Coming soon: hash join with bloom filter and sort-merge.
                Parameters
        ----------
            left : NumPy array
                The partitions of left Modin Frame
            right : NumPy array
                The partitions of right Modin Frame.
            on: List or String
                In which column/s based the merge operation.
            how: {‘left’, ‘outer’, ‘inner’}
                Type of merged performed.

        Returns
        -------
        NumPy array
            A new cuDFOnRayFramePartition object, the type of object that called this.
        """
        ## MUST BE ROW PARTITION OR BROADCAST DATA
        #TODO(lepl3): Implement other join functions
        #TODO(lepl3): Load Balancing
        func = cls.preprocess_func(func)
        left_flatten = left.flatten()
        right_flatten = right.flatten()
        new_partitions = []
        for left_partitition in left_flatten:
            right_ids = []
            left_gpu_manager = left_partitition.get_gpu_manager()
            left_key = left_partitition.get_key()
            for right_partitition in right_flatten:
                if left_gpu_manager == right_partitition.get_gpu_manager():
                    right_ids.append(right_partitition.get_key())
                else:
                    right_ids.append(right_partitition.get_object_id())
            new_partitions.append([left_gpu_manager, left_gpu_manager.brute_force_merge.remote(left_key, right_ids, func)])
        results = np.array([cuDFOnRayFramePartition(x[0], ray.get(x[1])) for x in new_partitions]).reshape(left.shape)
        return results

    @classmethod
    def column_partitions(cls, partitions):
        return [cls._column_partitions_class(col) for col in partitions.T]

    @classmethod
    def row_partitions(cls, partitions):
        return [cls._row_partition_class(row) for row in partitions]

    @classmethod
    def copy(cls, partitions):
        old_shape = partitions.shape
        partitions = list(partitions.flatten())
        return np.array([obj.copy() for obj in partitions]).reshape(old_shape)

    # TODO (kvu35): Hardcoded for row partitions. Add multiaxis support.
    @classmethod
    def trickle_down(cls, func, partitions, upstream):
        keys = [partition[0].get_key() for partition in partitions]
        gpu_managers = [partition[0].get_gpu_manager() for partition in partitions]
        if not upstream:
            upstream_partitions = partitions
        else:
            upstream_partitions = upstream._partitions
        cudf_dataframe_object_ids = [
            up[0].get_object_id() for up in upstream_partitions
        ]
        oids = [cudf_dataframe_object_ids[:i] for i in range(len(cudf_dataframe_object_ids))]
        keys = [
            gpu.apply_with_key_oid_list.remote(key, oids[i], func, join_type="concat")
            for i, (gpu, key) in enumerate(zip(gpu_managers, keys))
        ]
        keys = ray.get(keys)
        new_partitions = [
            cls._partition_class(gpu, key)
            for gpu, key in zip(gpu_managers, keys)
        ]
        return new_partitions
