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

"""Module houses class that implements ``GenericRayDataframePartitionManager`` using cuDF."""

import numpy as np
import ray

from .axis_partition import (
    cuDFOnRayDataframeColumnPartition,
    cuDFOnRayDataframeRowPartition,
)
from .partition import cuDFOnRayDataframePartition
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.config import GpuCount
from modin.core.execution.ray.generic.partitioning import (
    GenericRayDataframePartitionManager,
)

# Global view of GPU Actors
GPU_MANAGERS = []


# TODO: Check the need for this func
@ray.remote(num_cpus=1, num_gpus=0.5)
def func(df, other, apply_func):
    """
    Perform remotely `apply_func` on `df` and `other` objects.

    Parameters
    ----------
    df : cuDFOnRayDataframePartition
        Object to be processed.
    other : cuDFOnRayDataframePartition
        Object to be processed.
    apply_func : callable
        Function to apply.

    Returns
    -------
    The type of return of `apply_func`
        The result of the `apply_func`
        (will be a ``ray.ObjectRef`` in outside level).
    """
    return apply_func(ray.get(df.get.remote()), ray.get(other.get.remote()))


class cuDFOnRayDataframePartitionManager(GenericRayDataframePartitionManager):
    """The class implements the interface in ``GenericRayDataframePartitionManager`` using cuDF on Ray."""

    _partition_class = cuDFOnRayDataframePartition
    _column_partitions_class = cuDFOnRayDataframeColumnPartition
    _row_partition_class = cuDFOnRayDataframeRowPartition

    @classmethod
    def _create_partitions(cls, keys, gpu_managers):
        """
        Create NumPy array of partitions.

        Parameters
        ----------
        keys : list
            List of keys associated with dataframes in
            `gpu_managers`.
        gpu_managers : list
            List of ``GPUManager`` objects, which store
            dataframes.

        Returns
        -------
        np.ndarray
            A NumPy array of ``cuDFOnRayDataframePartition`` objects.
        """
        return np.array(
            [
                cls._partition_class(gpu_managers[i], keys[i])
                for i in range(len(gpu_managers))
            ]
        )

    @classmethod
    def _get_gpu_managers(cls):
        """
        Get list of gpu managers.

        Returns
        -------
        list
        """
        return GPU_MANAGERS

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        """
        Create partitions from ``pandas.DataFrame/pandas.Series``.

        Parameters
        ----------
        df : pandas.DataFrame/pandas.Series
            A ``pandas.DataFrame`` to add.
        return_dims : boolean, default: False
            Is return dimensions or not.

        Returns
        -------
        list or tuple
            List of partitions in case `return_dims` == False,
            tuple (partitions, row lengths, col widths) in other case.
        """
        num_splits = GpuCount.get()
        put_func = cls._partition_class.put
        # For now, we default to row partitioning
        pandas_dfs = split_result_of_axis_func_pandas(0, num_splits, df)
        keys = [
            put_func(cls._get_gpu_managers()[i], pandas_dfs[i])
            for i in range(num_splits)
        ]
        keys = ray.get(keys)
        parts = cls._create_partitions(keys, cls._get_gpu_managers()).reshape(
            (num_splits, 1)
        )
        if not return_dims:
            return parts
        else:
            row_lengths = [len(df.index) for df in pandas_dfs]
            col_widths = [
                len(df.columns)
            ]  # single value since we only have row partitions
            return parts, row_lengths, col_widths

    @classmethod
    def lazy_map_partitions(cls, partitions, map_func):
        """
        Apply `map_func` to every partition lazily.

        Compared to Modin-CPU, Modin-GPU lazy version represents:

        (1) A scheduled function in the Ray task graph.

        (2) A non-materialized key.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with partitions.
        map_func : callable
           The function to apply.

        Returns
        -------
        np.ndarray
            A NumPy array of ``cuDFOnRayDataframePartition`` objects.
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        partitions_flat = partitions.flatten()
        key_futures = [
            partition.apply(preprocessed_map_func) for partition in partitions_flat
        ]
        gpu_managers = [partition.get_gpu_manager() for partition in partitions_flat]
        return cls._create_partitions(key_futures, gpu_managers).reshape(
            partitions.shape
        )

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """
        Apply `func` to a list of remote partitions from `partitions`.

        Parameters
        ----------
        func : callable
            The function to apply.
        partitions : np.ndarray
            NumPy array with partitions.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        np.ndarray
            A NumPy array of ``cuDFOnRayDataframePartition`` objects.

        Notes
        -----
        This preprocesses the `func` first before applying it to the partitions.
        """
        preprocessed_map_func = cls.preprocess_func(func)
        key_futures = ray.get(
            [
                partition.apply(preprocessed_map_func, **kwargs)
                for partition in partitions
            ]
        )
        gpu_managers = [partition.get_gpu_manager() for partition in partitions]
        return cls._create_partitions(key_futures, gpu_managers)
