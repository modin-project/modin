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

"""Module houses classes of axis partitions implemented using Ray and cuDF."""

import cudf
import ray

from .partition import cuDFOnRayDataframePartition


class cuDFOnRayDataframeAxisPartition(object):
    """
    Base class for any axis partition class for cuDF storage format.

    Parameters
    ----------
    partitions : np.ndarray
        NumPy array with ``cuDFOnRayDataframePartition``-s.
    """

    def __init__(self, partitions):
        self.partitions = [obj for obj in partitions]

    partition_type = cuDFOnRayDataframePartition
    instance_type = cudf.DataFrame


class cuDFOnRayDataframeColumnPartition(cuDFOnRayDataframeAxisPartition):
    """
    The column partition implementation of ``cuDFOnRayDataframeAxisPartition``.

    Parameters
    ----------
    partitions : np.ndarray
        NumPy array with ``cuDFOnRayDataframePartition``-s.
    """

    axis = 0

    def reduce(self, func):
        """
        Reduce partitions along `self.axis` and apply `func`.

        Parameters
        ----------
        func : callable
            A func to apply.

        Returns
        -------
        cuDFOnRayDataframePartition
        """
        keys = [partition.get_key() for partition in self.partitions]
        gpu_managers = [partition.get_gpu_manager() for partition in self.partitions]
        head_gpu_manager = gpu_managers[0]
        cudf_dataframe_object_ids = [
            gpu_manager.get.remote(key) for gpu_manager, key in zip(gpu_managers, keys)
        ]

        # FIXME: The signature of `head_gpu_manager.reduce` requires
        # (first, others, func, axis=0, **kwargs) parameters, but `first`
        # parameter isn't present.
        key = head_gpu_manager.reduce.remote(
            cudf_dataframe_object_ids, axis=self.axis, func=func
        )
        key = ray.get(key)
        result = cuDFOnRayDataframePartition(gpu_manager=head_gpu_manager, key=key)
        return result


class cuDFOnRayDataframeRowPartition(cuDFOnRayDataframeAxisPartition):
    """
    The row partition implementation of ``cuDFOnRayDataframeAxisPartition``.

    Parameters
    ----------
    partitions : np.ndarray
        NumPy array with ``cuDFOnRayDataframePartition``-s.
    """

    axis = 1

    def reduce(self, func):
        """
        Reduce partitions along `self.axis` and apply `func`.

        Parameters
        ----------
        func : calalble
            A func to apply.

        Returns
        -------
        cuDFOnRayDataframePartition

        Notes
        -----
        Since we are using row partitions, we can bypass the Ray plasma
        store during axis reduce functions.
        """
        keys = [partition.get_key() for partition in self.partitions]
        gpu = self.partitions[0].get_gpu_manager()

        # FIXME: Method `gpu_manager.reduce_key_list` does not exist.
        key = gpu.reduce_key_list.remote(keys, func)
        key = ray.get(key)
        return cuDFOnRayDataframePartition(gpu_manager=gpu, key=key)
