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

"""Module houses class that wraps data (block partition) and its metadata."""

import ray
import cudf
import cupy
import numpy as np
import cupy as cp
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from pandas.core.dtypes.common import is_list_like


class cuDFOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition`` using cuDF on Ray.

    Parameters
    ----------
    gpu_manager : modin.core.execution.ray.implementations.cudf_on_ray.partitioning.GPUManager
        A gpu manager to store cuDF dataframes.
    key : ray.ObjectRef or int
        An integer key (or reference to key) associated with
        ``cudf.DataFrame`` stored in `gpu_manager`.
    length : ray.ObjectRef or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    """

    @property
    def __constructor__(self):
        """
        Create a new instance of this object.

        Returns
        -------
        cuDFOnRayDataframePartition
            New instance of cuDF partition.
        """
        return type(self)

    def __init__(self, gpu_manager, key, length=None, width=None):
        self.gpu_manager = gpu_manager
        self.key = key
        self._length_cache = length
        self._width_cache = width

    def __copy__(self):
        """
        Create a copy of this object.

        Returns
        -------
        cuDFOnRayDataframePartition
            A copy of this object.
        """
        # Shallow copy.
        return cuDFOnRayDataframePartition(
            self.gpu_manager, self.key, self._length_cache, self._width_cache
        )

    @classmethod
    def put(cls, gpu_manager, pandas_dataframe):
        """
        Put `pandas_dataframe` to `gpu_manager`.

        Parameters
        ----------
        gpu_manager : modin.core.execution.ray.implementations.cudf_on_ray.partitioning.GPUManager
            A gpu manager to store cuDF dataframes.
        pandas_dataframe : pandas.DataFrame/pandas.Series
            A ``pandas.DataFrame/pandas.Series`` to put.

        Returns
        -------
        ray.ObjectRef
            A reference to integer key of added pandas.DataFrame
            to internal dict-storage in `gpu_manager`.
        """
        return gpu_manager.put.remote(pandas_dataframe)

    def apply(self, func, **kwargs):
        """
        Apply `func` to this partition.

        Parameters
        ----------
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        ray.ObjectRef
            A reference to integer key of result
            in internal dict-storage of `self.gpu_manager`.
        """
        return self.gpu_manager.apply.remote(self.get_key(), None, func, **kwargs)

    # TODO: Check the need of this method
    def apply_result_not_dataframe(self, func, **kwargs):
        """
        Apply `func` to this partition.

        Parameters
        ----------
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        ray.ObjectRef
            A reference to integer key of result
            in internal dict-storage of `self.gpu_manager`.
        """
        # FIXME: Can't find `gpu_manager.apply_result_not_dataframe` method.
        return self.gpu_manager.apply_result_not_dataframe.remote(
            self.get_key(), func, **kwargs
        )

    def add_to_apply_calls(self, func, **kwargs):
        """
        Apply `func` to this partition and create new.

        Parameters
        ----------
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        cuDFOnRayDataframePartition
            New partition based on result of `func`.

        Notes
        -----
        We eagerly schedule the apply `func` and produce a new ``cuDFOnRayDataframePartition``.
        """
        return cuDFOnRayDataframePartition(self.gpu_manager, self.apply(func, **kwargs))

    @classmethod
    def preprocess_func(cls, func):
        """
        Put `func` to Ray object store.

        Parameters
        ----------
        func : callable
            Function to put.

        Returns
        -------
        ray.ObjectRef
            A reference to `func` in Ray object store.
        """
        return ray.put(func)

    def length(self):
        """
        Get the length of the object wrapped by this partition.

        Returns
        -------
        int or ray.ObjectRef
            The length (or reference to length) of the object.
        """
        if self._length_cache:
            return self._length_cache
        return self.gpu_manager.length.remote(self.get_key())

    def width(self):
        """
        Get the width of the object wrapped by this partition.

        Returns
        -------
        int or ray.ObjectRef
            The width (or reference to width) of the object.
        """
        if self._width_cache:
            return self._width_cache
        return self.gpu_manager.width.remote(self.get_key())

    def mask(self, row_labels, col_labels):
        """
        Select columns or rows from given indices.

        Parameters
        ----------
        row_labels : list of hashable
            The row labels to extract.
        col_labels : list of hashable
            The column labels to extract.

        Returns
        -------
        ray.ObjectRef
            A reference to integer key of result
            in internal dict-storage of `self.gpu_manager`.
        """
        if (
            (isinstance(row_labels, slice) and row_labels == slice(None))
            or (
                not isinstance(row_labels, slice)
                and self._length_cache is not None
                and len(row_labels) == self._length_cache
            )
        ) and (
            (isinstance(col_labels, slice) and col_labels == slice(None))
            or (
                not isinstance(col_labels, slice)
                and self._width_cache is not None
                and len(col_labels) == self._width_cache
            )
        ):
            return self.__copy__()

        # CuDF currently does not support indexing multiindices with arrays,
        # so we have to create a boolean array where the desire indices are true.
        # TODO(kvu35): Check if this functionality is fixed in the latest version of cudf
        def iloc(df, row_labels, col_labels):
            if isinstance(df.index, cudf.core.multiindex.MultiIndex) and is_list_like(
                row_labels
            ):
                new_row_labels = cp.full(
                    (1, df.index.size), False, dtype=bool
                ).squeeze()
                new_row_labels[row_labels] = True
                row_labels = new_row_labels
            return df.iloc[row_labels, col_labels]

        iloc = cuDFOnRayDataframePartition.preprocess_func(iloc)
        return self.gpu_manager.apply.remote(
            self.key,
            None,
            iloc,
            col_labels=col_labels,
            row_labels=row_labels,
        )

    def get_gpu_manager(self):
        """
        Get gpu manager associated with this partition.

        Returns
        -------
        modin.core.execution.ray.implementations.cudf_on_ray.partitioning.GPUManager
            ``GPUManager`` associated with this object.
        """
        return self.gpu_manager

    def get_key(self):
        """
        Get integer key of this partition in dict-storage of `self.gpu_manager`.

        Returns
        -------
        int
        """
        return ray.get(self.key) if isinstance(self.key, ray.ObjectRef) else self.key

    def get_object_id(self):
        """
        Get object stored for this partition from `self.gpu_manager`.

        Returns
        -------
        ray.ObjectRef
        """
        # FIXME: Can't find `gpu_manager.get_object_id` method. Probably, method
        # `gpu_manager.get_oid` should be used.
        return self.gpu_manager.get_object_id.remote(self.get_key())

    def get(self):
        """
        Get object stored by this partition from `self.gpu_manager`.

        Returns
        -------
        ray.ObjectRef
        """
        # FIXME: Can't find `gpu_manager.get` method. Probably, method
        # `gpu_manager.get_oid` should be used.
        return self.gpu_manager.get.remote(self.get_key())

    def to_pandas(self):
        """
        Convert this partition to pandas.DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        return ray.get(
            self.gpu_manager.apply_non_persistent.remote(
                self.get_key(), None, cudf.DataFrame.to_pandas
            )
        )

    def to_numpy(self):
        """
        Convert this partition to NumPy array.

        Returns
        -------
        NumPy array
        """

        def convert(df):
            """Convert `df` to NumPy array."""
            if len(df.columns == 1):
                df = df.iloc[:, 0]
            if isinstance(df, cudf.Series):  # convert to column vector
                return cupy.asnumpy(df.to_array())[:, np.newaxis]
            elif isinstance(
                df, cudf.DataFrame
            ):  # dataframes do not support df.values with strings
                return cupy.asnumpy(df.values)

        # FIXME: Can't find `gpu_manager.apply_result_not_dataframe` method.
        return self.gpu_manager.apply_result_not_dataframe.remote(
            self.get_key(),
            convert,
        )

    def free(self):
        """Free the dataFrame and associated `self.key` out of `self.gpu_manager`."""
        self.gpu_manager.free.remote(self.get_key())

    def copy(self):
        """
        Create a full copy of this object.

        Returns
        -------
        cuDFOnRayDataframePartition
        """
        new_key = self.gpu_manager.apply.remote(
            self.get_key(),
            lambda x: x,
        )
        new_key = ray.get(new_key)
        return self.__constructor__(self.gpu_manager, new_key)

    # TODO(kvu35): buggy garbage collector reference issue #43
    # def __del__(self):
    #     self.free()
