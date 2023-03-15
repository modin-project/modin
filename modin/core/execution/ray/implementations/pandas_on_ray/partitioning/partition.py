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
from ray.util import get_node_ip_address

from modin.core.execution.ray.common.utils import deserialize, ObjectIDType, wait
from modin.core.execution.ray.common import RayWrapper
from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.pandas.indexing import compute_sliced_len
from modin.logging import get_logger
from modin.core.execution.utils import (
    _get_index_and_columns_size,
    get_apply_func,
    get_apply_list_of_funcs,
)


class PandasOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    data : ray.ObjectRef
        A reference to ``pandas.DataFrame`` that need to be wrapped with this class.
    length : ray.ObjectRef or int, optional
        Length or reference to it of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width or reference to it of wrapped ``pandas.DataFrame``.
    ip : ray.ObjectRef or str, optional
        Node IP address or reference to it that holds wrapped ``pandas.DataFrame``.
    call_queue : list
        Call queue that needs to be executed on wrapped ``pandas.DataFrame``.
    """

    execution_wrapper = RayWrapper
    _compute_sliced_len = ray.remote(compute_sliced_len)
    _get_index_and_columns_size = ray.remote(num_returns=2)(_get_index_and_columns_size)
    _apply_func = ray.remote(num_returns=4)(get_apply_func(get_node_ip_address))
    _apply_list_of_funcs = ray.remote(num_returns=4)(
        get_apply_list_of_funcs(get_node_ip_address, deserialize)
    )

    def __init__(self, data, length=None, width=None, ip=None, call_queue=None):
        assert isinstance(data, ObjectIDType)
        self._data = data
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip

        log = get_logger()
        self._is_debug(log) and log.debug(
            "Partition ID: {}, Height: {}, Width: {}, Node IP: {}".format(
                self._identity,
                str(self._length_cache),
                str(self._width_cache),
                str(self._ip_cache),
            )
        )

    def apply(self, func, *args, **kwargs):
        """
        Apply a function to the object wrapped by this partition.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.

        Notes
        -----
        It does not matter if `func` is callable or an ``ray.ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.apply::{self._identity}")
        data = self._data
        call_queue = self.call_queue + [[func, args, kwargs]]
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(
                f"SUBMIT::_apply_list_of_funcs::{self._identity}"
            )
            result, length, width, ip = self._apply_list_of_funcs.remote(
                call_queue, data
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, f_args, f_kwargs = call_queue[0]
            result, length, width, ip = self._apply_func.remote(
                data, func, *f_args, **f_kwargs
            )
            self._is_debug(log) and log.debug(f"SUBMIT::_apply_func::{self._identity}")
        self._is_debug(log) and log.debug(f"EXIT::Partition.apply::{self._identity}")
        return self.__constructor__(result, length, width, ip)

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        log = get_logger()
        self._is_debug(log) and log.debug(
            f"ENTER::Partition.drain_call_queue::{self._identity}"
        )
        if len(self.call_queue) == 0:
            return
        data = self._data
        call_queue = self.call_queue
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(
                f"SUBMIT::_apply_list_of_funcs::{self._identity}"
            )
            (
                self._data,
                new_length,
                new_width,
                self._ip_cache,
            ) = self._apply_list_of_funcs.remote(call_queue, data)
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, f_args, f_kwargs = call_queue[0]
            self._is_debug(log) and log.debug(f"SUBMIT::_apply_func::{self._identity}")
            (
                self._data,
                new_length,
                new_width,
                self._ip_cache,
            ) = self._apply_func.remote(data, func, *f_args, **f_kwargs)
        self._is_debug(log) and log.debug(
            f"EXIT::Partition.drain_call_queue::{self._identity}"
        )
        self.call_queue = []

        # GH#4732 if we already have evaluated width/length cached as ints,
        #  don't overwrite that cache with non-evaluated values.
        if not isinstance(self._length_cache, int):
            self._length_cache = new_length
        if not isinstance(self._width_cache, int):
            self._width_cache = new_width

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait([self._data])

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnRayDataframePartition
            A copy of this partition.
        """
        return self.__constructor__(
            self._data,
            length=self._length_cache,
            width=self._width_cache,
            ip=self._ip_cache,
            call_queue=self.call_queue,
        )

    # If Ray has not been initialized yet by Modin,
    # it will be initialized when calling `RayWrapper.put`.
    _iloc = execution_wrapper.put(PandasDataframePartition._iloc)

    def mask(self, row_labels, col_labels):
        """
        Lazily create a mask that extracts the indices provided.

        Parameters
        ----------
        row_labels : list-like, slice or label
            The row labels for the rows to extract.
        col_labels : list-like, slice or label
            The column labels for the columns to extract.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.mask::{self._identity}")
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and isinstance(
            self._length_cache, ObjectIDType
        ):
            if row_labels == slice(None):
                # fast path - full axis take
                new_obj._length_cache = self._length_cache
            else:
                new_obj._length_cache = self._compute_sliced_len.remote(
                    row_labels, self._length_cache
                )
        if isinstance(col_labels, slice) and isinstance(
            self._width_cache, ObjectIDType
        ):
            if col_labels == slice(None):
                # fast path - full axis take
                new_obj._width_cache = self._width_cache
            else:
                new_obj._width_cache = self._compute_sliced_len.remote(
                    col_labels, self._width_cache
                )
        self._is_debug(log) and log.debug(f"EXIT::Partition.mask::{self._identity}")
        return new_obj

    @classmethod
    def put(cls, obj):
        """
        Put an object into Plasma store and wrap it with partition object.

        Parameters
        ----------
        obj : any
            An object to be put.

        Returns
        -------
        PandasOnRayDataframePartition
            A new ``PandasOnRayDataframePartition`` object.
        """
        return cls(cls.execution_wrapper.put(obj), len(obj.index), len(obj.columns))

    @classmethod
    def preprocess_func(cls, func):
        """
        Put a function into the Plasma store to use in ``apply``.

        Parameters
        ----------
        func : callable
            A function to preprocess.

        Returns
        -------
        ray.ObjectRef
            A reference to `func`.
        """
        return cls.execution_wrapper.put(func)

    def length(self, materialize=True):
        """
        Get the length of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The length of the object.
        """
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                (
                    self._length_cache,
                    self._width_cache,
                ) = self._get_index_and_columns.remote(self._data)
        if isinstance(self._length_cache, ObjectIDType) and materialize:
            self._length_cache = RayWrapper.materialize(self._length_cache)
        return self._length_cache

    def width(self, materialize=True):
        """
        Get the width of the object wrapped by the partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        int or ray.ObjectRef
            The width of the object.
        """
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                (
                    self._length_cache,
                    self._width_cache,
                ) = self._get_index_and_columns.remote(self._data)
        if isinstance(self._width_cache, ObjectIDType) and materialize:
            self._width_cache = RayWrapper.materialize(self._width_cache)
        return self._width_cache

    def ip(self):
        """
        Get the node IP address of the object wrapped by this partition.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if self._ip_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._ip_cache = self.apply(lambda df: df)._ip_cache
        if isinstance(self._ip_cache, ObjectIDType):
            self._ip_cache = RayWrapper.materialize(self._ip_cache)
        return self._ip_cache
