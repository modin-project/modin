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

import pandas
from distributed import Future
from distributed.utils import get_ip

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.dask.common import DaskWrapper
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len


class PandasOnDaskDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    data : distributed.Future
        A reference to pandas DataFrame that need to be wrapped with this class.
    length : distributed.Future or int, optional
        Length or reference to it of wrapped pandas DataFrame.
    width : distributed.Future or int, optional
        Width or reference to it of wrapped pandas DataFrame.
    ip : distributed.Future or str, optional
        Node IP address or reference to it that holds wrapped pandas DataFrame.
    call_queue : list, optional
        Call queue that needs to be executed on wrapped pandas DataFrame.
    """

    execution_wrapper = DaskWrapper

    def __init__(self, data, length=None, width=None, ip=None, call_queue=None):
        super().__init__()
        assert isinstance(data, Future)
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
        func : callable or distributed.Future
            A function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.apply::{self._identity}")
        call_queue = self.call_queue + [[func, args, kwargs]]
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(
                f"SUBMIT::_apply_list_of_funcs::{self._identity}"
            )
            futures = self.execution_wrapper.deploy(
                func=apply_list_of_funcs,
                f_args=(call_queue, self._data),
                num_returns=2,
                pure=False,
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this improves performance a bit.
            func, f_args, f_kwargs = call_queue[0]
            futures = self.execution_wrapper.deploy(
                func=apply_func,
                f_args=(self._data, func, *f_args),
                f_kwargs=f_kwargs,
                num_returns=2,
                pure=False,
            )
            self._is_debug(log) and log.debug(f"SUBMIT::_apply_func::{self._identity}")
        self._is_debug(log) and log.debug(f"EXIT::Partition.apply::{self._identity}")
        return self.__constructor__(futures[0], ip=futures[1])

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        log = get_logger()
        self._is_debug(log) and log.debug(
            f"ENTER::Partition.drain_call_queue::{self._identity}"
        )
        if len(self.call_queue) == 0:
            return
        call_queue = self.call_queue
        if len(call_queue) > 1:
            self._is_debug(log) and log.debug(
                f"SUBMIT::_apply_list_of_funcs::{self._identity}"
            )
            futures = self.execution_wrapper.deploy(
                func=apply_list_of_funcs,
                f_args=(call_queue, self._data),
                num_returns=2,
                pure=False,
            )
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this improves performance a bit.
            func, f_args, f_kwargs = call_queue[0]
            self._is_debug(log) and log.debug(f"SUBMIT::_apply_func::{self._identity}")
            futures = self.execution_wrapper.deploy(
                func=apply_func,
                f_args=(self._data, func, *f_args),
                f_kwargs=f_kwargs,
                num_returns=2,
                pure=False,
            )
        self._data = futures[0]
        self._ip_cache = futures[1]
        self._is_debug(log) and log.debug(
            f"EXIT::Partition.drain_call_queue::{self._identity}"
        )
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        self.execution_wrapper.wait(self._data)

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
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.
        """
        log = get_logger()
        self._is_debug(log) and log.debug(f"ENTER::Partition.mask::{self._identity}")
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and isinstance(self._length_cache, Future):
            if row_labels == slice(None):
                # fast path - full axis take
                new_obj._length_cache = self._length_cache
            else:
                new_obj._length_cache = self.execution_wrapper.deploy(
                    func=compute_sliced_len, f_args=(row_labels, self._length_cache)
                )
        if isinstance(col_labels, slice) and isinstance(self._width_cache, Future):
            if col_labels == slice(None):
                # fast path - full axis take
                new_obj._width_cache = self._width_cache
            else:
                new_obj._width_cache = self.execution_wrapper.deploy(
                    func=compute_sliced_len, f_args=(col_labels, self._width_cache)
                )
        self._is_debug(log) and log.debug(f"EXIT::Partition.mask::{self._identity}")
        return new_obj

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnDaskDataframePartition
            A copy of this partition.
        """
        return self.__constructor__(
            self._data,
            length=self._length_cache,
            width=self._width_cache,
            ip=self._ip_cache,
            call_queue=self.call_queue,
        )

    @classmethod
    def put(cls, obj):
        """
        Put an object into distributed memory and wrap it with partition object.

        Parameters
        ----------
        obj : any
            An object to be put.

        Returns
        -------
        PandasOnDaskDataframePartition
            A new ``PandasOnDaskDataframePartition`` object.
        """
        return cls(
            cls.execution_wrapper.put(obj, hash=False),
            len(obj.index),
            len(obj.columns),
        )

    @classmethod
    def preprocess_func(cls, func):
        """
        Preprocess a function before an ``apply`` call.

        Parameters
        ----------
        func : callable
            The function to preprocess.

        Returns
        -------
        callable
            An object that can be accepted by ``apply``.
        """
        return cls.execution_wrapper.put(func, hash=False, broadcast=True)

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
        int or distributed.Future
            The length of the object.
        """
        if self._length_cache is None:
            self._length_cache = self.apply(len)._data
        if isinstance(self._length_cache, Future) and materialize:
            self._length_cache = self.execution_wrapper.materialize(self._length_cache)
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
        int or distributed.Future
            The width of the object.
        """
        if self._width_cache is None:
            self._width_cache = self.apply(lambda df: len(df.columns))._data
        if isinstance(self._width_cache, Future) and materialize:
            self._width_cache = self.execution_wrapper.materialize(self._width_cache)
        return self._width_cache

    def ip(self, materialize=True):
        """
        Get the node IP address of the object wrapped by this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

        Returns
        -------
        str
            IP address of the node that holds the data.
        """
        if self._ip_cache is None:
            self._ip_cache = self.apply(lambda df: pandas.DataFrame([]))._ip_cache
        if materialize and isinstance(self._ip_cache, Future):
            self._ip_cache = self.execution_wrapper.materialize(self._ip_cache)
        return self._ip_cache


def apply_func(partition, func, *args, **kwargs):
    """
    Execute a function on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the function needs to be executed on.
    func : callable
        The function to perform.
    *args : list
        Positional arguments to pass to ``func``.
    **kwargs : dict
        Keyword arguments to pass to ``func``.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.

    Notes
    -----
    Directly passing a call queue entry (i.e. a list of [func, args, kwargs]) instead of
    destructuring it causes a performance penalty.
    """
    result = func(partition, *args, **kwargs)
    return result, get_ip()


def apply_list_of_funcs(call_queue, partition):
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    call_queue : list
        A call queue of ``[func, args, kwargs]`` triples that needs to be executed on the partition.
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    for func, f_args, f_kwargs in call_queue:
        partition = func(partition, *f_args, **f_kwargs)
    return partition, get_ip()
