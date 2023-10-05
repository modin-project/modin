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

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.logging import get_logger
from modin.pandas.indexing import compute_sliced_len

compute_sliced_len = ray.remote(compute_sliced_len)


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

    def __init__(self, data, length=None, width=None, ip=None, call_queue=None):
        super().__init__()
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

    @staticmethod
    def _apply_call_queue(call_queue, data):
        """
        Execute call queue over the given `data`.

        Parameters
        ----------
        call_queue : list[list[func, args, kwargs], ...]
        data : ray.ObjectRef

        Returns
        -------
        ray.ObjectRef of pandas.DataFrame
            The resulting pandas DataFrame.
        ray.ObjectRef of int
            The number of rows of the resulting pandas DataFrame.
        ray.ObjectRef of int
            The number of columns of the resulting pandas DataFrame.
        ray.ObjectRef of str
            The node IP address of the worker process.
        """
        (
            num_funcs,
            arg_lengths,
            kw_key_lengths,
            kw_value_lengths,
            unfolded_queue,
        ) = deconstruct_call_queue(call_queue)
        return _apply_list_of_funcs.remote(
            data,
            num_funcs,
            arg_lengths,
            kw_key_lengths,
            kw_value_lengths,
            *unfolded_queue,
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
            result, length, width, ip = self._apply_call_queue(call_queue, data)
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, f_args, f_kwargs = call_queue[0]
            result, length, width, ip = _apply_func.remote(
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
            ) = self._apply_call_queue(call_queue, data)
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
            ) = _apply_func.remote(data, func, *f_args, **f_kwargs)
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
        RayWrapper.wait(self._data)

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
                new_obj._length_cache = compute_sliced_len.remote(
                    row_labels, self._length_cache
                )
        if isinstance(col_labels, slice) and isinstance(
            self._width_cache, ObjectIDType
        ):
            if col_labels == slice(None):
                # fast path - full axis take
                new_obj._width_cache = self._width_cache
            else:
                new_obj._width_cache = compute_sliced_len.remote(
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
                self._length_cache, self._width_cache = _get_index_and_columns.remote(
                    self._data
                )
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
                self._length_cache, self._width_cache = _get_index_and_columns.remote(
                    self._data
                )
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


@ray.remote(num_returns=2)
def _get_index_and_columns(df):  # pragma: no cover
    """
    Get the number of rows and columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame which dimensions are needed.

    Returns
    -------
    int
        The number of rows.
    int
        The number of columns.
    """
    return len(df.index), len(df.columns)


def deconstruct_call_queue(call_queue):
    """
    Deconstruct the passed call queue into a 1D list.

    This is required, so the call queue can be then passed to a Ray's kernel
    as a variable-length argument ``kernel(*queue)`` so the Ray engine
    automatically materialize all the futures that the queue might have contained.

    Parameters
    ----------
    call_queue : list[list[func, args, kwargs], ...]

    Returns
    -------
    num_funcs : int
        The number of functions in the call queue.
    arg_lengths : list of ints
        The number of positional arguments for each function in the call queue.
    kw_key_lengths : list of ints
        The number of key-word arguments for each function in the call queue.
    kw_value_lengths : 2D list of dict{"len": int, "was_iterable": bool}
        Description of keyword arguments for each function. For example, `kw_value_lengths[i][j]`
        describes the j-th keyword argument of the i-th function in the call queue.
        The describtion contains of the lengths of the argument and whether it's a list at all
        (for example, {"len": 1, "was_iterable": False} describes a non-list argument).
    unfolded_queue : list
        A 1D call queue that can be reconstructed using ``reconstruct_call_queue`` function.
    """
    num_funcs = len(call_queue)
    arg_lengths = []
    kw_key_lengths = []
    kw_value_lengths = []
    unfolded_queue = []
    for call in call_queue:
        unfolded_queue.append(call[0])
        unfolded_queue.extend(call[1])
        unfolded_queue.extend(call[2].keys())
        value_lengths = []
        for value in call[2].values():
            was_iterable = True
            if not isinstance(value, (list, tuple)):
                was_iterable = False
                value = [value]
            unfolded_queue.extend(value)
            value_lengths.append({"len": len(value), "was_iterable": was_iterable})

        arg_lengths.append(len(call[1]))
        kw_key_lengths.append(len(call[2]))
        kw_value_lengths.append(value_lengths)

    return num_funcs, arg_lengths, kw_key_lengths, kw_value_lengths, unfolded_queue


def reconstruct_call_queue(
    num_funcs, arg_lengths, kw_key_lengths, kw_value_lengths, unfolded_queue
):
    """
    Reconstruct original call queue from the result of the ``deconstruct_call_queue()``.

    Parameters
    ----------
    num_funcs : int
        The number of functions in the call queue.
    arg_lengths : list of ints
        The number of positional arguments for each function in the call queue.
    kw_key_lengths : list of ints
        The number of key-word arguments for each function in the call queue.
    kw_value_lengths : 2D list of dict{"len": int, "was_iterable": bool}
        Description of keyword arguments for each function. For example, `kw_value_lengths[i][j]`
        describes the j-th keyword argument of the i-th function in the call queue.
        The describtion contains of the lengths of the argument and whether it's a list at all
        (for example, {"len": 1, "was_iterable": False} describes a non-list argument).
    unfolded_queue : list
        A 1D call queue that is result of the ``deconstruct_call_queue()`` function.

    Returns
    -------
    list[list[func, args, kwargs], ...]
        Original call queue.
    """
    items_took = 0

    def take_n_items(n):
        nonlocal items_took
        res = unfolded_queue[items_took : items_took + n]
        items_took += n
        return res

    call_queue = []
    for i in range(num_funcs):
        func = take_n_items(1)[0]
        args = take_n_items(arg_lengths[i])
        kw_keys = take_n_items(kw_key_lengths[i])
        kwargs = {}
        value_lengths = kw_value_lengths[i]
        for j, key in enumerate(kw_keys):
            vals = take_n_items(value_lengths[j]["len"])
            if value_lengths[j]["len"] == 1 and not value_lengths[j]["was_iterable"]:
                vals = vals[0]
            kwargs[key] = vals

        call_queue.append((func, args, kwargs))

    return call_queue


@ray.remote(num_returns=4)
def _apply_func(partition, func, *args, **kwargs):  # pragma: no cover
    """
    Execute a function on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the function needs to be executed on.
    func : callable
        The function to perform on the partition.
    *args : list
        Positional arguments to pass to ``func``.
    **kwargs : dict
        Keyword arguments to pass to ``func``.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    int
        The number of rows of the resulting pandas DataFrame.
    int
        The number of columns of the resulting pandas DataFrame.
    str
        The node IP address of the worker process.

    Notes
    -----
    Directly passing a call queue entry (i.e. a list of [func, args, kwargs]) instead of
    destructuring it causes a performance penalty.
    """
    try:
        result = func(partition, *args, **kwargs)
    # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
    # don't want the error to propagate to the user, and we want to avoid copying unless
    # we absolutely have to.
    except ValueError:
        result = func(partition.copy(), *args, **kwargs)
    return (
        result,
        len(result) if hasattr(result, "__len__") else 0,
        len(result.columns) if hasattr(result, "columns") else 0,
        get_node_ip_address(),
    )


@ray.remote(num_returns=4)
def _apply_list_of_funcs(
    partition, num_funcs, arg_lengths, kw_key_lengths, kw_value_lengths, *futures
):  # pragma: no cover
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.
    num_funcs : int
        The number of functions in the call queue.
    arg_lengths : list of ints
        The number of positional arguments for each function in the call queue.
    kw_key_lengths : list of ints
        The number of key-word arguments for each function in the call queue.
    kw_value_lengths : 2D list of dict{"len": int, "was_iterable": bool}
        Description of keyword arguments for each function. For example, `kw_value_lengths[i][j]`
        describes the j-th keyword argument of the i-th function in the call queue.
        The describtion contains of the lengths of the argument and whether it's a list at all
        (for example, {"len": 1, "was_iterable": False} describes a non-list argument).
    *futures : list
        A 1D call queue that is result of the ``deconstruct_call_queue()`` function.

    Returns
    -------
    pandas.DataFrame
        The resulting pandas DataFrame.
    int
        The number of rows of the resulting pandas DataFrame.
    int
        The number of columns of the resulting pandas DataFrame.
    str
        The node IP address of the worker process.
    """
    call_queue = reconstruct_call_queue(
        num_funcs, arg_lengths, kw_key_lengths, kw_value_lengths, futures
    )
    for func, args, kwargs in call_queue:
        try:
            partition = func(partition, *args, **kwargs)
        # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
        # don't want the error to propagate to the user, and we want to avoid copying unless
        # we absolutely have to.
        except ValueError:
            partition = func(partition.copy(), *args, **kwargs)

    return (
        partition,
        len(partition) if hasattr(partition, "__len__") else 0,
        len(partition.columns) if hasattr(partition, "columns") else 0,
        get_node_ip_address(),
    )
