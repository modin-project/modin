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
import uuid
from modin.core.execution.ray.common.utils import deserialize, ObjectIDType

from modin.core.dataframe.pandas.partitioning.partition import PandasDataframePartition
from modin.pandas.indexing import compute_sliced_len
from modin.logging import get_logger

compute_sliced_len = ray.remote(compute_sliced_len)


class PandasOnRayDataframePartition(PandasDataframePartition):
    """
    The class implements the interface in ``PandasDataframePartition``.

    Parameters
    ----------
    object_id : ray.ObjectRef
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

    def __init__(self, object_id, length=None, width=None, ip=None, call_queue=None):
        assert isinstance(object_id, ObjectIDType)
        self.oid = object_id
        if call_queue is None:
            call_queue = []
        self.call_queue = call_queue
        self._length_cache = length
        self._width_cache = width
        self._ip_cache = ip
        self._identity = uuid.uuid4().hex

        logger = get_logger()
        logger.debug(
            "Partition ID: {}, Height: {}, Width: {}, Node IP: {}".format(
                self._identity,
                str(self._length_cache),
                str(self._width_cache),
                str(self._ip_cache),
            )
        )

    def get(self):
        """
        Get the object wrapped by this partition out of the Plasma store.

        Returns
        -------
        pandas.DataFrame
            The object from the Plasma store.
        """
        logger = get_logger()
        logger.debug(f"ENTER::Partition.get::{self._identity}")
        if len(self.call_queue):
            self.drain_call_queue()
        result = ray.get(self.oid)
        logger.debug(f"EXIT::Partition.get::{self._identity}")
        return result

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
        logger = get_logger()
        logger.debug(f"ENTER::Partition.apply::{self._identity}")
        oid = self.oid
        call_queue = self.call_queue + [(func, args, kwargs)]
        if len(call_queue) > 1:
            logger.debug(f"SUBMIT::_apply_list_of_funcs::{self._identity}")
            result, length, width, ip = _apply_list_of_funcs.remote(call_queue, oid)
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, args, kwargs = call_queue[0]
            result, length, width, ip = _apply_func.remote(oid, func, *args, **kwargs)
            logger.debug(f"SUBMIT::_apply_func::{self._identity}")
        logger.debug(f"EXIT::Partition.apply::{self._identity}")
        return PandasOnRayDataframePartition(result, length, width, ip)

    def add_to_apply_calls(self, func, *args, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            Function to be added to the call queue.
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
        return PandasOnRayDataframePartition(
            self.oid, call_queue=self.call_queue + [(func, args, kwargs)]
        )

    def drain_call_queue(self):
        """Execute all operations stored in the call queue on the object wrapped by this partition."""
        logger = get_logger()
        logger.debug(f"ENTER::Partition.drain_call_queue::{self._identity}")
        if len(self.call_queue) == 0:
            return
        oid = self.oid
        call_queue = self.call_queue
        if len(call_queue) > 1:
            logger.debug(f"SUBMIT::_apply_list_of_funcs::{self._identity}")
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = _apply_list_of_funcs.remote(call_queue, oid)
        else:
            # We handle `len(call_queue) == 1` in a different way because
            # this dramatically improves performance.
            func, args, kwargs = call_queue[0]
            logger.debug(f"SUBMIT::_apply_func::{self._identity}")
            (
                self.oid,
                self._length_cache,
                self._width_cache,
                self._ip_cache,
            ) = _apply_func.remote(oid, func, *args, **kwargs)
        logger.debug(f"EXIT::Partition.drain_call_queue::{self._identity}")
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        ray.wait([self.oid])

    def __copy__(self):
        """
        Create a copy of this partition.

        Returns
        -------
        PandasOnRayDataframePartition
            A copy of this partition.
        """
        return PandasOnRayDataframePartition(
            self.oid,
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
        logger = get_logger()
        logger.debug(f"ENTER::Partition.mask::{self._identity}")
        new_obj = super().mask(row_labels, col_labels)
        if isinstance(row_labels, slice) and isinstance(
            self._length_cache, ObjectIDType
        ):
            new_obj._length_cache = compute_sliced_len.remote(
                row_labels, self._length_cache
            )
        if isinstance(col_labels, slice) and isinstance(
            self._width_cache, ObjectIDType
        ):
            new_obj._width_cache = compute_sliced_len.remote(
                col_labels, self._width_cache
            )
        logger.debug(f"EXIT::Partition.mask::{self._identity}")
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
        return PandasOnRayDataframePartition(
            ray.put(obj), len(obj.index), len(obj.columns)
        )

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
        return ray.put(func)

    def length(self):
        """
        Get the length of the object wrapped by this partition.

        Returns
        -------
        int
            The length of the object.
        """
        if self._length_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = _get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._length_cache, ObjectIDType):
            self._length_cache = ray.get(self._length_cache)
        return self._length_cache

    def width(self):
        """
        Get the width of the object wrapped by the partition.

        Returns
        -------
        int
            The width of the object.
        """
        if self._width_cache is None:
            if len(self.call_queue):
                self.drain_call_queue()
            else:
                self._length_cache, self._width_cache = _get_index_and_columns.remote(
                    self.oid
                )
        if isinstance(self._width_cache, ObjectIDType):
            self._width_cache = ray.get(self._width_cache)
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
            self._ip_cache = ray.get(self._ip_cache)
        return self._ip_cache


@ray.remote(num_returns=2)
def _get_index_and_columns(df):
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


@ray.remote(num_returns=4)
def _apply_func(partition, func, *args, **kwargs):  # pragma: no cover
    """
    Execute a function on the partition in a worker process.

    Parameters
    ----------
    partition : pandas.DataFrame
        A pandas DataFrame the function needs to be executed on.
    func : callable
        Function that needs to be executed on the partition.
    *args : iterable
        Additional positional arguments to be passed in `func`.
    **kwargs : dict
        Additional keyword arguments to be passed in `func`.

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
def _apply_list_of_funcs(funcs, partition):  # pragma: no cover
    """
    Execute all operations stored in the call queue on the partition in a worker process.

    Parameters
    ----------
    funcs : list
        A call queue that needs to be executed on the partition.
    partition : pandas.DataFrame
        A pandas DataFrame the call queue needs to be executed on.

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
    for func, args, kwargs in funcs:
        func = deserialize(func)
        args = deserialize(args)
        kwargs = deserialize(kwargs)
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
